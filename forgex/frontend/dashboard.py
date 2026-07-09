from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from forgex.config import load_settings
from forgex.errors import ModelSchemaError
from forgex.explain.shap_explainer import ShapExplainer, top_shap_drivers
from forgex.explain.narrative import generate_narrative
from forgex.models.hazard import load_model_artifact
from forgex.models.baseline import hazard_to_survival
from forgex.logging_setup import get_logger

logger = get_logger(__name__)

st.set_page_config(
    page_title="ForgeX — Portfolio Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_artifacts():
    settings = load_settings()
    try:
        model = load_model_artifact(
            str(settings.artifacts_dir / "hazard_model.pkl")
        )
        explainer = ShapExplainer.load(settings.artifacts_dir / "shap_explainer.pkl")
        person_period = pd.read_parquet(
            settings.data_dir / "processed" / "person_period.parquet"
        )
        features = pd.read_parquet(
            settings.data_dir / "processed" / "feature_table.parquet"
        )
        tenants = pd.read_parquet(
            settings.data_dir / "raw" / "tenants.parquet"
        )
    except (FileNotFoundError, ModelSchemaError) as e:
        st.error(
            f"Could not load model artifacts: {e}\n\n"
            f"Run `python -m forgex.models.hazard train` first."
        )
        st.stop()
        return None, None, None, None, None
    return model, explainer, person_period, features, tenants


def compute_risk_table(
    model, person_period: pd.DataFrame, threshold_pct: float = 50.0
) -> pd.DataFrame:
    test = person_period[person_period["fold"] == "test"].copy()
    if test.empty:
        test = person_period.copy()

    latest = test.loc[
        test.groupby("tenant_id")["month_of_lease"].idxmax()
    ].copy()

    hazards = model.model.predict(latest[model.feature_names].fillna(0))
    if hasattr(model.model, "_calibrator"):
        hazards = model.model._calibrator.predict(hazards)

    latest["risk_pct"] = (hazards * 100).round(1)
    latest["risk_band"] = pd.cut(
        latest["risk_pct"],
        bins=[0, 20, 40, 60, 80, 100],
        labels=["Very Low", "Low", "Moderate", "High", "Critical"],
    )

    return latest[["tenant_id", "unit_id", "risk_pct", "risk_band", "month_of_lease"]]


def render_portfolio_heatmap(risk_table: pd.DataFrame):
    if risk_table.empty:
        st.warning("No tenants found in the current portfolio snapshot.")
        return

    missing = {"tenant_id", "risk_pct", "risk_band"} - set(risk_table.columns)
    if missing:
        st.error(f"risk_table is missing columns: {missing}")
        return

    risk_table = risk_table.sort_values("risk_pct", ascending=False)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Portfolio Risk Overview")

        color_map = {
            "Critical": "#dc3545",
            "High": "#fd7e14",
            "Moderate": "#ffc107",
            "Low": "#28a745",
            "Very Low": "#20c997",
        }
        risk_table["color"] = risk_table["risk_band"].map(color_map)

        fig = px.bar(
            risk_table.head(50),
            x="tenant_id",
            y="risk_pct",
            color="risk_band",
            color_discrete_map=color_map,
            labels={"risk_pct": "90-Day Churn Risk (%)", "tenant_id": "Tenant"},
            title="Top 50 At-Risk Tenants",
            height=400,
        )
        fig.update_layout(showlegend=True, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Risk Distribution")
        band_counts = risk_table["risk_band"].value_counts().reindex(
            ["Very Low", "Low", "Moderate", "High", "Critical"], fill_value=0
        )
        fig2 = px.pie(
            values=band_counts.values,
            names=band_counts.index,
            color=band_counts.index,
            color_discrete_map=color_map,
            hole=0.4,
        )
        fig2.update_layout(height=350, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("All Tenants")
    st.dataframe(
        risk_table.drop(columns=["color"]),
        column_config={
            "risk_pct": st.column_config.ProgressColumn(
                "90-day churn risk",
                min_value=0,
                max_value=100,
                format="%.1f%%",
            ),
        },
        use_container_width=True,
        hide_index=True,
    )


def render_tenant_detail(
    tenant_id: str,
    model,
    explainer,
    person_period: pd.DataFrame,
    tenants: pd.DataFrame,
):
    st.subheader(f"Tenant Detail: {tenant_id}")

    tenant_info = tenants[tenants["tenant_id"] == tenant_id]
    if not tenant_info.empty:
        with st.expander("Tenant Profile", expanded=False):
            st.json(tenant_info.iloc[0].to_dict())

    tenant_rows = person_period[person_period["tenant_id"] == tenant_id].sort_values(
        "month_of_lease"
    )

    if tenant_rows.empty:
        st.warning(f"No lease data found for tenant {tenant_id}")
        return

    # Survival curve
    hazards = model.model.predict(tenant_rows[model.feature_names].fillna(0))
    if hasattr(model.model, "_calibrator"):
        hazards = model.model._calibrator.predict(hazards)

    survival_df = hazard_to_survival(
        pd.Series(hazards),
        tenant_rows["tenant_id"],
        tenant_rows["month_of_lease"],
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=survival_df["month"],
            y=survival_df["survival_prob"] * 100,
            mode="lines+markers",
            name="Survival Probability",
            line=dict(color="#28a745", width=3),
        ))
        fig.add_trace(go.Scatter(
            x=survival_df["month"],
            y=survival_df["cum_churn_prob"] * 100,
            mode="lines+markers",
            name="Cumulative Churn Probability",
            line=dict(color="#dc3545", width=3),
        ))
        fig.update_layout(
            title="Survival & Churn Curve",
            xaxis_title="Month of Lease",
            yaxis_title="Probability (%)",
            height=350,
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # SHAP driver bar
        latest_row = tenant_rows.iloc[-1:]

        try:
            shap_values = explainer.explain(latest_row)
            shap_series = pd.Series(
                shap_values[0], index=model.feature_names
            )
            feature_row = latest_row[model.feature_names].iloc[0]
            drivers = top_shap_drivers(shap_series, feature_row, k=5)

            driver_df = pd.DataFrame(drivers)
            driver_df["abs_shap"] = driver_df["shap_value"].abs()
            driver_df = driver_df.sort_values("abs_shap")

            fig2 = go.Figure()
            colors = [
                "#dc3545" if d["direction"] == "increases_risk" else "#28a745"
                for d in drivers
            ]
            fig2.add_trace(go.Bar(
                y=[d["label"] for d in reversed(drivers)],
                x=[d["shap_value"] for d in reversed(drivers)],
                orientation="h",
                marker_color=list(reversed(colors)),
            ))
            fig2.update_layout(
                title="Top Risk Drivers (SHAP)",
                xaxis_title="SHAP Value",
                yaxis_title=None,
                height=300,
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Narrative
            risk_pct = float(hazards[-1] * 100) if len(hazards) > 0 else 0.0
            narrative = generate_narrative(
                tenant_id, risk_pct, drivers, llm_client=None,
            )
            st.info(f"**Narrative:** {narrative['narrative']}")
        except Exception as e:
            st.warning(f"Could not generate explanation: {e}")


def main():
    st.title(" ForgeX — Vital-Signs Monitoring")
    st.markdown(
        "Portfolio intelligence for the landlord-tenant relationship. "
        "Diagnose risk, understand causes, and optimize interventions."
    )

    model, explainer, person_period, features, tenants = load_artifacts()
    if model is None:
        return

    risk_table = compute_risk_table(model, person_period)

    tab1, tab2 = st.tabs([" Portfolio Heatmap", " Tenant Detail"])

    with tab1:
        render_portfolio_heatmap(risk_table)

    with tab2:
        tenant_list = risk_table.sort_values("risk_pct", ascending=False)[
            "tenant_id"
        ].tolist()
        selected_tenant = st.selectbox(
            "Select a tenant to inspect:",
            tenant_list,
            index=0,
        )
        if selected_tenant:
            render_tenant_detail(
                selected_tenant,
                model,
                explainer,
                person_period,
                tenants,
            )

    st.sidebar.title("ForgeX")
    st.sidebar.markdown("---")
    st.sidebar.metric(
        "Total Tenants",
        len(risk_table),
    )
    st.sidebar.metric(
        "High Risk (60%+)",
        len(risk_table[risk_table["risk_pct"] >= 60]),
        delta=None,
        delta_color="inverse",
    )
    st.sidebar.metric(
        "Critical (80%+)",
        len(risk_table[risk_table["risk_pct"] >= 80]),
        delta=None,
        delta_color="inverse",
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Built for the hackathon — Phase 7+ roadmap includes causal uplift, what-if simulation, and fairness audit.")


if __name__ == "__main__":
    main()
