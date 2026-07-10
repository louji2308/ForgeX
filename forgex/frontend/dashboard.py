from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from forgex.config import load_settings
from forgex.errors import ModelSchemaError
from forgex.explain.shap_explainer import ShapExplainer
from forgex.models.hazard import load_model_artifact
from forgex.logging_setup import get_logger

logger = get_logger(__name__)

API_BASE = "http://localhost:8002"

DESIGN_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    :root {
        color-scheme: light;
        --blue: #2f6bff;
        --blue-strong: #1d4ed8;
        --blue-dark: #1e40af;
        --blue-soft: #eaf0ff;
        --ink: #0f1e3d;
        --ink-2: #1b2a4a;
        --muted: #64748b;
        --muted-2: #8b97ac;
        --line: #e6eaf2;
        --bg: #f4f6fb;
        --card: #ffffff;
        --green: #16b981;
        --red: #ef4444;
        --amber: #f59e0b;
        --radius: 18px;
        --radius-sm: 12px;
        --shadow: 0 10px 30px rgba(30,64,175,0.08);
        --shadow-lg: 0 24px 60px rgba(30,64,175,0.14);
        --font-head: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
        --font-body: 'Inter', system-ui, -apple-system, sans-serif;
    }

    .stApp {
        background: var(--bg) !important;
        background-image:
            radial-gradient(1200px 600px at 80% -5%, #eaf1ff 0%, rgba(234, 241, 255, 0) 55%),
            radial-gradient(900px 500px at 0% 100%, #eef2fb 0%, rgba(238, 242, 251, 0) 60%) !important;
        color: var(--ink) !important;
    }

    .stApp > header { display: none !important; }

    html, body, .stApp, .main, .block-container, .stMarkdown, p, span, div, li {
        color: var(--ink) !important;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: var(--font-head) !important;
        color: var(--ink) !important;
    }

    .st-emotion-cache-1avcm0n, .st-emotion-cache-18ni7ap, .st-emotion-cache-1qg05tj {
        color: var(--ink) !important;
    }

    .stButton button {
        font-family: var(--font-head);
        font-weight: 700;
        border-radius: 999px;
        background: var(--blue);
        color: white !important;
        border: none;
        padding: 8px 24px;
        transition: background 0.15s, box-shadow 0.15s;
    }
    .stButton button:hover { background: var(--blue-strong); }

    .stTextInput label, .stSelectbox label, .stTextArea label {
        color: var(--ink) !important;
        font-family: var(--font-head);
        font-weight: 600;
    }

    .stTextInput input, .stTextInput textarea {
        border-radius: var(--radius-sm);
        border: 1px solid var(--line);
        font-family: var(--font-body);
        background: var(--card);
        color: var(--ink) !important;
    }
    .stTextInput input:focus { border-color: var(--blue); box-shadow: 0 0 0 3px var(--blue-soft); }

    .stSelectbox div[data-baseweb="select"] {
        border-radius: var(--radius-sm);
        border-color: var(--line);
        color: var(--ink) !important;
    }
    .stSelectbox div[data-baseweb="select"] * {
        color: var(--ink) !important;
    }

    .stMetric {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: var(--radius);
        padding: 20px;
        box-shadow: var(--shadow);
    }
    div[data-testid="stMetricValue"] {
        font-family: var(--font-head);
        font-weight: 800;
        color: var(--ink) !important;
    }
    div[data-testid="stMetricLabel"] {
        font-family: var(--font-body);
        color: var(--muted) !important;
    }

    .stDataFrame {
        background: var(--card);
        border-radius: var(--radius);
        border: 1px solid var(--line);
    }
    .stDataFrame * {
        color: var(--ink) !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: var(--radius);
        padding: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: var(--font-head);
        font-weight: 700;
        border-radius: var(--radius-sm);
        padding: 8px 20px;
        color: var(--muted) !important;
    }
    .stTabs [aria-selected="true"] {
        background: var(--blue) !important;
        color: white !important;
    }

    .stSpinner > div { border-color: var(--blue-soft) var(--blue-soft) var(--blue) !important; }

    div[data-testid="stChatMessage"] { border-radius: var(--radius-sm); }
    div[data-testid="stChatMessageContent"] {
        font-family: var(--font-body);
        color: var(--ink) !important;
    }

    section[data-testid="stSidebar"] {
        background: var(--card);
        border-right: 1px solid var(--line);
    }
    section[data-testid="stSidebar"] * {
        color: var(--ink) !important;
    }

    .stAlert {
        color: var(--ink) !important;
    }

    .topbar-brand {
        display: flex; align-items: center; gap: 10px;
        font-family: var(--font-head); font-weight: 800; font-size: 1.5rem;
        letter-spacing: -0.02em; color: var(--ink); padding: 10px 0;
    }
    .topbar-brand .brand-x { color: var(--blue); }

    .card {
        background: var(--card); border: 1px solid var(--line);
        border-radius: var(--radius); box-shadow: var(--shadow);
        padding: 24px; margin-bottom: 20px;
    }

    .section-title {
        font-family: var(--font-head); font-weight: 800; font-size: 1.2rem;
        color: var(--ink); margin: 0 0 8px 0;
    }
    .section-sub {
        color: var(--muted); font-size: 0.9rem; margin-bottom: 18px;
    }

    .admin-badge {
        display: inline-flex; align-items: center; gap: 6px;
        background: var(--blue-soft); color: var(--blue-strong) !important;
        font-weight: 700; font-size: 0.74rem; letter-spacing: 0.13em;
        padding: 6px 14px; border-radius: 999px; text-transform: uppercase;
    }

    .stChatInputContainer { border-radius: var(--radius-sm); border: 1px solid var(--line); }

    @media (max-width: 768px) {
        .stTabs [data-baseweb="tab"] { font-size: 0.85rem; padding: 6px 14px; }
    }
</style>
"""

st.set_page_config(
    page_title="ForgeX — Admin Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(DESIGN_CSS, unsafe_allow_html=True)


def restore_session():
    """Restore admin session from query param token. Returns True if logged in."""

    if "admin_token" in st.session_state:
        return True

    token = st.query_params.get("token", "")
    if not token:
        st.session_state._auth_error = "No token found in URL. Sign in from the main page."
        return False

    try:
        with httpx.Client() as client:
            resp = client.get(f"{API_BASE}/auth/me", headers={"Authorization": f"Bearer {token}"}, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("role") == "admin":
                    st.session_state.admin_token = token
                    st.session_state.admin_user = data
                    st.session_state.authenticated = True
                    return True
                st.session_state._auth_error = f"Expected admin role, got '{data.get('role')}'"
            else:
                st.session_state._auth_error = f"API returned status {resp.status_code} — token may be invalid or expired"
    except httpx.ConnectError:
        st.session_state._auth_error = f"Cannot connect to API at {API_BASE}/auth/me — is the API server running?"
    except Exception as e:
        st.session_state._auth_error = f"Error: {e}"
        logger.warning(f"Failed to validate admin token: {e}")

    return False


def admin_logout():
    if st.session_state.get("admin_token"):
        try:
            with httpx.Client() as client:
                client.post(f"{API_BASE}/auth/logout", headers={"Authorization": f"Bearer {st.session_state.admin_token}"})
        except Exception:
            pass
    for key in ["admin_token", "admin_user", "authenticated"]:
        st.session_state.pop(key, None)
    st.markdown("""<script>window.location.href='http://localhost:3000';</script>""", unsafe_allow_html=True)
    st.rerun()


@st.cache_resource
def load_artifacts():
    settings = load_settings()
    try:
        model = load_model_artifact(str(settings.artifacts_dir / "hazard_model.pkl"))
        explainer = ShapExplainer.load(settings.artifacts_dir / "shap_explainer.pkl")
        person_period = pd.read_parquet(settings.data_dir / "processed" / "person_period.parquet")
        features = pd.read_parquet(settings.data_dir / "processed" / "feature_table.parquet")
        tenants = pd.read_parquet(settings.data_dir / "raw" / "tenants.parquet")
        return model, explainer, person_period, features, tenants
    except (FileNotFoundError, ModelSchemaError) as e:
        st.warning(f"Model artifacts not found ({e}) — showing user management only")
        return None, None, None, None, None


def compute_risk_table(model, person_period):
    test = person_period[person_period["fold"] == "test"].copy()
    if test.empty:
        test = person_period.copy()
    latest = test.loc[test.groupby("tenant_id")["month_of_lease"].idxmax()].copy()
    if hasattr(model.model, "predict_proba"):
        probs = model.model.predict_proba(latest[model.feature_names].fillna(0))[:, 1]
    else:
        probs = model.model.predict(latest[model.feature_names].fillna(0))
    if hasattr(model.model, "_calibrator"):
        probs = model.model._calibrator.predict(probs)
    latest["risk_pct"] = (probs * 100).round(1)
    latest["risk_band"] = pd.cut(latest["risk_pct"], bins=[0, 20, 40, 60, 80, 100], labels=["Very Low", "Low", "Moderate", "High", "Critical"])
    return latest[["tenant_id", "risk_pct", "risk_band"]]


def render_portfolio(risk_table):
    color_map = {"Critical": "#dc3545", "High": "#fd7e14", "Moderate": "#ffc107", "Low": "#28a745", "Very Low": "#20c997"}

    st.markdown('<div class="section-title">Portfolio Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Risk distribution across all tenants in your portfolio.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        sorted_table = risk_table.sort_values("risk_pct", ascending=False).head(50)
        fig = px.bar(sorted_table, x="tenant_id", y="risk_pct", color="risk_band", color_discrete_map=color_map,
                     labels={"risk_pct": "Churn Risk (%)", "tenant_id": "Tenant"}, title="Top 50 At-Risk Tenants", height=400)
        fig.update_layout(
            paper_bgcolor="white", plot_bgcolor="white",
            font=dict(family="Inter", color="#0f1e3d"),
            title_font=dict(family="Plus Jakarta Sans", size=16, color="#0f1e3d"),
            legend=dict(font=dict(family="Inter", color="#0f1e3d")),
            xaxis=dict(color="#0f1e3d", gridcolor="#e6eaf2"),
            yaxis=dict(color="#0f1e3d", gridcolor="#e6eaf2"),
            showlegend=True, xaxis_tickangle=-45,
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        band_counts = risk_table["risk_band"].value_counts().reindex(["Very Low", "Low", "Moderate", "High", "Critical"], fill_value=0)
        fig2 = px.pie(values=band_counts.values, names=band_counts.index, color=band_counts.index,
                      color_discrete_map=color_map, hole=0.4)
        fig2.update_layout(
            paper_bgcolor="white",
            font=dict(family="Inter", color="#0f1e3d"),
            title=dict(text="Risk Breakdown", font=dict(family="Plus Jakarta Sans", size=16, color="#0f1e3d")),
            margin=dict(t=0, b=0, l=0, r=0), height=350,
        )
        fig2.update_traces(textfont=dict(color="#0f1e3d"))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div style="margin-top: 8px"></div>', unsafe_allow_html=True)
    sort_by = st.selectbox("Sort tenants by", ["risk_pct descending", "risk_pct ascending", "tenant_id"])
    display = risk_table.copy()
    if sort_by == "risk_pct descending":
        display = display.sort_values("risk_pct", ascending=False)
    elif sort_by == "risk_pct ascending":
        display = display.sort_values("risk_pct", ascending=True)
    else:
        display = display.sort_values("tenant_id")
    st.dataframe(display, column_config={"risk_pct": st.column_config.ProgressColumn("Churn Risk", min_value=0, max_value=100, format="%.1f%%")},
                 use_container_width=True, hide_index=True)


def render_users():
    st.markdown('<div class="section-title">Registered Users</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">All users registered in the system.</div>', unsafe_allow_html=True)

    if not st.session_state.get("admin_token"):
        st.warning("Not authenticated")
        return

    users = []
    try:
        with httpx.Client() as client:
            resp = client.get(f"{API_BASE}/admin/users", headers={"Authorization": f"Bearer {st.session_state.admin_token}"})
            if resp.status_code == 200:
                users = resp.json()
            else:
                st.error(f"Failed to fetch users: {resp.text}")
    except Exception as e:
        st.error(f"Connection error: {e}")

    if users:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Users", len(users))
        landlords = sum(1 for u in users if u.get("role") == "landlord")
        tenants_c = sum(1 for u in users if u.get("role") == "tenant")
        c2.metric("Landlords", landlords)
        c3.metric("Tenants", tenants_c)

        df = pd.DataFrame(users)
        if "created_at" in df.columns and not df.empty:
            df["created_at"] = pd.to_datetime(df["created_at"], unit="s")
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No users registered yet")

    st.markdown('<div style="margin-top: 28px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">System Health</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">API server and model status.</div>', unsafe_allow_html=True)
    try:
        resp = httpx.get(f"{API_BASE}/health")
        if resp.status_code == 200:
            health = resp.json()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Tenants Indexed", health.get("tenants_indexed", 0))
            c2.metric("Model Loaded", "Yes" if health.get("is_model_loaded") else "No")
            c3.metric("Explainer Loaded", "Yes" if health.get("is_explainer_loaded") else "No")
            c4.metric("CATE Loaded", "Yes" if health.get("is_cate_loaded") else "No")
    except Exception as e:
        st.error(f"Cannot reach API: {e}")


def render_ai_search():
    st.markdown('<div class="section-title">AI Smart Search</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Ask questions about your portfolio in natural language.</div>', unsafe_allow_html=True)

    if "ai_search_history" not in st.session_state:
        st.session_state.ai_search_history = []

    for msg in st.session_state.ai_search_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input('e.g. "Which tenants are at high risk of churning?"')
    if question:
        st.session_state.ai_search_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    with httpx.Client(timeout=60) as client:
                        resp = client.post(
                            f"{API_BASE}/ai-search",
                            json={"question": question, "context": ""},
                        )
                        data = resp.json()
                        answer = data.get("answer", "No answer received")
                except Exception:
                    answer = "Could not reach the API server."
            st.markdown(answer)
            st.session_state.ai_search_history.append({"role": "assistant", "content": answer})


def main():
    if not restore_session():
        err = st.session_state.pop("_auth_error", None)
        st.markdown(f"""
        <style>
        * {{ color-scheme: light; }}
        #admin-please-signin {{
            position: fixed; inset: 0; z-index: 99999;
            display: flex; align-items: center; justify-content: center;
            background: #f4f6fb; font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background-image: radial-gradient(1200px 600px at 80% -5%, #eaf1ff 0%, rgba(234,241,255,0) 55%),
                              radial-gradient(900px 500px at 0% 100%, #eef2fb 0%, rgba(238,242,251,0) 60%);
        }}
        #admin-please-signin .card {{
            background: #fff; border: 1px solid #e6eaf2; border-radius: 18px;
            box-shadow: 0 24px 60px rgba(30,64,175,0.14);
            padding: 40px; width: 100%; max-width: 420px; text-align: center;
        }}
        #admin-please-signin svg {{
            filter: drop-shadow(0 4px 8px rgba(47,107,255,0.25));
        }}
        #admin-please-signin h1 {{
            font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 800;
            font-size: 1.35rem; color: #0f1e3d; margin: 16px 0 8px;
        }}
        #admin-please-signin p {{ color: #64748b; font-size: 0.92rem; margin-bottom: 24px; }}
        #admin-please-signin .auth-detail {{
            color: #dc2626; font-size: 0.82rem; margin-bottom: 20px; padding: 10px 14px;
            background: #fef2f2; border-radius: 10px; border: 1px solid #fecaca;
            font-weight: 500;
        }}
        #admin-please-signin a {{
            display: inline-flex; align-items: center; gap: 8px;
            border-radius: 999px; background: #2f6bff; color: #fff; border: none;
            padding: 12px 32px; font-weight: 700; font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 0.9rem; text-decoration: none;
            box-shadow: 0 12px 26px rgba(47,107,255,0.35);
            transition: background 0.15s, box-shadow 0.15s;
        }}
        #admin-please-signin a:hover {{ background: #1d4ed8; box-shadow: 0 16px 32px rgba(47,107,255,0.42); }}
        </style>
        <div id="admin-please-signin">
        <div class="card">
            <svg width="40" height="40" viewBox="0 0 36 36" fill="none">
                <rect width="36" height="36" rx="10" fill="#2f6bff"/>
                <path d="M10 26V14l8-6 8 6v12H18v-6h-4v6h-4z" fill="#fff"/>
            </svg>
            <h1>Admin Dashboard</h1>
            <p>Sign in with an admin account from the main page to access the dashboard.</p>
            {f'<p class="auth-detail">{err}</p>' if err else ''}
            <a href="http://localhost:3000" target="_top">Go to Sign In →</a>
        </div>
        </div>
        """, unsafe_allow_html=True)
        return

    handle = st.session_state.admin_user.get("handle", "")

    model, explainer, person_period, features, tenants = load_artifacts()
    has_model = model is not None and person_period is not None

    with st.sidebar:
        st.markdown(
            '<div class="topbar-brand">'
            '<svg width="30" height="30" viewBox="0 0 36 36" fill="none"><rect width="36" height="36" rx="10" fill="#2f6bff"/><path d="M10 26V14l8-6 8 6v12H18v-6h-4v6h-4z" fill="#fff"/></svg>'
            '<span>Forge<span class="brand-x">X</span></span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown(f'<span class="admin-badge">Admin</span>', unsafe_allow_html=True)
        st.markdown(f"**@{handle}**")
        st.markdown("---")

        if has_model:
            risk_table = compute_risk_table(model, person_period)
            st.metric("Total Tenants", len(risk_table))
            st.metric("High Risk (60%+)", len(risk_table[risk_table["risk_pct"] >= 60]))
            st.metric("Critical (80%+)", len(risk_table[risk_table["risk_pct"] >= 80]))
            st.markdown("---")

        if st.button("Sign Out", type="primary", use_container_width=True):
            admin_logout()

    if has_model:
        risk_table = compute_risk_table(model, person_period)
        tab1, tab2, tab3 = st.tabs([" Portfolio Overview", " User Management", " AI Smart Search"])
        with tab1:
            render_portfolio(risk_table)
        with tab2:
            render_users()
        with tab3:
            render_ai_search()
    else:
        tab1, tab2 = st.tabs([" User Management", " AI Smart Search"])
        with tab1:
            render_users()
        with tab2:
            render_ai_search()


if __name__ == "__main__":
    main()
