from __future__ import annotations

import sys
from pathlib import Path

import httpx
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

API_BASE = "http://localhost:8000"

LIGHT_CSS = """
<style>
    .stApp { background: #f4f6fb; }
    h1, h2, h3 { font-family: 'Plus Jakarta Sans', sans-serif !important; color: #0f1e3d !important; }
    .stButton button { font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; border-radius: 999px; background: #2f6bff; color: white; border: none; padding: 8px 24px; }
    .stButton button:hover { background: #1d4ed8; }
    .stTextInput input { border-radius: 12px; border: 1px solid #e6eaf2; font-family: 'Inter', sans-serif; }
    .stSelectbox div[data-baseweb="select"] { border-radius: 12px; border-color: #e6eaf2; }
    .stMetric { background: white; border-radius: 18px; padding: 16px; box-shadow: 0 10px 30px rgba(30,64,175,0.08); }
    div[data-testid="stMetricValue"] { font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 800; color: #0f1e3d; }
    div[data-testid="stMetricLabel"] { font-family: 'Inter', sans-serif; color: #64748b; }
    .stDataFrame { background: white; border-radius: 18px; border: 1px solid #e6eaf2; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; border-radius: 12px; padding: 8px 20px; }
</style>
"""

st.set_page_config(
    page_title="ForgeX — Admin Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(LIGHT_CSS, unsafe_allow_html=True)


def admin_login():
    token = st.query_params.get("token", "")
    error_msg = st.query_params.get("error", "")
    if token:
        for k in list(st.query_params.keys()):
            try:
                del st.query_params[k]
            except Exception:
                pass
        try:
            with httpx.Client() as client:
                resp = client.get(f"{API_BASE}/auth/me", headers={"Authorization": f"Bearer {token}"}, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("role") == "admin":
                        st.session_state.admin_token = token
                        st.session_state.admin_user = data
                        st.rerun()
                        return
        except Exception:
            pass
        try:
            st.query_params["error"] = "Session expired - sign in again"
        except Exception:
            pass

    err_block = ""
    if error_msg:
        err_block = '<div id="login-err" style="background:#fdecec;border:1px solid #f7b9b9;border-radius:12px;padding:12px 14px;color:#b42318;font-size:0.9rem;margin-bottom:16px;text-align:center">' + error_msg + '</div>'

    html = """<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body { height: 100%; }
#admin-shell {
    position: fixed; inset: 0; z-index: 99999;
    display: flex; align-items: center; justify-content: center;
    background: #f4f6fb;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}
.card {
    background: #fff; border: 1px solid #e6eaf2; border-radius: 18px;
    box-shadow: 0 24px 60px rgba(30,64,175,0.14);
    padding: 28px 28px 36px; width: 100%; max-width: 400px;
}
.card .logo { text-align: center; margin-bottom: 12px; }
.card .logo span { font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 800; font-size: 1.3rem; letter-spacing: -0.02em; color: #0f1e3d; vertical-align: middle; margin-left: 8px; }
.card .logo .x { color: #2f6bff; }
.card h1 { font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 800; font-size: 1.3rem; color: #0f1e3d; margin: 0; text-align: center; }
.card .sub { color: #64748b; font-size: 0.82rem; margin: 4px 0 18px; text-align: center; }
.input-wrap { margin-bottom: 14px; }
.input-wrap label { font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; font-size: 0.78rem; color: #1b2a4a; display: block; margin-bottom: 6px; }
.input-wrap input {
    width: 100%; padding: 10px 12px; border: 1px solid #e6eaf2; border-radius: 10px;
    font-family: 'Inter', sans-serif; font-size: 0.85rem; color: #0f1e3d; background: #fff; outline: none;
    transition: border-color 0.15s, box-shadow 0.15s;
}
.input-wrap input:focus { border-color: #2f6bff; box-shadow: 0 0 0 3px #eaf0ff; }
.input-wrap input::placeholder { color: #94a3b8; font-size: 0.8rem; }
.btn-wrap { text-align: center; }
.btn-wrap button {
    border-radius: 999px; background: #2f6bff; color: #fff; border: none; padding: 10px 36px;
    font-weight: 700; font-family: 'Plus Jakarta Sans', sans-serif; font-size: 0.88rem; min-width: 160px;
    box-shadow: 0 12px 26px rgba(47,107,255,0.35); cursor: pointer; transition: background 0.15s, box-shadow 0.15s;
}
.btn-wrap button:hover { background: #1d4ed8; box-shadow: 0 16px 32px rgba(47,107,255,0.42); }
.btn-wrap button:disabled { opacity: 0.6; cursor: not-allowed; }
.login-err {
    background: #fdecec; border: 1px solid #f7b9b9; border-radius: 12px;
    padding: 12px 14px; color: #b42318; font-size: 0.9rem; margin-bottom: 16px; text-align: center;
}
</style>
<div id="admin-shell">
<div class="card">
<div class="logo">
<svg width="32" height="32" viewBox="0 0 36 36" fill="none" style="display:inline-block;vertical-align:middle"><rect width="36" height="36" rx="10" fill="#2f6bff"/><path d="M10 26V14l8-6 8 6v12H18v-6h-4v6h-4z" fill="#fff"/></svg>
<span>Forge<span class="x">X</span></span>
</div>
<h1>Welcome back</h1>
<p class="sub">Sign in to your admin dashboard</p>
""" + err_block + """<div id="login-err" class="login-err" style="display:none"></div>
<form id="login-form" onsubmit="return false">
<div class="input-wrap"><label>Handle</label><input type="text" id="login-handle" placeholder="your_admin_handle" required></div>
<div class="input-wrap"><label>Password</label><input type="password" id="login-password" placeholder="\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022" required></div>
<div class="btn-wrap"><button type="submit" id="login-submit">Sign In</button></div>
</form>
</div>
</div>
<script>
var BASE = '""" + API_BASE + """';
document.getElementById('login-form').onsubmit = function() {
    var btn = document.getElementById('login-submit');
    var errEl = document.getElementById('login-err');
    var handle = document.getElementById('login-handle').value.trim();
    var password = document.getElementById('login-password').value;
    if (!handle || !password) {
        errEl.style.display = 'block';
        errEl.textContent = 'Handle and password are required';
        return;
    }
    btn.disabled = true;
    btn.textContent = 'Signing in...';
    errEl.style.display = 'none';
    fetch(BASE + '/auth/login', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({handle: handle, password: password})
    }).then(function(r) { return r.json(); }).then(function(data) {
        if (data.role !== 'admin') {
            errEl.style.display = 'block';
            errEl.textContent = data.detail || 'Access denied';
            btn.disabled = false;
            btn.textContent = 'Sign In';
            return;
        }
        window.location.href = window.location.pathname + '?token=' + encodeURIComponent(data.token);
    }).catch(function() {
        errEl.style.display = 'block';
        errEl.textContent = 'Cannot connect to API at ' + BASE + ' - is the server running?';
        btn.disabled = false;
        btn.textContent = 'Sign In';
    });
};
</script>"""

    st.markdown(html, unsafe_allow_html=True)


def admin_logout():
    if st.sidebar.button("Sign Out", type="primary"):
        if st.session_state.get("admin_token"):
            try:
                with httpx.Client() as client:
                    client.post(f"{API_BASE}/auth/logout", headers={"Authorization": f"Bearer {st.session_state.admin_token}"})
            except Exception:
                pass
        for key in ["admin_token", "admin_user", "authenticated"]:
            st.session_state.pop(key, None)
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

    col1, col2 = st.columns([2, 1])
    with col1:
        sorted_table = risk_table.sort_values("risk_pct", ascending=False).head(50)
        fig = px.bar(sorted_table, x="tenant_id", y="risk_pct", color="risk_band", color_discrete_map=color_map,
                     labels={"risk_pct": "Churn Risk (%)", "tenant_id": "Tenant"}, title="Top 50 At-Risk Tenants", height=400)
        fig.update_layout(showlegend=True, xaxis_tickangle=-45, plot_bgcolor="white", font_family="Inter")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        band_counts = risk_table["risk_band"].value_counts().reindex(["Very Low", "Low", "Moderate", "High", "Critical"], fill_value=0)
        fig2 = px.pie(values=band_counts.values, names=band_counts.index, color=band_counts.index,
                      color_discrete_map=color_map, hole=0.4)
        fig2.update_layout(height=350, margin=dict(t=0, b=0, l=0, r=0), font_family="Inter")
        st.plotly_chart(fig2, use_container_width=True)

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
    st.subheader("Registered Users")
    if not st.session_state.get("admin_token"):
        st.warning("Not authenticated")
        return
    try:
        with httpx.Client() as client:
            resp = client.get(f"{API_BASE}/admin/users", headers={"Authorization": f"Bearer {st.session_state.admin_token}"})
            if resp.status_code == 200:
                users = resp.json()
                if users:
                    df = pd.DataFrame(users)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    st.metric("Total Users", len(users))
                else:
                    st.info("No users registered yet")
            else:
                st.error(f"Failed to fetch users: {resp.text}")
    except Exception as e:
        st.error(f"Connection error: {e}")

    st.subheader("System Health")
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
    st.subheader("AI Smart Search")
    st.caption("Ask questions about your portfolio in natural language.")

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
    if "admin_token" not in st.session_state:
        admin_login()
        return

    admin_logout()
    st.sidebar.markdown(f"**Admin:** @{st.session_state.admin_user.get('handle', '')}")
    st.sidebar.markdown("---")

    model, explainer, person_period, features, tenants = load_artifacts()
    has_model = model is not None and person_period is not None

    if has_model:
        risk_table = compute_risk_table(model, person_period)
        tab1, tab2, tab3 = st.tabs([" Portfolio Overview", " User Management", " AI Smart Search"])
        with tab1:
            render_portfolio(risk_table)
        with tab2:
            render_users()
        with tab3:
            render_ai_search()
        st.sidebar.markdown("---")
        st.sidebar.metric("Total Tenants", len(risk_table))
        st.sidebar.metric("High Risk (60%+)", len(risk_table[risk_table["risk_pct"] >= 60]))
        st.sidebar.metric("Critical (80%+)", len(risk_table[risk_table["risk_pct"] >= 80]))
    else:
        tab1, tab2 = st.tabs([" User Management", " AI Smart Search"])
        with tab1:
            render_users()
        with tab2:
            render_ai_search()


if __name__ == "__main__":
    main()
