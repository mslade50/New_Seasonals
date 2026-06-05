import os
import streamlit as st

st.set_page_config(page_title="Multi-Page Dashboard", page_icon=":chart_with_upwards_trend:")


def _is_cloud():
    """True when running on the deployed (Streamlit Community Cloud) app.

    Detection is explicit-cloud, default-local: local dev needs no config and
    can never accidentally hide pages. The deployed app must carry a marker.

    Priority:
      1. IS_LOCAL env var (escape hatch — forces local even on a server).
      2. IS_CLOUD secret set to true in the Community Cloud app's secrets.
      3. Otherwise assume local (no secrets file locally -> st.secrets raises).
    """
    env = os.environ.get("IS_LOCAL")
    if env is not None and env.strip().lower() in ("1", "true", "yes", "on"):
        return False
    try:
        return bool(st.secrets.get("IS_CLOUD", False))
    except Exception:
        # No secrets.toml at all -> running locally.
        return False


# Pages visible everywhere (local + deployed web app).
PUBLIC_PAGES = [
    st.Page("pages/user_input.py", title="Seasonality", default=True),
    st.Page("pages/seasonal_sigs.py", title="Seasonal Signals"),
    st.Page("pages/macro_seasonality.py", title="Macro Seasonality"),
    st.Page("pages/macro_trend.py", title="Macro Trend"),
    st.Page("pages/risk_dashboard_v2.py", title="Risk Dashboard"),
    st.Page("pages/spx_breadth.py", title="SPX Breadth"),
    st.Page("pages/heatmaps.py", title="Heatmaps"),
    st.Page("pages/correlation_heatmaps.py", title="Correlation Heatmaps"),
    st.Page("pages/fx_sizer.py", title="FX Sizer"),
    st.Page("pages/fragility_sizing_lab.py", title="Fragility Sizing Lab"),
    st.Page("pages/exposure_backtester.py", title="Exposure Backtester"),
    st.Page("pages/rotation_backtester.py", title="Rotation Backtester"),
    st.Page("pages/signal_backtester.py", title="Signal Backtester"),
]

# Pages shown only on the local machine, hidden from the deployed web app.
LOCAL_ONLY_PAGES = [
    st.Page("pages/backtester.py", title="Backtester"),
    st.Page("pages/strat_backtester.py", title="Strat Backtester"),
]

pages = PUBLIC_PAGES if _is_cloud() else PUBLIC_PAGES + LOCAL_ONLY_PAGES

st.navigation(pages).run()
