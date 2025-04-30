import streamlit as st
from pages import user_input_2, variance_sim, trade_dashboard)

st.set_page_config(page_title="Multi-Page Dashboard", page_icon=":chart_with_upwards_trend:")

def variance_sim_page():
    st.sidebar.markdown("# Variance Simulation")
    variance_sim.variance_sim_app()
def trade_dashboard_page():
    st.sidebar.markdown("# Trade Dashboard")
    trade_dashboard.trade_dashboard_app()
def user_input_2_page():
    st.sidebar.markdown("# User Input 2")
    user_input_2.user_input_2_app()


page_names_to_funcs = {
    "Variance Simulation": variance_sim_page,
    "Trade Dashboard": trade_dashboard_page
    "User Input": user_input_2_page,
}
