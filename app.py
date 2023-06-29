import streamlit as st
from pages import bull_signals, positions, bullish_seasonals, bearish_seasonals, user_input, filtered_bull, filtered_bear

st.set_page_config(page_title="Multi-Page Dashboard", page_icon=":chart_with_upwards_trend:")

def bull_signals_page():
    st.sidebar.markdown("# Daily Signals")
    bull_signals.daily_signals_app()

def positions_page():
    st.sidebar.markdown("# Positions")
    positions.positions_app()

def bullish_seasonals_page():
    st.sidebar.markdown("# Bullish Seasonals")
    bullish_seasonals.bullish_seasonals_app()

def bearish_seasonals_page():
    st.sidebar.markdown("# Bearish Seasonals")
    bearish_seasonals.bearish_seasonals_app()

def user_input_page():
    st.sidebar.markdown("# User Input")
    user_input.user_input_app()

def filtered_bull_page():
    st.sidebar.markdown("# Filtered Bull")
    filtered_bull.filtered_bull_app()

def filtered_bear_page():
    st.sidebar.markdown("# Filtered Bear")
    filtered_bear.filtered_bear_app()

page_names_to_funcs = {
    "Daily Signals": bull_signals_page,
    "Positions": positions_page,
    "Bullish Seasonals": bullish_seasonals_page,
    "Bearish Seasonals": bearish_seasonals_page,
    "User Input": user_input_page,
    "Filtered Bull": filtered_bull_page,
    "Filtered Bear": filtered_bear_page,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()


