import streamlit as st
from pages import indicies, positions, single_names, currency_pairs, commodities, watchlist

st.set_page_config(page_title="Multi-Page Dashboard", page_icon=":chart_with_upwards_trend:")

def indicies_page():
    st.sidebar.markdown("# Indices")
    indicies.indicies_app()

def positions_page():
    st.sidebar.markdown("# Positions")
    positions.positions_app()

def single_names_page():
    st.sidebar.markdown("# Single Names")
    single_names.single_names_app()

def currency_pairs_page():
    st.sidebar.markdown("# Currency Pairs")
    currency_pairs.currency_pairs_app()

def commodities_page():
    st.sidebar.markdown("# Commodities")
    commodities.commodities_app()

def watchlist_page():
    st.sidebar.markdown("# Watchlist")
    watchlist.watchlist_app()

page_names_to_funcs = {
    "Indices": indicies_page,
    "Positions": positions_page,
    "Single Names": single_names_page,
    "Currency Pairs": currency_pairs_page,
    "Commodities": commodities_page,
    "Watchlist": watchlist_page,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
