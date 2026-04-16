import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Map common pair names to yfinance tickers
def pair_to_yf_ticker(pair):
    """Convert a pair like EURUSD to yfinance format."""
    pair = pair.upper().replace("/", "")
    if len(pair) != 6:
        return None
    # yfinance FX format: EURUSD=X, but JPY pairs are inverted (USDJPY=X -> JPY=X)
    # Try the direct mapping first
    return f"{pair[:3]}{pair[3:]}=X"


@st.cache_data(ttl=300, show_spinner=False)
def fetch_pair_data(yf_ticker):
    """Fetch price data and compute ATR% for an FX pair."""
    try:
        df = yf.download(yf_ticker, period="1y", progress=False, auto_adjust=True)
        if df.empty:
            return None, None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.capitalize() for c in df.columns]
        if 'Close' not in df.columns:
            return None, None

        prev = df['Close'].shift(1)
        tr = pd.concat([
            df['High'] - df['Low'],
            (df['High'] - prev).abs(),
            (df['Low'] - prev).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        atr_pct = (atr / df['Close']) * 100.0

        last_close = float(df['Close'].iloc[-1])
        last_atr_pct = float(atr_pct.dropna().iloc[-1])
        return last_close, last_atr_pct
    except Exception:
        return None, None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_usd_conversion(quote_ccy):
    """Fetch USD/quote_ccy rate for cross pair conversion."""
    try:
        ticker = f"USD{quote_ccy}=X"
        df = yf.download(ticker, period="5d", progress=False, auto_adjust=True)
        if df.empty:
            # Try inverse
            ticker = f"{quote_ccy}USD=X"
            df = yf.download(ticker, period="5d", progress=False, auto_adjust=True)
            if df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.capitalize() for c in df.columns]
            return 1.0 / float(df['Close'].iloc[-1])
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.capitalize() for c in df.columns]
        return float(df['Close'].iloc[-1])
    except Exception:
        return None


def main():
    st.set_page_config(page_title="FX Position Sizer", layout="centered")
    st.title("FX Position Sizer")

    pair = st.text_input("Pair (e.g. EURUSD, EURCHF, GBPJPY)", value="EURUSD").upper().replace("/", "")

    # Auto-fetch price and ATR
    live_price = None
    live_atr = None
    live_usdx = None

    if len(pair) == 6:
        yf_ticker = pair_to_yf_ticker(pair)
        if yf_ticker:
            with st.spinner("Fetching..."):
                live_price, live_atr = fetch_pair_data(yf_ticker)

            quote_ccy = pair[3:]
            if quote_ccy != "USD":
                live_usdx = fetch_usd_conversion(quote_ccy)

    col1, col2 = st.columns(2)

    with col1:
        entry = st.number_input(
            "Entry Price",
            value=live_price if live_price else 0.0,
            format="%.5f"
        )
        atr_pct = st.number_input(
            "ATR (%)",
            value=live_atr if live_atr else 0.0,
            format="%.2f"
        )

    with col2:
        direction = st.selectbox("Direction", ["Short", "Long"])
        atr_mult = st.number_input("Stop (ATR multiple)", value=2.0, format="%.1f")
        risk_usd = st.number_input("Risk ($USD)", value=2500.0, format="%.0f")

    quote_ccy = pair[3:] if len(pair) >= 6 else "USD"
    needs_conversion = quote_ccy != "USD"

    usdx_rate = 1.0
    if needs_conversion:
        usdx_rate = st.number_input(
            f"USD{quote_ccy} rate",
            value=live_usdx if live_usdx else 0.0,
            format="%.5f"
        )

    if st.button("Calculate", type="primary", use_container_width=True):
        if entry <= 0 or atr_pct <= 0 or risk_usd <= 0:
            st.error("Entry, ATR, and Risk must be > 0")
            return
        if needs_conversion and usdx_rate <= 0:
            st.error(f"USD{quote_ccy} rate must be > 0")
            return

        stop_dist = entry * (atr_pct / 100.0) * atr_mult

        if needs_conversion:
            risk_quote = risk_usd * usdx_rate
        else:
            risk_quote = risk_usd

        base_ccy = pair[:3] if len(pair) >= 6 else "???"
        position = risk_quote / stop_dist

        st.markdown("---")
        st.markdown("### Result")

        c1, c2, c3 = st.columns(3)
        c1.metric("Stop Distance", f"{stop_dist:.5f}")
        c2.metric(f"Risk ({quote_ccy})", f"{risk_quote:,.0f}")
        c3.metric(f"Position ({base_ccy})", f"{position:,.0f}")

        stop_price = entry - stop_dist if direction == "Long" else entry + stop_dist
        st.caption(
            f"{direction} **{position:,.0f} {base_ccy}** at {entry:.5f} | "
            f"Stop: {stop_price:.5f} ({atr_mult:.1f}x ATR) | "
            f"Risk: ${risk_usd:,.0f}"
        )

if __name__ == "__main__":
    main()
