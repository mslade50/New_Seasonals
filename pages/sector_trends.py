# pages/sector_trends.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf


SECTOR_ETFS = [
    # From screenshot
    "IBB", "IHI", "ITA", "ITB", "IYR", "KRE", "OIH", "SMH", "VNQ",
    "XBI", "XHB", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP",
    "XLU", "XLV", "XLY", "XME", "XOP", "XRT","GLD","CEF","SLV","BTC-USD",
    "ETH-USD", "UNG", "UVXY"
    # Extra index/sector leaders
    "SPY", "QQQ", "IWM", "DIA", "SMH",
]


def percentile_rank(series: pd.Series, value) -> float:
    """Return percentile rank (0–100) of value within series."""
    s = series.dropna().values
    if s.size == 0:
        return np.nan

    # Normalize value to a scalar float if it's array-like / Series
    if isinstance(value, (pd.Series, np.ndarray, list, tuple)):
        arr = np.asarray(value).ravel()
        if arr.size == 0:
            return np.nan
        v = float(arr[-1])
    else:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return np.nan

    if np.isnan(v):
        return np.nan

    pct = (s <= v).sum() / s.size * 100.0
    return float(pct)



@st.cache_data(show_spinner=True)

def load_sector_metrics(tickers):
    rows = []

    for t in tickers:
        try:
            df = yf.download(
                t,
                period="max",
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
        except Exception as e:
            st.write(f"⚠️ Failed download for {t}: {e}")
            continue

        if df.empty:
            st.write(f"⚠️ No data for {t}, skipping")
            continue

        # Use Adj Close if present, otherwise Close
        if "Adj Close" in df.columns:
            close = df["Adj Close"]
        elif "Close" in df.columns:
            close = df["Close"]
        else:
            st.write(f"⚠️ No Close/Adj Close for {t}, skipping")
            continue

        close = close.dropna()
        if close.empty:
            continue

        # Moving averages
        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()

        # Distances to MA in %
        dist20 = (close - ma20) / ma20 * 100.0
        dist50 = (close - ma50) / ma50 * 100.0
        dist200 = (close - ma200) / ma200 * 100.0

        # Today's distance values
        d20_today = dist20.dropna().iloc[-1] if not dist20.dropna().empty else np.nan
        d50_today = dist50.dropna().iloc[-1] if not dist50.dropna().empty else np.nan
        d200_today = dist200.dropna().iloc[-1] if not dist200.dropna().empty else np.nan

        # Percentile ranks over life of each distance series
        p20 = percentile_rank(dist20, d20_today)
        p50 = percentile_rank(dist50, d50_today)
        p200 = percentile_rank(dist200, d200_today)

        rows.append(
            {
                "Ticker": t,
                "Price": float(close.iloc[-1]),
                "PctRank20": p20,
                "PctRank50": p50,
                "PctRank200": p200,
            }
        )

    if not rows:
        return pd.DataFrame()

    df_out = pd.DataFrame(rows)

    # Ensure numeric dtypes before rounding
    num_cols = ["Price", "PctRank20", "PctRank50", "PctRank200"]
    for col in num_cols:
        if col in df_out.columns:
            df_out[col] = pd.to_numeric(df_out[col], errors="coerce")

    # Rounding / formatting
    if "Price" in df_out.columns:
        df_out["Price"] = df_out["Price"].round(2)
    for col in ["PctRank20", "PctRank50", "PctRank200"]:
        if col in df_out.columns:
            df_out[col] = df_out[col].round(1)

    # Sort (e.g., by PctRank200 descending)
    if "PctRank200" in df_out.columns:
        df_out = df_out.sort_values("PctRank200", ascending=False, ignore_index=True)
    else:
        df_out = df_out.sort_values("Ticker", ignore_index=True)

    return df_out



def main():
    st.title("Sector ETF Trend Dashboard")

    st.write(
        "Distance to 20/50/200-day moving averages and the percentile rank of "
        "today's distance vs each ETF's full trading history."
    )

    # Optional: refresh button to bust cache
    if st.button("Refresh data"):
        load_sector_metrics.clear()

    with st.spinner("Loading sector ETF data from Yahoo Finance..."):
        table = load_sector_metrics(sorted(set(SECTOR_ETFS)))

    if table.empty:
        st.error("No data available. Try refreshing or relaxing the ticker list.")
        return

    st.subheader("Sector & Index ETFs")

    def highlight_pct(val):
        if pd.isna(val):
            return ""
        if val > 90:
            # light red bg, dark red text
            return "background-color: #ffcccc; color: #8b0000;"
        if val < 15:
            # light green bg, dark green text
            return "background-color: #ccffcc; color: #006400;"
        return ""

    styled = (
        table.style
        .format(
            {
                "Price": "{:.2f}",
                "PctRank20": "{:.1f}",
                "PctRank50": "{:.1f}",
                "PctRank200": "{:.1f}",
            }
        )
        .applymap(
            highlight_pct,
            subset=["PctRank20", "PctRank50", "PctRank200"],
        )
    )

    st.dataframe(styled, use_container_width=True)


if __name__ == "__main__":
    main()
