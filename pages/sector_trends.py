# pages/sector_trends.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf


SECTOR_ETFS = [
    # From screenshot
    "IBB", "IHI", "ITA", "ITB", "IYR", "KRE", "OIH", "SMH", "VNQ",
    "XBI", "XHB", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP",
    "XLU", "XLV", "XLY", "XME", "XOP", "XRT",
    # Extra index/sector leaders
    "SPY", "QQQ", "IWM", "DIA", "SMH",
]


def percentile_rank(series: pd.Series, value: float) -> float:
    """Return percentile rank (0–100) of value within series."""
    s = series.dropna().values
    if s.size == 0 or pd.isna(value):
        return np.nan
    pct = (s <= value).sum() / s.size * 100.0
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

        # Today's values (last valid point)
        price_today = close.iloc[-1]
        d20_today = dist20.iloc[-1]
        d50_today = dist50.iloc[-1]
        d200_today = dist200.iloc[-1]

        # Percentile ranks over life of each distance series
        p20 = percentile_rank(dist20, d20_today)
        p50 = percentile_rank(dist50, d50_today)
        p200 = percentile_rank(dist200, d200_today)

        rows.append(
            {
                "Ticker": t,
                "Price": price_today,
                "D20%": d20_today,
                "PctRank20": p20,
                "D50%": d50_today,
                "PctRank50": p50,
                "D200%": d200_today,
                "PctRank200": p200,
            }
        )

    if not rows:
        return pd.DataFrame()

    df_out = pd.DataFrame(rows)

    # Rounding / formatting
    df_out["Price"] = df_out["Price"].round(2)
    for col in ["D20%", "D50%", "D200%"]:
        df_out[col] = df_out[col].round(1)
    for col in ["PctRank20", "PctRank50", "PctRank200"]:
        df_out[col] = df_out[col].round(1)

    # Sort by 200d distance or ticker, as you prefer
    df_out = df_out.sort_values("D200%", ascending=False, ignore_index=True)

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
    st.dataframe(table, use_container_width=True)


if __name__ == "__main__":
    main()
