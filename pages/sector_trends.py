# pages/sector_trends.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf


SECTOR_ETFS = [
    # From screenshot
    "IBB", "IHI", "ITA", "ITB", "IYR", "KRE", "OIH", "SMH", "VNQ",
    "XBI", "XHB", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP",
    "XLU", "XLV", "XLY", "XME", "XOP", "XRT", "GLD", "CEF", "SLV", "BTC-USD",
    "ETH-USD", "UNG", "UVXY",
    # Extra index/sector leaders
    "SPY", "QQQ", "IWM", "DIA", "SMH",
]

CORE_TICKERS = ["SPY", "QQQ", "IWM", "SMH", "DIA"]


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
        ma5 = close.rolling(5).mean()
        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()

        # Distances to MA in %
        dist5 = (close - ma5) / ma5 * 100.0
        dist20 = (close - ma20) / ma20 * 100.0
        dist50 = (close - ma50) / ma50 * 100.0
        dist200 = (close - ma200) / ma200 * 100.0

        # Today's distance values
        d5_today = dist5.dropna().iloc[-1] if not dist5.dropna().empty else np.nan
        d20_today = dist20.dropna().iloc[-1] if not dist20.dropna().empty else np.nan
        d50_today = dist50.dropna().iloc[-1] if not dist50.dropna().empty else np.nan
        d200_today = dist200.dropna().iloc[-1] if not dist200.dropna().empty else np.nan

        # Percentile ranks over life of each distance series
        p5 = percentile_rank(dist5, d5_today)
        p20 = percentile_rank(dist20, d20_today)
        p50 = percentile_rank(dist50, d50_today)
        p200 = percentile_rank(dist200, d200_today)

        rows.append(
            {
                "Ticker": t,
                "Price": float(close.iloc[-1]),
                "PctRank5": p5,
                "PctRank20": p20,
                "PctRank50": p50,
                "PctRank200": p200,
            }
        )

    if not rows:
        return pd.DataFrame()

    df_out = pd.DataFrame(rows)

    # Ensure numeric dtypes before rounding
    num_cols = ["Price", "PctRank5", "PctRank20", "PctRank50", "PctRank200"]
    for col in num_cols:
        if col in df_out.columns:
            df_out[col] = pd.to_numeric(df_out[col], errors="coerce")

    # Rounding / formatting
    if "Price" in df_out.columns:
        df_out["Price"] = df_out["Price"].round(2)
    for col in ["PctRank5", "PctRank20", "PctRank50", "PctRank200"]:
        if col in df_out.columns:
            df_out[col] = df_out[col].round(1)

    # Sort (e.g., by PctRank200 descending)
    if "PctRank200" in df_out.columns:
        df_out = df_out.sort_values("PctRank200", ascending=False, ignore_index=True)
    else:
        df_out = df_out.sort_values("Ticker", ignore_index=True)

    return df_out


@st.cache_data(show_spinner=True)
def load_core_distance_frame():
    """
    Build a daily feature frame for SPY/QQQ/IWM/SMH/DIA using
    distance-to-MA (5/20/50/200) for each, plus SPY forward returns.
    """
    all_feats = []
    spy_close = None

    for t in CORE_TICKERS:
        try:
            df = yf.download(
                t,
                period="max",
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
        except Exception as e:
            st.write(f"⚠️ Failed core download for {t}: {e}")
            continue

        if df.empty:
            st.write(f"⚠️ No data for {t} in core set, skipping")
            continue

        if "Adj Close" in df.columns:
            close = df["Adj Close"].dropna()
        elif "Close" in df.columns:
            close = df["Close"].dropna()
        else:
            st.write(f"⚠️ No Close/Adj Close for {t}, skipping")
            continue

        if close.empty:
            continue

        if t == "SPY":
            spy_close = close.copy()

        ma5 = close.rolling(5).mean()
        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()

        dist5 = (close - ma5) / ma5 * 100.0
        dist20 = (close - ma20) / ma20 * 100.0
        dist50 = (close - ma50) / ma50 * 100.0
        dist200 = (close - ma200) / ma200 * 100.0

        feats = pd.DataFrame(
            {
                f"{t}_dist5": dist5,
                f"{t}_dist20": dist20,
                f"{t}_dist50": dist50,
                f"{t}_dist200": dist200,
            }
        )
        all_feats.append(feats)

    if not all_feats:
        return pd.DataFrame()

    # Inner join on dates across all core tickers
    core_df = all_feats[0]
    for feats in all_feats[1:]:
        core_df = core_df.join(feats, how="inner")

    core_df = core_df.dropna()
    core_df = core_df.sort_index()
    core_df.index = pd.to_datetime(core_df.index)
    core_df.index.name = "Date"

    # Add forward returns for SPY
    if spy_close is not None:
        spy_close = spy_close.reindex(core_df.index).dropna()
        spy_close = spy_close.reindex(core_df.index)  # align index exactly

        horizons = [2, 5, 10, 21, 63, 126, 252]
        for h in horizons:
            core_df[f"SPY_fwd_{h}d"] = spy_close.shift(-h) / spy_close - 1.0

    return core_df


def compute_distance_matches(core_df: pd.DataFrame,
                             n_matches: int = 20,
                             exclude_last_n: int = 63) -> pd.DataFrame:
    """
    Use equal-weight Euclidean distance on all distance features
    across SPY/QQQ/IWM/SMH/DIA to find closest historical dates.
    """
    if core_df.empty:
        return pd.DataFrame()

    df = core_df.copy().sort_index()

    # Feature columns are all dist* columns for the 5 core tickers
    feature_cols = [c for c in df.columns if "dist" in c]
    fwd_cols = [c for c in df.columns if c.startswith("SPY_fwd_")]

    # Require all features + forward returns to be present
    df = df.dropna(subset=feature_cols + fwd_cols)

    if len(df) <= exclude_last_n + 1:
        return pd.DataFrame()

    # Target vector = latest row
    target = df.iloc[-1][feature_cols].astype(float).values

    # Exclude most recent N rows for matching
    hist = df.iloc[:-exclude_last_n].copy()

    # Optional: restrict to post-1997 and exclude 2020 (similar flavor to your other code)
    hist = hist[hist.index.year > 1997]
    hist = hist[hist.index.year != 2020]

    if hist.empty:
        return pd.DataFrame()

    X = hist[feature_cols].astype(float).values
    dists = np.sqrt(((X - target) ** 2).sum(axis=1))
    hist["distance"] = dists

    # Sort by distance and take top N
    matches = hist.sort_values("distance").head(n_matches).copy()
    matches = matches.reset_index()  # bring Date out as a column

    # Keep just Date, distance, and SPY forward returns
    keep_cols = ["Date", "distance"] + fwd_cols
    matches = matches[keep_cols]

    # Format returns as percents
    for c in fwd_cols:
        matches[c] = (matches[c] * 100.0).round(2)

    matches["distance"] = matches["distance"].round(4)

    return matches


def main():
    st.title("Sector ETF Trend Dashboard")

    st.write(
        "Distance to 5/20/50/200-day moving averages and the percentile rank of "
        "today's distance vs each ETF's full trading history."
    )

    # Optional: refresh button to bust cache
    if st.button("Refresh data"):
        load_sector_metrics.clear()
        load_core_distance_frame.clear()

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
                "PctRank5": "{:.1f}",
                "PctRank20": "{:.1f}",
                "PctRank50": "{:.1f}",
                "PctRank200": "{:.1f}",
            }
        )
        .applymap(
            highlight_pct,
            subset=["PctRank5", "PctRank20", "PctRank50", "PctRank200"],
        )
    )

    st.dataframe(styled, use_container_width=True)

    # -------- Distance-matching section --------
    st.subheader("Historical Distance Matches (SPY/QQQ/IWM/SMH/DIA)")

    with st.spinner("Computing distance matches vs history..."):
        core_df = load_core_distance_frame()
        match_table = compute_distance_matches(core_df, n_matches=20, exclude_last_n=63)

    if match_table.empty:
        st.warning("Not enough historical data to compute distance matches.")
    else:
        st.write(
            "20 closest historical dates to today's joint MA-distance profile "
            "using equal-weight distances across SPY, QQQ, IWM, SMH, and DIA. "
            "Returns are forward % changes in SPY."
        )
        st.dataframe(match_table, use_container_width=True)


if __name__ == "__main__":
    main()
