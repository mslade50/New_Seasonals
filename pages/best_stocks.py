import streamlit as st
import pandas as pd
import yfscreen as yfs
import yfinance as yf


# ---------- Helpers ----------

def get_momentum_field() -> str:
    df_filters = yfs.data_filters
    return df_filters.loc[
        (df_filters["sec_type"] == "equity")
        & (df_filters["name"] == "52 Week Price % Change"),
        "field",
    ].iloc[0]


def extract_price_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Handles common yfinance return formats and extracts a wide
    Ticker x Close/Adj Close matrix.
    """
    cols = prices.columns

    # Case A: MultiIndex (Price, Ticker)
    if isinstance(cols, pd.MultiIndex) and cols.names == ["Price", "Ticker"]:
        lvl0 = cols.get_level_values(0)
        if "Adj Close" in lvl0:
            return prices["Adj Close"]
        elif "Close" in lvl0:
            return prices["Close"]
        else:
            raise ValueError(f"No Adj Close or Close in first level: {lvl0.unique()}")

    # Case B: MultiIndex (Ticker, Price)
    if isinstance(cols, pd.MultiIndex) and cols.names == ["Ticker", "Price"]:
        lvl1 = cols.get_level_values(1)
        if "Adj Close" in lvl1:
            return prices.xs("Adj Close", axis=1, level=1)
        elif "Close" in lvl1:
            return prices.xs("Close", axis=1, level=1)
        else:
            raise ValueError(f"No Adj Close or Close in second level: {lvl1.unique()}")

    # Case C: SingleIndex
    if "Adj Close" in cols:
        return prices[["Adj Close"]]
    if "Close" in cols:
        return prices[["Close"]]

    raise ValueError(f"Could not extract prices from columns: {cols}")


def build_meta(screener_df: pd.DataFrame) -> pd.DataFrame:
    meta_cols = [
        "symbol",
        "regularMarketPrice.raw",
        "marketCap.raw",
        "averageDailyVolume10Day.raw",
        "averageDailyVolume3Month.raw",
        "sector",
    ]
    meta_cols = [c for c in meta_cols if c in screener_df.columns]

    meta = (
        screener_df[meta_cols]
        .drop_duplicates(subset="symbol")
        .set_index("symbol")
    )

    # Avg daily $ volume
    vol_col = None
    if "averageDailyVolume10Day.raw" in meta.columns:
        vol_col = "averageDailyVolume10Day.raw"
    elif "averageDailyVolume3Month.raw" in meta.columns:
        vol_col = "averageDailyVolume3Month.raw"

    if vol_col and "regularMarketPrice.raw" in meta.columns:
        meta["avg_dollar_volume"] = meta[vol_col] * meta["regularMarketPrice.raw"]

    # Shorter names
    rename_map = {}
    if "regularMarketPrice.raw" in meta.columns:
        rename_map["regularMarketPrice.raw"] = "Price"
    if "marketCap.raw" in meta.columns:
        rename_map["marketCap.raw"] = "Mkt_Cap"
    if "sector" in meta.columns:
        rename_map["sector"] = "Sector"

    meta = meta.rename(columns=rename_map)
    return meta


def compute_atr_from_prices(prices: pd.DataFrame, tickers, window: int = 14) -> pd.Series:
    """
    Compute ATR for each ticker using High/Low/Close from the multi-ticker
    price DataFrame returned by yfinance.
    Assumes columns are MultiIndex with levels ['Ticker','Price'].
    """
    cols = prices.columns
    if not isinstance(cols, pd.MultiIndex):
        raise ValueError("ATR helper expects a MultiIndex columns DataFrame.")

    # Try to infer which level is ticker / which is price
    if cols.names == ["Ticker", "Price"]:
        ticker_level = 0
        price_level = 1
    elif cols.names == ["Price", "Ticker"]:
        ticker_level = 1
        price_level = 0
    else:
        # fallback: assume level with 'Close' in it is price level
        lvl0 = cols.get_level_values(0)
        lvl1 = cols.get_level_values(1)
        if "Close" in lvl0 or "High" in lvl0:
            price_level, ticker_level = 0, 1
        else:
            price_level, ticker_level = 1, 0

    atr_vals = {}

    for t in tickers:
        try:
            df_t = prices.xs(t, axis=1, level=ticker_level)
        except KeyError:
            # ticker might have been dropped / missing
            continue

        # Need High, Low, Close
        needed = [c for c in ["High", "Low", "Close"] if c in df_t.columns]
        if len(needed) < 3:
            continue

        hlc = df_t[["High", "Low", "Close"]].dropna()
        if hlc.empty:
            continue

        high = hlc["High"]
        low = hlc["Low"]
        close = hlc["Close"]

        # True range components
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(window).mean().iloc[-1]
        atr_vals[t] = float(atr)

    return pd.Series(atr_vals, name="ATR")


def run_pipeline():
    # ---------- 1. Screener ----------
    momentum_field = get_momentum_field()

    filters = [
        ["eq", ["region", "us"]],
        ["gt", ["intradayprice", 8]],
        ["gt", ["avgdailyvol3m", 300000]],
        ["gt", ["intradaymarketcap", 3_000_000_000]],
    ]

    query = yfs.create_query(filters)
    payload = yfs.create_payload(
        sec_type="equity",
        query=query,
        size=500,
        sort_field=momentum_field,
        sort_type="DESC",
    )

    screener_df = yfs.get_data(payload)

    # Drop 5-letter symbols (often OTC/illiquid)
    screener_df = screener_df[screener_df["symbol"].str.len() != 5].copy()

    tickers = screener_df["symbol"].dropna().unique().tolist()

    # ---------- 2. Prices ----------
    prices = yf.download(
        tickers=tickers,
        period="13mo",
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    if prices is None or prices.empty:
        raise ValueError("No price data downloaded from yfinance.")

    # Close / Adj Close matrix
    px = extract_price_matrix(prices)

    # Drop tickers with insufficient history
    px = px.dropna(axis=1, thresh=180)

    # ---------- 3. Returns (63 / 126 / 252 trading days) ----------
    returns = pd.DataFrame(
        {
            "R63D": px.pct_change(63).iloc[-1],
            "R126D": px.pct_change(126).iloc[-1],
            "R252D": px.pct_change(252).iloc[-1],
        }
    ).dropna()

    # ---------- 3B. ATR + atr_to50 ----------
    # ATR on same tickers as px
    atr_series = compute_atr_from_prices(prices, px.columns, window=14)

    # 50d SMA and last price from px
    sma50 = px.rolling(50).mean().iloc[-1]
    last_price = px.iloc[-1]

    # Align indices and avoid divide-by-zero
    atr_aligned = atr_series.reindex(returns.index)
    atr_safe = atr_aligned.replace(0, pd.NA)

    atr_to50 = (last_price.reindex(returns.index) - sma50.reindex(returns.index)) / atr_safe

    # ---------- 4. Top lists & combined universe ----------
    top_63_idx = returns.sort_values("R63D", ascending=False).head(20).index
    top_126_idx = returns.sort_values("R126D", ascending=False).head(20).index
    top_252_idx = returns.sort_values("R252D", ascending=False).head(20).index

    selected = top_63_idx.union(top_126_idx).union(top_252_idx)
    combined = returns.loc[selected]

    # Convert returns to % and round
    for col in ["R63D", "R126D", "R252D"]:
        combined[col] = (combined[col] * 100).round(1)

    # Attach ATR & atr_to50
    combined["ATR"] = atr_aligned.reindex(combined.index)
    combined["atr_to50"] = atr_to50.reindex(combined.index)

    combined["ATR"] = combined["ATR"].round(2)
    combined["atr_to50"] = combined["atr_to50"].round(1)

    # Attach meta
    meta = build_meta(screener_df)
    combined = combined.join(meta, how="left")

    # Price 2 decimals
    if "Price" in combined.columns:
        combined["Price"] = combined["Price"].round(2)

    # Market cap in $B
    if "Mkt_Cap" in combined.columns:
        combined["MCap_$B"] = (combined["Mkt_Cap"] / 1e9).round(2)
        combined = combined.drop(columns=["Mkt_Cap"])

    # Avg dollar volume in $M
    if "avg_dollar_volume" in combined.columns:
        combined["ADVol_$M"] = (combined["avg_dollar_volume"] / 1e6).round(2)
        combined = combined.drop(columns=["avg_dollar_volume"])

    combined.index.name = "Ticker"
    combined = combined.reset_index()

    # Order columns nicely
    desired_cols = [
        "Ticker",
        "R63D",
        "R126D",
        "R252D",
        "Price",
        "ATR",
        "atr_to50",
        "MCap_$B",
        "ADVol_$M",
        "Sector",
    ]
    combined = combined[[c for c in desired_cols if c in combined.columns]]

    # Sort final table by 252d return
    if "R252D" in combined.columns:
        combined = combined.sort_values("R252D", ascending=False)

    # Ticker list string for copy/paste
    ticker_list = ", ".join(f"'{t}'" for t in combined["Ticker"].tolist())

    return combined, ticker_list


# ---------- Streamlit UI ----------

def load_data_once(force: bool = False):
    # If user clicked refresh, drop existing cached data
    if force:
        st.session_state.pop("table_df", None)
        st.session_state.pop("ticker_list", None)

    if "table_df" not in st.session_state or "ticker_list" not in st.session_state:
        with st.spinner("Running screener and fetching data..."):
            table_df, ticker_list = run_pipeline()
        st.session_state["table_df"] = table_df
        st.session_state["ticker_list"] = ticker_list


import traceback

def main():
    st.title("Best Performing US Stocks (Multi-Horizon)")

    try:
        load_data_once()
    except Exception as e:
        st.error("Error while building table:")
        st.exception(e)  # <- shows stack trace & location
        st.code(traceback.format_exc())
        return

    table_df = st.session_state["table_df"]
    ticker_list = st.session_state["ticker_list"]

    st.subheader("Combined Top Names (63D / 126D / 252D)")
    st.dataframe(table_df, use_container_width=True)

    st.sidebar.header("Options")
    show_tickers = st.sidebar.checkbox("Show copy-paste ticker list")

    if show_tickers:
        st.subheader("Ticker list (copy/paste friendly)")
        st.code(ticker_list, language=None)


if __name__ == "__main__":
    main()
