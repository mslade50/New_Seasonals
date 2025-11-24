import streamlit as st
import pandas as pd
import yfscreen as yfs
import yfinance as yf
from yahooquery import Ticker
import traceback

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
    Compute ATR for each ticker using High/Low/Close.
    """
    cols = prices.columns
    if not isinstance(cols, pd.MultiIndex):
        # Fallback if single index passed (unexpected but safe to handle)
        return pd.Series(index=tickers, dtype=float)

    # Try to infer which level is ticker / which is price
    if cols.names == ["Ticker", "Price"]:
        ticker_level = 0
        price_level = 1
    elif cols.names == ["Price", "Ticker"]:
        ticker_level = 1
        price_level = 0
    else:
        lvl0 = cols.get_level_values(0)
        if "Close" in lvl0 or "High" in lvl0:
            price_level, ticker_level = 0, 1
        else:
            price_level, ticker_level = 1, 0

    atr_vals = {}

    for t in tickers:
        try:
            df_t = prices.xs(t, axis=1, level=ticker_level)
        except KeyError:
            continue

        needed = [c for c in ["High", "Low", "Close"] if c in df_t.columns]
        if len(needed) < 3:
            continue

        hlc = df_t[["High", "Low", "Close"]].dropna()
        if hlc.empty:
            continue

        high = hlc["High"]
        low = hlc["Low"]
        close = hlc["Close"]

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(window).mean().iloc[-1]
        atr_vals[t] = float(atr)

    return pd.Series(atr_vals, name="ATR")


def download_ohlc_with_fallback(tickers, period="13mo", interval="1d"):
    """
    Try fast batch yf.download. If it fails, fall back to per-ticker.
    FIX: Explicitly sets auto_adjust=True to suppress FutureWarnings.
    """
    try:
        prices = yf.download(
            tickers=tickers,
            period=period,
            interval=interval,
            auto_adjust=True,  # FIX: Set explicit default
            group_by="ticker",
            threads=True,
            progress=False,
            ignore_tz=True, 
        )
    except Exception as e:
        st.warning(f"Batch download failed ({e}); falling back to per-ticker.")
        prices = None

    if prices is not None and not prices.empty:
        return prices

    # Fallback: per-ticker
    st.warning("Using per-ticker yfinance downloads due to timezone/index issues.")

    frames = []
    for t in tickers:
        try:
            df = yf.download(
                t,
                period=period,
                interval=interval,
                auto_adjust=True,  # FIX: Set explicit default
                progress=False,
            )
        except Exception as e:
            st.write(f"⚠️ Failed download for {t}: {e}")
            continue

        if df.empty:
            continue

        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)

        cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
        if not cols:
            continue
        df = df[cols]

        df.columns = pd.MultiIndex.from_product([[t], df.columns], names=["Ticker", "Price"])
        frames.append(df)

    if not frames:
        raise RuntimeError("No valid OHLC series downloaded in fallback.")

    prices = pd.concat(frames, axis=1).sort_index()
    return prices


import time

def run_pipeline():
    # ---------- 1. Screener (With Retry Logic) ----------
    momentum_field = get_momentum_field()

    filters = [
        ["eq", ["region", "us"]],
        ["gt", ["intradayprice", 8]],
        ["gt", ["avgdailyvol3m", 300000]],
        ["gt", ["totalrevenues.lasttwelvemonths", 30]],
        ["gt", ["lastclosemarketcap.lasttwelvemonths", 3_000_000_000]],
    ]

    query = yfs.create_query(filters)
    payload = yfs.create_payload(
        sec_type="equity",
        query=query,
        size=500,
        sort_field=momentum_field,
        sort_type="DESC",
    )

    screener_df = None
    max_retries = 3
    
    # --- RETRY LOOP START ---
    for attempt in range(max_retries):
        try:
            screener_df = yfs.get_data(payload)
            # If we get here and it's valid, break the loop
            if screener_df is not None and not screener_df.empty:
                break
        except Exception as e:
            if attempt < max_retries - 1:
                sleep_time = 2 * (attempt + 1)
                st.warning(f"Screener connection failed ({e}). Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                raise RuntimeError(f"Yahoo Screener failed after {max_retries} attempts. API might be down.")
    # --- RETRY LOOP END ---

    # Double check we actually got data
    if screener_df is None or screener_df.empty:
         raise RuntimeError("Screener returned no data.")

    # Drop 5-letter symbols
    screener_df = screener_df[screener_df["symbol"].str.len() != 5].copy()

    # ---------- 1B. Drop Biotech ----------
    all_tickers = screener_df["symbol"].dropna().unique().tolist()

    try:
        tq = Ticker(all_tickers)
        profile = tq.asset_profile

        if isinstance(profile, pd.DataFrame):
            prof_df = profile
        else:
            prof_df = pd.DataFrame.from_dict(profile, orient="index")

        keep_cols = [c for c in ["sector", "industry"] if c in prof_df.columns]
        prof_df = prof_df[keep_cols]

        prof_df.index = prof_df.index.astype(str).str.upper()
        screener_df["symbol_upper"] = screener_df["symbol"].str.upper()

        screener_df = screener_df.merge(
            prof_df,
            left_on="symbol_upper",
            right_index=True,
            how="left",
            suffixes=("", "_yq"),
        )

        biotech_mask = False
        if "industry" in screener_df.columns:
            biotech_mask = screener_df["industry"].str.contains("Biotechnology", case=False, na=False)
        elif "sector" in screener_df.columns:
            biotech_mask = screener_df["sector"].eq("Biotechnology")

        screener_df = screener_df[~biotech_mask].copy()
        screener_df = screener_df.drop(columns=["symbol_upper"])
    except Exception:
        pass

    tickers = screener_df["symbol"].dropna().unique().tolist()

    # ---------- 2. Prices ----------
    prices = download_ohlc_with_fallback(tickers=tickers)

    if prices is None or prices.empty:
        raise ValueError("No price data downloaded from yfinance.")

    px = extract_price_matrix(prices)
    px = px.dropna(axis=1, thresh=180)

    # ---------- 3. Returns ----------
    # RAW FLOATS (No rounding, no multiplying by 100 yet)
    returns = pd.DataFrame(
        {
            "R63D": px.pct_change(63).iloc[-1],
            "R126D": px.pct_change(126).iloc[-1],
            "R252D": px.pct_change(252).iloc[-1],
        }
    ).dropna()

    atr_series = compute_atr_from_prices(prices, px.columns, window=14)
    sma50 = px.rolling(50).mean().iloc[-1]
    last_price = px.iloc[-1]

    atr_aligned = atr_series.reindex(returns.index)
    atr_safe = atr_aligned.replace(0, pd.NA)

    atr_to50 = (last_price.reindex(returns.index) - sma50.reindex(returns.index)) / atr_safe

    # ---------- 4. Combine ----------
    top_63_idx = returns.sort_values("R63D", ascending=False).head(20).index
    top_126_idx = returns.sort_values("R126D", ascending=False).head(20).index
    top_252_idx = returns.sort_values("R252D", ascending=False).head(20).index

    selected = top_63_idx.union(top_126_idx).union(top_252_idx)
    combined = returns.loc[selected].copy()

    combined["ATR"] = atr_aligned.reindex(combined.index)
    combined["atr_to50"] = atr_to50.reindex(combined.index)

    # ---------- 4B. Meta ----------
    meta = build_meta(screener_df)
    combined = combined.join(meta, how="left")

    if "Mkt_Cap" in combined.columns:
        combined["MCap_$B"] = (combined["Mkt_Cap"] / 1e9)
        combined = combined.drop(columns=["Mkt_Cap"])

    if "avg_dollar_volume" in combined.columns:
        combined["ADVol_$M"] = (combined["avg_dollar_volume"] / 1e6)
        combined = combined.drop(columns=["avg_dollar_volume"])

    # ---------- 4C. Revenue Growth ----------
    try:
        final_tickers = combined.index.astype(str).tolist()
        tf = Ticker(final_tickers)
        fin = tf.financial_data
        fin_df = pd.DataFrame.from_dict(fin, orient="index")

        if "revenueGrowth" in fin_df.columns:
            rg = pd.to_numeric(fin_df["revenueGrowth"], errors="coerce")
            rg.index = rg.index.astype(str)
            combined["RevGrowth_%"] = rg.reindex(combined.index).values
    except Exception:
        pass

    combined.index.name = "Ticker"
    combined = combined.reset_index()

    desired_cols = [
        "Ticker", "R63D", "R126D", "R252D", "Price",
        "ATR", "atr_to50", "MCap_$B", "ADVol_$M", "Sector", "RevGrowth_%",
    ]
    combined = combined[[c for c in desired_cols if c in combined.columns]]

    if "R252D" in combined.columns:
        combined = combined.sort_values("R252D", ascending=False)

    ticker_list = ", ".join(f"'{t}'" for t in combined["Ticker"].tolist())

    return combined, ticker_list

# ---------- Streamlit UI ----------

def load_data_once(force: bool = False):
    if force:
        st.session_state.pop("table_df", None)
        st.session_state.pop("ticker_list", None)

    if "table_df" not in st.session_state or "ticker_list" not in st.session_state:
        with st.spinner("Running screener and fetching data..."):
            table_df, ticker_list = run_pipeline()
        st.session_state["table_df"] = table_df
        st.session_state["ticker_list"] = ticker_list


def main():
    st.set_page_config(layout="wide") # Use wide mode
    st.title("Best Performing US Stocks (Multi-Horizon)")

    if st.sidebar.button("Refresh Data"):
        load_data_once(force=True)

    try:
        load_data_once()
    except Exception as e:
        st.error("Error while building table. This may be due to a Yahoo Finance API timeout (503).")
        st.exception(e)
        st.code(traceback.format_exc())
        return

    table_df = st.session_state["table_df"]
    ticker_list = st.session_state["ticker_list"]

    st.subheader("Combined Top Names (63D / 126D / 252D)")

    # FIX: Use column_config to handle formatting safely without breaking PyArrow
    # FIX: Use width="stretch" instead of use_container_width
    st.dataframe(
        table_df,
        width="stretch", 
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker"),
            "R63D": st.column_config.NumberColumn("63D Return", format="%.1f%%"),
            "R126D": st.column_config.NumberColumn("126D Return", format="%.1f%%"),
            "R252D": st.column_config.NumberColumn("252D Return", format="%.1f%%"),
            "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
            "ATR": st.column_config.NumberColumn("ATR", format="%.2f"),
            "atr_to50": st.column_config.NumberColumn("Dist to 50d (ATR)", format="%.1f"),
            "MCap_$B": st.column_config.NumberColumn("Market Cap ($B)", format="%.2f"),
            "ADVol_$M": st.column_config.NumberColumn("Avg Vol ($M)", format="%.1f"),
            "RevGrowth_%": st.column_config.NumberColumn("Rev Growth (YoY)", format="%.1f%%"),
        },
        hide_index=True
    )

    st.sidebar.header("Options")
    show_tickers = st.sidebar.checkbox("Show copy-paste ticker list")

    if show_tickers:
        st.subheader("Ticker list")
        st.code(ticker_list, language=None)


if __name__ == "__main__":
    main()
