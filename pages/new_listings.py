import streamlit as st
import pandas as pd
import yfscreen as yfs
import yfinance as yf
from yahooquery import Ticker
import traceback


# ======================================================
# Helpers
# ======================================================

def get_momentum_field() -> str:
    """
    Return the Yahoo screener field name for 52 Week Price % Change,
    to use as a valid sort_field in yfs.create_payload.
    """
    df_filters = yfs.data_filters
    row = df_filters.loc[
        (df_filters["sec_type"] == "equity")
        & (df_filters["name"] == "52 Week Price % Change")
    ].iloc[0]
    return row["field"]

def get_ipo_field() -> str:
    """Find the Yahoo screener field used for IPO date."""
    df_filters = yfs.data_filters
    row = df_filters.loc[
        (df_filters["sec_type"] == "equity")
        & (df_filters["name"].str.contains("IPO", case=False, na=False)),
        :
    ].iloc[0]
    return row["field"]


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
    """
    Build a metadata table keyed by symbol:
    Price, Mkt_Cap, avg dollar volume, Sector (when available).
    """
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
    Assumes columns are MultiIndex with levels ['Ticker','Price'] or ['Price','Ticker'].
    """
    cols = prices.columns
    if not isinstance(cols, pd.MultiIndex):
        raise ValueError("ATR helper expects a MultiIndex columns DataFrame.")

    # Try to infer which level is ticker / which is price
    if cols.names == ["Ticker", "Price"]:
        ticker_level = 0
    elif cols.names == ["Price", "Ticker"]:
        ticker_level = 1
    else:
        lvl0 = cols.get_level_values(0)
        lvl1 = cols.get_level_values(1)
        if "Close" in lvl0 or "High" in lvl0:
            ticker_level = 1
        else:
            ticker_level = 0

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
    Try fast batch yf.download. If it fails, fall back to per-ticker downloads
    and build a MultiIndex [Ticker, Price] frame.
    """
    try:
        prices = yf.download(
            tickers=tickers,
            period=period,
            interval=interval,
            auto_adjust=True,
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
    st.warning("Using per-ticker yfinance downloads due to index/timezone issues.")
    frames = []

    for t in tickers:
        try:
            df = yf.download(
                t,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
            )
        except Exception as e:
            st.write(f"⚠️ Failed download for {t}: {e}")
            continue

        if df.empty:
            continue

        # Normalize datetime index to tz-naive
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


# ======================================================
# New Listings Pipeline (Last 12m IPOs)
# ======================================================

def run_new_listings_pipeline():
    """
    Build table of IPOs from last 12 calendar months with:
    R63D, R126D, Price, ATR, atr_to50, MCap_$B, ADVol_$M, Sector, RevGrowth_%.
    """
    # --- 1. Screener: broad US equity universe ---
    # Just reuse your 52w momentum field for sorting; it's valid & cheap.
    momentum_field = get_momentum_field()

    filters = [
        ["eq", ["region", "us"]],
        # no price/vol/mktcap/IPO filters here – we'll filter locally
    ]

    query = yfs.create_query(filters)
    payload = yfs.create_payload(
        sec_type="equity",
        query=query,
        size=1000,                # grab a decent chunk of the universe
        sort_field=momentum_field,
        sort_type="DESC",
    )

    screener_df = yfs.get_data(payload)

    if screener_df is None or screener_df.empty:
        raise ValueError("No data returned from Yahoo screener for US equities.")

    # Ensure we have symbols
    screener_df = screener_df[screener_df["symbol"].notna()].copy()

    # --- 1B. Find IPO date column in screener_df & filter last 12 months ---
    ipo_cols = [c for c in screener_df.columns if "ipo" in c.lower()]
    if not ipo_cols:
        raise ValueError(
            "No IPO-like column found in screener_df. "
            f"Columns were: {list(screener_df.columns)}"
        )

    ipo_col = ipo_cols[0]  # e.g. 'ipoExpectedDate'
    screener_df["ipo_date"] = pd.to_datetime(
        screener_df[ipo_col], errors="coerce"
    )

    cutoff = pd.Timestamp.today() - pd.DateOffset(months=12)
    screener_df = screener_df[screener_df["ipo_date"] >= cutoff].copy()

    if screener_df.empty:
        raise ValueError("No IPOs found in the last 12 months after filtering.")

    # --- 1C. Filter by market cap >= $300M using screener columns ---
    mc_col = None
    for c in ["marketCap.raw", "lastclosemarketcap.lasttwelvemonths"]:
        if c in screener_df.columns:
            mc_col = c
            break

    if mc_col is not None:
        screener_df = screener_df[screener_df[mc_col] >= 300_000_000].copy()

    if screener_df.empty:
        raise ValueError("No IPOs with market cap ≥ $300M found.")

    tickers = screener_df["symbol"].dropna().unique().tolist()
    if not tickers:
        raise ValueError("No IPO tickers remaining after market cap filter.")

    # --- 2. Prices (13mo window so we can get 126d returns & 50d SMA) ---
    prices = download_ohlc_with_fallback(
        tickers=tickers,
        period="13mo",
        interval="1d",
    )

    if prices is None or prices.empty:
        raise ValueError("No price data downloaded for IPO tickers.")

    px = extract_price_matrix(prices)

    # Need enough history for 126d returns
    px = px.dropna(axis=1, thresh=130)

    returns = pd.DataFrame(
        {
            "R63D": px.pct_change(63).iloc[-1],
            "R126D": px.pct_change(126).iloc[-1],
        }
    ).dropna()

    if returns.empty:
        raise ValueError("No IPO tickers have sufficient price history for 63/126d returns.")

    # --- 3. ATR + atr_to50 ---
    atr_series = compute_atr_from_prices(prices, px.columns, window=14)

    sma50 = px.rolling(50).mean().iloc[-1]
    last_price = px.iloc[-1]

    atr_aligned = atr_series.reindex(returns.index)
    atr_safe = atr_aligned.replace(0, pd.NA)

    atr_to50 = (last_price.reindex(returns.index) - sma50.reindex(returns.index)) / atr_safe

    # Convert returns to % and round
    for col in ["R63D", "R126D"]:
        returns[col] = (returns[col] * 100).round(1)

    out = returns.copy()
    out["ATR"] = atr_aligned.reindex(out.index).round(2)
    out["atr_to50"] = atr_to50.reindex(out.index).round(1)

    # --- 4. Meta (Price, MCap, ADVol, Sector) from screener ---
    meta = build_meta(screener_df)
    out = out.join(meta, how="left")

    if "Price" in out.columns:
        out["Price"] = out["Price"].round(2)

    if "Mkt_Cap" in out.columns:
        out["MCap_$B"] = (out["Mkt_Cap"] / 1e9).round(2)
        out = out.drop(columns=["Mkt_Cap"])

    if "avg_dollar_volume" in out.columns:
        out["ADVol_$M"] = (out["avg_dollar_volume"] / 1e6).round(2)
        out = out.drop(columns=["avg_dollar_volume"])

    # --- 5. Revenue growth via yahooquery (optional but nice) ---
    try:
        final_tickers = out.index.astype(str).tolist()
        tf = Ticker(final_tickers)
        fin = tf.financial_data
        fin_df = pd.DataFrame.from_dict(fin, orient="index")

        if "revenueGrowth" in fin_df.columns:
            rg = pd.to_numeric(fin_df["revenueGrowth"], errors="coerce") * 100.0
            rg = rg.round(1)
            rg.index = rg.index.astype(str)
            out["RevGrowth_%"] = rg.reindex(out.index).values
    except Exception:
        # If yahooquery fails, just skip RevGrowth_% rather than error out
        pass

    out.index.name = "Ticker"
    out = out.reset_index()

    desired_cols = [
        "Ticker",
        "R63D",
        "R126D",
        "Price",
        "ATR",
        "atr_to50",
        "MCap_$B",
        "ADVol_$M",
        "Sector",
        "RevGrowth_%",
    ]
    out = out[[c for c in desired_cols if c in out.columns]]

    # Sort by 126d return
    if "R126D" in out.columns:
        out = out.sort_values("R126D", ascending=False)

    return out


# ======================================================
# Streamlit page
# ======================================================

@st.cache_data(show_spinner=False)
def load_new_listings_table():
    return run_new_listings_pipeline()

def main():
    st.title("New Listings (Last 12m IPOs)")

    try:
        if st.button("Refresh IPO Table"):
            load_new_listings_table.clear()  # clear cache on demand
        table = load_new_listings_table()
    except Exception as e:
        st.error("Error while building New Listings table:")
        st.exception(e)
        st.code(traceback.format_exc())
        return

    st.subheader("IPO Universe (Mkt Cap ≥ $300M, Last 12m)")
    st.dataframe(table, use_container_width=True)

    if "Ticker" in table.columns:
        ticker_list = ", ".join(f"'{t}'" for t in table["Ticker"])
        st.subheader("Ticker list (copy/paste friendly)")
        st.code(ticker_list, language=None)


if __name__ == "__main__":
    main()
