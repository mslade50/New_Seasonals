# pages/new_listings.py

import streamlit as st
import pandas as pd
import yfinance as yf
from yahooquery import Ticker
import traceback
import re
from pathlib import Path

def get_last12m_ipo_tickers():
    """
    Return unique IPO tickers from:
      1) IPOScoop 'Last 12 Months' page
      2) Local HTML snippet saved in file 'ipos' at project root
         (symbols pulled from Yahoo links like ...q?s=SYMB)

    Output is a sorted list of unique ticker strings.
    """
    # -------- 1. Live scrape from IPOScoop --------
    url = "https://www.iposcoop.com/last-12-months/"
    tables = pd.read_html(url)
    if not tables:
        raise ValueError("No tables found on IPOScoop last-12-months page.")

    df = tables[0]

    # Find the Symbol column robustly
    symbol_col_candidates = [c for c in df.columns if "symbol" in str(c).lower()]
    if not symbol_col_candidates:
        raise ValueError(f"No Symbol column found. Columns: {list(df.columns)}")
    symbol_col = symbol_col_candidates[0]

    base_symbols = (
        df[symbol_col]
        .astype(str)
        .str.strip()
    )

    base_tickers = {s.upper() for s in base_symbols.unique() if s and s != "nan"}

    # -------- 2. Extra tickers from local 'ipos' HTML snippet --------
    extra_tickers = set()
    try:
        # project root = one level up from /pages/new_listings.py
        project_root = Path(__file__).resolve().parents[1]
        ipos_path = project_root / "ipos"   # your txt file with the HTML

        if ipos_path.exists():
            html = ipos_path.read_text(encoding="utf-8", errors="ignore")
            # Grab symbols from links like http://finance.yahoo.com/q?s=MSW
            matches = re.findall(r"q\?s=([A-Za-z0-9\.\-]+)", html)
            extra_tickers = {m.upper() for m in matches if m}
    except Exception:
        # if anything goes wrong, just ignore the local file
        pass

    # -------- 3. Union + return --------
    all_tickers = sorted(base_tickers | extra_tickers)
    st.write("📌 IPO Tickers Found:", all_tickers)
    return all_tickers


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


def compute_atr_from_prices(prices: pd.DataFrame, tickers, window: int = 14) -> pd.Series:
    """
    Compute ATR for each ticker using High/Low/Close from the multi-ticker
    price DataFrame returned by yfinance.
    Assumes columns are MultiIndex with levels ['Ticker','Price'] or ['Price','Ticker'].
    """
    cols = prices.columns
    if not isinstance(cols, pd.MultiIndex):
        raise ValueError("ATR helper expects a MultiIndex columns DataFrame.")

    # Identify ticker level
    if cols.names == ["Ticker", "Price"]:
        ticker_level = 0
    elif cols.names == ["Price", "Ticker"]:
        ticker_level = 1
    else:
        lvl0 = cols.get_level_values(0)
        if "Close" in lvl0 or "High" in lvl0 or "Low" in lvl0:
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
# New Listings Pipeline (IPOScoop source, no filters)
# ======================================================

def run_new_listings_pipeline():
    """
    Build table of IPOs from IPOScoop 'Last 12 Months' page with:
    R63D, R126D, Price, ATR, atr_to50, MCap_$B, ADVol_$M, Sector, RevGrowth_%.
    No extra filters; every scraped IPO ticker appears, even if returns/ATR are NaN.
    """
    # 1. Get IPO tickers from IPOScoop (+ local ipos file)
    ipo_tickers = get_last12m_ipo_tickers()
    if not ipo_tickers:
        raise ValueError("No IPO symbols scraped from IPOScoop/local file.")
    
    # Master index = ALL IPO tickers (what we want to display)
    master_index = pd.Index(sorted(set(ipo_tickers)), name="Ticker")

    # 2. Download prices (up to ~13mo if available)
    prices = download_ohlc_with_fallback(
        tickers=master_index.tolist(),
        period="13mo",
        interval="1d",
    )

    if prices is None or prices.empty:
        raise ValueError("No price data downloaded for IPO tickers.")

    px = extract_price_matrix(prices)
    if px is None or px.empty:
        raise ValueError("Could not extract price matrix for IPO tickers.")

    # Ensure px has only columns that are actual tickers in master_index
    px = px.loc[:, [c for c in px.columns if c in master_index]]

    # 3. Returns (63 / 126 trading days)
    # We'll compute returns only for tickers in px and then reindex to master_index.
    present_tickers = px.columns.tolist()

    returns = pd.DataFrame(index=pd.Index(present_tickers, name="Ticker"))
    returns["R63D"] = px[present_tickers].pct_change(63).iloc[-1]
    returns["R126D"] = px[present_tickers].pct_change(126).iloc[-1]

    # Convert returns to % and round (NaNs stay NaNs)
    for col in ["R63D", "R126D"]:
        returns[col] = (returns[col] * 100).round(1)

    # 4. ATR + atr_to50 (only for tickers we have OHLC for)
    atr_series = compute_atr_from_prices(prices, present_tickers, window=14)
    atr_series = pd.to_numeric(atr_series, errors="coerce")  # numeric only

    sma50 = px.rolling(50).mean().iloc[-1]
    last_price = px.iloc[-1]

    atr_aligned = atr_series.reindex(present_tickers)
    atr_safe = atr_aligned.replace(0, pd.NA)

    atr_to50 = (last_price.reindex(present_tickers) - sma50.reindex(present_tickers)) / atr_safe
    atr_to50 = pd.to_numeric(atr_to50, errors="coerce")

    # 5. Build signals frame on MASTER index (all IPOs)
    out = pd.DataFrame(index=master_index)

    # join returns (some tickers will be NaN for both)
    out = out.join(returns, how="left")

    # ATR & atr_to50
    out["ATR"] = atr_series.reindex(master_index).round(2)
    out["atr_to50"] = atr_to50.reindex(master_index).round(1)

    # 6. Fundamentals/meta via yahooquery (using ALL IPO tickers)
    try:
        tq = Ticker(master_index.tolist())

        # Price info (for Price, MCap, current volume)
        price_dict = tq.price
        price_df = pd.DataFrame.from_dict(price_dict, orient="index")

        meta = pd.DataFrame(index=master_index)

        if "regularMarketPrice" in price_df.columns:
            meta["Price"] = price_df["regularMarketPrice"]
        if "marketCap" in price_df.columns:
            meta["MCap_$B"] = (price_df["marketCap"] / 1e9).round(2)
        if {"regularMarketVolume", "regularMarketPrice"}.issubset(price_df.columns):
            adv = price_df["regularMarketVolume"] * price_df["regularMarketPrice"]
            meta["ADVol_$M"] = (adv / 1e6).round(2)

        # Sector / industry
        try:
            prof_dict = tq.asset_profile
            prof_df = pd.DataFrame.from_dict(prof_dict, orient="index")
            if "sector" in prof_df.columns:
                meta["Sector"] = prof_df["sector"]
            elif "industry" in prof_df.columns:
                meta["Sector"] = prof_df["industry"]
        except Exception:
            pass

        # Revenue growth
        try:
            fin_dict = tq.financial_data
            fin_df = pd.DataFrame.from_dict(fin_dict, orient="index")
            if "revenueGrowth" in fin_df.columns:
                rg = pd.to_numeric(fin_df["revenueGrowth"], errors="coerce") * 100.0
                meta["RevGrowth_%"] = rg.round(1)
        except Exception:
            pass

        out = out.join(meta, how="left")

        if "Price" in out.columns:
            out["Price"] = out["Price"].round(2)
    except Exception:
        # If yahooquery fails entirely, keep only tech columns
        pass

    out = out.reset_index()  # Ticker as column

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
        "RevGrowth_%"
    ]
    out = out[[c for c in desired_cols if c in out.columns]]

    # Optional: sort by R126D but don't lose the NaN rows
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
    st.title("New Listings (Last 12 Months IPOs)")

    try:
        if st.button("Refresh IPO Table"):
            load_new_listings_table.clear()  # clear cache
        table = load_new_listings_table()
    except Exception as e:
        st.error("Error while building New Listings table:")
        st.exception(e)
        st.code(traceback.format_exc())
        return

    st.subheader("IPO Universe (Last 12 Months, tech + fundamentals)")
    st.dataframe(table, use_container_width=True)

    if "Ticker" in table.columns:
        ticker_list = ", ".join(f"'{t}'" for t in table["Ticker"])
        st.subheader("Ticker list (copy/paste friendly)")
        st.code(ticker_list, language=None)


if __name__ == "__main__":
    main()
