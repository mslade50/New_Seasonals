"""
Absolute Return Dispersion — Full S&P 500 Constituents
========================================================
Replicates the Nomura Vol "S&P 500 Absolute Return Dispersion" metric:

    Dispersion = Mean(|constituent 1-month return|) - |index 1-month return|

Uses the full S&P 500 constituent list (~503 tickers) for accurate
cross-sectional dispersion, with batched yfinance downloads.

RUNNING LOCALLY:
    python abs_return_dispersion.py

    This will:
      1. Download ~505 S&P 500 tickers via yfinance (~30-45 sec first run)
      2. Compute the dispersion series back to 1998
      3. Print current reading + historical stats to console
      4. Save dispersion_output.csv (daily time series)
      5. Save dispersion_chart.html (interactive Plotly chart, opens in browser)
      6. Optionally cache price data to parquet for fast re-runs

    Output CSV columns:
      - avg_abs_ret      : mean |constituent 21d return| (decimal, e.g. 0.108 = 10.8%)
      - index_abs_ret    : |SPY 21d return| (decimal)
      - dispersion       : avg_abs_ret - index_abs_ret (the Nomura metric)
      - dispersion_rank  : expanding percentile rank (0-100)
      - n_constituents   : how many tickers had data that day

YFINANCE MULTIINDEX HANDLING:
    yfinance now returns MultiIndex columns: (Price, Ticker) for multi-ticker
    downloads. Every function in this module handles this explicitly.
    The defensive pattern used everywhere is:

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

    For the batch downloader, we extract Close prices using:
        raw["Close"]  → gives a DataFrame with ticker columns

Author: McKinley trading system
Source: Nomura Vol (via @ConnorJBates_)
"""

import pandas as pd
import numpy as np
import time
from typing import Optional, Dict
from pathlib import Path

# ============================================================================
# S&P 500 CONSTITUENTS (as of early 2025)
# Update periodically — additions/deletions don't materially affect the metric
# since we're averaging across ~500 names.
#
# To refresh: scrape Wikipedia's "List of S&P 500 companies" page.
# ============================================================================
SP500_TICKERS = [
    "A", "AAPL", "ABBV", "ABNB", "ABT", "ACGL", "ACN", "ADBE", "ADI", "ADM",
    "ADP", "ADSK", "AEE", "AEP", "AES", "AFL", "AIG", "AIZ", "AJG", "AKAM",
    "ALB", "ALGN", "ALL", "ALLE", "AMAT", "AMCR", "AMD", "AME", "AMGN", "AMP",
    "AMT", "AMZN", "ANET", "ANSS", "AON", "AOS", "APA", "APD", "APH", "APTV",
    "ARE", "ATO", "ATVI", "AVB", "AVGO", "AVY", "AWK", "AXP", "AZO",
    "BA", "BAC", "BAX", "BBWI", "BBY", "BDX", "BEN", "BF-B", "BG", "BIIB",
    "BIO", "BK", "BKNG", "BKR", "BLDR", "BLK", "BMY", "BR", "BRK-B", "BRO",
    "BSX", "BWA", "BX", "BXP",
    "C", "CAG", "CAH", "CARR", "CAT", "CB", "CBOE", "CBRE", "CCI", "CCL",
    "CDAY", "CDNS", "CDW", "CE", "CEG", "CF", "CFG", "CHD", "CHRW", "CHTR",
    "CI", "CINF", "CL", "CLX", "CMCSA", "CME", "CMG", "CMI", "CMS", "CNC",
    "CNP", "COF", "COO", "COP", "COR", "COST", "CPAY", "CPB", "CPRT", "CPT",
    "CRL", "CRM", "CRWD", "CSCO", "CSGP", "CSX", "CTAS", "CTLT", "CTRA",
    "CTSH", "CTVA", "CVS", "CVX", "CZR",
    "D", "DAL", "DAY", "DD", "DE", "DECK", "DFS", "DG", "DGX", "DHI",
    "DHR", "DIS", "DISH", "DLR", "DLTR", "DOV", "DOW", "DPZ", "DRI",
    "DTE", "DUK", "DVA", "DVN",
    "DXCM", "EA", "EBAY", "ECL", "ED", "EFX", "EIX", "EL", "EMN", "EMR",
    "ENPH", "EOG", "EPAM", "EQIX", "EQR", "EQT", "ES", "ESS", "ETN", "ETR",
    "ETSY", "EVRG", "EW", "EXC", "EXPD", "EXPE", "EXR",
    "F", "FANG", "FAST", "FBHS", "FCX", "FDS", "FDX", "FE", "FFIV", "FI",
    "FICO", "FIS", "FISV", "FITB", "FLT", "FMC", "FOX", "FOXA", "FRT",
    "FSLR", "FTNT", "FTV",
    "GD", "GDDY", "GE", "GEHC", "GEN", "GEV", "GILD", "GIS", "GL", "GLW",
    "GM", "GNRC", "GOOG", "GOOGL", "GPC", "GPN", "GRMN", "GS", "GWW",
    "HAL", "HAS", "HBAN", "HCA", "HD", "HOLX", "HON", "HPE", "HPQ", "HRL",
    "HSIC", "HST", "HSY", "HUBB", "HUM", "HWM", "HII",
    "IBM", "ICE", "IDXX", "IEX", "IFF", "ILMN", "INCY", "INTC", "INTU",
    "INVH", "IP", "IPG", "IQV", "IR", "IRM", "ISRG", "IT", "ITW",
    "J", "JBHT", "JBL", "JCI", "JKHY", "JNJ", "JNPR", "JPM",
    "K", "KDP", "KEY", "KEYS", "KHC", "KIM", "KLAC", "KMB", "KMI", "KMX",
    "KO", "KR",
    "L", "LDOS", "LEN", "LH", "LHX", "LIN", "LKQ", "LLY", "LMT", "LNT",
    "LOW", "LRCX", "LULU", "LUV", "LVS", "LW", "LYB", "LYV",
    "MA", "MAA", "MAR", "MAS", "MCD", "MCHP", "MCK", "MCO", "MDLZ", "MDT",
    "MET", "META", "MGM", "MHK", "MKC", "MKTX", "MLM", "MMC", "MMM", "MNST",
    "MO", "MOH", "MOS", "MPC", "MPWR", "MRK", "MRNA", "MRO", "MS", "MSCI",
    "MSFT", "MSI", "MTB", "MTCH", "MTD", "MU",
    "NCLH", "NDAQ", "NDSN", "NEE", "NEM", "NFLX", "NI", "NKE", "NOC",
    "NOW", "NRG", "NSC", "NTAP", "NTRS", "NUE", "NVDA", "NVR", "NWS", "NWSA",
    "NXPI",
    "O", "ODFL", "OKE", "OMC", "ON", "ORCL", "ORLY", "OTIS", "OXY",
    "PARA", "PAYC", "PAYX", "PCAR", "PCG", "PEAK", "PEG", "PEP", "PFE",
    "PFG", "PG", "PGR", "PH", "PHM", "PKG", "PLD", "PLTR", "PM", "PNC",
    "PNR", "PNW", "PODD", "POOL", "PPG", "PPL", "PRU", "PSA", "PSX", "PTC",
    "PVH", "PWR", "PXD", "PYPL",
    "QCOM", "QRVO",
    "RCL", "REG", "REGN", "RF", "RHI", "RJF", "RL", "RMD", "ROK", "ROL",
    "ROP", "ROST", "RSG", "RTX",
    "SBAC", "SBUX", "SCHW", "SEE", "SHW", "SIVB", "SJM", "SLB", "SMCI",
    "SNA", "SNPS", "SO", "SPG", "SPGI", "SRE", "STE", "STLD", "STT", "STX",
    "STZ", "SWK", "SWKS", "SYF", "SYK", "SYY",
    "T", "TAP", "TDG", "TDY", "TECH", "TEL", "TER", "TFC", "TFX", "TGT",
    "TJX", "TMO", "TMUS", "TPR", "TRGP", "TRMB", "TROW", "TRV", "TSCO",
    "TSLA", "TSN", "TT", "TTWO", "TXN", "TXT", "TYL",
    "UAL", "UBER", "UDR", "UHS", "ULTA", "UNH", "UNP", "UPS", "URI", "USB",
    "V", "VICI", "VLO", "VLTO", "VMC", "VRSK", "VRSN", "VRTX", "VST",
    "VTR", "VTRS", "VZ",
    "WAB", "WAT", "WBA", "WBD", "WDC", "WEC", "WELL", "WFC", "WHR", "WM",
    "WMB", "WMT", "WRB", "WRK", "WST", "WTW", "WY", "WYNN",
    "XEL", "XOM", "XRAY", "XYL",
    "YUM",
    "ZBH", "ZBRA", "ZION", "ZTS",
]

_SP500_SET = set(SP500_TICKERS)


# ============================================================================
# YFINANCE MULTIINDEX HELPER
# ============================================================================
def _extract_close_from_yf_download(raw: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """
    Extract a clean Close-price DataFrame from a yfinance multi-ticker download.
    
    yfinance returns MultiIndex columns: (Price, Ticker) for multi-ticker downloads.
    e.g. ("Close", "AAPL"), ("Close", "MSFT"), ("Open", "AAPL"), ...
    
    For single-ticker downloads, returns flat columns: "Close", "Open", ...
    
    Returns: DataFrame with index=date, columns=ticker symbols, values=Close price.
    """
    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        # Multi-ticker download → MultiIndex (Price, Ticker)
        # Level 0 = Price type (Close, Open, High, Low, Volume)
        # Level 1 = Ticker symbol
        level_0_values = raw.columns.get_level_values(0).unique().tolist()
        
        if "Close" in level_0_values:
            close_df = raw["Close"].copy()
        elif "close" in level_0_values:
            close_df = raw["close"].copy()
        elif "Adj Close" in level_0_values:
            close_df = raw["Adj Close"].copy()
        else:
            raise ValueError(
                f"Cannot find 'Close' in MultiIndex level 0. "
                f"Available levels: {level_0_values}"
            )
    else:
        # Single-ticker download → flat columns
        flat_cols = [str(c).capitalize() for c in raw.columns]
        raw.columns = flat_cols
        if "Close" in raw.columns:
            close_df = raw[["Close"]].copy()
            # Name the column after the single ticker
            close_df.columns = [tickers[0]] if len(tickers) == 1 else ["UNKNOWN"]
        else:
            raise ValueError(f"No 'Close' column found. Columns: {list(raw.columns)}")

    # Normalize column names to uppercase ticker symbols
    close_df.columns = [str(c).strip().upper() for c in close_df.columns]

    # Normalize timezone
    if close_df.index.tz is not None:
        close_df.index = close_df.index.tz_localize(None)

    return close_df


def _clean_single_ticker_df(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Extract a Close price Series from a single-ticker DataFrame.
    Handles both MultiIndex (from raw yf.download) and flat columns
    (from data_dict after slicing).
    
    Returns: pd.Series of Close prices, or None if extraction fails.
    """
    if df is None or df.empty:
        return None

    # Handle MultiIndex columns (yfinance trap)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    # Normalize column names
    col_map = {c: str(c).strip().capitalize() for c in df.columns}
    df = df.rename(columns=col_map)

    if "Close" in df.columns:
        series = df["Close"].dropna()
        # Normalize timezone
        if series.index.tz is not None:
            series.index = series.index.tz_localize(None)
        return series
    
    return None


# ============================================================================
# CORE CALCULATION
# ============================================================================
def compute_dispersion_series(
    price_matrix: pd.DataFrame,
    index_col: str = "SPY",
    window: int = 21,
    rank_min_periods: int = 252,
) -> pd.DataFrame:
    """
    Core dispersion calculation. Works on any Close-price matrix.

    Parameters
    ----------
    price_matrix : DataFrame
        Index = dates, columns = tickers (close prices). Must include `index_col`.
    index_col : str
        Column name for the benchmark. Default "SPY".
    window : int
        Return lookback in trading days. Default 21 (~1 month).
    rank_min_periods : int
        Minimum observations before percentile rank is computed. Default 252.

    Returns
    -------
    DataFrame with columns:
        - avg_abs_ret      : mean |constituent return| over the window
        - index_abs_ret    : |index return| over the window
        - dispersion       : avg_abs_ret - index_abs_ret
        - dispersion_rank  : expanding percentile rank (0-100) of dispersion
        - n_constituents   : how many constituents had data that day
    """
    if index_col not in price_matrix.columns:
        raise ValueError(
            f"Index column '{index_col}' not found. "
            f"Available: {sorted(price_matrix.columns.tolist())[:10]}..."
        )

    constituent_cols = [c for c in price_matrix.columns if c != index_col]
    if not constituent_cols:
        raise ValueError("Price matrix has no constituent columns (only the index).")

    # Window returns (simple pct_change, not log — matches Nomura)
    constituent_returns = price_matrix[constituent_cols].pct_change(window)
    index_returns = price_matrix[index_col].pct_change(window)

    # Absolute returns
    abs_constituent_returns = constituent_returns.abs()
    abs_index_return = index_returns.abs()

    # Cross-sectional mean of absolute constituent returns (each day)
    avg_abs_ret = abs_constituent_returns.mean(axis=1, skipna=True)

    # How many constituents contributed (data quality check)
    n_constituents = abs_constituent_returns.count(axis=1)

    # The Nomura metric
    dispersion = avg_abs_ret - abs_index_return

    # Expanding percentile rank (matches "Nth percentile over past 30 years")
    dispersion_rank = (
        dispersion
        .expanding(min_periods=rank_min_periods)
        .rank(pct=True) * 100.0
    )

    return pd.DataFrame({
        "avg_abs_ret": avg_abs_ret,
        "index_abs_ret": abs_index_return,
        "dispersion": dispersion,
        "dispersion_rank": dispersion_rank,
        "n_constituents": n_constituents,
    }, index=price_matrix.index)


# ============================================================================
# DICT-BASED INTERFACE (for daily_scan.py / strat_backtester.py)
# ============================================================================
def compute_dispersion_from_dict(
    data_dict: Dict[str, pd.DataFrame],
    index_ticker: str = "SPY",
    constituent_tickers: Optional[list] = None,
    window: int = 21,
    rank_min_periods: int = 252,
) -> pd.DataFrame:
    """
    Build a price matrix from your existing {ticker: DataFrame} dict
    and compute dispersion. This is the main integration point for daily_scan.py.

    Parameters
    ----------
    data_dict : dict
        {ticker: DataFrame} — standard format from download_historical_data().
        Each DataFrame must have a 'Close' column. Handles MultiIndex columns.
    index_ticker : str
        Benchmark ticker. Default "SPY".
    constituent_tickers : list or None
        If None, uses SP500_TICKERS.
    """
    if constituent_tickers is None:
        constituent_tickers = SP500_TICKERS

    # Build price matrix by extracting Close from each ticker's DataFrame
    close_series = {}

    # Always include the index
    all_tickers_to_check = list(set(constituent_tickers + [index_ticker]))

    for ticker in all_tickers_to_check:
        df = data_dict.get(ticker)
        series = _clean_single_ticker_df(df)
        if series is not None and len(series) > 0:
            close_series[ticker] = series

    if index_ticker not in close_series:
        raise ValueError(f"Index ticker '{index_ticker}' not found in data_dict.")

    price_matrix = pd.DataFrame(close_series)

    available = [c for c in price_matrix.columns if c != index_ticker]
    total_requested = len(constituent_tickers)
    found = len(available)

    if found < 50:
        print(f"[dispersion] WARNING: Only {found}/{total_requested} constituents. "
              f"Results may not be representative.")
    else:
        print(f"[dispersion] {found}/{total_requested} constituents "
              f"({found/total_requested*100:.0f}% coverage)")

    return compute_dispersion_series(
        price_matrix, index_col=index_ticker,
        window=window, rank_min_periods=rank_min_periods,
    )


# ============================================================================
# BATCHED YFINANCE DOWNLOADER
# ============================================================================
def _download_close_prices(
    tickers: list,
    start_date: str = "1998-01-01",
    chunk_size: int = 50,
    sleep_between: float = 0.3,
) -> pd.DataFrame:
    """
    Download Close prices for a large ticker list via yfinance in batches.
    Returns: DataFrame with index=date, columns=tickers, values=Close.
    """
    import yfinance as yf

    clean_tickers = sorted(set(
        str(t).strip().upper().replace(".", "-") for t in tickers
    ))
    total = len(clean_tickers)
    all_frames = []

    print(f"[dispersion] Downloading {total} tickers in batches of {chunk_size}...")

    for i in range(0, total, chunk_size):
        chunk = clean_tickers[i : i + chunk_size]
        batch_num = i // chunk_size + 1
        total_batches = (total + chunk_size - 1) // chunk_size
        print(f"  Batch {batch_num}/{total_batches} ({len(chunk)} tickers)...", end=" ")

        try:
            raw = yf.download(
                chunk, start=start_date,
                auto_adjust=True, progress=False, threads=True,
            )

            if raw.empty:
                print("empty")
                continue

            close_df = _extract_close_from_yf_download(raw, chunk)
            if not close_df.empty:
                all_frames.append(close_df)
                print(f"{len(close_df.columns)} tickers OK")
            else:
                print("no Close data")

        except Exception as e:
            print(f"ERROR: {e}")
            if "rate" in str(e).lower():
                print("  Rate limited — waiting 5s...")
                time.sleep(5)

        time.sleep(sleep_between)

    if not all_frames:
        raise RuntimeError("No data downloaded. Check yfinance / network connectivity.")

    combined = pd.concat(all_frames, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]

    # Final timezone normalization
    if combined.index.tz is not None:
        combined.index = combined.index.tz_localize(None)

    print(f"[dispersion] Total: {len(combined.columns)} tickers, "
          f"{len(combined)} days "
          f"({combined.index[0].strftime('%Y-%m-%d')} → "
          f"{combined.index[-1].strftime('%Y-%m-%d')})")

    return combined


# ============================================================================
# STANDALONE ENTRY POINTS
# ============================================================================
def download_and_compute(
    start_date: str = "1998-01-01",
    window: int = 21,
    index_ticker: str = "SPY",
    constituent_tickers: Optional[list] = None,
    chunk_size: int = 50,
    rank_min_periods: int = 252,
) -> pd.DataFrame:
    """Download full S&P 500 and compute dispersion."""
    if constituent_tickers is None:
        constituent_tickers = SP500_TICKERS

    all_tickers = sorted(set(constituent_tickers + [index_ticker]))
    price_matrix = _download_close_prices(all_tickers, start_date, chunk_size)

    idx = index_ticker.upper()
    if idx not in price_matrix.columns:
        raise ValueError(f"Index '{idx}' not in downloaded data.")

    return compute_dispersion_series(
        price_matrix, index_col=idx,
        window=window, rank_min_periods=rank_min_periods,
    )


def download_and_compute_cached(
    cache_path: str = "data/sp500_dispersion_prices.parquet",
    start_date: str = "1998-01-01",
    window: int = 21,
    max_cache_age_days: int = 1,
    **kwargs,
) -> pd.DataFrame:
    """
    Same as download_and_compute but caches the PRICE MATRIX to parquet.
    You can change window/rank_min_periods without re-downloading.
    """
    cache_file = Path(cache_path)
    use_cache = False

    if cache_file.exists():
        cache_age = time.time() - cache_file.stat().st_mtime
        if cache_age < max_cache_age_days * 86400:
            use_cache = True

    if use_cache:
        print(f"[dispersion] Loading cached prices from {cache_path}")
        price_matrix = pd.read_parquet(cache_path)
    else:
        all_tickers = sorted(set(SP500_TICKERS + ["SPY"]))
        price_matrix = _download_close_prices(all_tickers, start_date)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        price_matrix.to_parquet(cache_path)
        print(f"[dispersion] Cached prices → {cache_path}")

    index_ticker = kwargs.get("index_ticker", "SPY")
    rank_min_periods = kwargs.get("rank_min_periods", 252)

    return compute_dispersion_series(
        price_matrix, index_col=index_ticker,
        window=window, rank_min_periods=rank_min_periods,
    )


# ============================================================================
# DAILY_SCAN HELPER
# ============================================================================
def get_dispersion_for_daily_scan(
    master_dict: Dict[str, pd.DataFrame],
    window: int = 21,
) -> pd.DataFrame:
    """
    Call from daily_scan.py after master_dict is populated.
    Downloads any missing S&P 500 tickers, then computes dispersion.
    """
    available = [t for t in SP500_TICKERS if t in master_dict]
    missing = [t for t in SP500_TICKERS if t not in master_dict]

    print(f"[dispersion] {len(available)}/{len(SP500_TICKERS)} already in master_dict, "
          f"{len(missing)} to download")

    if missing:
        # Match the start date of existing data
        sample_dates = []
        for df in list(master_dict.values())[:5]:
            if df is not None and not df.empty:
                sample_dates.append(df.index[0])
        start = min(sample_dates).strftime("%Y-%m-%d") if sample_dates else "2020-01-01"

        extra = _download_close_prices(missing, start, chunk_size=50, sleep_between=0.3)
        for ticker in extra.columns:
            if ticker not in master_dict:
                master_dict[ticker] = pd.DataFrame({"Close": extra[ticker]})

    return compute_dispersion_from_dict(
        master_dict, index_ticker="SPY",
        constituent_tickers=SP500_TICKERS, window=window,
    )


# ============================================================================
# HTML CHART GENERATOR (no Plotly dependency — pure HTML/JS with Chart.js CDN)
# ============================================================================
def _generate_html_chart(df: pd.DataFrame, output_path: str, window: int = 21):
    """
    Generate a self-contained HTML file with an interactive dispersion chart.
    Uses Chart.js via CDN — no local dependencies needed.
    """
    # Subsample for chart performance (every 5th trading day)
    chart_df = df.dropna(subset=["dispersion"]).iloc[::5].copy()

    dates = chart_df.index.strftime("%Y-%m-%d").tolist()
    disp_vals = (chart_df["dispersion"] * 100).round(2).tolist()
    rank_vals = chart_df["dispersion_rank"].round(1).tolist()

    p95 = df["dispersion"].quantile(0.95) * 100
    p99 = df["dispersion"].quantile(0.99) * 100
    latest = df.dropna(subset=["dispersion"]).iloc[-1]

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>S&P 500 Absolute Return Dispersion</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.0.1/dist/chartjs-plugin-annotation.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Courier New', monospace; background: #0d0d1a; color: #e0e0e0; padding: 24px; }}
        h1 {{ font-size: 20px; color: #fff; margin-bottom: 4px; }}
        .subtitle {{ color: #888; font-size: 13px; margin-bottom: 20px; }}
        .stats {{ display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }}
        .stat {{ background: #151528; border: 1px solid #252540; border-radius: 6px; padding: 10px 16px; min-width: 120px; }}
        .stat-label {{ font-size: 10px; color: #666; text-transform: uppercase; letter-spacing: 0.05em; }}
        .stat-value {{ font-size: 20px; font-weight: 700; margin-top: 2px; }}
        .callout {{ background: rgba(255,217,61,0.08); border: 1px solid rgba(255,217,61,0.2);
                    border-radius: 8px; padding: 12px 16px; margin-bottom: 20px; font-size: 13px; }}
        .chart-container {{ background: #111122; border: 1px solid #252540; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
        canvas {{ max-height: 400px; }}
        .tabs {{ display: flex; gap: 4px; margin-bottom: 16px; }}
        .tab {{ padding: 6px 14px; font-size: 11px; font-weight: 600; border-radius: 4px; border: none;
                cursor: pointer; font-family: inherit; text-transform: uppercase; letter-spacing: 0.04em; }}
        .tab.active {{ background: #3366ff; color: #fff; }}
        .tab.inactive {{ background: #1a1a2e; color: #888; }}
        .source {{ font-size: 10px; color: #444; text-align: right; margin-top: 12px; }}
    </style>
</head>
<body>
    <h1>S&P 500 Absolute Return Dispersion</h1>
    <p class="subtitle">Mean |Constituent {window}-Day Return| &minus; |Index {window}-Day Return| &nbsp; ({int(latest['n_constituents'])} constituents)</p>

    <div class="callout">
        Current: <strong style="color:#fff">{latest['dispersion']*100:.1f}%</strong> dispersion
        &mdash; <strong style="color:{'#ff6b6b' if latest['dispersion_rank'] > 95 else '#ffd93d' if latest['dispersion_rank'] > 80 else '#6bff6b'}">{latest['dispersion_rank']:.0f}th percentile</strong>
        &nbsp; (avg constituent moved {latest['avg_abs_ret']*100:.1f}%, index moved {latest['index_abs_ret']*100:.1f}%)
    </div>

    <div class="stats">
        <div class="stat"><div class="stat-label">Current</div><div class="stat-value" style="color:#fff">{latest['dispersion']*100:.1f}%</div></div>
        <div class="stat"><div class="stat-label">Rank</div><div class="stat-value" style="color:{'#ff6b6b' if latest['dispersion_rank'] > 95 else '#ffd93d'}">{latest['dispersion_rank']:.0f}th</div></div>
        <div class="stat"><div class="stat-label">Mean</div><div class="stat-value" style="color:#aaa">{df['dispersion'].mean()*100:.1f}%</div></div>
        <div class="stat"><div class="stat-label">95th Pctl</div><div class="stat-value" style="color:#ffa500">{p95:.1f}%</div></div>
        <div class="stat"><div class="stat-label">99th Pctl</div><div class="stat-value" style="color:#ff4444">{p99:.1f}%</div></div>
        <div class="stat"><div class="stat-label">Constituents</div><div class="stat-value" style="color:#aaa">{int(latest['n_constituents'])}</div></div>
    </div>

    <div class="tabs">
        <button class="tab active" id="btn-disp" onclick="showChart('disp')">Dispersion %</button>
        <button class="tab inactive" id="btn-rank" onclick="showChart('rank')">Percentile Rank</button>
    </div>

    <div class="chart-container">
        <canvas id="dispChart"></canvas>
    </div>
    <div class="chart-container" style="display:none">
        <canvas id="rankChart"></canvas>
    </div>

    <div class="source">Source: Nomura Vol methodology &middot; S&P 500 constituents via yfinance &middot; Generated {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</div>

<script>
const dates = {dates};
const dispVals = {disp_vals};
const rankVals = {rank_vals};

const dispCtx = document.getElementById('dispChart').getContext('2d');
const dispChart = new Chart(dispCtx, {{
    type: 'line',
    data: {{
        labels: dates,
        datasets: [{{
            label: 'Dispersion %',
            data: dispVals,
            borderColor: '#8888cc',
            backgroundColor: 'rgba(58, 58, 92, 0.4)',
            fill: true,
            pointRadius: 0,
            borderWidth: 1,
            tension: 0.1,
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{
            annotation: {{
                annotations: {{
                    p95: {{ type: 'line', yMin: {p95:.2f}, yMax: {p95:.2f}, borderColor: 'orange', borderDash: [6,4], borderWidth: 1, label: {{ content: '95th', display: true, position: 'start', font: {{size:10}}, color: 'orange' }} }},
                    p99: {{ type: 'line', yMin: {p99:.2f}, yMax: {p99:.2f}, borderColor: 'red', borderDash: [6,4], borderWidth: 1, label: {{ content: '99th', display: true, position: 'start', font: {{size:10}}, color: 'red' }} }},
                    zero: {{ type: 'line', yMin: 0, yMax: 0, borderColor: '#444', borderWidth: 0.5 }},
                }}
            }},
            legend: {{ display: false }},
            tooltip: {{ callbacks: {{ label: (ctx) => ctx.parsed.y.toFixed(1) + '%' }} }}
        }},
        scales: {{
            x: {{ ticks: {{ maxTicksLimit: 10, color: '#666', font: {{size:10}} }}, grid: {{ color: '#1a1a35' }} }},
            y: {{ ticks: {{ callback: (v) => v + '%', color: '#666', font: {{size:10}} }}, grid: {{ color: '#1a1a35' }} }}
        }}
    }}
}});

const rankCtx = document.getElementById('rankChart').getContext('2d');
const rankChart = new Chart(rankCtx, {{
    type: 'line',
    data: {{
        labels: dates,
        datasets: [{{
            label: 'Percentile Rank',
            data: rankVals,
            borderColor: '#4a8adf',
            backgroundColor: 'rgba(42, 74, 122, 0.4)',
            fill: true,
            pointRadius: 0,
            borderWidth: 1,
            tension: 0.1,
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{
            annotation: {{
                annotations: {{
                    p95line: {{ type: 'line', yMin: 95, yMax: 95, borderColor: 'red', borderDash: [6,4], borderWidth: 1 }},
                    p80line: {{ type: 'line', yMin: 80, yMax: 80, borderColor: 'orange', borderDash: [6,4], borderWidth: 1 }},
                }}
            }},
            legend: {{ display: false }},
        }},
        scales: {{
            x: {{ ticks: {{ maxTicksLimit: 10, color: '#666', font: {{size:10}} }}, grid: {{ color: '#1a1a35' }} }},
            y: {{ min: 0, max: 105, ticks: {{ color: '#666', font: {{size:10}} }}, grid: {{ color: '#1a1a35' }} }}
        }}
    }}
}});

function showChart(which) {{
    const containers = document.querySelectorAll('.chart-container');
    containers[0].style.display = which === 'disp' ? 'block' : 'none';
    containers[1].style.display = which === 'rank' ? 'block' : 'none';
    document.getElementById('btn-disp').className = 'tab ' + (which === 'disp' ? 'active' : 'inactive');
    document.getElementById('btn-rank').className = 'tab ' + (which === 'rank' ? 'active' : 'inactive');
}}
</script>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)


# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    import argparse
    import webbrowser
    import os

    parser = argparse.ArgumentParser(
        description="S&P 500 Absolute Return Dispersion (Nomura Vol methodology)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output files:
  dispersion_output.csv    Daily time series (all columns)
  dispersion_chart.html    Interactive chart (opens in browser)

Examples:
  python abs_return_dispersion.py                          # Download & compute, open chart
  python abs_return_dispersion.py --cache data/prices.parquet  # Cache prices for fast re-runs
  python abs_return_dispersion.py --window 42              # 2-month window instead of 1-month
  python abs_return_dispersion.py --no-browser             # Don't auto-open chart
        """
    )
    parser.add_argument("--start", default="1998-01-01", help="Start date (default: 1998-01-01)")
    parser.add_argument("--window", type=int, default=21, help="Return window in trading days (default: 21)")
    parser.add_argument("--cache", default=None, help="Parquet cache path for price matrix")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open the HTML chart")
    parser.add_argument("--csv-path", default="dispersion_output.csv", help="Output CSV path")
    parser.add_argument("--html-path", default="dispersion_chart.html", help="Output HTML path")
    args = parser.parse_args()

    print("=" * 60)
    print("S&P 500 ABSOLUTE RETURN DISPERSION")
    print(f"Window: {args.window} trading days | Start: {args.start}")
    print(f"Constituents: {len(SP500_TICKERS)} S&P 500 tickers + SPY")
    print("=" * 60)

    # Compute
    if args.cache:
        df = download_and_compute_cached(
            cache_path=args.cache, start_date=args.start, window=args.window,
        )
    else:
        df = download_and_compute(start_date=args.start, window=args.window)

    df_clean = df.dropna(subset=["dispersion"])

    # --- Console output ---
    latest = df_clean.iloc[-1]
    print(f"\n{'─'*40}")
    print(f"LATEST READING ({df_clean.index[-1].strftime('%Y-%m-%d')})")
    print(f"{'─'*40}")
    print(f"  Constituents reporting:      {int(latest['n_constituents'])}")
    print(f"  Avg stock |{args.window}d return|:     {latest['avg_abs_ret']*100:.1f}%")
    print(f"  SPY |{args.window}d return|:           {latest['index_abs_ret']*100:.1f}%")
    print(f"  DISPERSION:                  {latest['dispersion']*100:.1f}%")
    print(f"  PERCENTILE RANK:             {latest['dispersion_rank']:.0f}th")

    print(f"\n{'─'*40}")
    print(f"HISTORICAL DISTRIBUTION")
    print(f"{'─'*40}")
    print(f"  Period: {df_clean.index[0].strftime('%Y-%m-%d')} → {df_clean.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Mean:         {df_clean['dispersion'].mean()*100:.1f}%")
    print(f"  Median:       {df_clean['dispersion'].median()*100:.1f}%")
    print(f"  Std Dev:      {df_clean['dispersion'].std()*100:.1f}%")
    print(f"  5th pctl:     {df_clean['dispersion'].quantile(0.05)*100:.1f}%")
    print(f"  25th pctl:    {df_clean['dispersion'].quantile(0.25)*100:.1f}%")
    print(f"  75th pctl:    {df_clean['dispersion'].quantile(0.75)*100:.1f}%")
    print(f"  95th pctl:    {df_clean['dispersion'].quantile(0.95)*100:.1f}%")
    print(f"  99th pctl:    {df_clean['dispersion'].quantile(0.99)*100:.1f}%")

    # --- Save CSV ---
    df_clean.to_csv(args.csv_path)
    print(f"\n✅ CSV saved → {args.csv_path}")
    print(f"   ({len(df_clean)} rows, columns: {list(df_clean.columns)})")

    # --- Generate HTML chart ---
    _generate_html_chart(df_clean, args.html_path, window=args.window)
    print(f"✅ Chart saved → {args.html_path}")

    # --- Open in browser ---
    if not args.no_browser:
        abs_path = os.path.abspath(args.html_path)
        webbrowser.open(f"file://{abs_path}")
        print(f"🌐 Opened in browser")