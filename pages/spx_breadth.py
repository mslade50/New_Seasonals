"""
SPX Market Breadth — Bloomberg-style 6-panel dashboard
======================================================
Stacked time series mirroring the Bloomberg SPX breadth template:

  1. SPX candlesticks + 100-week MA + 200 DMA + 50 DMA
  2. Advancers minus Decliners (S&P 500 constituents)
  3. % New 52-week Highs / % New 52-week Lows
  4. % Above 200 DMA / % Above 50 DMA
  5. % with 14-day RSI > 70 / % with 14-day RSI < 30
  6. CBOE S&P 500 Dispersion Index (DSPX) + CBOE Put/Call Ratio when available

Standalone page. No imports from strategy modules. Reuses the same SP500 close
cache produced by `pages/risk_dashboard_v2.py` (`data/rd2_sp500_closes.parquet`).
"""

import os
import sys
import time
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    st.set_page_config(page_title="SPX Breadth", layout="wide")
except Exception:
    pass

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

DATA_DIR = os.path.join(parent_dir, "data")
CACHE_SP500 = os.path.join(DATA_DIR, "rd2_sp500_closes.parquet")

LOOKBACK_OPTIONS = {
    "6M": 126,
    "1Y": 252,
    "2Y": 504,
    "5Y": 1260,
    "10Y": 2520,
    "Max": None,
}


# ---------------------------------------------------------------------------
# DATA LOADERS
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_spx_ohlc(start: str) -> pd.DataFrame:
    raw = yf.download("^GSPC", start=start, auto_adjust=False,
                      progress=False, threads=False)
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [str(c).capitalize() for c in raw.columns]
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in raw.columns]
    df = raw[keep].copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_dspx(start: str) -> pd.Series:
    raw = yf.download("^DSPX", start=start, auto_adjust=False,
                      progress=False, threads=False)
    if raw is None or raw.empty:
        return pd.Series(dtype=float, name="DSPX")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [str(c).capitalize() for c in raw.columns]
    s = raw["Close"].copy() if "Close" in raw.columns else pd.Series(dtype=float)
    s.name = "DSPX"
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    return s


@st.cache_data(ttl=1800, show_spinner=False)
def load_cpc(start: str, ratio: str = "index") -> pd.Series:
    """
    CBOE Put/Call Ratio from the scraped daily-statistics cache.
    `ratio` is one of: total, index, equity, etp, spx, oex.
    Returns empty Series if the cache is missing.
    """
    try:
        from cboe_putcall import load_series
    except ImportError:
        return pd.Series(dtype=float, name="CPC")
    s = load_series(start=start, column=ratio)
    s.name = "CPC"
    return s


def _load_sp500_closes_cache() -> pd.DataFrame:
    if not os.path.exists(CACHE_SP500):
        return pd.DataFrame()
    df = pd.read_parquet(CACHE_SP500)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def _refresh_sp500_closes(start: str) -> pd.DataFrame:
    """Re-download S&P 500 closes via yfinance and overwrite the parquet cache."""
    try:
        from abs_return_dispersion import SP500_TICKERS
    except ImportError:
        st.error("SP500_TICKERS not importable from abs_return_dispersion.")
        return pd.DataFrame()

    clean = sorted(set(str(t).strip().upper().replace(".", "-") for t in SP500_TICKERS))
    chunk_size = 50
    total = (len(clean) + chunk_size - 1) // chunk_size
    frames = []
    pbar = st.progress(0.0, text=f"Downloading S&P 500 closes (0/{total})...")

    for i in range(0, len(clean), chunk_size):
        chunk = clean[i:i + chunk_size]
        batch = i // chunk_size + 1
        try:
            raw = yf.download(chunk, start=start, auto_adjust=True,
                              progress=False, threads=True)
            if raw is None or raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                lvl0 = raw.columns.get_level_values(0).unique().tolist()
                key = "Close" if "Close" in lvl0 else None
                if key is None:
                    continue
                close_df = raw[key].copy()
            else:
                cols = [str(c).capitalize() for c in raw.columns]
                raw.columns = cols
                if "Close" not in raw.columns:
                    continue
                close_df = raw[["Close"]].copy()
                close_df.columns = [chunk[0]]
            close_df.columns = [str(c).strip().upper() for c in close_df.columns]
            if close_df.index.tz is not None:
                close_df.index = close_df.index.tz_localize(None)
            frames.append(close_df)
        except Exception as e:
            print(f"sp500 batch {batch} failed: {e}")
        pbar.progress(batch / total, text=f"Downloading S&P 500 closes ({batch}/{total})...")
        time.sleep(0.2)

    pbar.empty()
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    combined.to_parquet(CACHE_SP500)
    return combined


# ---------------------------------------------------------------------------
# BREADTH COMPUTATIONS (vectorized over wide closes DataFrame)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def compute_breadth(closes_key: float, _closes: pd.DataFrame) -> pd.DataFrame:
    """
    All breadth panels from a wide [date x ticker] closes frame.

    Returns DataFrame with columns:
      adv_minus_dec, pct_new_52wh, pct_new_52wl,
      pct_above_50, pct_above_200, pct_rsi_over70, pct_rsi_under30
    """
    closes = _closes.sort_index()
    valid = closes.notna()

    # Daily change direction
    chg = closes.diff()
    advancers = (chg > 0).sum(axis=1)
    decliners = (chg < 0).sum(axis=1)
    adv_minus_dec = advancers - decliners

    # 52-week (252 trading days) rolling high/low on a per-ticker basis.
    # min_periods=200 lets the indicator start before the full year fills,
    # matching how Bloomberg renders it from inception.
    high_252 = closes.rolling(252, min_periods=200).max()
    low_252 = closes.rolling(252, min_periods=200).min()

    at_high = (closes >= high_252) & valid
    at_low = (closes <= low_252) & valid
    denom = valid.sum(axis=1).replace(0, np.nan)
    pct_new_52wh = 100.0 * at_high.sum(axis=1) / denom
    pct_new_52wl = -100.0 * at_low.sum(axis=1) / denom  # plotted as negative

    # Moving averages
    sma50 = closes.rolling(50, min_periods=40).mean()
    sma200 = closes.rolling(200, min_periods=150).mean()
    above50 = (closes > sma50) & valid
    above200 = (closes > sma200) & valid
    pct_above_50 = 100.0 * above50.sum(axis=1) / denom
    pct_above_200 = 100.0 * above200.sum(axis=1) / denom

    # RSI(14), Wilder smoothing
    delta = closes.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    # Wilder == EMA with alpha = 1/14 == EWM with com=13
    roll_up = up.ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean()
    roll_down = down.ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.where(valid)

    over70 = (rsi > 70) & valid
    under30 = (rsi < 30) & valid
    pct_over70 = 100.0 * over70.sum(axis=1) / denom
    pct_under30 = -100.0 * under30.sum(axis=1) / denom

    out = pd.DataFrame({
        "adv_minus_dec": adv_minus_dec,
        "pct_new_52wh": pct_new_52wh,
        "pct_new_52wl": pct_new_52wl,
        "pct_above_50": pct_above_50,
        "pct_above_200": pct_above_200,
        "pct_rsi_over70": pct_over70,
        "pct_rsi_under30": pct_under30,
    })
    return out


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("SPX Market Breadth")
st.caption("Bloomberg-style stack — SPX with 100-week / 200 DMA / 50 DMA above five constituent-level breadth panels.")

col_a, col_b, col_c, col_d = st.columns([1, 1, 1, 3])
with col_a:
    lookback = st.selectbox("Lookback", list(LOOKBACK_OPTIONS.keys()), index=1)
with col_b:
    pc_ratio_label = st.selectbox(
        "P/C series",
        ["Index", "Total", "Equity", "ETP", "SPX+SPXW", "OEX"],
        index=0,
        help="Which CBOE put/call ratio to overlay on the bottom panel. Bloomberg's R1 = Index.",
    )
PC_RATIO_KEY = {
    "Index": "index", "Total": "total", "Equity": "equity",
    "ETP": "etp", "SPX+SPXW": "spx", "OEX": "oex",
}[pc_ratio_label]
with col_c:
    if st.button("Refresh data", help="Re-download SP500 closes + scrape any missing CBOE put/call days."):
        with st.spinner("Refreshing SP500 closes..."):
            sp = _refresh_sp500_closes("2015-01-01")
        if sp.empty:
            st.error("SP500 refresh failed.")
        else:
            st.success(f"SP500 closes: {sp.shape[1]} tickers x {sp.shape[0]} days.")
        with st.spinner("Backfilling CBOE put/call (this can take a minute)..."):
            try:
                from cboe_putcall import backfill as _cboe_backfill
                pc = _cboe_backfill(start="2024-01-01", sleep_between=0.4)
                st.success(f"CBOE P/C cache: {len(pc)} rows.")
            except Exception as e:
                st.error(f"CBOE P/C refresh failed: {e}")
        st.cache_data.clear()
        st.rerun()

closes = _load_sp500_closes_cache()
if closes.empty:
    st.error(
        "`data/rd2_sp500_closes.parquet` not found. "
        "Open the Risk Dashboard V2 page and click its data refresh, "
        "or click 'Refresh S&P 500 closes' above."
    )
    st.stop()

# Decide start date based on lookback. We pull a buffer for 252d/200d windows.
last_date = closes.index.max()
buffer_td = 260  # need at least 252d back to warm up the 52w rolling windows
n = LOOKBACK_OPTIONS[lookback]
if n is None:
    plot_start = closes.index.min()
else:
    plot_start = closes.index[max(0, len(closes) - n - 1)]

compute_start = closes.index[max(0, len(closes) - (n + buffer_td if n else len(closes)) - 1)]
window_closes = closes.loc[compute_start:]

# SPX OHLC — pull enough history for the 100-week MA (500 trading days)
spx_start = (plot_start - pd.Timedelta(days=int(500 * 1.6))).strftime("%Y-%m-%d")
spx = load_spx_ohlc(spx_start)
if spx.empty:
    st.error("Could not download ^GSPC from yfinance.")
    st.stop()

dspx = load_dspx(plot_start.strftime("%Y-%m-%d"))
cpc = load_cpc(plot_start.strftime("%Y-%m-%d"), ratio=PC_RATIO_KEY)

# Breadth computations
cache_key = os.path.getmtime(CACHE_SP500) if os.path.exists(CACHE_SP500) else 0.0
breadth = compute_breadth(cache_key, window_closes)

# Slice to plot window
spx_plot = spx.loc[plot_start:].copy()
breadth_plot = breadth.loc[plot_start:].copy()

# SPX moving averages (computed on the longer SPX frame so they are valid on day 1 of the plot window)
spx["MA50"] = spx["Close"].rolling(50, min_periods=40).mean()
spx["MA200"] = spx["Close"].rolling(200, min_periods=150).mean()
# 100-week MA: 100 weekly closes — resample to W-FRI, take 100 SMA, reindex back to daily
weekly_close = spx["Close"].resample("W-FRI").last()
ma_100w = weekly_close.rolling(100, min_periods=80).mean()
spx["MA100W"] = ma_100w.reindex(spx.index, method="ffill")
spx_plot = spx.loc[plot_start:].copy()


# ---------------------------------------------------------------------------
# FIGURE — 6 stacked rows
# ---------------------------------------------------------------------------
fig = make_subplots(
    rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.012,
    row_heights=[0.30, 0.13, 0.13, 0.15, 0.14, 0.15],
    specs=[
        [{}], [{}], [{}], [{}], [{}],
        [{"secondary_y": True}],
    ],
)

# Row 1 — SPX candles + MAs
fig.add_trace(
    go.Candlestick(
        x=spx_plot.index,
        open=spx_plot["Open"], high=spx_plot["High"],
        low=spx_plot["Low"], close=spx_plot["Close"],
        increasing_line_color="#d9d9d9", decreasing_line_color="#d9d9d9",
        increasing_fillcolor="rgba(255,255,255,0.0)",
        decreasing_fillcolor="rgba(255,255,255,0.0)",
        name="SPX", showlegend=False,
    ),
    row=1, col=1,
)
fig.add_trace(go.Scatter(x=spx_plot.index, y=spx_plot["MA100W"],
                         line=dict(color="#22d3ee", width=1.4),
                         name="100w MA"), row=1, col=1)
fig.add_trace(go.Scatter(x=spx_plot.index, y=spx_plot["MA200"],
                         line=dict(color="#a3e635", width=1.4),
                         name="200 DMA"), row=1, col=1)
fig.add_trace(go.Scatter(x=spx_plot.index, y=spx_plot["MA50"],
                         line=dict(color="#fbbf24", width=1.4),
                         name="50 DMA"), row=1, col=1)

# Row 2 — Advancers minus Decliners
amd = breadth_plot["adv_minus_dec"]
colors = np.where(amd >= 0, "#22c55e", "#ef4444")
fig.add_trace(
    go.Bar(x=amd.index, y=amd.values,
           marker_color=colors, marker_line_width=0,
           name="Adv - Dec", showlegend=False),
    row=2, col=1,
)
fig.add_hline(y=0, line=dict(color="#444", width=0.5), row=2, col=1)

# Row 3 — % new 52w highs / lows
fig.add_trace(
    go.Bar(x=breadth_plot.index, y=breadth_plot["pct_new_52wh"],
           marker_color="#22d3ee", marker_line_width=0,
           name="% New 52wk Highs"),
    row=3, col=1,
)
fig.add_trace(
    go.Bar(x=breadth_plot.index, y=breadth_plot["pct_new_52wl"],
           marker_color="#ef4444", marker_line_width=0,
           name="% New 52wk Lows"),
    row=3, col=1,
)
fig.add_hline(y=0, line=dict(color="#444", width=0.5), row=3, col=1)

# Row 4 — % above 200 / 50 DMA
fig.add_trace(
    go.Scatter(x=breadth_plot.index, y=breadth_plot["pct_above_200"],
               line=dict(color="#a3e635", width=1.4),
               name="% Above 200 DMA"),
    row=4, col=1,
)
fig.add_trace(
    go.Scatter(x=breadth_plot.index, y=breadth_plot["pct_above_50"],
               line=dict(color="#fbbf24", width=1.4),
               name="% Above 50 DMA"),
    row=4, col=1,
)

# Row 5 — % RSI > 70 / < 30
fig.add_trace(
    go.Bar(x=breadth_plot.index, y=breadth_plot["pct_rsi_over70"],
           marker_color="#22d3ee", marker_line_width=0,
           name="% RSI > 70"),
    row=5, col=1,
)
fig.add_trace(
    go.Bar(x=breadth_plot.index, y=breadth_plot["pct_rsi_under30"],
           marker_color="#fbbf24", marker_line_width=0,
           name="% RSI < 30"),
    row=5, col=1,
)
fig.add_hline(y=0, line=dict(color="#444", width=0.5), row=5, col=1)

# Row 6 — DSPX (left axis, red) + CPC (right axis, white)
if not dspx.empty:
    dplot = dspx.loc[plot_start:]
    fig.add_trace(
        go.Scatter(x=dplot.index, y=dplot.values,
                   line=dict(color="#f43f5e", width=1.2),
                   name="CBOE SPX Dispersion (DSPX)"),
        row=6, col=1, secondary_y=False,
    )
if not cpc.empty:
    # 10-day rolling mean (readable trend) + trailing 500d percentile bands:
    #   red where smoothed P/C < 10th pctile (call-heavy / complacent)
    #   green where smoothed P/C > 85th pctile (put-heavy / fear)
    cpc_smoothed = cpc.rolling(10, min_periods=3).mean()
    pct_rank = cpc_smoothed.rolling(500, min_periods=100).rank(pct=True)
    cplot = cpc_smoothed.loc[plot_start:]
    rplot = pct_rank.loc[plot_start:]

    # Base line — white, drawn everywhere.
    fig.add_trace(
        go.Scatter(x=cplot.index, y=cplot.values,
                   line=dict(color="#e5e7eb", width=1.2),
                   name=f"CBOE Put/Call ({pc_ratio_label}, 10d MA)"),
        row=6, col=1, secondary_y=True,
    )
    # Extend each in-condition run by one trailing neighbor so visual segments
    # connect into the transition rather than terminating one point short.
    red_mask = (rplot < 0.10).fillna(False)
    grn_mask = (rplot > 0.85).fillna(False)
    red_mask_ext = red_mask | red_mask.shift(-1, fill_value=False)
    grn_mask_ext = grn_mask | grn_mask.shift(-1, fill_value=False)

    red_seg = cplot.where(red_mask_ext)
    grn_seg = cplot.where(grn_mask_ext)

    if red_seg.notna().any():
        fig.add_trace(
            go.Scatter(x=red_seg.index, y=red_seg.values,
                       line=dict(color="#ef4444", width=2.0),
                       name="P/C < 10th pctile (500d)", showlegend=True,
                       connectgaps=False),
            row=6, col=1, secondary_y=True,
        )
    if grn_seg.notna().any():
        fig.add_trace(
            go.Scatter(x=grn_seg.index, y=grn_seg.values,
                       line=dict(color="#22c55e", width=2.0),
                       name="P/C > 85th pctile (500d)", showlegend=True,
                       connectgaps=False),
            row=6, col=1, secondary_y=True,
        )

# Layout
fig.update_layout(
    template="plotly_dark",
    height=1200,
    margin=dict(l=40, r=60, t=30, b=30),
    legend=dict(orientation="h", yanchor="bottom", y=1.01,
                xanchor="left", x=0.0, font=dict(size=10)),
    barmode="overlay",
    xaxis_rangeslider_visible=False,
    hovermode="x unified",
    paper_bgcolor="#0b0f17",
    plot_bgcolor="#0b0f17",
)

# Per-axis cosmetic tweaks
for r in range(1, 7):
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.04)", row=r, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", row=r, col=1)

# Y axis titles (compact)
fig.update_yaxes(title_text="SPX", row=1, col=1)
fig.update_yaxes(title_text="Adv-Dec", row=2, col=1)
fig.update_yaxes(title_text="% 52wk H/L", row=3, col=1)
fig.update_yaxes(title_text="% > MA", row=4, col=1, range=[0, 100])
fig.update_yaxes(title_text="% RSI", row=5, col=1)
fig.update_yaxes(title_text="DSPX", row=6, col=1, secondary_y=False)
if not cpc.empty:
    fig.update_yaxes(title_text="P/C", row=6, col=1, secondary_y=True,
                     showgrid=False)

# Hide candlestick rangebreaks (weekends) for a tidy axis
fig.update_xaxes(
    rangebreaks=[dict(bounds=["sat", "mon"])],
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Latest values strip + notes
# ---------------------------------------------------------------------------
last = breadth_plot.dropna().iloc[-1] if not breadth_plot.dropna().empty else None
last_spx = spx_plot["Close"].iloc[-1] if not spx_plot.empty else None
last_ma50 = spx_plot["MA50"].iloc[-1] if not spx_plot.empty else None
last_ma200 = spx_plot["MA200"].iloc[-1] if not spx_plot.empty else None
last_ma100w = spx_plot["MA100W"].iloc[-1] if not spx_plot.empty else None
last_dspx = dspx.iloc[-1] if not dspx.empty else None
last_cpc = cpc.rolling(10, min_periods=3).mean().iloc[-1] if not cpc.empty else None

st.markdown("**Latest values**")
mcols = st.columns(7)
def _fmt(v, p=1):
    return "n/a" if v is None or pd.isna(v) else f"{v:,.{p}f}"

mcols[0].metric("SPX", _fmt(last_spx, 2))
mcols[1].metric("50 / 200 / 100w", f"{_fmt(last_ma50,0)} / {_fmt(last_ma200,0)} / {_fmt(last_ma100w,0)}")
if last is not None:
    mcols[2].metric("Adv-Dec", _fmt(last["adv_minus_dec"], 0))
    mcols[3].metric("% 52wH / 52wL",
                    f"{_fmt(last['pct_new_52wh'],2)} / {_fmt(-last['pct_new_52wl'],2)}")
    mcols[4].metric("% >50 / >200",
                    f"{_fmt(last['pct_above_50'],1)} / {_fmt(last['pct_above_200'],1)}")
    mcols[5].metric("RSI>70 / <30",
                    f"{_fmt(last['pct_rsi_over70'],2)} / {_fmt(-last['pct_rsi_under30'],2)}")
mcols[6].metric("DSPX / CPC", f"{_fmt(last_dspx,2)} / {_fmt(last_cpc,2)}")

note_lines = [
    f"Constituent universe: {window_closes.shape[1]} S&P 500 tickers, "
    f"closes through {window_closes.index.max().date()}.",
    "RSI uses Wilder smoothing (EMA alpha=1/14). 52-week thresholds are rolling 252 trading days.",
    "52wk H/L panel is close-based (intraday H/L unavailable in the SP500 close cache) — "
    "expect lower magnitudes than Bloomberg's intraday-based version, but the same shape.",
]
if cpc.empty:
    note_lines.append(
        "CBOE Put/Call cache is empty — click 'Refresh data' above to backfill from cboe.com. "
        "First backfill (2 years) takes ~2-5 minutes; subsequent runs are incremental."
    )
else:
    note_lines.append(
        f"CBOE Put/Call: scraped daily from cboe.com market_statistics, "
        f"cached at data/cboe_putcall.parquet ({len(cpc)} {pc_ratio_label} rows shown)."
    )
st.caption("  \n".join(note_lines))
