"""
seasonal_edge.py - shared core for the daily discretionary trade-idea process.

Pure stats/data helpers with NO daily_scan / strategy_config import, so this
module stays lightweight and is safely importable (and unit-testable) on its
own. The heavier engine (daily_seasonal_ideas.py) wires these together with
the live-book negative filter and the detector channels.

Building blocks
---------------
- load_seasonal_ranks / seasonal_cross_section / seasonal_series
    ATR-normalized 0-100 seasonal ranks at 6 horizons (atr_seasonal_ranks.parquet)
- load_prices
    daily OHLCV from master_prices.parquet (+ overflow_prices.parquet fallback)
- run_event_study(price, condition, ...)
    generalized lift of risk_dashboard_v2._run_fragility_event_study: forward
    returns on condition=True dates vs unconditional, declustered, Welch t-test
- seasonal_window_returns(price, asof, fwd, cycle_phase=...)
    TRUE presidential-cycle re-derivation from raw prices (the blended rank
    parquet cannot answer "worked 5/6 in midterm years"; this can)
- benjamini_hochberg
    FDR control across the day's candidate set (the dominant multiple-comparisons guard)
- make_candidate / render_markdown
    common candidate schema + markdown output

All data sources confirmed present 2026-06-09.
"""

from __future__ import annotations

import os
import json
import sqlite3
from functools import lru_cache

import numpy as np
import pandas as pd

try:
    from scipy import stats as _stats
except Exception:  # scipy should be present (used by risk_dashboard); degrade if not
    _stats = None

# -----------------------------------------------------------------------------
# Paths (resolve relative to repo root = parent of scripts/)
# -----------------------------------------------------------------------------
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_SCRIPTS_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data")

ATR_SZNL_PATH = os.path.join(REPO_ROOT, "atr_seasonal_ranks.parquet")
MASTER_PRICES_PATH = os.path.join(DATA_DIR, "master_prices.parquet")
OVERFLOW_PRICES_PATH = os.path.join(DATA_DIR, "overflow_prices.parquet")
SZNL_FORECAST_DB = os.path.join(DATA_DIR, "sznl_forecast.db")
OVERFLOW_UNIVERSE_PATH = os.path.join(DATA_DIR, "overflow_universe.parquet")

ATR_SZNL_WINDOWS = [5, 10, 21, 63, 126, 252]
ATR_SZNL_COLS = [f"atr_sznl_{w}d" for w in ATR_SZNL_WINDOWS]

DEFAULT_FWD_WINDOWS = [5, 10, 21, 63]

CYCLE_LABELS = {0: "Election Year", 1: "Post-Election", 2: "Midterm Year", 3: "Pre-Election"}


def cycle_phase(year: int) -> int:
    """Presidential-cycle phase: year % 4 (2 == Midterm). 2026 -> 2."""
    return int(year) % 4


def cycle_label(year: int) -> str:
    return CYCLE_LABELS.get(cycle_phase(year), "Unknown")


def _norm_ticker(t: str) -> str:
    """Match master_prices storage convention (upper, '.'->'-'); leave ^ and =F."""
    return str(t).strip().upper().replace(".", "-")


# -----------------------------------------------------------------------------
# Seasonal ranks
# -----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_seasonal_ranks(path: str | None = None) -> pd.DataFrame:
    """Long DataFrame [Date(datetime), atr_sznl_*, ticker(upper)]. Cached."""
    path = path or ATR_SZNL_PATH
    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df["ticker"] = df["ticker"].astype(str).str.upper()
    return df


def seasonal_cross_section(asof=None, ranks: pd.DataFrame | None = None) -> pd.DataFrame:
    """Latest seasonal ranks per ticker as of `asof` (default: max date).

    Returns a frame indexed by ticker with the 6 rank columns + the as-of Date
    each ticker's row was taken from. Uses each ticker's most recent row <= asof
    (so a ticker that stopped updating still reports its last known rank, with
    its own Date stamped).
    """
    ranks = load_seasonal_ranks() if ranks is None else ranks
    if asof is not None:
        ranks = ranks[ranks["Date"] <= pd.Timestamp(asof).normalize()]
    if ranks.empty:
        return pd.DataFrame(columns=ATR_SZNL_COLS)
    idx = ranks.groupby("ticker")["Date"].idxmax()
    cs = ranks.loc[idx].set_index("ticker")
    return cs[["Date"] + ATR_SZNL_COLS].sort_index()


def seasonal_series(ticker: str, ranks: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-ticker seasonal-rank history indexed by Date."""
    ranks = load_seasonal_ranks() if ranks is None else ranks
    t = str(ticker).upper()
    sub = ranks[ranks["ticker"] == t]
    if sub.empty:
        return pd.DataFrame(columns=ATR_SZNL_COLS)
    return sub.set_index("Date")[ATR_SZNL_COLS].sort_index()


# -----------------------------------------------------------------------------
# Prices
# -----------------------------------------------------------------------------
def load_prices(tickers, include_overflow: bool = True) -> dict[str, pd.DataFrame]:
    """{ticker: OHLCV DataFrame indexed by normalized date}.

    Reads master_prices first, then overflow_prices for any still-missing
    tickers. Uses pyarrow predicate pushdown when available, falls back to a
    full read + in-memory filter otherwise.
    """
    wanted = {_norm_ticker(t) for t in tickers}
    out: dict[str, pd.DataFrame] = {}
    cols = ["ticker", "date", "Open", "High", "Low", "Close", "Volume"]
    paths = [MASTER_PRICES_PATH] + ([OVERFLOW_PRICES_PATH] if include_overflow else [])
    for path in paths:
        if not os.path.exists(path):
            continue
        need = wanted - set(out)
        if not need:
            break
        try:
            df = pd.read_parquet(path, columns=cols, filters=[("ticker", "in", list(need))])
        except Exception:
            df = pd.read_parquet(path)
            df = df[[c for c in cols if c in df.columns]]
            df = df[df["ticker"].astype(str).str.upper().isin(need)]
        if df is None or df.empty:
            continue
        df = df.copy()
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        df["date"] = pd.to_datetime(df["date"])
        for t, g in df.groupby("ticker"):
            if t in out:
                continue
            sub = g.set_index("date").sort_index()[["Open", "High", "Low", "Close", "Volume"]]
            if getattr(sub.index, "tz", None) is not None:
                sub.index = sub.index.tz_localize(None)
            sub.index = sub.index.normalize()
            out[t] = sub
    return out


def load_one_price(ticker: str) -> pd.DataFrame | None:
    d = load_prices([ticker])
    return d.get(_norm_ticker(ticker))


def recent_dollar_volume(price_df: pd.DataFrame, window: int = 63) -> float:
    """Mean Close*Volume over the trailing `window` bars (liquidity screen)."""
    if price_df is None or price_df.empty or "Volume" not in price_df:
        return float("nan")
    tail = price_df.tail(window)
    return float((tail["Close"] * tail["Volume"]).mean())


# -----------------------------------------------------------------------------
# Event study (generalized from risk_dashboard_v2._run_fragility_event_study)
# -----------------------------------------------------------------------------
def run_event_study(
    price: pd.Series,
    condition: pd.Series,
    forward_windows: list | None = None,
    decluster: bool = True,
    min_gap: int = 5,
) -> dict:
    """Forward returns on condition=True dates vs the unconditional baseline.

    price      : Series of closes (DatetimeIndex)
    condition  : boolean Series (DatetimeIndex) — True on candidate signal dates
    Returns {n_episodes, episode_dates, windows: {w: {...}}} where each window
    carries n, mean, median, pct_neg, pct_pos, uncond_mean, diff_mean, p_value,
    worst, best. Forward windows that overlap the (incomplete) tail are dropped
    via the shift/dropna, so there is no look-ahead on the most recent dates.
    """
    if forward_windows is None:
        forward_windows = DEFAULT_FWD_WINDOWS

    price = price.dropna().sort_index()
    condition = condition.reindex(price.index).fillna(False).astype(bool)

    sig = condition.copy()
    if decluster and min_gap > 0:
        fire = np.where(sig.values)[0]
        keep = np.ones(len(sig), dtype=bool)
        last = -min_gap - 1
        for pos in fire:
            if pos - last <= min_gap:
                keep[pos] = False
            else:
                last = pos
        sig = sig & pd.Series(keep, index=sig.index)

    episode_dates = sig[sig].index
    windows: dict[int, dict] = {}
    for w in forward_windows:
        fwd = (price.shift(-w) / price - 1.0).dropna()
        s = fwd.reindex(episode_dates).dropna()
        if len(s) == 0:
            windows[w] = None
            continue
        if _stats is not None and len(s) > 2 and len(fwd) > 2:
            try:
                _, p_val = _stats.ttest_ind(s.values, fwd.values, equal_var=False)
            except Exception:
                p_val = float("nan")
        else:
            p_val = float("nan")
        windows[w] = {
            "n": int(len(s)),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "pct_neg": float((s < 0).mean()),
            "pct_pos": float((s > 0).mean()),
            "uncond_mean": float(fwd.mean()),
            "diff_mean": float(s.mean() - fwd.mean()),
            "worst": float(s.min()),
            "best": float(s.max()),
            "p_value": float(p_val),
        }
    return {
        "n_episodes": int(len(episode_dates)),
        "episode_dates": list(episode_dates),
        "windows": windows,
    }


# -----------------------------------------------------------------------------
# True presidential-cycle re-derivation (the blended rank parquet cannot do this)
# -----------------------------------------------------------------------------
def _trading_doy(index: pd.DatetimeIndex) -> pd.Series:
    """Trading-day-of-year (1-based) for each date — mirrors the rank builder's
    day_count index, so cross-year matching aligns with the seasonal grid."""
    yr = pd.Series(index.year, index=index)
    doy = yr.groupby(yr.values).cumcount() + 1
    return pd.Series(doy.values, index=index)


def _window_pick_positions(doy_arr, years_arr, target_doy, asof_year,
                           cycle_phase, tol, exclude_current):
    """Vectorized replacement for the per-year doy-matching loop: one pick per
    prior year (the bar with day-of-year closest to target, ties to the earliest),
    returned as integer positions in year-ascending order. Mirrors the loop's
    `(cand - target).abs().idxmin()` exactly (lexsort by year, |doy-target|, pos)."""
    m = (doy_arr >= target_doy - tol) & (doy_arr <= target_doy + tol)
    if exclude_current:
        m &= years_arr < asof_year
    if cycle_phase is not None:
        m &= (years_arr % 4) == cycle_phase
    idx = np.flatnonzero(m)
    if idx.size == 0:
        return idx
    order = np.lexsort((idx, np.abs(doy_arr[idx] - target_doy), years_arr[idx]))
    ys = years_arr[idx][order]
    first = np.ones(ys.size, dtype=bool)
    first[1:] = ys[1:] != ys[:-1]
    return idx[order][first]


def seasonal_window_returns(
    price_df: pd.DataFrame,
    asof,
    forward_window: int,
    cycle_phase_filter: int | None = None,
    doy_tol: int = 2,
    min_years: int = 3,
    exclude_current_year: bool = True,
) -> dict | None:
    """Realized forward returns for THIS calendar window across prior years.

    Picks, in each prior year, the trading day whose day-of-year is closest to
    `asof`'s day-of-year (within +/-doy_tol), and records the realized
    `forward_window`-day return from there. With cycle_phase_filter set to
    asof.year%4 you get the literal "in midterm years it did X" — the stat the
    blended rank cannot express because the cycle weighting is collapsed in.

    Returns {n, mean, median, n_down, n_up, pct_down, years, rets} or
    {n, insufficient: True} when fewer than `min_years` matches exist.

    Numpy-vectorized (2026-06; ~6x faster than the prior per-year loop). Float
    math runs in float64 vs the old float32 path, so window means/medians can
    differ at ~1e-8 — economically identical and verified to leave candidate
    generation unchanged (tests/regression: candidate-equality vs the loop).
    """
    if price_df is None or price_df.empty:
        return None
    close = price_df["Close"].dropna().sort_index()
    if close.empty:
        return None
    asof = pd.Timestamp(asof).normalize()
    doy = _trading_doy(close.index).values
    years = close.index.year.values.astype(np.int64)
    le = close.index.values <= np.datetime64(asof)
    if not le.any():
        return None
    target_doy = int(doy[le][-1])
    cv = close.values.astype(np.float64)
    N = int(forward_window)
    fwd = np.full(cv.shape, np.nan)
    if 0 < N < cv.size:
        with np.errstate(divide="ignore", invalid="ignore"):
            fwd[:-N] = cv[N:] / cv[:-N] - 1.0
    picks = _window_pick_positions(doy, years, target_doy, asof.year,
                                   cycle_phase_filter, doy_tol, exclude_current_year)
    r = fwd[picks]
    valid = ~np.isnan(r)
    rets = r[valid]
    yrs = years[picks][valid]
    if rets.size < min_years:
        return {"n": int(rets.size), "insufficient": True}
    return {
        "n": int(rets.size),
        "mean": float(rets.mean()),
        "median": float(np.median(rets)),
        "n_down": int((rets < 0).sum()),
        "n_up": int((rets > 0).sum()),
        "pct_down": float((rets < 0).mean()),
        "years": [int(y) for y in yrs],
        "rets": [round(float(x), 4) for x in rets],
    }


def expected_seasonal_path(price_df, asof, forward_window, doy_tol=2, min_years=3):
    """Average ATR-NORMALIZED per-day path (length forward_window) over the same
    calendar window in prior years (same pick logic as seasonal_window_returns,
    all-years). Each prior year's cumulative move is divided by that year's Wilder
    ATR at the anchor bar — identical to expected_atr_move / the atr_sznl ranks —
    so one high-volatility year can't skew where the AVERAGE path bottoms. The day
    the path BOTTOMS (long) / PEAKS (short) is the ex-ante entry-timing nadir/peak.
    Returns the np.ndarray path (ATR units) or None when fewer than `min_years`
    prior matches exist."""
    if price_df is None or len(price_df) == 0:
        return None
    close = price_df["Close"].dropna().sort_index()
    if close.empty:
        return None
    atr = atr_wilder(price_df).reindex(close.index).values.astype(np.float64)
    asof = pd.Timestamp(asof).normalize()
    doy = _trading_doy(close.index).values
    years = close.index.year.values.astype(np.int64)
    le = close.index.values <= np.datetime64(asof)
    if not le.any():
        return None
    target = int(doy[le][-1])
    picks = _window_pick_positions(doy, years, target, asof.year, None, doy_tol, True)
    if picks.size < min_years:
        return None
    cv = close.values.astype(np.float64)
    N = int(forward_window)
    paths = [(cv[p + 1:p + N + 1] - cv[p]) / atr[p]
             for p in picks if p + N < cv.size and np.isfinite(atr[p]) and atr[p] > 0]
    if len(paths) < min_years:
        return None
    return np.nanmean(np.vstack(paths), axis=0)


# -----------------------------------------------------------------------------
# Multiple-comparisons control
# -----------------------------------------------------------------------------
def binom_p_greater(k: int, n: int, p0: float = 0.5) -> float:
    """One-sided binomial p-value: P(X >= k) under Binomial(n, p0).

    This is the honest test for a "this window closed lower k of n years" claim
    (k = directional count: n_down for a short, n_up for a long). NaN if n < 1.
    Note: when k is selected on the same seasonal signal used to pick the name,
    this p is post-selection-optimistic -- treat it as descriptive support for a
    discretionary idea, not an out-of-sample guarantee.
    """
    if n is None or n < 1:
        return float("nan")
    k = int(k)
    n = int(n)
    if _stats is not None:
        try:
            return float(_stats.binomtest(k, n, p0, alternative="greater").pvalue)
        except Exception:
            pass
    from math import sqrt, erf
    mu = n * p0
    sd = sqrt(n * p0 * (1 - p0))
    if sd == 0:
        return 1.0
    z = (k - 0.5 - mu) / sd  # continuity correction
    return float(0.5 * (1 - erf(z / sqrt(2))))


def benjamini_hochberg(pvals, alpha: float = 0.10):
    """Benjamini-Hochberg FDR. Returns (reject_mask: np.bool array, crit_p: float).

    crit_p is the largest p-value still rejected (0.0 if none survive). NaN
    p-values never reject.
    """
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return np.zeros(0, dtype=bool), 0.0
    valid = ~np.isnan(p)
    pv = p[valid]
    m = len(pv)
    if m == 0:
        return np.zeros(n, dtype=bool), 0.0
    order = np.argsort(pv)
    ranked = pv[order]
    crit = alpha * (np.arange(1, m + 1) / m)
    passed = ranked <= crit
    if not passed.any():
        return np.zeros(n, dtype=bool), 0.0
    kmax = int(np.max(np.where(passed)[0]))
    thresh = float(ranked[kmax])
    reject = valid & (p <= thresh)
    return reject, thresh


# -----------------------------------------------------------------------------
# Candidate schema + markdown rendering
# -----------------------------------------------------------------------------
def make_candidate(
    channel: str,
    ticker: str,
    direction: str,
    headline: str,
    *,
    horizon: str | None = None,
    evidence: dict | None = None,
    conviction: str | None = None,
    p_value: float | None = None,
    sort_key: float = 0.0,
    asof=None,
    notes: str | None = None,
) -> dict:
    """Common candidate record consumed by the engine and the markdown renderer."""
    return {
        "channel": channel,
        "ticker": str(ticker).upper(),
        "direction": direction,  # 'long' | 'short' | 'context'
        "headline": headline,
        "horizon": horizon,
        "evidence": evidence or {},
        "conviction": conviction,  # 'A' | 'B' | 'C' | None
        "p_value": p_value,
        "sort_key": float(sort_key),
        "asof": str(pd.Timestamp(asof).date()) if asof is not None else None,
        "notes": notes,
    }


_DIR_TAG = {"long": "[LONG]", "short": "[SHORT]", "context": "[CONTEXT]"}


def render_markdown(candidates: list[dict], meta: dict | None = None) -> str:
    """Render the ranked candidate set to a committed-markdown digest."""
    meta = meta or {}
    asof = meta.get("asof", "")
    lines = []
    lines.append(f"# Daily Seasonal / Whitespace Ideas - {asof}")
    lines.append("")
    if meta.get("regime"):
        lines.append(f"_Regime: {meta['regime']}_")
        lines.append("")
    if meta.get("summary"):
        lines.append(meta["summary"])
        lines.append("")
    if meta.get("stale_notes"):
        lines.append("> Data staleness: " + "; ".join(meta["stale_notes"]))
        lines.append("")

    if not candidates:
        lines.append("**No setups cleared the bar today.** (This is a feature, not a bug -- "
                     "the statistical gates are meant to emit zero on quiet days.)")
        return "\n".join(lines)

    # group by channel, preserve channel order of first appearance
    order = []
    by_channel: dict[str, list] = {}
    for c in candidates:
        ch = c["channel"]
        if ch not in by_channel:
            by_channel[ch] = []
            order.append(ch)
        by_channel[ch].append(c)

    for ch in order:
        rows = sorted(by_channel[ch], key=lambda x: -x["sort_key"])
        lines.append(f"## {ch}")
        lines.append("")
        for c in rows:
            tag = _DIR_TAG.get(c["direction"], "")
            conv = f" ({c['conviction']})" if c.get("conviction") else ""
            hz = f" [{c['horizon']}]" if c.get("horizon") else ""
            lines.append(f"- {tag} **{c['ticker']}**{hz}{conv} - {c['headline']}")
            ev = c.get("evidence") or {}
            if ev:
                ev_str = "; ".join(f"{k}: {v}" for k, v in ev.items())
                lines.append(f"  - {ev_str}")
            if c.get("notes"):
                lines.append(f"  - _{c['notes']}_")
        lines.append("")
    if meta.get("footer"):
        lines.append("---")
        lines.append("")
        lines.append(f"_{meta['footer']}_")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Curated idea universes (megacap equities + macro/cross-asset). Mirrors
# pages/equity_seasonals.py and pages/macro_seasonality.py, copied here so the
# core has no streamlit dependency.
# -----------------------------------------------------------------------------
MEGACAP_TICKERS = [
    "MMM", "AXP", "AMGN", "AMZN", "AAPL", "BA", "CAT", "CVX", "CSCO", "KO",
    "DIS", "GS", "HD", "HON", "IBM", "JNJ", "JPM", "MCD", "MRK", "MSFT",
    "NKE", "NVDA", "PG", "CRM", "SHW", "TRV", "UNH", "VZ", "V", "WMT",
    "GOOGL", "XOM", "LLY", "ORCL", "ADBE", "TXN", "QCOM", "PEP", "COST", "LOW",
    "NFLX", "CMCSA", "T", "NEE", "UNP", "MA", "BAC", "AMT",
]
MACRO_TICKERS = [
    "^GSPC", "^NDX", "^IXIC", "^DJI", "^DJT", "^RUT", "^MID", "^SOX", "GLD",
    "CEF", "SLV", "BTC-USD", "ETH-USD", "UNG", "UVXY", "EURUSD=X", "JPY=X",
    "GBPUSD=X", "AUDUSD=X", "NZDUSD=X", "CAD=X", "CHF=X", "DX-Y.NYB", "CL=F",
    "NG=F", "GC=F", "HG=F", "KC=F", "PL=F", "ZC=F", "ZW=F", "CC=F", "SB=F",
    "PA=F", "ZS=F", "CT=F", "SI=F", "^FTSE", "^GDAXI", "^FCHI", "^N225", "^HSI",
    "^STI", "^AXJO", "^KS11", "^TWII", "^BSESN", "^GSPTSE", "^MXX", "^BVSP",
    "^STOXX50E", "TLT", "IEF", "TIP", "LQD", "HYG", "AGG", "^VIX",
]
IDEA_UNIVERSE = MEGACAP_TICKERS + MACRO_TICKERS


# -----------------------------------------------------------------------------
# Trade tickets: expected seasonal move (ATR units) + structure -> entry/stop/target
# -----------------------------------------------------------------------------
def atr_wilder(price_df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder ATR series."""
    h, l, c = price_df["High"], price_df["Low"], price_df["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()


def expected_atr_move(price_df, asof, forward_window, cycle_phase_filter=None,
                      doy_tol: int = 2, min_years: int = 3,
                      exclude_current_year: bool = True):
    """Mean ATR-normalized seasonal forward move ((Close[t+N]-Close[t])/ATR) for
    this calendar window across prior years. This is the magnitude estimate that
    sizes a ticket's target. Returns float (ATR units) or None."""
    if price_df is None or price_df.empty:
        return None
    close = price_df["Close"].dropna().sort_index()
    if close.empty:
        return None
    atr = atr_wilder(price_df).reindex(close.index).values.astype(np.float64)
    asof = pd.Timestamp(asof).normalize()
    doy = _trading_doy(close.index).values
    years = close.index.year.values.astype(np.int64)
    le = close.index.values <= np.datetime64(asof)
    if not le.any():
        return None
    target_doy = int(doy[le][-1])
    cv = close.values.astype(np.float64)
    N = int(forward_window)
    fwd = np.full(cv.shape, np.nan)
    if 0 < N < cv.size:
        with np.errstate(divide="ignore", invalid="ignore"):
            fwd[:-N] = (cv[N:] - cv[:-N]) / atr[:-N]
    picks = _window_pick_positions(doy, years, target_doy, asof.year,
                                   cycle_phase_filter, doy_tol, exclude_current_year)
    v = fwd[picks]
    v = v[~np.isnan(v)]
    if v.size < min_years:
        return None
    return float(v.mean())


def build_trade_ticket(price_df, asof, direction, forward_window, expected_move_atr,
                       *, min_rr: float = 2.0, min_stop_atr: float = 0.8,
                       swing_lookback: int = 20) -> dict | None:
    """Concrete trade ticket from TA structure + the expected seasonal move.

    target = full expected seasonal move (ATR units); stop sits at the recent
    swing (structure) but is bounded so reward/risk >= min_rr, floored at
    min_stop_atr ATR; time-stop = the seasonal window. is_ticket flags whether
    R/R clears min_rr (below that it's a 'lean', not a standalone swing)."""
    if price_df is None or len(price_df) < 30:
        return None
    atr = float(atr_wilder(price_df).iloc[-1])
    if not np.isfinite(atr) or atr <= 0:
        return None
    entry = float(price_df["Close"].iloc[-1])
    hi = float(price_df["High"].tail(swing_lookback).max())
    lo = float(price_df["Low"].tail(swing_lookback).min())
    em = abs(float(expected_move_atr))
    if em <= 0:
        return None
    if direction == "short":
        struct_dist = (hi - entry) / atr if hi > entry else 99.0
    else:
        struct_dist = (entry - lo) / atr if lo < entry else 99.0
    stop_dist = max(min(struct_dist, em / min_rr), min_stop_atr)
    rr = em / stop_dist
    if direction == "short":
        stop = entry + stop_dist * atr
        target = entry - em * atr
    else:
        stop = entry - stop_dist * atr
        target = entry + em * atr
    return {
        "entry": round(entry, 4), "stop": round(stop, 4), "target": round(target, 4),
        "atr": round(atr, 4), "stop_atr": round(stop_dist, 2),
        "time_stop_days": int(forward_window), "rr": round(rr, 2),
        "is_ticket": bool(rr >= min_rr - 1e-9),
        "swing_hi": round(hi, 4), "swing_lo": round(lo, 4),
        "expected_move_atr": round(float(expected_move_atr), 2),
    }


def _confirms(stat, direction, min_n: int = 8) -> bool:
    """Realized day-of-year window confirms the claimed direction."""
    if not stat or stat.get("insufficient") or stat.get("n", 0) < min_n:
        return False
    if direction == "short":
        return stat["pct_down"] >= 0.60 and stat["mean"] < -0.003
    return stat["pct_down"] <= 0.40 and stat["mean"] > 0.003


def cycle_blended_expected_move(px, asof, N, blend: float = 0.75):
    """Expected ATR move blended `blend` current-cycle + (1-blend) all-years,
    matching the rank engine's 75/25 cycle weighting. Falls back to whichever
    component is available."""
    allm = expected_atr_move(px, asof, N)
    cyc = expected_atr_move(px, asof, N, cycle_phase_filter=pd.Timestamp(asof).year % 4)
    if allm is None:
        return cyc
    if cyc is None:
        return allm
    return blend * cyc + (1 - blend) * allm


def seasonal_window_blended(px, asof, N, blend: float = 0.75) -> dict | None:
    """Blend current-cycle and all-years realized window stats (cycle gets
    `blend` weight - the rank engine's 75/25). Confirmation and sizing both key
    off this so the tool screens and sizes on the SAME seasonal the rank is
    built from. Returns blended mean/pct_down, the blended ATR expected move,
    both component stat dicts, and a sign-conflict `disagree` flag."""
    s_all = seasonal_window_returns(px, asof, N)
    if not s_all or s_all.get("insufficient"):
        return None
    s_cyc = seasonal_window_returns(px, asof, N, cycle_phase_filter=pd.Timestamp(asof).year % 4)
    cyc_ok = bool(s_cyc and not s_cyc.get("insufficient"))
    if cyc_ok:
        b_mean = blend * s_cyc["mean"] + (1 - blend) * s_all["mean"]
        b_pct_down = blend * s_cyc["pct_down"] + (1 - blend) * s_all["pct_down"]
    else:
        b_mean, b_pct_down = s_all["mean"], s_all["pct_down"]
    disagree = cyc_ok and (np.sign(s_cyc["mean"]) != np.sign(s_all["mean"]))
    ea_all = expected_atr_move(px, asof, N)
    ea_cyc = expected_atr_move(px, asof, N, cycle_phase_filter=pd.Timestamp(asof).year % 4)
    if ea_all is None:
        ea = ea_cyc
    elif ea_cyc is None:
        ea = ea_all
    else:
        ea = blend * ea_cyc + (1 - blend) * ea_all
    return {"mean": b_mean, "pct_down": b_pct_down, "ea": ea,
            "ea_all": ea_all, "ea_cyc": ea_cyc,
            "all": s_all, "cyc": s_cyc, "cyc_ok": cyc_ok, "disagree": disagree}


def _confirms_blended(blend, direction, min_n: int = 8) -> bool:
    """The 75/25-blended realized window confirms the claimed direction. Needs a
    baseline of all-years history (n >= min_n) so the cycle isn't trusted alone."""
    if blend is None or blend["all"].get("n", 0) < min_n:
        return False
    if direction == "short":
        return blend["pct_down"] >= 0.60 and blend["mean"] < -0.003
    return blend["pct_down"] <= 0.40 and blend["mean"] > 0.003


# Short-duration mandate: 5/10/21d only. The 63d swing bucket is retired.
TACTICAL_HORIZONS = (5, 10, 21)
SWING_HORIZONS = ()


_CYCLE_NAME = {0: "election", 1: "post-elec", 2: "midterm", 3: "pre-elec"}

# Signal-quality 2x2 (magnitude x consistency). A leg is STRONG when the move is
# BIG and the hit-rate is RELIABLE; OK when both are decent; else WEAK. The
# magnitude bars are anchored at the 21d horizon and scaled DOWN by sqrt(time)
# for shorter windows -- a 5d move can't be held to a 21d move's size (price
# dispersion grows ~ sqrt(N)).
SZN_HIT_STRONG = 0.66
SZN_MAG_STRONG = 1.5    # ATR at the reference horizon
SZN_HIT_OK = 0.58
SZN_MAG_OK = 0.7        # ATR at the reference horizon
SZN_REF_HORIZON = 21


def _horizon_scale(N) -> float:
    """sqrt(time) scaling of magnitude expectations, anchored at SZN_REF_HORIZON."""
    return (float(N) / SZN_REF_HORIZON) ** 0.5


def _leg_grade(hit, ea, N) -> str:
    """Grade one sample's leg by magnitude AND consistency, with the magnitude
    bar scaled to the horizon (shorter window -> smaller required move)."""
    if hit is None or not np.isfinite(hit) or ea is None:
        return "WEAK"
    mag = abs(ea)
    sc = _horizon_scale(N)
    if hit >= SZN_HIT_STRONG and mag >= SZN_MAG_STRONG * sc:
        return "STRONG"
    if hit >= SZN_HIT_OK and mag >= SZN_MAG_OK * sc:
        return "OK"
    return "WEAK"


def _grade_2x2(cyc_leg: str, all_leg: str, disagree: bool) -> str:
    """Conviction from the cycle (primary) + all-years (upgrade) legs.

    A = STRONG cycle confirmed by all-years (OK or STRONG) - big + reliable in
        the cycle AND it carries to the full sample.
    B = STRONG cycle alone, or STRONG all-years with a supporting cycle - a
        cycle-specific (or all-years-driven) bet.
    C = marginal, or the two samples conflict in direction.
    """
    if disagree:
        return "C"
    if cyc_leg == "STRONG" and all_leg in ("STRONG", "OK"):
        return "A"
    if (cyc_leg == "STRONG" and all_leg == "WEAK") or (all_leg == "STRONG" and cyc_leg in ("STRONG", "OK")):
        return "B"
    return "C"


def _seasonal_candidate(channel, t, px, asof, h, direction, blend, ticket, rk, bucket):
    """Build one ticketed candidate from a confirmed (direction, horizon). Sizing
    and screening are 75/25 cycle-blended; both the cycle and all-years realized
    counts are shown, and a sign-conflict between them is flagged."""
    s_all, s_cyc = blend["all"], blend["cyc"]
    ea, ea_all, ea_cyc = blend["ea"], blend["ea_all"], blend["ea_cyc"]
    phase = pd.Timestamp(asof).year % 4
    cyc_name = _CYCLE_NAME[phase]
    word = "lower" if direction == "short" else "higher"
    ndir_all = s_all["n_down"] if direction == "short" else s_all["n_up"]
    all_hit = ndir_all / s_all["n"]
    binom_p = binom_p_greater(ndir_all, s_all["n"])  # all-years significance, for FDR
    cyc_shown = blend["cyc_ok"] and s_cyc.get("n", 0) >= 3
    if cyc_shown:
        ndir_cyc = s_cyc["n_down"] if direction == "short" else s_cyc["n_up"]
        cyc_hit = ndir_cyc / s_cyc["n"]
    else:
        ndir_cyc = None
        cyc_hit = float("nan")

    # conviction = the magnitude x consistency 2x2, cycle primary + all-years
    # upgrade; magnitude bars scaled to the horizon (shorter -> smaller move)
    cyc_leg = _leg_grade(cyc_hit, ea_cyc, h)
    all_leg = _leg_grade(all_hit, ea_all, h)
    conviction = _grade_2x2(cyc_leg, all_leg, blend["disagree"])

    verb = "SELL" if direction == "short" else "BUY"
    head = f"{verb} {t} - {h}d {bucket} window"
    if cyc_shown:
        head += f", {cyc_name} {ndir_cyc}/{s_cyc['n']}"
    head += f" + all-yrs {ndir_all}/{s_all['n']} {word}"
    ev = {
        "TICKET": (f"{verb} ~{ticket['entry']:.2f} | stop {ticket['stop']:.2f} ({ticket['stop_atr']:.1f} ATR) | "
                   f"target {ticket['target']:.2f} | time-stop {h}td | R/R {ticket['rr']:.1f}"),
        "quality": f"cycle {cyc_leg} / all-years {all_leg} (magnitude x consistency)",
        f"{cyc_name} cycle": (f"{ndir_cyc}/{s_cyc['n']} {word}, {ea_cyc:+.1f} ATR"
                              if cyc_shown and ea_cyc is not None else "insufficient cycle history"),
        "all-years": (f"{ndir_all}/{s_all['n']} {word}, {ea_all:+.1f} ATR"
                      if ea_all is not None else f"{ndir_all}/{s_all['n']} {word}"),
        "blended target": f"{ea:+.2f} ATR over {h}td (75/25 cycle-blend)",
        "rank": f"atr_sznl_{h}d = {rk:.0f}",
        "binomial p (all-yrs)": f"{binom_p:.3f}" if np.isfinite(binom_p) else "n/a",
    }
    # Expected seasonal-path entry timing: the day the average prior-years path
    # bottoms (long) / peaks (short). Enter there instead of T+1. Best-effort —
    # a failure or short history just leaves the default T+1. Displayed only for
    # now; the live order path still stages T+1 (delayed execution is a separate
    # step). 0-indexed offset (0 = T+1 = day 1).
    entry_off = 0
    try:
        _pth = expected_seasonal_path(px, asof, h)  # full OHLC: ATR-normalized path
        if _pth is not None and len(_pth):
            entry_off = int(np.argmin(_pth)) if direction == "long" else int(np.argmax(_pth))
    except Exception:
        entry_off = 0
    ev["entry timing"] = (
        f"enter T+{entry_off + 1} (expected path {'nadir' if direction == 'long' else 'peak'} day)"
        if entry_off > 0 else "enter T+1 (path bottoms day 1)")

    notes = None
    if blend["disagree"]:
        notes = f"all-years seasonal disagrees in sign with {cyc_name} - graded C (conflict)"

    # rank by grade tier, then magnitude x consistency (cycle-weighted), then R/R
    base = {"A": 300.0, "B": 200.0, "C": 100.0}[conviction]
    cyc_s = (cyc_hit if np.isfinite(cyc_hit) else 0.0) * abs(ea_cyc or 0.0)
    all_s = all_hit * abs(ea_all or 0.0)
    sort_key = base + 6.5 * cyc_s + 3.5 * all_s + ticket["rr"]
    cand = make_candidate(
        channel, t, direction, head, horizon=f"{h}d", evidence=ev,
        conviction=conviction, p_value=float(binom_p) if np.isfinite(binom_p) else None,
        sort_key=sort_key, asof=asof, notes=notes,
    )
    cand["entry_offset_days"] = entry_off
    return cand


def scan_seasonal_tickets(universe, asof, channel, *, min_rr: float = 2.0,
                          ranks: pd.DataFrame | None = None,
                          horizons=(5, 10, 21), short_thr: float = 15,
                          long_thr: float = 85, min_dollar_vol: float = 5e6,
                          cycle_blend: float = 0.75, min_all_hit: float = 0.667) -> list:
    """Scan `universe` across `horizons`; emit swing tickets (R/R >= min_rr).

    Horizon selection: a name can yield up to TWO tickets - one tactical (<=21d)
    and one swing (>=63d) - when it clears the R/R + realized gate in both
    buckets. Within a bucket the horizon with the best R/R wins, ties broken
    toward the SHORTER window (so a tight tactical setup is not buried by a
    larger-but-sprawling long-horizon move). Names whose seasonal move is too
    small to clear the R/R bar at any horizon are dropped."""
    asof = pd.Timestamp(asof).normalize()
    cs = seasonal_cross_section(asof=asof, ranks=ranks)
    names = [t for t in universe if t in cs.index]
    prices = load_prices(names)
    out = []
    for t in names:
        px = prices.get(_norm_ticker(t))
        if px is None or len(px) < 300:
            continue
        is_fut_or_idx = t.startswith("^") or t.endswith("=F") or t.endswith("=X")
        if not is_fut_or_idx:
            dvol = recent_dollar_volume(px)
            if np.isfinite(dvol) and dvol < min_dollar_vol:
                continue

        # collect every (direction, horizon) that clears the 75/25-blended
        # realized gate AND the R/R bar (confirmation + sizing both cycle-blended)
        quals = []
        for h in horizons:
            col = f"atr_sznl_{h}d"
            if col not in cs.columns:
                continue
            rk = float(cs.loc[t, col])
            for direction, ext in (("short", rk < short_thr), ("long", rk > long_thr)):
                if not ext:
                    continue
                blend = seasonal_window_blended(px, asof, h, blend=cycle_blend)
                if not _confirms_blended(blend, direction):
                    continue
                # All-years directional hit-rate gate (2026-06-25). Walk-forward
                # dose-response: the bottom hit-rate quintile (~58-63%, barely
                # above the 60% confirmation floor) carried little realized edge;
                # >=0.667 keeps the dose-responsive top (per-trade avgR 0.139 ->
                # 0.171, PF 1.24 -> 1.30; ~42% fewer trades at the same Sharpe, so
                # the gain is fewer/higher-conviction ideas, not a higher ratio).
                # Shared by the live engine AND backtest_seasonal_ideas.
                _s_all = blend["all"]
                _ndir = _s_all.get("n_down") if direction == "short" else _s_all.get("n_up")
                _n = _s_all.get("n", 0)
                if not _n or (_ndir / _n) < min_all_hit:
                    continue
                ea = blend["ea"]
                if ea is None:
                    continue
                if (direction == "short" and ea >= 0) or (direction == "long" and ea <= 0):
                    continue  # blended expected move sign must match the trade
                ticket = build_trade_ticket(px, asof, direction, h, ea, min_rr=min_rr,
                                            min_stop_atr=max(0.5, 0.8 * _horizon_scale(h)))
                if ticket is None or not ticket["is_ticket"]:
                    continue  # swing tickets only
                quals.append({"h": h, "direction": direction, "blend": blend,
                              "ticket": ticket, "rk": rk})
        if not quals:
            continue

        # one ticket per bucket: best R/R, tie-break toward the shorter horizon
        for bucket_hz, bucket_name in ((TACTICAL_HORIZONS, "tactical"), (SWING_HORIZONS, "swing")):
            bucket = [q for q in quals if q["h"] in bucket_hz]
            if not bucket:
                continue
            q = sorted(bucket, key=lambda x: (-x["ticket"]["rr"], x["h"]))[0]
            out.append(_seasonal_candidate(channel, t, px, asof, q["h"], q["direction"],
                                           q["blend"], q["ticket"], q["rk"], bucket_name))
    return out


if __name__ == "__main__":
    # Smoke test against today's data.
    ranks = load_seasonal_ranks()
    cs = seasonal_cross_section()
    asof = cs["Date"].max()
    print(f"seasonal ranks: {ranks.shape}, cross-section asof {asof.date()}, {len(cs)} tickers")
    low5 = cs[cs["atr_sznl_5d"] < 5].sort_values("atr_sznl_5d")
    print(f"names at 5d rank < 5: {len(low5)} -> {list(low5.index[:10])}")
    # event study + cycle re-derivation on the most extreme name
    if not low5.empty:
        t = low5.index[0]
        px = load_one_price(t)
        if px is not None and len(px) > 300:
            rk = seasonal_series(t)["atr_sznl_5d"].reindex(px.index).ffill()
            es = run_event_study(px["Close"], rk < 15, forward_windows=[5, 10, 21])
            w5 = es["windows"].get(5)
            print(f"{t}: rank<15 event study 5d -> n={w5['n'] if w5 else 0} "
                  f"mean={w5['mean']:+.2%} pct_neg={w5['pct_neg']:.0%} p={w5['p_value']:.3f}" if w5 else f"{t}: no 5d window")
            mt = seasonal_window_returns(px, asof, 5, cycle_phase_filter=2)
            allp = seasonal_window_returns(px, asof, 5)
            print(f"{t}: midterm window 5d -> {mt}")
            print(f"{t}: all-years window 5d -> {allp}")
