"""Dual-qualifier replay: dial >= 50 AND SPY within 2% of its 252-session high.

Research only. Nothing here is imported by production and nothing is written
outside scratch/dual_qualifier_replay/.

Question
--------
The six fragility-band carriers (FAMILY4 + 3x Bear ETF Overbot Fade + Monthly
Weak Close) currently select a band TABLE by the lag-1 P/C fear state
(CLAUDE.md "P/C Fear-Conditioned Family Bands"). McKinley wants a second
qualifier on the dial >= 50 restriction: SPY within 2% of its trailing
252-session closing high. Two readings:

  Variant A (STACK)  incumbent sizing as today, PLUS zero any signal where
                     dial >= 50 AND near-high, regardless of the fear state.
  Variant B (NARROW) the dial >= 50 restriction applies ONLY when also
                     near-high; at dial >= 50 and NOT near-high the signal
                     trades at 1.0x (fear tables ignored there). Below 50
                     unchanged.

Method
------
Rather than re-weighting the shipped ledger (whose pcfear shadow companion is a
2026-08-07 vintage that predates the 2026-09-04 base-bps tilt and the retired
WCDS size tiers, and which stops at the 2026-07-29 signal), this script replays
the production engine four times on ONE set of candidates, monkeypatching
`pages.strat_backtester.frag_band_mult_at` with each scheme. Everything else --
per-strategy 250 bps daily cap, cross-strategy overlap clamp, gap derate,
same-day derate, fill logic, stop-fill convention -- is the production engine,
so cap interactions and dropped (zero-share) rows are handled by the engine
instead of by an arithmetic re-weighting.

The incumbent pass is asserted against data/backtest_trades_full.parquet as a
provenance check (printed, not enforced).

Inputs
------
  data/master_prices.parquet   adjusted OHLCV (SPY close -> near-high flag)
  data/rd2_fragility.parquet   the live PIT sizing dial, read through the
                               production `_frag_score_series()` (main_score
                               where present, else the 10-session mean of 63d)
  data/cboe_putcall.parquet    via pc_fear.fear_state_asof (lag-1, stale > 3 bd)

Usage
-----
  python scratch/dual_qualifier_replay/replay.py            # engine + analysis
  python scratch/dual_qualifier_replay/replay.py --analysis-only
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)

import data_provider  # noqa: E402
import pc_fear  # noqa: E402
import pages.strat_backtester as sbt  # noqa: E402
from pages.strat_backtester import (  # noqa: E402
    load_seasonal_map,
    load_atr_seasonal_map,
    precompute_all_indicators,
    generate_candidates_fast,
    process_signals_fast,
    get_daily_mtm_series,
)
from daily_portfolio_report import build_full_strategy_book  # noqa: E402
from scripts.build_trade_ledger import shape_flat_trades  # noqa: E402

try:
    from scipy import stats as _sps
except Exception:  # pragma: no cover
    _sps = None

CARRIERS = [
    "Weak Close Decent Sznls",
    "SPY QQQ MonFri Reversion",
    "Monday Dip",
    "Indices Oversold Bounce",
    "3x Bear ETF Overbot Fade",
    "Monthly Weak Close",
]
DATA_START = datetime.date(2000, 1, 1)
BT_START = datetime.date(2003, 1, 1)
STUDY_START = pd.Timestamp("2016-01-01")
FLAT_EQUITY = 750_000.0
DIAL_THRESH = 50.0
NEAR_THRESH = 0.02
NEAR_WINDOW = 252
NEAR_MINP = 60
EPISODE_GAP = 10  # sessions

OUT = os.path.join(_HERE, "results")
CACHE = os.path.join(_HERE, "cache")
SCHEMES = ["incumbent", "variant_a", "variant_b", "nobands"]


# ---------------------------------------------------------------------------
# Context series: dial (PIT), near-high, fear state
# ---------------------------------------------------------------------------
_CTX: dict = {}


def dial_series() -> pd.Series:
    """The production sizing dial, read exactly as the engine reads it."""
    if "dial" not in _CTX:
        s = sbt._frag_score_series()
        if s is None:
            raise RuntimeError("rd2_fragility.parquet unreadable")
        _CTX["dial"] = s
    return _CTX["dial"]


def dial_at(ts) -> float | None:
    s = dial_series()
    ts = pd.Timestamp(ts).normalize()
    if ts not in s.index:
        return None
    v = s.loc[ts]
    return None if pd.isna(v) else float(v)


def build_near_series(spy_close: pd.Series) -> pd.DataFrame:
    """Distance to the trailing 252-session closing high, on SPY sessions,
    then forward-filled onto a daily grid (limit 5) the way the dial is."""
    c = spy_close.dropna().sort_index()
    c.index = pd.to_datetime(c.index).normalize()
    roll = c.rolling(NEAR_WINDOW, min_periods=NEAR_MINP).max()
    drawdown = 1.0 - (c / roll)  # 0 at a new high
    grid = pd.date_range(c.index.min(), c.index.max(), freq="D")
    out = pd.DataFrame({"spy_close": c, "spy_252max": roll, "spy_dd": drawdown})
    return out.reindex(grid).ffill(limit=5)


def near_at(ts, thresh: float = NEAR_THRESH) -> bool | None:
    df = _CTX["near"]
    ts = pd.Timestamp(ts).normalize()
    if ts not in df.index:
        return None
    dd = df.loc[ts, "spy_dd"]
    return None if pd.isna(dd) else bool(dd <= thresh)


_FEAR_CACHE: dict = {}


def fear_state(ts) -> str:
    ts = pd.Timestamp(ts).normalize()
    if ts not in _FEAR_CACHE:
        _FEAR_CACHE[ts] = pc_fear.fear_state_asof(ts)["state"]
    return _FEAR_CACHE[ts]


# ---------------------------------------------------------------------------
# Scheme multipliers
# ---------------------------------------------------------------------------
def scheme_mult(execution, ts, scheme, dial_thresh=DIAL_THRESH,
                near_thresh=NEAR_THRESH) -> float:
    """Band multiplier under one scheme. Mirrors frag_band_mult_at for the
    incumbent (same table selection, same first-match band lookup, same
    'no dial -> 1.0x' rule) and layers the qualifier on top."""
    if not execution:
        return 1.0
    if not (execution.get("frag_risk_bands") or execution.get("pc_fear_bands")):
        return 1.0
    dial = dial_at(ts)
    if dial is None:
        return 1.0  # pre-2016 / gap in the dial -> 1.0x in every scheme
    state = fear_state(ts) if execution.get("pc_fear_bands") else "stale"
    inc = pc_fear.band_mult(pc_fear.select_bands(execution, state), dial)
    if scheme == "incumbent":
        return inc
    if scheme == "nobands":
        return 1.0
    hi = dial >= dial_thresh
    near = bool(near_at(ts, near_thresh))
    if scheme == "variant_a":
        return 0.0 if (hi and near) else inc
    if scheme == "variant_b":
        if not hi:
            return inc
        return inc if near else 1.0
    raise ValueError(scheme)


def make_patch(scheme):
    def _patched(execution, signal_ts, pc_fear_enabled=True):
        return scheme_mult(execution, signal_ts, scheme)
    return _patched


# ---------------------------------------------------------------------------
# Engine stage
# ---------------------------------------------------------------------------
def run_engine():
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)

    book = [s for s in build_full_strategy_book() if s["name"] in CARRIERS]
    names = sorted(s["name"] for s in book)
    if len(book) != len(CARRIERS):
        raise RuntimeError(f"expected {len(CARRIERS)} carriers, got {names}")
    print(f"  Book: {names}")

    tickers = set()
    for s in book:
        tickers.update(s["universe_tickers"])
    tickers.update(["SPY", "^VIX"])
    print(f"  Loading {len(tickers)} tickers from master_prices ...")
    md = data_provider.get_history(sorted(tickers),
                                   start=DATA_START.strftime("%Y-%m-%d"))

    vix = md.get("^VIX")
    vix_series = None
    if vix is not None and not vix.empty:
        vd = vix.copy()
        if isinstance(vd.columns, pd.MultiIndex):
            vd.columns = vd.columns.get_level_values(0)
        vd.columns = [c.capitalize() for c in vd.columns]
        vix_series = vd["Close"]

    spy = md.get("SPY").copy()
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy.columns = [c.capitalize() for c in spy.columns]
    _CTX["near"] = build_near_series(spy["Close"])
    _CTX["near"].to_parquet(os.path.join(CACHE, "near_high.parquet"))
    print(f"  SPY near-high series {_CTX['near'].index.min().date()} -> "
          f"{_CTX['near'].index.max().date()}")
    d = dial_series()
    print(f"  Dial series {d.index.min().date()} -> {d.index.max().date()} "
          f"(n={d.notna().sum()})")

    sznl_map = load_seasonal_map()
    atr_sznl_map = load_atr_seasonal_map()
    print("  Precomputing indicators ...")
    processed = precompute_all_indicators(md, book, sznl_map, vix_series,
                                          atr_sznl_map)
    print("  Generating candidates ...")
    candidates, signal_data = generate_candidates_fast(processed, book,
                                                       sznl_map, BT_START)
    print(f"    {len(candidates)} candidate signal-dates")

    cand = pd.DataFrame(
        [(pd.Timestamp(c[0]), c[1], book[c[3]]["name"]) for c in candidates],
        columns=["Signal Date", "Ticker", "Strategy"])
    cand.to_parquet(os.path.join(CACHE, "candidates.parquet"), index=False)

    _orig = sbt.frag_band_mult_at
    daily = {}
    try:
        for scheme in SCHEMES:
            sbt.frag_band_mult_at = make_patch(scheme)
            sig = process_signals_fast(
                list(candidates), signal_data, processed, book, FLAT_EQUITY,
                cap_bps=250, overflow_active=True, flat_sizing=True,
                max_long_risk_bps=None, max_short_risk_bps=None,
            )
            df = shape_flat_trades(sig)
            df.to_parquet(os.path.join(CACHE, f"trades_{scheme}.parquet"),
                          index=False)
            sub = sig[pd.to_datetime(sig["Date"]) >= STUDY_START]
            ser = get_daily_mtm_series(sub, md, start_date=STUDY_START)
            daily[scheme] = ser
            print(f"    {scheme:10s} {len(df):5d} trades  "
                  f"PnL ${df['PnL_flat_750k'].sum():,.0f}")
    finally:
        sbt.frag_band_mult_at = _orig

    dl = pd.DataFrame(daily).fillna(0.0)
    dl.index.name = "date"
    dl.to_parquet(os.path.join(CACHE, "daily_pnl.parquet"))
    print(f"  Daily MTM rows: {len(dl)}")


# ---------------------------------------------------------------------------
# Analysis stage
# ---------------------------------------------------------------------------
def load_trades(scheme) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(CACHE, f"trades_{scheme}.parquet"))


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df["Signal Date"])
    df["dial"] = [dial_at(t) for t in ts]
    df["fear"] = [fear_state(t) for t in ts]
    for lab, th in (("near1", 0.01), ("near2", 0.02), ("near3", 0.03)):
        df[lab] = [near_at(t, th) for t in ts]
    df["spy_dd"] = [
        (_CTX["near"].loc[pd.Timestamp(t).normalize(), "spy_dd"]
         if pd.Timestamp(t).normalize() in _CTX["near"].index else np.nan)
        for t in ts]
    df["near"] = df["near2"]
    df["hi50"] = df["dial"].apply(lambda x: bool(x is not None and x >= 50.0))
    df["hi65"] = df["dial"].apply(lambda x: bool(x is not None and x >= 65.0))
    df["year"] = ts.dt.year
    for scheme in SCHEMES:
        df[f"mult_{scheme}"] = [
            scheme_mult({"frag_risk_bands": [[50, 999, 0.25]],
                         "pc_fear_bands": _PCB}, t, scheme) for t in ts]
    return df


def _cell_stats(g: pd.DataFrame) -> pd.Series:
    r = g["R_Multiple"].dropna()
    return pd.Series({
        "N": int(len(g)),
        "avgR": round(float(r.mean()), 4) if len(r) else np.nan,
        "medR": round(float(r.median()), 4) if len(r) else np.nan,
        "hit": round(float((r > 0).mean()), 4) if len(r) else np.nan,
        "totR": round(float(r.sum()), 3) if len(r) else np.nan,
        "avg_$": round(float(g["PnL_flat_750k"].mean()), 1),
        "tot_$": round(float(g["PnL_flat_750k"].sum()), 1),
    })


def sessions_index() -> pd.DatetimeIndex:
    if "sessions" not in _CTX:
        idx = _CTX["near"].dropna(subset=["spy_close"]).index
        _CTX["sessions"] = pd.DatetimeIndex(sorted(set(idx)))
    return _CTX["sessions"]


def episodes(dates) -> list[int]:
    """Episode id per date: a gap of more than EPISODE_GAP sessions starts a
    new episode (pitch_lab's declustering rule, sessions not calendar days)."""
    sess = sessions_index()
    pos = []
    for d in dates:
        d = pd.Timestamp(d).normalize()
        i = sess.searchsorted(d)
        pos.append(int(i))
    order = np.argsort(pos, kind="stable")
    ids = [0] * len(pos)
    ep, prev = 0, None
    for k in order:
        if prev is not None and pos[k] - prev > EPISODE_GAP:
            ep += 1
        ids[k] = ep
        prev = pos[k]
    return ids


def episode_means(df: pd.DataFrame) -> np.ndarray:
    if df.empty:
        return np.array([])
    d = df.copy()
    d["ep"] = episodes(d["Signal Date"])
    return d.groupby("ep")["R_Multiple"].mean().values


def ttest(a: np.ndarray, b: np.ndarray | None = None):
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    if b is None:
        if len(a) < 2:
            return np.nan, np.nan
        t = a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))
        p = (2 * (1 - _sps.t.cdf(abs(t), len(a) - 1))) if _sps else np.nan
        return float(t), float(p)
    b = np.asarray(b, dtype=float)
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    if _sps:
        t, p = _sps.ttest_ind(a, b, equal_var=False)
        return float(t), float(p)
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se), np.nan


def sign_test(a: np.ndarray, mu: float = 0.0):
    """Two-sided exact sign test of P(x > mu) = 0.5 (ties dropped)."""
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    pos = int((a > mu).sum())
    neg = int((a < mu).sum())
    n = pos + neg
    if n == 0:
        return pos, neg, np.nan
    from fractions import Fraction
    from math import comb
    tail = sum(comb(n, k) for k in range(0, min(pos, neg) + 1))
    p = float(min(1.0, 2 * Fraction(tail, 2 ** n)))
    return pos, neg, p


def curve_stats(pnl: pd.Series) -> dict:
    cum = pnl.cumsum()
    dd = cum - cum.cummax()
    return {
        "total_$": round(float(pnl.sum()), 0),
        "maxDD_$": round(float(dd.min()), 0),
        "maxDD_pctNAV": round(float(dd.min() / FLAT_EQUITY * 100), 2),
        "worst21d_$": round(float(pnl.rolling(21).sum().min()), 0),
        "worst_day_$": round(float(pnl.min()), 0),
        "ann_vol_$": round(float(pnl.std(ddof=1) * np.sqrt(252)), 0),
    }


_PCB = None

DIST_BINS = [-np.inf, 0.02, 0.03, 0.05, np.inf]
DIST_LABELS = ["<2%", "2-3%", "3-5%", ">5%"]


def _dist_bucket(dd):
    """Positional (not index-aligned) bucket labels for a distance array."""
    return pd.cut(pd.Series(np.asarray(dd, dtype=float)), bins=DIST_BINS,
                  labels=DIST_LABELS, right=False).astype(object).values


def _bucket_row(label, g):
    r = g["R_Multiple"].dropna()
    yr = g.groupby("year").size()
    return {
        "bucket": label,
        "N_trades": int(len(g)),
        "avgR": round(float(r.mean()), 4) if len(r) else np.nan,
        "medR": round(float(r.median()), 4) if len(r) else np.nan,
        "hit": round(float((r > 0).mean()), 4) if len(r) else np.nan,
        "totR": round(float(r.sum()), 3) if len(r) else np.nan,
        "tot_$_at_1.0x": round(float(g["PnL_flat_750k"].sum()), 0),
        "max_year_share": round(float(yr.max() / yr.sum()), 3) if len(yr) else np.nan,
        "top_year": int(yr.idxmax()) if len(yr) else None,
    }


def removed_by_distance(b, cand, lines):
    """The trades the CURRENT (incumbent) rule removes or cuts at dial >= 50,
    bucketed by SPY's distance from its 252-session closing high, read against
    the fear-ON cell it keeps and the whole dial < 50 book."""
    b = b.copy()
    b["dist"] = _dist_bucket(b["spy_dd"].values)
    cand = cand.copy()
    cand["spy_dd"] = [
        float(_CTX["near"].loc[pd.Timestamp(t).normalize(), "spy_dd"])
        for t in cand["Signal Date"]]
    cand["dist"] = _dist_bucket(cand["spy_dd"].values)
    cand["mult_incumbent"] = [
        scheme_mult({"frag_risk_bands": [[50, 999, 0.25]],
                     "pc_fear_bands": _PCB}, t, "incumbent")
        for t in cand["Signal Date"]]

    hi = b["hi50"]
    sets = {
        "REMOVED (dial>=50, incumbent 0.0x/0.25x)":
            (b[hi & b["mult_incumbent"].isin([0.0, 0.25])],
             cand[cand["hi50"] & cand["mult_incumbent"].isin([0.0, 0.25])]),
        "KEPT full size (dial>=50, fear ON)":
            (b[hi & (b["mult_incumbent"] >= 1.0)],
             cand[cand["hi50"] & (cand["mult_incumbent"] >= 1.0)]),
        "ALL dial>=50 (any fear state)": (b[hi], cand[cand["hi50"]]),
        "dial<50 (all carrier trades)": (b[~hi], cand[~cand["hi50"]]),
    }
    rows = []
    for name, (tr, sg) in sets.items():
        sig_by = sg.groupby("dist").size().to_dict()
        for lab in DIST_LABELS:
            g = tr[tr["dist"] == lab]
            r = _bucket_row(lab, g) if len(g) else {
                "bucket": lab, "N_trades": 0, "avgR": np.nan, "medR": np.nan,
                "hit": np.nan, "totR": np.nan, "tot_$_at_1.0x": 0.0,
                "max_year_share": np.nan, "top_year": None}
            r = {"set": name, **r, "N_signals": int(sig_by.get(lab, 0))}
            rows.append(r)
        tot = _bucket_row("ALL", tr) if len(tr) else {"bucket": "ALL"}
        rows.append({"set": name, **tot, "N_signals": int(len(sg))})
    out = pd.DataFrame(rows)[
        ["set", "bucket", "N_signals", "N_trades", "avgR", "medR", "hit",
         "totR", "tot_$_at_1.0x", "max_year_share", "top_year"]]
    out.to_csv(os.path.join(OUT, "removed_by_distance.csv"), index=False)
    lines.append("\n### 4e. Removed/cut set by distance to the 252d high\n")
    lines.append(out.to_string(index=False))
    print("\n=== 4e. Removed/cut set by distance to the 252d high ===")
    print(out.to_string(index=False))

    rem = sets["REMOVED (dial>=50, incumbent 0.0x/0.25x)"][0]
    n_stale = int((rem["mult_incumbent"] == 0.25).sum())
    near = rem[rem["dist"] == "<2%"]
    far = rem[rem["dist"] != "<2%"]
    en, ef = episode_means(near), episode_means(far)
    t2, p2 = ttest(en, ef)
    t1, p1 = ttest(en)
    t1f, p1f = ttest(ef)
    pn, nn, psn = sign_test(en)
    pf, nf, psf = sign_test(ef)
    cmp_tbl = pd.DataFrame([
        {"cell": "removed, <2% from high", "trades": len(near),
         "episodes": len(en),
         "avgR_trade": round(float(near["R_Multiple"].mean()), 4),
         "avgR_episode": round(float(en.mean()), 4),
         "tot_$": round(float(near["PnL_flat_750k"].sum()), 0),
         "t_ep_vs_0": round(t1, 3), "sign": f"{pn}+/{nn}-",
         "p_sign": round(psn, 3)},
        {"cell": "removed, >=2% from high", "trades": len(far),
         "episodes": len(ef),
         "avgR_trade": round(float(far["R_Multiple"].mean()), 4),
         "avgR_episode": round(float(ef.mean()), 4),
         "tot_$": round(float(far["PnL_flat_750k"].sum()), 0),
         "t_ep_vs_0": round(t1f, 3), "sign": f"{pf}+/{nf}-",
         "p_sign": round(psf, 3)},
    ]).set_index("cell")
    print("\n=== 4f. Within the removed set: <2% vs >=2% from the high ===")
    print(cmp_tbl.to_string())
    cmp_tbl.to_csv(os.path.join(OUT, "removed_near_vs_far.csv"))
    msg = (f"\npooled episode-clustered Welch t (<2% vs >=2%): t={t2:.3f} "
           f"p={p2:.3f}\nstale-P/C (0.25x) rows inside the removed set: "
           f"{n_stale}\n")
    lines.append("\n" + cmp_tbl.to_string() + msg)
    print(msg)


def analysis():
    global _PCB
    import strategy_config
    _PCB = strategy_config.PC_FEAR_BANDS
    os.makedirs(OUT, exist_ok=True)
    if "near" not in _CTX:
        _CTX["near"] = pd.read_parquet(os.path.join(CACHE, "near_high.parquet"))

    base = annotate(load_trades("nobands"))
    base = base[pd.to_datetime(base["Signal Date"]) >= STUDY_START].copy()
    base.to_csv(os.path.join(OUT, "baseline_trades_annotated.csv"), index=False)

    lines = []

    def emit(title, frame, fname=None):
        lines.append(f"\n### {title}\n")
        lines.append(frame.to_string())
        if fname:
            frame.to_csv(os.path.join(OUT, fname))
        print(f"\n=== {title} ===")
        print(frame.to_string())

    # --- 1. universe -------------------------------------------------------
    cand = pd.read_parquet(os.path.join(CACHE, "candidates.parquet"))
    cand = cand[cand["Signal Date"] >= STUDY_START].copy()
    cand["dial"] = [dial_at(t) for t in cand["Signal Date"]]
    cand["near"] = [near_at(t) for t in cand["Signal Date"]]
    cand["fear"] = [fear_state(t) for t in cand["Signal Date"]]
    cand["hi50"] = cand["dial"].apply(lambda x: bool(x is not None and x >= 50))
    cand.to_csv(os.path.join(OUT, "signals_annotated.csv"), index=False)
    uni = pd.DataFrame({
        "signals_2016+": cand.groupby("Strategy").size(),
        "filled_trades_2016+": base.groupby("Strategy").size(),
        "signals_dial>=50": cand[cand.hi50].groupby("Strategy").size(),
        "signals_dial>=50_&_near": cand[cand.hi50 & (cand.near == True)]
                                   .groupby("Strategy").size(),
    }).fillna(0).astype(int)
    uni.loc["TOTAL"] = uni.sum()
    emit("1. Universe (2016+)", uni, "universe.csv")

    # --- 2. cell table -----------------------------------------------------
    b = base.copy()
    b["dial>=50"] = np.where(b["hi50"], "yes", "no")
    b["near-high"] = np.where(b["near"], "yes", "no")
    cell = (b.groupby(["dial>=50", "near-high", "fear"], dropna=False)
              .apply(_cell_stats, include_groups=False)
              .reset_index())
    emit("2a. Cell table (baseline 1.0x sizing, 2016+)", cell, "cell_table.csv")

    cell2 = (b.groupby(["dial>=50", "near-high"], dropna=False)
               .apply(_cell_stats, include_groups=False).reset_index())
    emit("2b. Cell table collapsed over fear state", cell2,
         "cell_table_collapsed.csv")

    zero = b[b["hi50"] & b["near"]]
    if not zero.empty:
        per_strat = (zero.groupby("Strategy").apply(_cell_stats,
                                                    include_groups=False))
        emit("2c. dial>=50 & near-high cell by strategy", per_strat,
             "zerocell_by_strategy.csv")
        per_year = (zero.groupby("year").apply(_cell_stats,
                                               include_groups=False))
        per_year["share_of_N"] = (per_year["N"] / per_year["N"].sum()).round(3)
        emit("2d. dial>=50 & near-high cell by year", per_year,
             "zerocell_by_year.csv")
        lines.append(f"\nmax single-year share of the cell's N: "
                     f"{per_year['share_of_N'].max():.1%}\n")

    # --- 3. scheme comparison ---------------------------------------------
    daily = pd.read_parquet(os.path.join(CACHE, "daily_pnl.parquet"))
    rows = []
    for scheme in SCHEMES:
        t = load_trades(scheme)
        t = t[pd.to_datetime(t["Signal Date"]) >= STUDY_START]
        r = t["R_Multiple"].dropna()
        d = daily[scheme].astype(float)
        stat = curve_stats(d)
        rows.append({
            "scheme": scheme,
            "trades": len(t),
            "total_PnL_$": round(float(t["PnL_flat_750k"].sum()), 0),
            "avgR": round(float(r.mean()), 4),
            "R_dollar_wtd": round(float(t["PnL_flat_750k"].sum()
                                        / t["Risk_flat_750k"].sum()), 4),
            "totR": round(float(r.sum()), 2),
            "risk_$_staged": round(float(t["Risk_flat_750k"].sum()), 0),
            **stat,
        })
    comp = pd.DataFrame(rows).set_index("scheme")
    comp["vs_incumbent_$"] = (comp["total_PnL_$"]
                              - comp.loc["incumbent", "total_PnL_$"]).round(0)
    emit("3. Scheme comparison, 2016+ carrier set, flat $750k", comp,
         "scheme_comparison.csv")

    # per-scheme x strategy PnL
    rows = []
    for scheme in SCHEMES:
        t = load_trades(scheme)
        t = t[pd.to_datetime(t["Signal Date"]) >= STUDY_START]
        rows.append(t.groupby("Strategy")["PnL_flat_750k"].sum().rename(scheme))
    bystrat = pd.concat(rows, axis=1).round(0)
    emit("3b. Scheme PnL by strategy", bystrat, "scheme_by_strategy.csv")

    # --- 4. episode clustering --------------------------------------------
    hi = b[b["hi50"]]
    zc = hi[hi["near"]]
    rest = hi[~hi["near"]]
    ez, er = episode_means(zc), episode_means(rest)
    t_two, p_two = ttest(ez, er)
    t_one, p_one = ttest(ez)
    pos, neg, p_sign = sign_test(ez)
    ep_tbl = pd.DataFrame([
        {"cell": "dial>=50 & near-high", "trades": len(zc),
         "episodes": len(ez), "avgR_trade": round(float(zc["R_Multiple"].mean()), 4)
         if len(zc) else np.nan,
         "avgR_episode": round(float(ez.mean()), 4) if len(ez) else np.nan},
        {"cell": "dial>=50 & NOT near-high", "trades": len(rest),
         "episodes": len(er),
         "avgR_trade": round(float(rest["R_Multiple"].mean()), 4) if len(rest) else np.nan,
         "avgR_episode": round(float(er.mean()), 4) if len(er) else np.nan},
    ]).set_index("cell")
    emit("4. Episode clustering, dial>=50 split by near-high", ep_tbl,
         "episodes.csv")
    lines.append(
        f"\nepisode-clustered Welch t (zeroed cell vs rest of dial>=50): "
        f"t={t_two:.3f} p={p_two:.3f}\n"
        f"zeroed cell episode means vs 0: t={t_one:.3f} p={p_one:.3f}; "
        f"sign test {pos}+/{neg}- p={p_sign:.3f}\n")
    print(lines[-1])

    # LOYO on the zeroed cell
    loyo = []
    for y in sorted(zc["year"].unique()):
        rem = zc[zc["year"] != y]
        em = episode_means(rem)
        t_, p_ = ttest(em)
        loyo.append({"drop_year": int(y), "episodes_left": len(em),
                     "avgR_episode": round(float(em.mean()), 4) if len(em) else np.nan,
                     "t": round(t_, 3), "p": round(p_, 4) if p_ == p_ else np.nan})
    loyo_df = pd.DataFrame(loyo).set_index("drop_year")
    emit("4b. LOYO on the dial>=50 & near-high cell", loyo_df, "loyo.csv")

    # drop-biggest-episode
    if len(zc):
        zz = zc.copy()
        zz["ep"] = episodes(zz["Signal Date"])
        big = zz.groupby("ep")["PnL_flat_750k"].sum().abs().idxmax()
        rem = zz[zz["ep"] != big]
        em = episode_means(rem)
        t_, p_ = ttest(em)
        lines.append(
            f"\ndrop-largest-|PnL|-episode: {len(rem)} trades / {len(em)} "
            f"episodes, avgR_episode {em.mean():.4f}, t={t_:.3f}\n")
        print(lines[-1])

    # --- 4c. the split WITHIN each fear state (the incumbent governs the
    # dial>=50 zone by fear state, so the qualifier has to beat it there) ---
    rows = []
    for st in ("off", "on"):
        h = hi[hi["fear"] == st]
        a_, c_ = h[h["near"]], h[~h["near"]]
        ea, ec = episode_means(a_), episode_means(c_)
        t_, p_ = ttest(ea, ec)
        rows.append({
            "fear": st,
            "N_near": len(a_), "ep_near": len(ea),
            "avgR_near": round(float(a_["R_Multiple"].mean()), 4) if len(a_) else np.nan,
            "N_notnear": len(c_), "ep_notnear": len(ec),
            "avgR_notnear": round(float(c_["R_Multiple"].mean()), 4) if len(c_) else np.nan,
            "welch_t_ep": round(t_, 3) if t_ == t_ else np.nan,
            "p": round(p_, 3) if p_ == p_ else np.nan,
        })
    emit("4c. dial>=50 split by near-high WITHIN each fear state",
         pd.DataFrame(rows).set_index("fear"), "episodes_by_fear.csv")

    # --- 4d. the exact cells each variant moves ---------------------------
    a_kill = hi[hi["near"] & (hi["fear"] != "off")]      # A zeroes these (inc 1.0x)
    b_add = hi[(~hi["near"]) & (hi["fear"] == "off")]    # B restores these (inc 0.0x)
    rows = []
    for lab, g, direction in (
            ("A zeroes: dial>=50 & near & fear ON", a_kill, "lose"),
            ("B restores: dial>=50 & NOT near & fear OFF", b_add, "gain")):
        em = episode_means(g)
        t_, p_ = ttest(em)
        pos, neg, ps = sign_test(em)
        rows.append({
            "cell": lab, "N": len(g), "episodes": len(em),
            "avgR": round(float(g["R_Multiple"].mean()), 4) if len(g) else np.nan,
            "totR": round(float(g["R_Multiple"].sum()), 3) if len(g) else np.nan,
            "PnL_at_1.0x_$": round(float(g["PnL_flat_750k"].sum()), 0),
            "t_ep_vs_0": round(t_, 3) if t_ == t_ else np.nan,
            "p": round(p_, 3) if p_ == p_ else np.nan,
            "sign": f"{pos}+/{neg}-", "p_sign": round(ps, 3) if ps == ps else np.nan,
            "effect_on_book": direction,
        })
    emit("4d. The cells each variant actually moves vs the incumbent",
         pd.DataFrame(rows).set_index("cell"), "variant_delta_cells.csv")

    # --- 4e. what the CURRENT rule removes, bucketed by distance to high ---
    removed_by_distance(b, cand, lines)

    # --- 5. sensitivity ----------------------------------------------------
    sens = []
    for dial_th, dcol in ((50, "hi50"), (65, "hi65")):
        for near_lab in ("near1", "near2", "near3"):
            hi_ = b[b[dcol]]
            zc_ = hi_[hi_[near_lab] == True]
            rest_ = hi_[hi_[near_lab] != True]
            ez_, er_ = episode_means(zc_), episode_means(rest_)
            t_, p_ = ttest(ez_, er_)
            sens.append({
                "dial_thresh": dial_th,
                "near_thresh": {"near1": "1%", "near2": "2%", "near3": "3%"}[near_lab],
                "N_zerocell": len(zc_), "ep_zerocell": len(ez_),
                "avgR_zerocell": round(float(zc_["R_Multiple"].mean()), 4) if len(zc_) else np.nan,
                "tot$_zerocell": round(float(zc_["PnL_flat_750k"].sum()), 0),
                "N_rest": len(rest_),
                "avgR_rest": round(float(rest_["R_Multiple"].mean()), 4) if len(rest_) else np.nan,
                "welch_t_ep": round(t_, 3) if t_ == t_ else np.nan,
            })
    sens_df = pd.DataFrame(sens)
    emit("5a. Sensitivity: near-high threshold x dial threshold", sens_df,
         "sensitivity_thresholds.csv")

    # what sits between the 1% and 2% near-high thresholds
    band = b[b["hi50"] & (b["near2"] == True) & (b["near1"] != True)]
    if not band.empty:
        bb = band[["Signal Date", "Strategy", "Ticker", "dial", "spy_dd",
                   "fear", "R_Multiple", "PnL_flat_750k"]].copy()
        bb["spy_dd"] = (bb["spy_dd"] * 100).round(2)
        bb["Signal Date"] = pd.to_datetime(bb["Signal Date"]).dt.date
        emit("5a2. Trades in the 1%-2% near-high band (dial>=50)",
             bb.sort_values("Signal Date").set_index("Signal Date"),
             "near_band_1to2pct.csv")

    era = []
    for lab, m in (("pre-2020", b["year"] < 2020), ("2020+", b["year"] >= 2020)):
        sub = b[m]
        hi_ = sub[sub["hi50"]]
        for cell_lab, cm in (("dial>=50 & near", hi_["near"] == True),
                             ("dial>=50 & NOT near", hi_["near"] != True),
                             ("dial<50", None)):
            g = sub[~sub["hi50"]] if cm is None else hi_[cm]
            if g.empty:
                continue
            s = _cell_stats(g)
            era.append({"era": lab, "cell": cell_lab, **s.to_dict()})
    era_df = pd.DataFrame(era)
    emit("5b. Sensitivity: era split", era_df, "sensitivity_era.csv")

    # second dial vintage (research-only recompute), where it covers
    try:
        ts_v = pd.read_parquet(os.path.join(_ROOT, "data",
                                            "rd2_fragility_ts.parquet"))
        v = ts_v["63d"].dropna().rolling(10, min_periods=1).mean()
        v.index = pd.to_datetime(v.index).normalize()
        cov = b[pd.to_datetime(b["Signal Date"]) <= v.index.max()].copy()
        cov["dial_ts"] = [float(v.loc[pd.Timestamp(t).normalize()])
                          if pd.Timestamp(t).normalize() in v.index else np.nan
                          for t in cov["Signal Date"]]
        cov = cov.dropna(subset=["dial_ts"])
        cov["hi50_ts"] = cov["dial_ts"] >= 50
        vt = (cov.groupby([np.where(cov["hi50_ts"], "yes", "no"),
                           np.where(cov["near"], "yes", "no")])
                 .apply(_cell_stats, include_groups=False))
        vt.index.names = ["dial>=50 (ts vintage)", "near-high"]
        emit("5c. Second dial vintage (rd2_fragility_ts, research recompute, "
             f"covers through {v.index.max().date()})", vt.reset_index(),
             "vintage_ts_cells.csv")
        agree = float((cov["hi50_ts"] == cov["hi50"]).mean())
        lines.append(f"\ndial>=50 agreement live vs ts vintage on covered "
                     f"trades: {agree:.1%} (n={len(cov)})\n")
    except Exception as e:  # pragma: no cover
        lines.append(f"\n(second vintage skipped: {e})\n")

    # --- 6. Aug-Sep 2026 ---------------------------------------------------
    recent = cand[cand["Signal Date"] >= pd.Timestamp("2026-08-01")].copy()
    r_base = base[pd.to_datetime(base["Signal Date"])
                  >= pd.Timestamp("2026-08-01")]
    rmap = {(pd.Timestamp(r["Signal Date"]), r["Strategy"], r["Ticker"]):
            (r["R_Multiple"], r["PnL_flat_750k"], r["Exit Type"])
            for _, r in r_base.iterrows()}
    rows = []
    for _, c in recent.iterrows():
        key = (pd.Timestamp(c["Signal Date"]), c["Strategy"], c["Ticker"])
        rv = rmap.get(key, (np.nan, np.nan, "no fill"))
        exe = {"frag_risk_bands": [[50, 999, 0.25]], "pc_fear_bands": _PCB}
        rows.append({
            "Signal Date": pd.Timestamp(c["Signal Date"]).date(),
            "Strategy": c["Strategy"], "Ticker": c["Ticker"],
            "dial": round(c["dial"], 1) if c["dial"] is not None else None,
            "near_high": c["near"],
            "spy_dd_pct": round(float(_CTX["near"].loc[
                pd.Timestamp(c["Signal Date"]).normalize(), "spy_dd"]) * 100, 2),
            "fear": c["fear"],
            "mult_incumbent": scheme_mult(exe, c["Signal Date"], "incumbent"),
            "mult_A": scheme_mult(exe, c["Signal Date"], "variant_a"),
            "mult_B": scheme_mult(exe, c["Signal Date"], "variant_b"),
            "R_at_1.0x": round(rv[0], 3) if rv[0] == rv[0] else None,
            "PnL_at_1.0x_$": round(rv[1], 0) if rv[1] == rv[1] else None,
            "exit": rv[2],
        })
    recent_df = pd.DataFrame(rows).sort_values(["Signal Date", "Strategy"])
    emit("6. Carrier signals since 2026-08-01", recent_df, "aug_sep_2026.csv")

    fill = recent_df.dropna(subset=["R_at_1.0x"])
    sub = []
    for lab, m in (("Aug 2026", fill["Signal Date"] < datetime.date(2026, 9, 1)),
                   ("Sep 2026", fill["Signal Date"] >= datetime.date(2026, 9, 1)),
                   ("Aug-Sep 2026", pd.Series(True, index=fill.index))):
        g = fill[m]
        sub.append({
            "window": lab, "filled": len(g),
            "totR_at_1.0x": round(float(g["R_at_1.0x"].sum()), 3),
            "tot$_at_1.0x": round(float(g["PnL_at_1.0x_$"].sum()), 0),
            "incumbent_$": round(float((g["PnL_at_1.0x_$"] * g["mult_incumbent"]).sum()), 0),
            "variantA_$": round(float((g["PnL_at_1.0x_$"] * g["mult_A"]).sum()), 0),
            "variantB_$": round(float((g["PnL_at_1.0x_$"] * g["mult_B"]).sum()), 0),
        })
    emit("6b. Aug-Sep 2026 subtotals (filled signals, flat basis)",
         pd.DataFrame(sub).set_index("window"), "aug_sep_2026_subtotals.csv")

    n_missing = int(base["near"].isna().sum()) if base["near"].dtype == object else 0
    lines.append(f"\nbaseline trades with an unresolvable near-high flag: "
                 f"{n_missing}\n")

    # --- provenance check vs the shipped ledger ---------------------------
    try:
        led = pd.read_parquet(os.path.join(_ROOT, "data",
                                           "backtest_trades_full.parquet"))
        led = led[led.Strategy.isin(CARRIERS)
                  & (led["Signal Date"] >= STUDY_START)]
        mine = load_trades("incumbent")
        mine = mine[pd.to_datetime(mine["Signal Date"]) >= STUDY_START]
        chk = pd.DataFrame({
            "shipped_ledger": [len(led), round(led["PnL_flat_750k"].sum(), 0)],
            "replay_incumbent": [len(mine),
                                 round(mine["PnL_flat_750k"].sum(), 0)],
        }, index=["trades", "PnL_flat_$"])
        emit("0. Provenance check: replayed incumbent vs shipped ledger", chk,
             "provenance_check.csv")
    except Exception as e:  # pragma: no cover
        lines.append(f"\n(provenance check skipped: {e})\n")

    with open(os.path.join(OUT, "analysis_log.txt"), "w") as fh:
        fh.write("\n".join(str(x) for x in lines))
    print(f"\nWrote results -> {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-only", action="store_true")
    a = ap.parse_args()
    if not a.analysis_only:
        run_engine()
    analysis()


if __name__ == "__main__":
    main()
