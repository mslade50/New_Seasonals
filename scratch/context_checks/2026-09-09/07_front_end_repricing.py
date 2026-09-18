"""DRILL 07 (2026-09-09) — front-end repricing (^IRX 63d-return rank >= 95)
while the S&P is still within 3% of its own 52w high. What follows on the
equity tape?

Convention: lag=0 close-to-close forward returns (fwd_ret) — CONTEXT, not a
trade entry. Fractions into summarize, percent out. ^VIX "return" is a percent
change in the index LEVEL. Every series is .dropna()'d before rolling stats.

CAVEAT computed and printed below, not assumed: ^IRX is a 13-week BILL YIELD
level. A 63-day percent CHANGE of a near-zero yield explodes (0.02 -> 0.04 is
+100%), so the ZIRP eras can manufacture rank>=95 readings off noise. The
script prints the year distribution of triggers and a diagnostic cut that
excludes sessions with ^IRX below 0.50.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

ASOF = pd.Timestamp("2026-09-09")
HORIZONS = (1, 5, 21)
RANK_MIN = 95.0
SPX_NEAR_HIGH = 0.97      # within 3% of trailing-252d max
GAP_TD = 10
FWD_TICKERS = ["^GSPC", "^VIX"]


def _fmt(x, nd=3):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.{nd}f}"


def leg_report(fwd: pd.Series, epi, label: str):
    s = fwd.reindex(pd.DatetimeIndex(epi)).dropna()
    if len(s) == 0:
        print(f"    {label:<26} n=0 (no measurable episodes)")
        return None
    d = pd.DatetimeIndex(s.index)
    v = s.values.astype(float)
    r = summarize(v, label)
    w = int((v > 0).sum())
    l = len(v) - w
    p_up = sign_test(w, len(v))       # P(>= w wins) -- the UP-side one-sided p
    p_dn = sign_test(l, len(v))       # P(>= l losses) -- the DOWN-side one-sided p
    print(f"    {label:<26} n={r['n']:<4} mean={_fmt(r['mean_pct'])}%  "
          f"med={_fmt(r['median_pct'])}%  hit={r['hit']:.1f}%  t={_fmt(r['t'], 2)}  "
          f"rec {w}-{l}  sign p(up)={p_up:.4f} p(dn)={p_dn:.4f}  "
          f"worst={_fmt(r['worst_pct'], 2)}%  best={_fmt(r['best_pct'], 2)}%")
    for e in era_split(d, v):
        if e.get("n"):
            print(f"        era {e['label']:<12} n={e['n']:<4} mean={_fmt(e['mean_pct'])}%  "
                  f"hit={e['hit']:.1f}%  t={_fmt(e['t'], 2)}")
        else:
            print(f"        era {e['label']:<12} n=0")
    print(f"        cluster: {cluster_note(d, v)}")
    return r


def ctrl_report(fwd: pd.Series, dates, label: str):
    s = fwd.reindex(pd.DatetimeIndex(dates)).dropna() if dates is not None else fwd.dropna()
    if len(s) == 0:
        print(f"    {label:<38} n=0")
        return None
    v = s.values.astype(float)
    r = summarize(v, label)
    print(f"    {label:<38} n={r['n']:<6} mean={_fmt(r['mean_pct'])}%  "
          f"med={_fmt(r['median_pct'])}%  hit={r['hit']:.1f}%  t={_fmt(r['t'], 2)}")
    return r


def run_cell(name, trig_raw, spine, fwd, irx_close, do_controls=True):
    print("\n" + "=" * 78)
    print(f"CELL: {name}")
    print("=" * 78)
    trig = pd.DatetimeIndex(trig_raw).intersection(spine)
    print(f"  trigger sessions: {len(trig_raw)}   on the ^GSPC session spine: {len(trig)}")
    if len(trig) == 0:
        print("  NO TRIGGERS. Dead.")
        return None
    epi = declusters(trig, GAP_TD, spine)
    print(f"  episodes after {GAP_TD}td declustering: {len(epi)}")
    print(f"  span: {trig[0].date()} .. {trig[-1].date()}")
    if len(epi) <= 30:
        print("  episode dates: " + ", ".join(str(d.date()) for d in epi))
    else:
        print("  (>30 episodes; dates suppressed)")
    yr = pd.Series(pd.DatetimeIndex(epi).year).value_counts().sort_index()
    print("  episodes by year: " + ", ".join(f"{y}:{n}" for y, n in yr.items()))
    lv = irx_close.reindex(pd.DatetimeIndex(epi)).dropna()
    if len(lv):
        print(f"  ^IRX level on episode dates: min={lv.min():.3f} med={lv.median():.3f} "
              f"max={lv.max():.3f}   episodes with ^IRX < 0.50: "
              f"{int((lv < 0.5).sum())} of {len(lv)}")

    for h in HORIZONS:
        print(f"\n  --- forward lag=0, h={h} td (episodes) ---")
        for t in FWD_TICKERS:
            leg_report(fwd[t][h], epi, t)

    if do_controls:
        print("\n  --- CONTROLS for the ^GSPC leg ---")
        for h in HORIZONS:
            print(f"   h={h}:")
            cell = leg_report(fwd["^GSPC"][h], epi, f"^GSPC cell h={h}")
            a = ctrl_report(fwd["^GSPC"][h], None, f"CTRL-a all days full history h={h}")
            gidx = fwd["^GSPC"][h].dropna().index
            loc = local_control(gidx, trig, 126)
            b = ctrl_report(fwd["^GSPC"][h], loc, f"CTRL-b local +/-126td ex-trigger h={h}")
            if cell and a:
                print(f"      EDGE vs all-days   = {cell['mean_pct'] - a['mean_pct']:+.3f} pp")
            if cell and b:
                print(f"      EDGE vs local ctrl = {cell['mean_pct'] - b['mean_pct']:+.3f} pp")
    return epi


def main():
    print("=" * 78)
    print("DRILL 07 — ^IRX 63d-return rank >= 95 x S&P near its 52w high  (asof 2026-09-09)")
    print("=" * 78)

    px = load_prices(["^IRX", "^GSPC", "^VIX"])
    irx = px["^IRX"]["Close"].dropna()
    gspc = px["^GSPC"]["Close"].dropna()
    vix = px["^VIX"]["Close"].dropna()

    # ---- 6. history adequacy ---------------------------------------------
    print(f"\n^IRX in cache: n={len(irx)} sessions, {irx.index[0].date()} .. {irx.index[-1].date()}")
    print(f"  first date with a defined 63d-return rank (needs 63 + 252 valid sessions): ", end="")
    r63 = pct_rank(px["^IRX"]["Close"], 63, 252).dropna()
    print(f"{r63.index[0].date()}  (n={len(r63)} ranked sessions)")
    print(f"^GSPC in cache: n={len(gspc)}, {gspc.index[0].date()} .. {gspc.index[-1].date()}")
    print(f"^VIX  in cache: n={len(vix)}, {vix.index[0].date()} .. {vix.index[-1].date()}")

    # ---- 1. today's readings ---------------------------------------------
    r21 = pct_rank(px["^IRX"]["Close"], 21, 252)
    r63_full = pct_rank(px["^IRX"]["Close"], 63, 252)
    print(f"\nTODAY {ASOF.date()}: ^IRX close = {irx.loc[ASOF]:.4f}")
    print(f"  63d-return rank (252 lookback) = {r63_full.loc[ASOF]:.2f}   "
          f"21d-return rank = {r21.loc[ASOF]:.2f}")
    i63 = (irx / irx.shift(63) - 1.0)
    i21 = (irx / irx.shift(21) - 1.0)
    print(f"  raw 63d change in the yield level = {100*i63.loc[ASOF]:+.2f}%   "
          f"raw 21d = {100*i21.loc[ASOF]:+.2f}%")
    rank_flag = (r63_full >= RANK_MIN)
    rf = rank_flag[r63_full.notna()]
    print(f"  sessions with 63d rank >= {RANK_MIN}: {int(rf.sum())} of {len(rf)} "
          f"({100*rf.mean():.2f}%)")

    # ---- 2. S&P near its own 252d high -----------------------------------
    g_hi = gspc.rolling(252).max()
    g_near = (gspc >= SPX_NEAR_HIGH * g_hi).where(g_hi.notna())
    print(f"\n^GSPC {ASOF.date()}: close = {gspc.loc[ASOF]:.2f}   252d max = {g_hi.loc[ASOF]:.2f}   "
          f"{100*(gspc.loc[ASOF]/g_hi.loc[ASOF] - 1):+.2f}% vs 252d high   "
          f"within-3% flag = {g_near.loc[ASOF]}")
    gn = g_near.dropna()
    print(f"  ^GSPC within 3% of its 252d high on {int(gn.sum())} of {len(gn)} sessions "
          f"({100*gn.mean():.2f}%)")

    # ---- forward return series -------------------------------------------
    spine = gspc.index
    fwd = {
        "^GSPC": {h: fwd_ret(gspc, h) for h in HORIZONS},
        "^VIX": {h: fwd_ret(vix, h) for h in HORIZONS},
    }

    # ---- MAIN cell --------------------------------------------------------
    rank_dates = pd.DatetimeIndex(r63_full.index[rank_flag.fillna(False).values])
    near_dates = pd.DatetimeIndex(g_near.index[g_near.astype(float).fillna(0.0).astype(bool).values])
    main_trig = rank_dates.intersection(near_dates)
    epi_main = run_cell(
        f"MAIN — ^IRX 63d rank >= {RANK_MIN} AND ^GSPC within 3% of its 252d high",
        main_trig, spine, fwd, irx, do_controls=True)

    # ---- 5. same cell WITHOUT the equity-proximity condition ---------------
    epi_norank = run_cell(
        f"NO-EQUITY-CONDITION — ^IRX 63d rank >= {RANK_MIN} only",
        rank_dates, spine, fwd, irx, do_controls=True)

    # ---- diagnostic: ZIRP artifact ---------------------------------------
    print("\n" + "=" * 78)
    print("DIAGNOSTIC — the same MAIN cell with ZIRP sessions removed (^IRX >= 0.50)")
    print("(a 63d PERCENT change of a near-zero bill yield explodes; this shows "
          "whether the cell is a rates story or a divide-by-small-number story)")
    print("=" * 78)
    hi_level = pd.DatetimeIndex(irx.index[(irx >= 0.5).values])
    run_cell(
        f"MAIN cell, ^IRX level >= 0.50 only",
        main_trig.intersection(hi_level), spine, fwd, irx, do_controls=True)

    print("\n" + "=" * 78)
    print("DRILL 07 done.")
    print("=" * 78)


if __name__ == "__main__":
    main()
