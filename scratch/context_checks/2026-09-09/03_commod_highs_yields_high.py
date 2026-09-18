"""DRILL 03 (2026-09-09) — REPLACEMENT VERSION.

The original question (a broad commodity panel at 252d highs) was withdrawn:
the 2026-09-09 "252d high" prints in ZC=F / ZS=F / SB=F / KC=F / CT=F are
CONTRACT-ROLL SEAMS, not prices (see 09_futures_data_integrity.py), and ZW=F
is not at a 252d high at all. Only CL=F, NG=F, HG=F and SI=F passed the
integrity test, plus cash indices / rates which have no roll concept.

QUESTION AS REBUILT: crude is up ~16% over 21 sessions on clean volume while
the 10-year yield sits AT a 252d high and IEF sits fractionally off a 252d
LOW, on the eve of PPI and CPI. What has that energy-plus-rates combination
done to the equity tape?

Conventions: lag=0 close-to-close forward returns (fwd_ret) — CONTEXT, not a
trade entry. Fractions into summarize, percent out. ^TNX and ^VIX "returns"
are percent changes in the YIELD / INDEX LEVEL, not price returns. Every
series is .dropna()'d before rolling stats.

INTEGRITY GUARD (requirement f): every trigger date must have a CL=F bar that
traded back THROUGH its prior close (Low <= prev_close <= High) OR gapped less
than 3%. Failures are dropped and counted, so the cell cannot be built on a
crude roll seam either.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

ASOF = pd.Timestamp("2026-09-09")
FWD_TICKERS = ["^GSPC", "^TNX", "^VIX", "IEF"]
HORIZONS = (1, 5, 21)
GAP_TD = 10
CRUDE_DECILE = 90.0        # top decile of trailing-252 21d returns
CRUDE_QUINTILE = 80.0      # loosened
TNX_STRICT = 0.995         # within 0.5% of trailing-252d max
TNX_LOOSE = 0.98           # within 2.0%
MAX_GAP = 0.03             # integrity guard: 3%


def _fmt(x, nd=3):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.{nd}f}"


def at_high_flag(close: pd.Series, frac: float) -> pd.Series:
    """1.0 / 0.0 / NaN(no 252d window yet), on the ticker's OWN valid sessions."""
    c = close.dropna()
    hi = c.rolling(252).max()
    return (c >= frac * hi).astype(float).where(hi.notna())


def crude_integrity(bars: pd.DataFrame) -> pd.DataFrame:
    """Per-session roll-seam test on CL=F. A clean bar either spans its prior
    close (Low <= prev_close <= High) or gaps less than MAX_GAP."""
    b = bars.dropna(subset=["Close"]).copy()
    prev = b["Close"].shift(1)
    b["prev_close"] = prev
    b["gap"] = (b["Open"] / prev - 1.0).abs()
    b["spans_prev"] = (b["Low"] <= prev) & (b["High"] >= prev)
    b["clean"] = b["spans_prev"] | (b["gap"] < MAX_GAP)
    return b


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
    p_up = sign_test(w, len(v))       # P(>= w wins) -- UP-side one-sided p
    p_dn = sign_test(l, len(v))       # P(>= l losses) -- DOWN-side one-sided p
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


def run_cell(name, trig_raw, spine, fwd, cl_clean, apply_guard=True,
             do_controls=True):
    print("\n" + "=" * 78)
    print(f"CELL: {name}")
    print("=" * 78)
    trig = pd.DatetimeIndex(trig_raw).intersection(spine)
    print(f"  trigger sessions: {len(pd.DatetimeIndex(trig_raw))}   "
          f"on the ^GSPC session spine: {len(trig)}")
    if apply_guard:
        ok = cl_clean.reindex(trig)
        bad = pd.DatetimeIndex(ok.index[ok.fillna(False).astype(bool).values == False])
        trig = pd.DatetimeIndex(ok.index[ok.fillna(False).astype(bool).values])
        print(f"  INTEGRITY GUARD (CL=F bar spans prior close OR gap < {100*MAX_GAP:.0f}%): "
              f"dropped {len(bad)} trigger date(s)"
              + (": " + ", ".join(str(d.date()) for d in bad[:20]) if len(bad) else ""))
    else:
        print("  INTEGRITY GUARD: not applicable (no crude condition in this cell)")
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
    print("DRILL 03 (REPLACEMENT) — crude 21d top decile x ^TNX at a 252d high")
    print("asof 2026-09-09.  Roll-contaminated grains/softs EXCLUDED by design.")
    print("=" * 78)

    px = load_prices(["CL=F", "^TNX", "^GSPC", "^VIX", "IEF"])
    for t in ["CL=F", "^TNX", "^GSPC", "^VIX", "IEF"]:
        c = px[t]["Close"].dropna()
        print(f"   {t:<6} n={len(c):<6} {c.index[0].date()} .. {c.index[-1].date()}")
    print("   NOTE: IEF starts 2002-07-30, so any trigger before that has NO "
          "measurable IEF forward return (its n will be lower).")

    cl = px["CL=F"]["Close"].dropna()
    tnx = px["^TNX"]["Close"].dropna()
    gspc = px["^GSPC"]["Close"].dropna()
    ief = px["IEF"]["Close"].dropna()
    spine = gspc.index

    # ---- integrity table on CL=F -----------------------------------------
    clb = crude_integrity(px["CL=F"])
    cl_clean = clb["clean"]
    print(f"\nCL=F integrity: {int(cl_clean.sum())} of {int(cl_clean.notna().sum())} "
          f"sessions clean ({100*cl_clean.mean():.2f}%)")
    row = clb.loc[ASOF]
    print(f"  TODAY {ASOF.date()}: O={row['Open']:.2f} H={row['High']:.2f} "
          f"L={row['Low']:.2f} C={row['Close']:.2f}  prev_close={row['prev_close']:.2f}  "
          f"gap={100*row['gap']:+.2f}%  spans_prev={bool(row['spans_prev'])}  "
          f"CLEAN={bool(row['clean'])}")

    # ---- today's readings -------------------------------------------------
    cl_r21 = pct_rank(px["CL=F"]["Close"], 21, 252)
    cl_ret21 = (cl / cl.shift(21) - 1.0)
    cl_z10 = zscore(px["CL=F"]["Close"], 10)
    print(f"\nCL=F {ASOF.date()}: close={cl.loc[ASOF]:.2f}   21d return="
          f"{100*cl_ret21.loc[ASOF]:+.2f}%   21d-return rank (252 lookback)="
          f"{cl_r21.loc[ASOF]:.2f}   z10={cl_z10.loc[ASOF]:+.2f}")
    print(f"   qualifies for top decile (rank >= {CRUDE_DECILE})? "
          f"{bool(cl_r21.loc[ASOF] >= CRUDE_DECILE)}   "
          f"top quintile (>= {CRUDE_QUINTILE})? {bool(cl_r21.loc[ASOF] >= CRUDE_QUINTILE)}")

    tnx_hi = tnx.rolling(252).max()
    tnx_s = at_high_flag(px["^TNX"]["Close"], TNX_STRICT)
    tnx_l = at_high_flag(px["^TNX"]["Close"], TNX_LOOSE)
    print(f"^TNX {ASOF.date()}: close={tnx.loc[ASOF]:.3f}   252d max={tnx_hi.loc[ASOF]:.3f}   "
          f"pct of max={100*tnx.loc[ASOF]/tnx_hi.loc[ASOF]:.3f}%   "
          f"within-0.5% flag={tnx_s.loc[ASOF]}   within-2% flag={tnx_l.loc[ASOF]}")
    ts = tnx_s.dropna()
    tl = tnx_l.dropna()
    print(f"   ^TNX within 0.5% of its 252d max on {int(ts.sum())} of {len(ts)} sessions "
          f"({100*ts.mean():.2f}%);  within 2.0% on {int(tl.sum())} of {len(tl)} "
          f"({100*tl.mean():.2f}%)")

    ief_lo = ief.rolling(252).min()
    print(f"IEF  {ASOF.date()}: close={ief.loc[ASOF]:.3f}   252d min={ief_lo.loc[ASOF]:.3f}   "
          f"{100*(ief.loc[ASOF]/ief_lo.loc[ASOF] - 1):+.3f}% above its 252d LOW")

    cl_dec = (cl_r21 >= CRUDE_DECILE)
    cl_qui = (cl_r21 >= CRUDE_QUINTILE)
    print(f"\nbase rates: CL=F 21d rank >= {CRUDE_DECILE} on "
          f"{int(cl_dec.sum())} of {int(cl_r21.notna().sum())} ranked sessions "
          f"({100*cl_dec[cl_r21.notna()].mean():.2f}%);  >= {CRUDE_QUINTILE} on "
          f"{int(cl_qui.sum())} ({100*cl_qui[cl_r21.notna()].mean():.2f}%)")

    # ---- forward series ---------------------------------------------------
    fwd = {t: {h: fwd_ret(px[t]["Close"].dropna(), h) for h in HORIZONS}
           for t in FWD_TICKERS}

    dec_dates = pd.DatetimeIndex(cl_r21.index[cl_dec.fillna(False).values])
    qui_dates = pd.DatetimeIndex(cl_r21.index[cl_qui.fillna(False).values])
    tnx_s_dates = pd.DatetimeIndex(tnx_s.index[tnx_s.fillna(0).astype(bool).values])
    tnx_l_dates = pd.DatetimeIndex(tnx_l.index[tnx_l.fillna(0).astype(bool).values])

    # ---- (a/b/c) STRICT cell ---------------------------------------------
    epi_strict = run_cell(
        f"STRICT — CL=F 21d rank >= {CRUDE_DECILE} AND ^TNX within 0.5% of 252d max",
        dec_dates.intersection(tnx_s_dates), spine, fwd, cl_clean,
        apply_guard=True, do_controls=True)
    n_strict = 0 if epi_strict is None else len(epi_strict)

    # ---- (d) each leg alone ----------------------------------------------
    run_cell(f"LEG A ONLY — CL=F 21d rank >= {CRUDE_DECILE} (no rates condition)",
             dec_dates, spine, fwd, cl_clean, apply_guard=True, do_controls=True)
    run_cell("LEG B ONLY — ^TNX within 0.5% of its 252d max (no crude condition)",
             tnx_s_dates, spine, fwd, cl_clean, apply_guard=False, do_controls=True)

    # ---- (e) loosened -----------------------------------------------------
    if n_strict < 15:
        print(f"\n\n!!! STRICT cell has {n_strict} episodes (< 15). "
              f"Running the LOOSENED version as well. !!!")
    run_cell(
        f"LOOSENED — CL=F 21d rank >= {CRUDE_QUINTILE} AND ^TNX within 2.0% of 252d max",
        qui_dates.intersection(tnx_l_dates), spine, fwd, cl_clean,
        apply_guard=True, do_controls=True)

    print("\n" + "=" * 78)
    print("DRILL 03 done.")
    print("=" * 78)


if __name__ == "__main__":
    main()
