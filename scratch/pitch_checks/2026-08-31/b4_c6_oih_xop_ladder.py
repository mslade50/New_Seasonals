"""C6 round 1 - long OIH outright at an OIH-minus-XOP 63-day spread extreme.

The watchlist (entry 24) parks this as a RECORD condition: at the 2.5th PIT
trailing-252 percentile rung, h=10, long OIH MOC lag=1 pays +0.934% on 28-23
and arms at 32-23 (sign p 0.046). Today reads PIT 4.0, so the literal rung is
NOT armed and the candidate is a LOOSENED rung.

Obligations discharged here:
  0. premise re-derivation under BOTH PIT definitions in the repo, plus the
     full-sample percentile for contrast, plus the recent trigger-day tape.
  1. reproduce the parked 2.5 cell exactly (confirm or refute 28-23 / +0.934%).
  2. threshold ladder 1.0 / 2.5 / 4.0 / 5.0 / 10.0 / 20 / all, TODAY'S 4.0
     marked, with the edge over OIH's own drift at each rung.
  3. the RECORD: what it stands at, whether it moved, what arms it.
  3b. FRESH vs CONTINUATION trigger days - today is not the first day of this
      episode, and the parked record is an EPISODE record.
  4. gate attribution: OIH forward return with no spread condition, decluster
     ladder, drop-best-2, year histogram, era split, MIDTERM split, local
     +/-126td control, cost at 6 bp one leg.
  6. SPY-beta and XLE-beta PIT residual; does the trigger just select falling
     energy?
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change  # noqa

import numpy as np
import pandas as pd

BAR = pd.Timestamp("2026-08-28")
H = 10
LAG = 1
COST_BPS = 6.0
NAMES = ["OIH", "XOP", "XLE", "USO", "SPY"]

px_all = load_prices(NAMES)
for t in NAMES:
    s = px_all[t]["Close"].dropna()
    print(f"  {t}: {s.index[0].date()} .. {s.index[-1].date()}  n={len(s)}")

oi = px_all["OIH"]["Close"].dropna()
xo = px_all["XOP"]["Close"].dropna()
CAL = oi.index.intersection(xo.index)
CAL = CAL[CAL <= BAR]
px = pd.DataFrame({t: px_all[t]["Close"] for t in NAMES}).reindex(CAL)
print(f"\ncommon OIH/XOP calendar: {CAL[0].date()} .. {CAL[-1].date()}  n={len(CAL)}")

# ------------------------------------------------------------------ 0. premise
print("\n" + "=" * 100)
print("0. PREMISE re-derivation (print the thing it is NAMED after)")
print("=" * 100)

r63_oih = px["OIH"].pct_change(63)
r63_xop = px["XOP"].pct_change(63)
spread = r63_oih - r63_xop


def pit_excl(s, look=252):
    """% of the PRIOR look-1 values <= current. The 00_recon.py definition."""
    return s.rolling(look).apply(lambda w: 100.0 * (w[:-1] <= w[-1]).mean(), raw=True)


def pit_rank(s, look=252):
    """rolling rank(pct) incl. self. The 2026-08-25 script's definition."""
    return s.rolling(look).rank(pct=True) * 100.0


pe = pit_excl(spread)
pr = pit_rank(spread)
print(f"  OIH r63 {100*r63_oih.iloc[-1]:+.2f}%   XOP r63 {100*r63_xop.iloc[-1]:+.2f}%"
      f"   spread {100*spread.iloc[-1]:+.2f}pp")
print(f"  PIT pctile (excl-self, recon defn) = {pe.iloc[-1]:.2f}")
print(f"  PIT pctile (rank incl-self, 08-25 defn) = {pr.iloc[-1]:.2f}")
full = 100.0 * (spread.dropna() <= spread.iloc[-1]).mean()
print(f"  FULL-SAMPLE pctile of today's spread = {full:.2f}  "
      f"(PIT is a trailing-252 statement, not a history statement)")

print("\n  last 20 sessions of the trigger state:")
tail = pd.DataFrame({"spread_pp": 100 * spread, "pit_excl": pe, "pit_rank": pr}).tail(20)
print(tail.round(2).to_string())

# ------------------------------------------------------- masks / helper machinery
fwd = fwd_lag(px["OIH"], H, LAG)
valid = fwd.notna()
oih_ret_all = fwd[valid]
UP_RATE = float((oih_ret_all > 0).mean())
print(f"\n  OIH unconditional h={H} lag={LAG} up-rate = {100*UP_RATE:.2f}%  "
      f"mean {100*oih_ret_all.mean():+.3f}%  n={len(oih_ret_all)}")

PIT = pr  # parked definition is the primary; excl-self reported alongside


def mask_at(thr, pit=PIT):
    return (pit <= thr) & pit.notna()


def cell(thr, h=H, pit=PIT, min_gap=None, label=None):
    m = mask_at(thr, pit)
    f = fwd_lag(px["OIH"], h, LAG)
    v = f.notna()
    days = px.index[m.reindex(px.index, fill_value=False).values & v.values]
    if len(days) == 0:
        return None
    epi = declusters(days, min_gap or h, px.index)
    vals = f.loc[epi].values
    base = f[v]
    w = int((vals > 0).sum())
    r = summarize(vals, label or f"pit<={thr}")
    r["n_days"] = len(days)
    r["edge_pp"] = r["mean_pct"] - 100 * base.mean()
    r["rec"] = f"{w}-{len(vals)-w}"
    r["signp_vs_uprate"] = sign_test(w, len(vals), float((base > 0).mean()))
    r["signp_vs_coin"] = sign_test(w, len(vals))
    r["x_cost"] = (100 * r["mean_pct"]) / COST_BPS
    return r, epi, vals, days


# ------------------------------------------------ 1. reproduce the parked cell
print("\n" + "=" * 100)
print("1. REPRODUCE the parked 2.5-pctile cell (claim: +0.934%, edge +0.706pp, 28-23)")
print("=" * 100)
res25 = cell(2.5)
show([res25[0]], "parked rung, episode level (min_gap = h = 10)")
print(f"  episode dates ({len(res25[1])}): "
      + ", ".join(str(d.date()) for d in res25[1]))
print(f"  concentration: {cluster_note(res25[1], res25[2])}")
v25 = res25[2]
order = np.argsort(-v25)
print(f"  drop-best   mean = {100*np.delete(v25, order[0]).mean():+.3f}%")
print(f"  drop-best-2 mean = {100*np.delete(v25, order[:2]).mean():+.3f}%")
print(f"  drop-best-3 mean = {100*np.delete(v25, order[:3]).mean():+.3f}%")

# also under the recon (excl-self) definition, for definitional robustness
res25e = cell(2.5, pit=pe)
show([res25e[0]], "same rung under the excl-self PIT definition")

# ---------------------------------------------------------- 2. threshold ladder
print("\n" + "=" * 100)
print("2. THRESHOLD LADDER  (*** = today's reading, PIT rank {:.2f})".format(pr.iloc[-1]))
print("=" * 100)
rows = []
for thr in [0.5, 1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 100.0]:
    c = cell(thr)
    if c is None:
        continue
    r = c[0]
    r["label"] = ("*** " if abs(thr - 4.0) < 1e-9 else "    ") + f"pit<={thr}"
    rows.append(r)
show(rows, "episode-level ladder, h=10, long OIH lag=1")

print("\n  same ladder at h=5 (the horizon the pair was killed at):")
rows5 = []
for thr in [1.0, 2.5, 4.0, 5.0, 10.0, 20.0, 100.0]:
    c = cell(thr, h=5)
    if c:
        rows5.append(c[0])
show(rows5, "h=5")

print("\n  HORIZON SCAN at the parked 2.5 rung and at the loosened 4.0 rung:")
for thr in (2.5, 4.0):
    rr = []
    for h in (1, 2, 3, 5, 10, 15, 21):
        c = cell(thr, h=h)
        if c:
            r = c[0]
            r["label"] = f"pit<={thr} h={h}"
            rr.append(r)
    show(rr, f"horizon scan, pit<={thr}")

# ---------------------------------------------------------------- 3. the record
print("\n" + "=" * 100)
print("3. THE RECORD - has it moved?")
print("=" * 100)
for thr in (2.5, 4.0):
    c = cell(thr)
    epi, vals = c[1], c[2]
    w = int((vals > 0).sum())
    n = len(vals)
    print(f"\n  pit<={thr}: record {w}-{n-w} over {n} episodes, "
          f"sign p (coin) {sign_test(w, n):.4f}, "
          f"sign p (vs OIH up-rate {100*UP_RATE:.1f}%) "
          f"{sign_test(w, n, UP_RATE):.4f}")
    need = None
    for extra in range(0, 15):
        if sign_test(w + extra, n + extra) <= 0.05:
            need = extra
            break
    print(f"    -> arms (sign p <= 0.05 vs a coin) after {need} more consecutive WINS")
    print(f"    last 6 scored episodes: "
          + ", ".join(f"{d.date()} {100*v:+.2f}%" for d, v in
                      zip(epi[-6:], vals[-6:])))
    # unscored triggers still inside the hold window
    m = mask_at(thr)
    trig_all = px.index[m.reindex(px.index, fill_value=False).values]
    unscored = trig_all[trig_all > (epi[-1] if len(epi) else px.index[0])]
    print(f"    trigger days AFTER the last scored episode ({len(unscored)}): "
          + ", ".join(str(d.date()) for d in unscored[-12:]))

# ------------------------------------------- 3b. fresh vs continuation entries
print("\n" + "=" * 100)
print("3b. FRESH vs CONTINUATION trigger days")
print("    (the parked record is an EPISODE record; today is not day 1)")
print("=" * 100)
for thr in (2.5, 4.0):
    m = mask_at(thr).reindex(px.index, fill_value=False)
    prev = m.shift(1, fill_value=False)
    fresh_mask = m & (~prev)
    cont_mask = m & prev
    fd = px.index[fresh_mask.values & valid.values]
    cd = px.index[cont_mask.values & valid.values]
    rows = [summarize(fwd.loc[fd].values, f"pit<={thr} FRESH day-1 (N={len(fd)})"),
            summarize(fwd.loc[cd].values, f"pit<={thr} CONTINUATION (N={len(cd)})"),
            summarize(oih_ret_all.values, "all days")]
    show(rows, f"day-level fresh vs continuation, pit<={thr}, h={H}")
    # how deep into the current run is today?
    run = 0
    for v in m.values[::-1]:
        if v:
            run += 1
        else:
            break
    print(f"  current consecutive run of pit<={thr} through {px.index[-1].date()}: "
          f"{run} sessions"
          + (f" (started {px.index[-run].date()})" if run else ""))
    # distribution of run length at trigger
    runs, cur = [], 0
    for v in m.values:
        if v:
            cur += 1
        else:
            if cur:
                runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    print(f"  historical run lengths: n_runs={len(runs)} median={np.median(runs):.0f} "
          f"max={max(runs) if runs else 0}")

# -------------------------------------------------------- 4. gate attribution
print("\n" + "=" * 100)
print("4. GATE ATTRIBUTION + battery")
print("=" * 100)
variants = {f"pit<={t}": mask_at(t) for t in (1.0, 2.5, 4.0, 5.0, 10.0, 20.0)}
battery(px, mask_at(2.5), [("OIH", 1.0)], H,
        "C6 parked rung: LONG OIH, PIT<=2.5", COST_BPS,
        variants=variants, lag=LAG, min_gap=H)
battery(px, mask_at(4.0), [("OIH", 1.0)], H,
        "C6 LOOSENED rung: LONG OIH, PIT<=4.0 (today's reading)", COST_BPS,
        variants=variants, lag=LAG, min_gap=H)

print("\n  decluster ladder (episode mean by min_gap), both rungs:")
rows = []
for thr in (2.5, 4.0):
    for g in (1, 5, 10, 21, 42, 63):
        c = cell(thr, min_gap=g)
        if c:
            r = c[0]
            r["label"] = f"pit<={thr} gap={g}"
            rows.append(r)
show(rows, "decluster sensitivity")

print("\n  YEAR histogram of episode returns, both rungs:")
for thr in (2.5, 4.0):
    c = cell(thr)
    epi, vals = c[1], c[2]
    by = pd.Series(100 * vals, index=pd.DatetimeIndex(epi).year)
    agg = by.groupby(level=0).agg(["count", "sum", "mean"])
    print(f"\n  pit<={thr}:")
    print(agg.round(2).to_string())

print("\n  ERA + MIDTERM split (episodes):")
for thr in (2.5, 4.0):
    c = cell(thr)
    epi, vals = c[1], c[2]
    yr = pd.DatetimeIndex(epi).year
    mid = (yr % 4 == 2)
    base = fwd[valid]
    b_yr = pd.DatetimeIndex(base.index).year
    b_mid = (b_yr % 4 == 2)
    rows = era_split(epi, vals)
    rows += [summarize(vals[mid], f"MIDTERM episodes (N={int(mid.sum())})"),
             summarize(vals[~mid], f"non-midterm episodes (N={int((~mid).sum())})"),
             summarize(base.values[b_mid], "CTRL all days, midterm yrs"),
             summarize(base.values[~b_mid], "CTRL all days, non-midterm")]
    show(rows, f"pit<={thr}")
    w = int((vals[mid] > 0).sum())
    print(f"   midterm record {w}-{int(mid.sum())-w}, "
          f"sign p vs midterm all-days up-rate "
          f"{sign_test(w, int(mid.sum()), float((base.values[b_mid] > 0).mean())):.4f}")

# ------------------------------------------ 6. beta residual + regime selection
print("\n" + "=" * 100)
print("6. BETA RESIDUAL (PIT trailing-252 beta) + regime over-selection")
print("=" * 100)
d_oih = px["OIH"].pct_change()
for bench in ("SPY", "XLE"):
    d_b = px[bench].pct_change()
    cov = d_oih.rolling(252).cov(d_b)
    var = d_b.rolling(252).var()
    beta = (cov / var).shift(1)  # PIT: known at the signal close
    f_b = fwd_lag(px[bench], H, LAG)
    resid = fwd - beta * f_b
    rv = resid.notna()
    rows = []
    for thr in (2.5, 4.0):
        c = cell(thr)
        epi = c[1]
        e = pd.DatetimeIndex(epi).intersection(resid.dropna().index)
        rows.append(summarize(resid.loc[e].values,
                              f"pit<={thr} residual vs {bench} (N={len(e)})"))
    rows.append(summarize(resid[rv].values, f"CTRL all days residual vs {bench}"))
    rows.append(summarize(f_b[valid].values, f"{bench} own h={H} all days"))
    for thr in (2.5, 4.0):
        c = cell(thr)
        rows.append(summarize(f_b.loc[c[1]].values,
                              f"{bench} h={H} ON pit<={thr} episodes"))
    show(rows, f"beta-neutral residual vs {bench}  (mean beta "
               f"{beta.dropna().mean():.2f}, today {beta.iloc[-1]:.2f})")

print("\n  does the trigger select falling energy?")
xle63 = px["XLE"].pct_change(63)
sma200 = px["SPY"].rolling(200).mean()
below = px["SPY"] < sma200
for thr in (2.5, 4.0):
    c = cell(thr)
    epi = c[1]
    print(f"   pit<={thr}: XLE trailing-63d at trigger mean "
          f"{100*xle63.loc[epi].mean():+.2f}%  vs all-days "
          f"{100*xle63.dropna().mean():+.2f}%   |  OIH trailing-63d "
          f"{100*r63_oih.loc[epi].mean():+.2f}% vs {100*r63_oih.dropna().mean():+.2f}%")
    print(f"      SPY<200d at trigger {100*below.loc[epi].mean():.1f}% vs base "
          f"{100*below.dropna().mean():.1f}%")
    sub = below.loc[epi].values.astype(bool)
    show([summarize(c[2][sub], f"pit<={thr} & SPY<200d (N={int(sub.sum())})"),
          summarize(c[2][~sub], f"pit<={thr} & SPY>=200d (N={int((~sub).sum())})")],
         f"regime split, pit<={thr}")

print("\n  TODAY: XLE r63 {:+.2f}%, SPY {} 200d, OIH r63 {:+.2f}%".format(
    100 * xle63.iloc[-1], "below" if bool(below.iloc[-1]) else "above",
    100 * r63_oih.iloc[-1]))
