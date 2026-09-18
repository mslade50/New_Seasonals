"""C3 round 2 -- tear down the ONE number in C3 that looked alive.

C3 round 1 found: SVXY entered at first-print-2 and held across BOTH prints
pays +3.194% at h=5 on 67 anchors, 77.6% hit, t 4.013, against an all-days
drift of +0.635%.  Everything else in C3 said no (set identity with the dead
runway-1 bucket, placebo rank 6 of 12, short ^VIX negative through h=4, the
runway-after conditioner discarding 0 of 67).  This script decides whether
that h=5 number is a finding or an artefact.

Round-1's beta check was DEGENERATE and is redone here: an OLS residual with
an intercept fitted ON THE TRIGGER SET has mean zero by construction, which
is why it printed -0.000%.  The honest version estimates beta on ALL days and
then asks what the conditional residual mean is.

Probes:
  a. beta estimated OUT of the trigger set -> conditional alpha
  b. gate attribution: pair anchors vs ALL PPI anchors vs ALL prints, same
     k=-3 / h=5 geometry.  If the pair does nothing, the cell is "hold SVXY
     for a week starting mid-month".
  c. month x trading-day-of-month control
  d. local +/-126td control (full battery)
  e. concentration + per-year decomposition
  f. the SEPTEMBER sub-cell (N=8, 100% hit) priced for the 12-month search
  g. the live compression band: today's VIX 21d rel-range pctile is 4.37,
     which is watchlist 33's DEAD half.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 240)
rng = np.random.default_rng(42)

px = close_panel(["SVXY", "^VIX", "SPY"])
cal = px["SPY"].dropna().index
pos = pd.Series(range(len(cal)), index=cal)

KINDS = ("nfp", "cpi", "ppi", "fomc_decision")
EV = {k: load_events([k])["date"] for k in KINDS}
ALL_PRINTS = pd.DatetimeIndex(sorted(pd.concat(list(EV.values())).unique()))
ppi = load_events(["ppi"])["date"]

p_all, kept = anchor_positions(cal, ppi, 0)
nxt_print = set(int(pos.get(d, -1)) for d in ALL_PRINTS)
pair_pos = [p for p in p_all if (p + 1) in nxt_print]
iso_pos = [p for p in p_all if (p + 1) not in nxt_print]
allpr_pos, _ = anchor_positions(cal, ALL_PRINTS, 0)

A_PAIR = cal[[p - 3 for p in pair_pos if p - 3 >= 0]]
A_ISO = cal[[p - 3 for p in iso_pos if p - 3 >= 0]]
A_ALLPPI = cal[[p - 3 for p in p_all if p - 3 >= 0]]
A_ALLPR = cal[sorted({p - 3 for p in allpr_pos if p - 3 >= 0})]

H = 5
sv = px["SVXY"].dropna()
spy = px["SPY"].dropna()
rs = fwd_lag(sv, H, lag=1)
rp = fwd_lag(spy, H, lag=1)

print("=" * 100)
print("a. BETA ESTIMATED OUT OF THE TRIGGER SET -> conditional alpha")
print("=" * 100)
both = pd.concat([rs, rp], axis=1).dropna()
both.columns = ["svxy", "spy"]
trig = both.index.intersection(A_PAIR)
off = both.index.difference(A_PAIR)
b, a0 = np.polyfit(both.loc[off, "spy"], both.loc[off, "svxy"], 1)
print(f"  beta on NON-trigger days: {b:.3f}  intercept {100*a0:+.3f}%  "
      f"(n_off={len(off)})")
resid = both["svxy"] - (b * both["spy"] + a0)
rt, ro = resid.loc[trig].values, resid.loc[off].values
print(f"  conditional residual on pair anchors: {100*rt.mean():+.3f}% "
      f"(n={len(rt)}, hit {100*(rt>0).mean():.1f}%, "
      f"t {rt.mean()/(rt.std(ddof=1)/np.sqrt(len(rt))):+.2f})")
print(f"  off-trigger residual: {100*ro.mean():+.4f}%")
print(f"  raw conditional {100*both.loc[trig,'svxy'].mean():+.3f}% of which "
      f"beta x SPY = {100*b*both.loc[trig,'spy'].mean():+.3f}% "
      f"({100*b*both.loc[trig,'spy'].mean()/both.loc[trig,'svxy'].mean():.0f}% of it)")
print(f"  SPY itself on the pair anchors at h={H}: "
      f"{100*both.loc[trig,'spy'].mean():+.3f}% against an all-days SPY drift "
      f"of {100*rp.dropna().mean():+.3f}%")

print("\n" + "=" * 100)
print(f"b. GATE ATTRIBUTION, k=-3 anchor, h={H}. Does 'pair' do any work?")
print("=" * 100)
rows = []
for lbl, dts in (("PAIR (PPI, print next session)", A_PAIR),
                 ("PPI, NO print next session", A_ISO),
                 ("ALL PPI anchors", A_ALLPPI),
                 ("ALL prints (nfp/cpi/ppi/fomc)", A_ALLPR)):
    v = rs.reindex(pd.DatetimeIndex(dts)).dropna().values
    if not len(v):
        continue
    r = summarize(v, lbl)
    r["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
    rows.append(r)
rows.append(summarize(rs.dropna().values, "ALL DAYS (drift)"))
# and: all days in the same YEARS as the pair anchors (SVXY-era only)
lo, hi = A_PAIR.min(), A_PAIR.max()
rows.append(summarize(rs.dropna().loc[lo:hi].values, "ALL DAYS, pair span only"))
show(rows, f"SVXY h={H}")

print("\n" + "=" * 100)
print("c. MONTH x TRADING-DAY-OF-MONTH control (mid-month is not an event)")
print("=" * 100)
idx = sv.index
tdom = pd.Series(pd.Series(idx, index=idx).groupby([idx.year, idx.month]).cumcount().values + 1,
                 index=idx)
cells = pd.DataFrame({"m": idx.month, "d": tdom.values, "r": rs.reindex(idx).values},
                     index=idx).dropna()
for keys, lbl in ((["d"], "tdom only"), (["m", "d"], "month x tdom")):
    cm = cells.groupby(keys)["r"].mean()
    sig = cells.reindex(pd.DatetimeIndex(A_PAIR).intersection(cells.index))
    exc = sig["r"].values - np.asarray(sig.set_index(keys).index.map(cm), float)
    exc = exc[~np.isnan(exc)]
    print(f"  {lbl:14s}: excess {100*exc.mean():+.3f}pp  n={len(exc)}  "
          f"hit {100*(exc>0).mean():.1f}%  "
          f"t {exc.mean()/(exc.std(ddof=1)/np.sqrt(len(exc))):+.2f}")
print("  tdom histogram of pair anchors:",
      dict(tdom.reindex(A_PAIR).dropna().astype(int).value_counts().sort_index()))

print("\n" + "=" * 100)
print("d. FULL BATTERY (adds the local +/-126td control the round 1 skipped)")
print("=" * 100)
pxs = pd.DataFrame({"SVXY": sv}).dropna()
mask = pd.Series(False, index=pxs.index)
mask.loc[pxs.index.intersection(A_PAIR)] = True
battery(pxs, mask, [("SVXY", 1.0)], h=H,
        title=f"C3b SVXY: entry first-print-2, h={H}", cost_bps=8.0,
        min_gap=5, event_kinds=("fomc_decision",))

print("\n" + "=" * 100)
print("e. CONCENTRATION + per-year")
print("=" * 100)
d = pd.DatetimeIndex(A_PAIR).intersection(rs.dropna().index)
v = rs.reindex(d).values
print("  " + cluster_note(d, v, k=3))
byy = pd.Series(v, index=d.year).groupby(level=0).agg(["count", "mean"])
byy["mean_pct"] = (100 * byy["mean"]).round(2)
print(byy[["count", "mean_pct"]].to_string())
srt = np.sort(v)
print(f"  drop the best 3 episodes: {100*srt[:-3].mean():+.3f}% "
      f"(from {100*v.mean():+.3f}%)")

print("\n" + "=" * 100)
print("f. THE SEPTEMBER SUB-CELL, priced for the 12-month search")
print("=" * 100)
mo = d.month
obs = {}
for m in range(1, 13):
    sub = v[mo == m]
    if len(sub) >= 3:
        obs[m] = sub.mean()
print("  per-month means (%):", {k: round(100 * x, 2) for k, x in obs.items()})
sep = v[mo == 9]
print(f"  SEPTEMBER n={len(sep)} mean {100*sep.mean():+.3f}% "
      f"hit {100*(sep>0).mean():.1f}% sign p {sign_test(int((sep>0).sum()), len(sep)):.4f}")
best = max(obs.values())
cnt = 0
NP = 5000
for _ in range(NP):
    perm = rng.permutation(v)
    mx = -9
    for m in obs:
        s = perm[mo == m]
        if len(s):
            mx = max(mx, s.mean())
    cnt += (mx >= best)
print(f"  permutation P(SOME month with n>=3 looks this good) = {cnt/NP:.4f}")
# and the SPY leg of the same September cell
spy_sep = rp.reindex(d).values[mo == 9]
print(f"  SPY on the SAME September anchors, h={H}: {100*np.nanmean(spy_sep):+.3f}% "
      f"-> the September cell is {100*b*np.nanmean(spy_sep)/sep.mean():.0f}% beta")

print("\n" + "=" * 100)
print("g. LIVE BAND -- where does today sit?")
print("=" * 100)
vix = px["^VIX"].dropna()
vix = vix[vix.index <= pd.Timestamp("2026-09-04")]
rng21 = (rolling_on_valid(vix, lambda x: x.rolling(21).max())
         - rolling_on_valid(vix, lambda x: x.rolling(21).min()))
REL = rolling_on_valid(rng21 / rolling_on_valid(vix, lambda x: x.rolling(21).mean()),
                       lambda x: x.rolling(252).rank(pct=True) * 100)
print(f"  today's VIX 21d rel-range pctile = {REL.iloc[-1]:.2f} "
      f"(watchlist 33: (0,5] is the DEAD half, (5,15] the live band)")
rel_at = REL.reindex(d)
rows = []
for lbl, m in (("rel <= 5 (TODAY)", rel_at <= 5), ("rel (5,15]", (rel_at > 5) & (rel_at <= 15)),
               ("rel > 15", rel_at > 15)):
    sub = v[m.fillna(False).values]
    if not len(sub):
        continue
    r = summarize(sub, lbl)
    r["sign_p"] = round(sign_test(int((sub > 0).sum()), len(sub)), 4)
    rows.append(r)
show(rows, f"SVXY h={H} pair anchors, by VIX 21d rel-range band")
