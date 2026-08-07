"""C12 "Pre-expiry vol carry": long SVXY, entry MOC k td before the last close
prior to VIX expiry, exit that pre-expiry close.

The controls that matter:
 (1) SVXY's own unconditional k-day drift (it is a short-vol carry vehicle, so
     EVERY long window is positive - the cell must beat this, not zero),
 (2) an all-macro-event-day baseline at the same k,
 (3) the repo's live V4 trade (opex MOC -> +3 sessions, ex-September) measured
     on the same sample, plus a calendar-day overlap count.
Era split is load-bearing: SVXY was -1.0x before Feb 2018 and -0.5x after.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import *  # noqa

import numpy as np
import pandas as pd

px = load_prices(["SVXY", "^VIX", "SPY"])
sv = px["SVXY"]["Close"].dropna()
svl = px["SVXY"]["Low"].reindex(sv.index)
vix = px["^VIX"]["Close"].dropna()
cal = sv.index
pos = pd.Series(range(len(cal)), index=cal)
ev = load_events()
vxp = sorted(ev.loc[ev.event == "vix_expiry", "date"])
opx = sorted(ev.loc[ev.event == "opex", "date"])
cpi_d = set(ev.loc[ev.event == "cpi", "date"])
allev = [d for d in ev["date"] if d in pos.index]
vr5 = pct_rank(vix, 5).reindex(cal)

print(f"SVXY bars {cal[0].date()} .. {cal[-1].date()} n={len(cal)}")
r1 = sv.pct_change()
print(f"leverage-regime check: worst single day = {100*r1.min():.1f}% on "
      f"{r1.idxmin().date()}  (Feb-2018 XIV event; SVXY went -1.0x -> -0.5x)")
print(f"  daily sd pre-2018 = {100*r1[cal<'2018-03-01'].std():.2f}%  "
      f"2018-06+ = {100*r1[cal>'2018-06-01'].std():.2f}%")


def pre_windows(k):
    """(entry_date, exit_date) with exit = last close before each VIX expiry.

    Only expiries that have ALREADY happened in the data are usable; the events
    file runs to 2027 and forward-dated expiries would otherwise all collapse
    onto the final 8 bars and be counted repeatedly.
    """
    out = []
    for E in vxp:
        if E > cal[-1]:
            continue
        prior = cal[cal < E]
        if len(prior) == 0:
            continue
        xi = pos[prior[-1]]
        if xi - k < 0:
            continue
        out.append((cal[xi - k], cal[xi]))
    return out


def wret(w):
    return np.array([sv.loc[b] / sv.loc[a] - 1.0 for a, b in w])


# --- 1) does the cell exist? ----------------------------------------------
rows = []
for k in [5, 6, 7, 8, 9, 10]:
    w = pre_windows(k)
    v = wret(w)
    s = summarize(v, f"pre-expiry k={k}")
    s["ctl_svxy_alldays"] = 100 * fwd_ret(sv, k).mean()
    s["ctl_allevent"] = 100 * np.nanmean(fwd_ret(sv, k).reindex(allev).to_numpy())
    s["edge_vs_carry"] = s["mean_pct"] - s["ctl_svxy_alldays"]
    rows.append(s)
show(rows, "1) pre-expiry window vs SVXY's own carry (the real control)")

K = 8
W = pre_windows(K)
V = wret(W)
ent = pd.DatetimeIndex([a for a, _ in W])

# conditioned on VIX rank5 <= 25 at entry
m = vr5.reindex(ent).to_numpy() <= 25
crows = [summarize(V, "k=8 unconditional"),
         summarize(V[m], "k=8 | VIX rank5<=25"),
         summarize(V[~m], "k=8 | VIX rank5>25")]
for r in crows:
    r["ctl_svxy_alldays"] = 100 * fwd_ret(sv, K).mean()
show(crows, "1b) the today conditioner")

# THE DECISIVE CONTROL: is the pre-expiry framing doing anything, or is this
# just "long SVXY on any calm-vol day"? Same gate, ALL days, not just the
# pre-expiry run-in. Era-matched so the -1.0x era cannot flatter it.
print("\n=== 1c) DECISIVE: pre-expiry window vs ANY 8td window, same VIX gate ===")
f8 = fwd_ret(sv, K)
drows = []
for lo, hi, nm in [("2011-01-01", "2030-01-01", "full"),
                   ("2011-01-01", "2018-03-01", "pre-2018 (-1.0x)"),
                   ("2018-06-01", "2030-01-01", "2018-06+ (-0.5x)")]:
    em = (cal >= lo) & (cal < hi)
    gate = (vr5 <= 25).to_numpy() & em & f8.notna().to_numpy()
    wm = np.array([(a >= pd.Timestamp(lo)) and (a < pd.Timestamp(hi)) for a in ent])
    gm = wm & (vr5.reindex(ent).to_numpy() <= 25)
    drows.append({"era": nm,
                  "preexp_gated_n": int(gm.sum()),
                  "preexp_gated_pct": 100 * np.nanmean(V[gm]) if gm.sum() else np.nan,
                  "ANYday_gated_n": int(gate.sum()),
                  "ANYday_gated_pct": 100 * f8.to_numpy()[gate].mean(),
                  "ANYday_all_n": int((em & f8.notna().to_numpy()).sum()),
                  "ANYday_all_pct": 100 * f8[em].mean()})
show(drows, "pre-expiry vs any-window, both gated on VIX rank5<=25")

# --- 3) V4 overlap --------------------------------------------------------
print("\n=== V4 collision test (repo already trades long SVXY opex -> +3td) ===")
v4days, v4rets = set(), []
for O in opx:
    if O.month == 9 or O not in pos.index:
        continue
    p = pos[O]
    if p + 3 >= len(cal):
        continue
    v4days |= set(cal[p + 1: p + 4])
    v4rets.append(sv.iloc[p + 3] / sv.iloc[p] - 1.0)
predays = set()
for a, b in W:
    predays |= set(cal[pos[a] + 1: pos[b] + 1])
ovl = predays & v4days
print(f"  pre-expiry k=8 held days: {len(predays)}   V4 held days: {len(v4days)}"
      f"   OVERLAP: {len(ovl)} ({100*len(ovl)/max(len(predays),1):.1f}% of the "
      f"pre-expiry window)")
show([summarize(np.array(v4rets), "V4 opex->+3td (ex-Sep)"),
      summarize(V, "C12 pre-expiry k=8")], "same-vehicle cells side by side")
# median calendar gap between the pre-expiry exit and the following opex
gaps = [int(pos[min([o for o in opx if o > b and o in pos.index],
                    default=b)] - pos[b]) for a, b in W[:-1]]
print(f"  median td from pre-expiry exit to the next opex: {int(np.median(gaps))}")

# --- 2) era, worst window, drawdown ---------------------------------------
show(era_split(ent, V), "2) era split k=8 (pre-2018 SVXY is -1.0x, 2018+ -0.5x)")
srt = pd.Series(V, index=ent).sort_values()
print(f"  worst 3: {[(str(d.date()), round(100*x,2)) for d, x in srt.head(3).items()]}")
print(f"  best  3: {[(str(d.date()), round(100*x,2)) for d, x in srt.tail(3).items()]}")

print("\n=== 6b) worst SVXY 8td tail (the real risk) ===")
r8 = fwd_ret(sv, 8).dropna()
print(f"  worst any-8td close-to-close: {100*r8.min():.1f}% on {r8.idxmin().date()}")
print(f"  worst 8td in the 2018-06+ (-0.5x) era: "
      f"{100*r8[r8.index>'2018-06-01'].min():.1f}% on "
      f"{r8[r8.index>'2018-06-01'].idxmin().date()}")
print(f"  1st/5th pctile of all 8td: {100*np.percentile(r8,1):.1f}% / "
      f"{100*np.percentile(r8,5):.1f}%")
tro = [100 * (svl.iloc[pos[a] + 1: pos[b] + 1].min() / sv.loc[a] - 1) for a, b in W]
print(f"  worst intra-window LOW trough across the pre-expiry windows: "
      f"{min(tro):.1f}%   median trough: {np.median(tro):.1f}%")

# --- 3) decluster + bootstrap ---------------------------------------------
dc = declusters(ent, K, cal)
vd = wret([w for w in W if w[0] in set(dc)])
s = summarize(vd, "episodes k=8 (monthly, gap>=8td)")
s["p_le0_boot"] = bootstrap_p_le0(vd)
sm = summarize(V[m], "episodes k=8 | VIX r5<=25")
sm["p_le0_boot"] = bootstrap_p_le0(V[m])
show([s, sm], "3) episode level + bootstrap")

# --- 4) sensitivity on the VIX gate ---------------------------------------
sens = []
for thr in [15, 25, 35, 50]:
    mm = vr5.reindex(ent).to_numpy() <= thr
    sens.append(summarize(V[mm], f"k=8 | VIX rank5<={thr}"))
show(sens, "4) VIX-gate sensitivity")

# --- 6) CPI inside ---------------------------------------------------------
ins = np.array([any(x in cpi_d for x in cal[pos[a] + 1: pos[b] + 1]) for a, b in W])
show([summarize(V[ins], "k=8 CPI inside"), summarize(V[~ins], "k=8 no CPI inside")],
     "6) CPI-in-window split")

print("\n=== 5) cost: SVXY ~5-8bp/side => 10-16bp round trip = 0.100-0.160% ===")
print("     5x hurdle = 0.500-0.800%, before the -0.5x ETP's path decay.")
