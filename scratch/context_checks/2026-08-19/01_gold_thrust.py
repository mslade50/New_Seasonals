"""Gold closed +4.92% on 2026-08-19. Price that against its own magnitude tail
rather than the engine's generic 2-ATR threshold, and follow the path out."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, cluster_note, declusters, era_split, fwd_ret, local_control,
    sign_test, summarize,
)

TK = ["GC=F", "SI=F", "DX-Y.NYB", "SPY", "TLT"]
px = close_panel(TK).dropna(subset=["GC=F"])
g = px["GC=F"]
r1 = g.pct_change()
print("gold history", g.index[0].date(), "->", g.index[-1].date(), "n", len(g))
print("today", g.index[-1].date(), "close", round(float(g.iloc[-1]), 2),
      "ret", round(float(r1.iloc[-1]) * 100, 2), "%")

today = float(r1.iloc[-1])
hist = r1.iloc[:-1].dropna()
print("\n--- where does +4.92% sit in gold's own distribution ---")
print("sessions strictly larger, full history:", int((hist > today).sum()),
      "of", len(hist), "=", round(100 * (hist > today).mean(), 3), "%")
for thr in (0.03, 0.035, 0.04, 0.045):
    print(f"  >= {thr*100:.1f}% : n={int((hist >= thr).sum())}")

# The cell: gold up 4% or more in one session. Declustered at 5 td.
for thr in (0.03, 0.04):
    trig_all = r1.index[(r1 >= thr).fillna(False)]
    trig_all = trig_all[trig_all < g.index[-1]]
    trig = declusters(trig_all, 5, g.index)
    print(f"\n=== gold single session >= {thr*100:.0f}%  (raw {len(trig_all)}, "
          f"declustered {len(trig)}) ===")
    for h in (1, 5, 10, 21):
        f = fwd_ret(g, h).reindex(trig).dropna()
        s = summarize(f.values, f"h{h}")
        nup = int((f > 0).sum())
        print(f"  h{h:<3} n={s['n']:<4} mean={s['mean_pct']:+.3f}%  "
              f"med={s['median_pct']:+.3f}%  {nup}-{len(f)-nup} up  "
              f"t={s['t']:+.2f}  sign_p={sign_test(nup, len(f)):.4f}")
    # controls for h1 and h5
    ctrl = local_control(g.index, trig, 126)
    for h in (1, 5):
        f = fwd_ret(g, h).reindex(trig).dropna()
        allc = summarize(fwd_ret(g, h).dropna().values, "all")
        loc = summarize(fwd_ret(g, h).reindex(ctrl).dropna().values, "local")
        print(f"  control h{h}: cell {summarize(f.values)['mean_pct']:+.3f}%  "
              f"all-days {allc['mean_pct']:+.3f}%  local+-126td {loc['mean_pct']:+.3f}%")
    f1 = fwd_ret(g, 1).reindex(trig).dropna()
    print("  era h1:", [(e["label"], e["n"], round(e["mean_pct"], 3)) for e in
                        era_split(f1.index, f1.values)])
    print("  concentration h1:", cluster_note(f1.index, f1.values))
    f5 = fwd_ret(g, 5).reindex(trig).dropna()
    print("  era h5:", [(e["label"], e["n"], round(e["mean_pct"], 3)) for e in
                        era_split(f5.index, f5.values)])
    print("  concentration h5:", cluster_note(f5.index, f5.values))
    if thr == 0.04:
        print("  episodes:")
        for d in trig:
            fwd = fwd_ret(g, 5).get(d, np.nan)
            print(f"    {d.date()}  day {r1[d]*100:+.2f}%  next5 "
                  f"{fwd*100 if fwd == fwd else float('nan'):+.2f}%")

# Was today's move dollar-driven? Split the 3%+ cell by same-session DXY.
print("\n=== gold >= 3% split by the dollar's same-session move ===")
dx = px["DX-Y.NYB"].pct_change()
trig = declusters(r1.index[(r1 >= 0.03).fillna(False)][:-1], 5, g.index)
both = pd.DataFrame({"g5": fwd_ret(g, 5), "dx": dx}).reindex(trig).dropna()
for lab, m in (("DXY down that session", both["dx"] < 0),
               ("DXY flat or up", both["dx"] >= 0)):
    v = both.loc[m, "g5"].values
    if len(v) == 0:
        continue
    s = summarize(v, lab)
    nup = int((v > 0).sum())
    print(f"  {lab:<24} n={s['n']:<3} gold next 5d {s['mean_pct']:+.3f}%  "
          f"{nup}-{len(v)-nup} up  t={s['t']:+.2f}")
print("  today's DXY move:", round(float(dx.iloc[-1]) * 100, 2), "%")

# Silver companion: both metals up 4%+ same session.
print("\n=== gold AND silver both >= 4% on the same session ===")
s1 = px["SI=F"].pct_change()
joint_all = r1.index[((r1 >= 0.04) & (s1 >= 0.04)).fillna(False)]
joint_all = joint_all[joint_all < g.index[-1]]
joint = declusters(joint_all, 5, g.index)
print("  raw", len(joint_all), "declustered", len(joint))
for d in joint:
    print(f"    {d.date()}  gold {r1[d]*100:+.2f}%  silver {s1[d]*100:+.2f}%")
for h in (1, 5, 21):
    f = fwd_ret(g, h).reindex(joint).dropna()
    if len(f) == 0:
        continue
    s = summarize(f.values, f"h{h}")
    nup = int((f > 0).sum())
    print(f"  gold h{h:<3} n={s['n']:<3} mean={s['mean_pct']:+.3f}%  "
          f"{nup}-{len(f)-nup} up  sign_p={sign_test(nup, len(f)):.4f}")
