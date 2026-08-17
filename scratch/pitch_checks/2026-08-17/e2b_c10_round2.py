"""C10 round 2 - teardown of the GDX rank-100 thrust.

Round 1 (e2_c10_gdx_thrust.py) left one number alive: GDX h=10 at +1.162%
over 18 episodes. Round 2 prices it.

Four things round 2 owes, plus the two the coordinator added:
  1. concentration / LOYO / drop-top-2
  2. definition neighbours (rank + magnitude), already partly in round 1
  3. era + REGIME split (gold bull vs bear), plus MIDTERM (2026 is midterm)
  4. gate attribution (done in round 1: the drawdown gate INVERTS)
  5. RANK TRAP (2026-08-14): a trailing-year rank on a series with secular
     drift. Quote +26.01% against the FULL-HISTORY distribution of 21d GDX
     moves, not just against the trailing year.
  6. Jackson Hole: zero in-sample episodes carry a JH inside the hold, and the
     registry's JH gold leg is midterm -1.213% at 1-4. Rebuild the independent
     late-August midterm control on GDX itself.
  7. cross-miner reference class (registry 2026-08-13): the identical rule on
     the peer group, because the cell names one instrument out of a natural
     peer group.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

PEERS = ["GDX", "GDXJ", "NEM", "AEM", "KGC", "GLD", "SLV", "SPY"]
px = close_panel(PEERS).dropna(subset=["GDX"])
idx = px.index

rk21 = pct_rank(px["GDX"], 21)
r21 = px["GDX"].pct_change(21)
dd = px["GDX"] / px["GDX"].rolling(252).max() - 1.0
trig = rk21 >= 100.0

H = 10
ret = fwd_lag(px["GDX"], H, 1)
epi = declusters(idx[trig.values & ret.notna().values], 21, idx)
ev = ret.loc[epi].values
print(f"parent cell: GDX rank100, h={H}, N={len(epi)} episodes, "
      f"mean {100*ev.mean():+.3f}%, record {(ev>0).sum()}-{(ev<=0).sum()}")

# ---------------------------------------------------------------------------
# 1. concentration / drop-top / LOYO
# ---------------------------------------------------------------------------
print("\n===== 1. concentration =====")
print("  ", cluster_note(epi, ev, k=2))
order = np.argsort(-ev)
for k in (1, 2, 3):
    keep = np.ones(len(ev), bool)
    keep[order[:k]] = False
    print(f"  drop top-{k}: {100*ev[keep].mean():+.3f}% on "
          f"{(ev[keep]>0).sum()}-{(ev[keep]<=0).sum()}")
yrs = pd.DatetimeIndex(epi).year
print("  LOYO (leave one year out):")
loyo = []
for y in sorted(set(yrs)):
    m = yrs != y
    loyo.append((y, 100 * ev[m].mean()))
print("   ", ", ".join(f"{y}:{v:+.2f}" for y, v in loyo))
print(f"    LOYO floor = {min(v for _, v in loyo):+.3f}%  "
      f"(dropping {min(loyo, key=lambda x: x[1])[0]})")
print("  per-episode:", ", ".join(f"{d.date()}:{100*v:+.1f}"
                                  for d, v in zip(epi, ev)))

# ---------------------------------------------------------------------------
# 5. RANK TRAP: today's +26.01% against FULL history, not a trailing year
# ---------------------------------------------------------------------------
print("\n===== 5. rank trap: the level behind the rank =====")
live = r21.iloc[-1]
full = r21.dropna()
mod = full[full.index >= "2018-01-01"]
print(f"  live 21d move {100*live:+.2f}%  |  trailing-year rank {rk21.iloc[-1]:.1f}")
print(f"  full-history pctile of that move: {100*(full < live).mean():.1f}  "
      f"(N={len(full)})")
print(f"  2018+ pctile:                     {100*(mod < live).mean():.1f}  "
      f"(N={len(mod)})")
print(f"  full-history 21d sd {100*full.std():.2f}%, "
      f"95th pctile {100*full.quantile(.95):+.2f}%, "
      f"99th {100*full.quantile(.99):+.2f}%")
print("  historical trigger days' 21d move: "
      f"min {100*r21.loc[epi].min():+.2f}%, median "
      f"{100*r21.loc[epi].median():+.2f}%, max {100*r21.loc[epi].max():+.2f}%")
# the magnitude cell WITHOUT the rank gate at all
print("\n  magnitude-only cell (no rank gate), h=10 episodes:")
rows = []
for thr in (0.15, 0.20, 0.26, 0.30):
    m = r21 >= thr
    e = declusters(idx[m.values & ret.notna().values], 21, idx)
    r = summarize(ret.loc[e].values, f"21d ret >= {100*thr:.0f}% (no rank gate)")
    r["n_days"] = int((m & ret.notna()).sum())
    rows.append(r)
rows.append(summarize(ret.dropna().values, "all days"))
show(rows, "magnitude gate alone")

# ---------------------------------------------------------------------------
# 3. era + regime + MIDTERM
# ---------------------------------------------------------------------------
print("\n===== 3. regime splits =====")
above200 = px["GDX"] > px["GDX"].rolling(200).mean()
gld_up = px["GLD"] > px["GLD"].rolling(200).mean()
mid = pd.Series(pd.DatetimeIndex(idx).year % 4 == 2, index=idx)
rows = []
for lbl, m in [("GDX above 200d (LIVE? see below)", above200),
               ("GDX below 200d", ~above200),
               ("GLD above 200d (gold bull)", gld_up),
               ("GLD below 200d (gold bear)", ~gld_up),
               ("MIDTERM year (2026)", mid),
               ("non-midterm", ~mid)]:
    sel = [i for i, d in enumerate(epi) if bool(m.loc[d])]
    rows.append(summarize(ev[sel], f"{lbl} (N={len(sel)})"))
show(rows, "GDX rank100 h=10 episodes by regime")
print(f"  LIVE: GDX above200d={bool(above200.iloc[-1])}, "
      f"GLD above200d={bool(gld_up.iloc[-1])}, midterm=True")

# ---------------------------------------------------------------------------
# 6. Jackson Hole / late-August midterm control on GDX itself
# ---------------------------------------------------------------------------
print("\n===== 6. the JH window the trade would actually sit in =====")
jh = load_events(["jackson_hole"])["date"]
pos = pd.Series(range(len(idx)), index=idx)
rows = []
for h in (5, 10):
    r = fwd_lag(px["GDX"], h, 1)
    anchors, mids = [], []
    for d in jh:
        # today sits 9 td before JH -> anchor 9 td before each JH
        p = idx.searchsorted(pd.Timestamp(d))
        q = p - 9
        if 0 <= q < len(idx):
            anchors.append(idx[q])
            mids.append(idx[q].year % 4 == 2)
    a = pd.DatetimeIndex(anchors)
    mids = np.array(mids)
    rows.append(summarize(r.loc[a].dropna().values, f"GDX h={h} JH-9td, all yrs"))
    rows.append(summarize(r.loc[a[mids]].dropna().values, f"GDX h={h} JH-9td, MIDTERM"))
    rl = fwd_lag(px["GLD"], h, 1)
    rows.append(summarize(rl.loc[a[mids]].dropna().values, f"GLD h={h} JH-9td, MIDTERM"))
show(rows, "the calendar slot the hold occupies (independent of the thrust)")

# ---------------------------------------------------------------------------
# 7. cross-miner reference class
# ---------------------------------------------------------------------------
print("\n===== 7. reference class: the identical rule on the peer group =====")
rows = []
for t in ["GDX", "GDXJ", "NEM", "AEM", "KGC", "SLV", "GLD"]:
    s = px[t].dropna()
    rk = pct_rank(s, 21)
    rr = fwd_lag(s, H, 1)
    m = (rk >= 100.0) & rr.notna()
    d = s.index[m.reindex(s.index, fill_value=False).values]
    e = declusters(d, 21, s.index)
    r = summarize(rr.loc[e].values, t)
    if r["n"]:
        base = 100 * rr.dropna().mean()
        r["ctl_pct"] = round(base, 3)
        r["excess_pp"] = round(r["mean_pct"] - base, 3)
    rows.append(r)
show(rows, f"21d rank==100, h={H}, episodes, per name")
ex = [r["excess_pp"] for r in rows if r.get("n")]
print(f"  cross-name excess: mean {np.mean(ex):+.3f}pp, sd {np.std(ex, ddof=1):.3f}pp, "
      f"GDX ranks {1 + sum(1 for v in ex if v > rows[0]['excess_pp'])} of {len(ex)}")
