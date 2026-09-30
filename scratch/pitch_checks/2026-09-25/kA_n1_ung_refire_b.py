"""kA N1 round 2: re-fire concentration (drop-best, cluster_note, one episode
per prior-firing cluster), definition neighbours (re-fire window, 1d and volume
thresholds, volume definition, horizon), era split, gate attribution (does the
re-fire state or the thrust carry it? +5% no-volume and volume-only days inside
the 5 sessions after a firing), and the run-in state (did the first thrust
already pay?) that separates today from the one big re-fire winner."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

d = load_prices(["UNG"])["UNG"].astype(float)
c, v = d["Close"], d["Volume"]
px = pd.DataFrame({"UNG": c})
idx = px.index
pos = pd.Series(range(len(idx)), index=idx)
r1 = c.pct_change()
VR = {"mean_incl": v / v.rolling(63).mean(), "mean_ex": v / v.shift(1).rolling(63).mean(),
      "median": v / v.rolling(63).median()}
rets = {h: vehicle_ret(px, [("UNG", 1.0)], h) for h in (1, 2, 3, 4, 5)}


def refires(th=0.05, vm=3.0, vdef="mean_incl", lo=1, hi=5):
    trig = ((r1 >= th) & (VR[vdef] >= vm)).fillna(False)
    td = idx[trig.values]
    p = pos[td].values
    out = []
    for i in range(1, len(td)):
        g = p[i] - p[i - 1]
        if lo <= g <= hi:
            out.append(td[i])
    first = [td[0]] + [td[i] for i in range(1, len(td)) if p[i] - p[i - 1] >= 10]
    return pd.DatetimeIndex(out), pd.DatetimeIndex(first)


def rec(x, lab):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    r = summarize(x, lab)
    if r["n"]:
        w = int((x > 0).sum())
        r["rec"] = f"{w}-{len(x)-w}"
        r["sign_p"] = round(sign_test(w, len(x)), 4)
        s = np.sort(x)
        r["drop_best1"] = round(100 * s[:-1].mean(), 3) if len(x) > 1 else np.nan
        r["drop_best2"] = round(100 * s[:-2].mean(), 3) if len(x) > 2 else np.nan
    return r


# (a) concentration at the pitched rung
rf, fs = refires()
x2 = rets[2].reindex(rf).dropna()
print("(a) re-fire h=2:", cluster_note(x2.index, x2.values))
show([rec(x2.values, "re-fire 1..5 h=2 day-level")], "(a) drop-best")
# one obs per prior-firing cluster (a cluster = firings chained with gaps < 10)
trig = ((r1 >= 0.05) & (VR["mean_incl"] >= 3)).fillna(False)
td = idx[trig.values]
cid, k, lastp = {}, 0, None
for t in td:
    if lastp is not None and pos[t] - lastp >= 10:
        k += 1
    cid[t] = k
    lastp = pos[t]
cl = pd.Series({t: cid[t] for t in x2.index})
cm = x2.groupby(cl.values).mean()
show([rec(cm.values, f"re-fire h=2, one mean per cluster ({len(cm)} clusters)")], "(a) cluster-level")
yrs = pd.Series(x2.values, index=x2.index.year).groupby(level=0).agg(["sum", "count"])
print("   year sums pp:", {y: (round(100 * r["sum"], 2), int(r["count"])) for y, r in yrs.iterrows()})

# (b) definition neighbours
rows = []
for lo, hi in [(1, 3), (1, 5), (1, 7), (1, 9), (2, 2), (2, 5)]:
    r_, _ = refires(lo=lo, hi=hi)
    rows.append(rec(rets[2].reindex(r_).values, f"window {lo}..{hi} h=2"))
for th in (0.04, 0.05, 0.06):
    for vm in (2.5, 3.0, 3.5):
        r_, _ = refires(th=th, vm=vm)
        rows.append(rec(rets[2].reindex(r_).values, f"1d>={th:.0%} vol>={vm}x h=2"))
for vd in ("mean_ex", "median"):
    r_, _ = refires(vdef=vd)
    rows.append(rec(rets[2].reindex(r_).values, f"vol def {vd} h=2"))
for h in (1, 2, 3, 4, 5):
    rows.append(rec(rets[h].reindex(rf).values, f"horizon h={h}"))
show(rows, "(b) definition neighbours (re-fire, long UNG, lag 1)")

# (c) era split
e = x2.index
show([rec(x2.values[e < "2018-01-01"], "pre-2018"), rec(x2.values[e >= "2018-01-01"], "2018+"),
      rec(x2.values[e >= "2011-01-01"], "2011+ (post creation halt)")], "(c) era split h=2")

# (d) gate attribution: inside 1..5 sessions of a firing, which legs carry it?
near = pd.Series(False, index=idx)
for t in td:
    p = pos[t]
    near.iloc[p + 1: min(len(idx), p + 6)] = True
vr = VR["mean_incl"]
rows = []
for lab, m in [("near & +5% & vol>=3x (RE-FIRE)", near & (r1 >= 0.05) & (vr >= 3)),
               ("near & +5% & vol<3x", near & (r1 >= 0.05) & (vr < 3)),
               ("near & vol>=3x & 1d<5%", near & (r1 < 0.05) & (vr >= 3)),
               ("near & 1d>=+2%", near & (r1 >= 0.02)),
               ("near & any day", near),
               ("FIRST firing (gap>=10)", pd.Series(idx.isin(fs), index=idx))]:
    ds = idx[m.fillna(False).values]
    rows.append(rec(rets[2].reindex(ds).values, f"h=2 {lab}"))
show(rows, "(d) gate attribution inside the 5 sessions after a firing (day-level)")

# run-in state: first-firing entry -> re-fire close
run = pd.Series({t: c.iloc[pos[t]] / c.iloc[pos[t] - int(g)] - 1 for t, g in
                 zip(rf, [pos[t] - pos[td[td < t][-1]] for t in rf])})
# note: run-in here = prior SIGNAL close -> re-fire close (includes the prior thrust day's follow-through only)
run_entry = pd.Series({t: c.iloc[pos[t]] / c.iloc[pos[td[td < t][-1]] + 1] - 1 for t in rf})
live = run_entry.iloc[-1]
print(f"\nlive run-in (prior entry close -> re-fire close) {100*live:+.2f}%")
xx = rets[2].reindex(rf)
for lab, m in [("run-in >= +3% (first thrust already paid)", run_entry >= 0.03),
               ("run-in < +3%", run_entry < 0.03)]:
    show([rec(xx[m.values].values, f"h=2 {lab}"), rec(rets[3].reindex(rf)[m.values].values, f"h=3 {lab}")])
print("\nsearch charge: this round ran 6 window + 9 threshold + 2 vol-def + 5 horizon cells = 22 neighbours"
      " plus 1 run-in split (post-hoc).")
