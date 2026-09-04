"""C10 - the post-Labor-Day driving-season boundary on crude and refiners.

Anchor derivation (no holiday list in the repo): the Labor Day boundary is the
first September session preceded by a calendar gap of >= 4 days (Fri -> Tue).
The PITCH anchor is the session immediately BEFORE that gap, i.e. the
pre-Labor-Day Friday, because today (2026-09-04) is exactly that session.
Entry lag=1 = the first post-Labor-Day session. Direction is decided by the
measurement, so every vehicle is measured LONG and the sign is read off.

Blockers: placebo anchor ladder k=-8..+8, reference class (energy complex plus
non-energy commodities) with Cochran Q / I^2 / rank, gate attribution against
the momentum state the tape is in today, midterm cross, cost, concentration.
"""
import sys
from math import erf, sqrt
from pathlib import Path

ROOTP = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOTP))
import numpy as np
import pandas as pd
from pitch_lab import *  # noqa

ENERGY = ["USO", "XLE", "XOP", "VLO", "CVX", "XOM", "OIH"]
NONE = ["DBC", "GLD", "SLV", "UNG", "SPY"]
px = close_panel(ENERGY + NONE)
cal = px["SPY"].dropna().index
pos = pd.Series(range(len(cal)), index=cal)

# ------------------------------------------------- Labor Day boundary
gaps = (cal.to_series().diff().dt.days)
post = [d for d in cal if d.month == 9 and d.day <= 8 and gaps.get(d, 0) >= 4]
post = pd.DatetimeIndex(sorted(set(post)))
# keep the FIRST such session per year
first = {}
for d in post:
    first.setdefault(d.year, d)
post = pd.DatetimeIndex(sorted(first.values()))
pre = pd.DatetimeIndex([cal[int(pos[d]) - 1] for d in post])
print(f"derived Labor Day boundaries: N={len(post)}")
print("  post-holiday session (Tue):",
      ", ".join(str(d.date()) for d in post))
print("  PITCH ANCHOR pre-holiday Friday:",
      ", ".join(str(d.date()) for d in pre))
bad = [str(d.date()) for d in post if d.weekday() != 1]
print(f"  sanity: non-Tuesday post-holiday sessions {bad if bad else 'none'}; "
      f"all anchors Friday: {all(d.weekday() == 4 for d in pre)}")
print("  live analogue: 2026-09-04 is the Friday before Labor Day 2026-09-07")


def cell(tkr, dates, h, lag=1):
    ret = vehicle_ret(px, [(tkr, 1.0)], h, lag)
    d = pd.DatetimeIndex(dates).intersection(ret.dropna().index)
    return d, ret


COST = {"USO": 5.0, "XLE": 3.0, "XOP": 4.0, "VLO": 4.0, "CVX": 3.0,
        "XOM": 3.0, "OIH": 4.0, "DBC": 5.0, "GLD": 3.0, "SLV": 4.0,
        "UNG": 6.0, "SPY": 2.0}

print("\n=== ROUND 1: the pitched vehicles, LONG, at h = 5 / 10 / 21 ===")
for h in (5, 10, 21):
    rows = []
    for t in ["USO", "XLE", "XOP", "VLO"]:
        d, ret = cell(t, pre, h)
        base = ret.dropna()
        s = summarize(ret.loc[d].values, f"{t} h={h} (N={len(d)})")
        s["ctrl_all_pct"] = round(100 * base.mean(), 3)
        s["edge_pp"] = round(s["mean_pct"] - 100 * base.mean(), 3)
        loc = local_control(base.index, d)
        s["ctrl_local_pct"] = round(100 * ret.loc[loc].mean(), 3)
        w = int((ret.loc[d].values > 0).sum())
        s["sign_p_long"] = round(sign_test(w, len(d)), 4)
        s["sign_p_short"] = round(sign_test(len(d) - w, len(d)), 4)
        rows.append(s)
    show(rows, f"pre-Labor-Day anchor, entry lag=1, h={h}")

print("\n=== BLOCKER 7: concentration + year histogram, h=10 ===")
for t in ["USO", "XLE", "XOP", "VLO"]:
    d, ret = cell(t, pre, 10)
    v = ret.loc[d].values
    print(f"  {t}: {cluster_note(d, v)}")
    print(f"       per-year %: "
          f"{ {int(x.year): round(100*y, 2) for x, y in zip(d, v)} }")

print("\n=== BLOCKER 8: midterm cross, h=10 ===")
for t in ["USO", "XLE", "XOP", "VLO"]:
    d, ret = cell(t, pre, 10)
    mid = d[d.year % 4 == 2]
    non = d[d.year % 4 != 2]
    print(f"  {t}: midterm N={len(mid)} {100*ret.loc[mid].mean():+.3f}% | "
          f"non-midterm N={len(non)} {100*ret.loc[non].mean():+.3f}%")

print("\n=== BLOCKER 1: placebo anchor ladder k=-8..+8 (h=10) ===")
lad = []
for k in range(-8, 9):
    a = pd.DatetimeIndex([cal[int(pos[d]) + k] for d in pre
                          if 0 <= int(pos[d]) + k < len(cal)])
    row = {"k": k}
    for t in ["USO", "XLE", "XOP", "VLO"]:
        d, ret = cell(t, a, 10)
        row[t] = round(100 * ret.loc[d].mean(), 3) if len(d) else np.nan
    lad.append(row)
L = pd.DataFrame(lad)
print(L.to_string(index=False))
for t in ["USO", "XLE", "XOP", "VLO"]:
    s = L.dropna(subset=[t])
    true = s.loc[s["k"] == 0, t].iloc[0]
    rk_hi = int((s[t] > true).sum()) + 1
    rk_lo = int((s[t] < true).sum()) + 1
    print(f"  {t}: true k=0 = {true:+.3f}%  rank {rk_hi} of {len(s)} from the top "
          f"/ {rk_lo} from the bottom; ladder range "
          f"[{s[t].min():+.3f}, {s[t].max():+.3f}], median {s[t].median():+.3f}")

print("\n=== BLOCKER 2: reference class, identical rule (h=10, long, lag=1) ===")
rows, eff, var = [], [], []
for t in ENERGY + NONE:
    d, ret = cell(t, pre, 10)
    if len(d) < 3:
        rows.append({"tkr": t, "n": len(d)})
        continue
    s = summarize(ret.loc[d].values, t)
    se = s["sd_pct"] / np.sqrt(s["n"])
    base = ret.dropna()
    rows.append({"tkr": t, "grp": "energy" if t in ENERGY else "other",
                 "n": s["n"], "mean_pct": round(s["mean_pct"], 3),
                 "edge_pp": round(s["mean_pct"] - 100 * base.mean(), 3),
                 "hit": round(s["hit"], 1), "se": round(se, 3),
                 "t": round(s["t"], 2)})
    eff.append(s["mean_pct"])
    var.append(se ** 2)
R = pd.DataFrame(rows).sort_values("mean_pct", ascending=False)
print(R.to_string(index=False))
eff = np.array(eff)
var = np.array(var)
w = 1 / var
mu = (w * eff).sum() / w.sum()
Q = float((w * (eff - mu) ** 2).sum())
dfree = len(eff) - 1
I2 = max(0.0, (Q - dfree) / Q) * 100 if Q > 0 else 0.0
pQ = 1 - 0.5 * (1 + erf((Q - dfree) / sqrt(2 * 2 * dfree)))
print(f"  pooled {mu:.3f}%  Cochran Q={Q:.2f} on {dfree} df (normal-approx "
      f"p~{pQ:.3f})  I^2={I2:.1f}%")
for t in ["USO", "XLE", "XOP", "VLO"]:
    r = R[R["tkr"] == t]
    if len(r):
        print(f"    {t} ranks {int((R['mean_pct'] > r['mean_pct'].iloc[0]).sum())+1}"
              f" of {int(R['mean_pct'].notna().sum())}")

print("\n=== BLOCKER 3: gate attribution vs the momentum state the tape is in ===")
print("Today XLE r21 = 93.3, XOP 94.8, DBC at a 52w high. Gate: member r21 >= 80.")
for t in ["USO", "XLE", "XOP", "VLO"]:
    d, ret = cell(t, pre, 10)
    rk = pct_rank(px[t], 21).reindex(d)
    g = d[(rk >= 80).values]
    c = d[(rk < 80).values]
    um = 100 * ret.loc[d].mean()
    gm = 100 * ret.loc[g].mean() if len(g) else np.nan
    cm = 100 * ret.loc[c].mean() if len(c) else np.nan
    print(f"  {t}: ungated N={len(d)} {um:+.3f}% | gated r21>=80 N={len(g)} "
          f"{gm:+.3f}% | complement N={len(c)} {cm:+.3f}%  -> gate worth "
          f"{gm-um:+.3f}pp   gated years {sorted(g.year.tolist())}")

print("\n=== BLOCKER 6: cost, h=10 ===")
for t in ["USO", "XLE", "XOP", "VLO"]:
    d, ret = cell(t, pre, 10)
    e = 100 * ret.loc[d].mean() * 100
    print(f"  {t}: episode mean {e:+.1f} bps vs {COST[t]} bps round trip -> "
          f"{abs(e)/COST[t]:.1f}x (need >=5x); short side would be "
          f"{-e:+.1f} bps -> {abs(e)/COST[t]:.1f}x")
