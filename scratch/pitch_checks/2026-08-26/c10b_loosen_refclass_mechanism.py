"""C10 round 1b: the literal XLRE rung is 2 episodes.  Loosen honestly, then
test the MECHANISM directly.

Three things settle this:
 (a) a loosening ladder that reaches a measurable N on XLRE,
 (b) the 10-sector reference class on the LOOSENED rung, with a permutation
     max-of-N null on the rates-gate value,
 (c) the mechanism itself: is XLRE actually the most duration-sensitive
     sector, and if so does the duration gate pay on the sector whose beta
     says it should?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change

SECT = ["XLRE", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
ALL = SECT + ["IYR", "VNQ", "TLT", "SPY"]
px = load_prices(ALL)
CAL = px["SPY"].index.intersection(px["TLT"].index)
P = pd.DataFrame({t: px[t]["Close"].reindex(CAL) for t in ALL})
tlt_r5 = pct_rank(P["TLT"], 5)


def stats(tkr, mask, h, gap=5):
    ret = fwd_lag(P[tkr], h)
    ok = mask.fillna(False).values & ret.notna().values & P[tkr].notna().values
    sig = P.index[ok]
    if len(sig) == 0:
        return np.array([]), pd.DatetimeIndex([])
    epi = declusters(sig, gap, P.index[P[tkr].notna()])
    return ret.loc[epi].values, epi


def rung(tkr, z_min, r21_max, tlt_min):
    z = zscore(P[tkr], 10)
    r21 = pct_rank(P[tkr], 21)
    bare = (z >= z_min) & (r21 <= r21_max)
    return bare, bare & (tlt_r5 >= tlt_min)


print("=" * 78)
print("1. LOOSENING LADDER on XLRE (and IYR, its 2000+ predecessor)")
print("=" * 78)
LAD = [(1.0, 30, 90), (1.0, 30, 80), (0.75, 40, 80), (0.5, 50, 75),
       (0.5, 50, 60), (0.0, 50, 75)]
for tkr in ("XLRE", "IYR", "VNQ"):
    print(f"\n--- {tkr} ---")
    for h in (3, 5):
        rows = []
        for z_, r_, t_ in LAD:
            bare, joint = rung(tkr, z_, r_, t_)
            j, ej = stats(tkr, joint, h)
            b, eb = stats(tkr, bare, h)
            base = fwd_lag(P[tkr], h).dropna()
            lbl = f"z>={z_}, r21<={r_}, TLTr5>={t_}"
            if len(j) < 3:
                rows.append({"label": lbl, "n": len(j)})
                continue
            r = summarize(j, lbl)
            r["gate_pp"] = round(100 * (j.mean() - b.mean()), 3)
            r["n_bare"] = len(eb)
            r["excess_pp"] = round(100 * (j.mean() - base.mean()), 3)
            r["x_cost"] = round(100 * (j.mean() - base.mean()) * 100 / 6, 1)
            rows.append(r)
        show(rows, f"{tkr} h={h}")

print("\n" + "=" * 78)
print("2. REFERENCE CLASS on a rung XLRE can actually reach (z>=0.5, r21<=50,")
print("   TLT r5 >= 75), 10 sectors, and a permutation null on the gate value")
print("=" * 78)
Z_, R_, T_ = 0.5, 50, 75
for h in (3, 5, 7):
    rows = []
    for t in SECT:
        bare, joint = rung(t, Z_, R_, T_)
        j, ej = stats(t, joint, h)
        b, eb = stats(t, bare, h)
        base = fwd_lag(P[t], h).dropna()
        if len(j) < 3:
            rows.append({"label": t, "n": len(j)})
            continue
        r = summarize(j, t)
        r["bare_pct"] = round(100 * b.mean(), 3)
        r["n_bare"] = len(eb)
        r["gate_pp"] = round(100 * (j.mean() - b.mean()), 3)
        r["excess_pp"] = round(100 * (j.mean() - base.mean()), 3)
        rows.append(r)
    df = pd.DataFrame(rows).sort_values("gate_pp", ascending=False)
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].round(3)
    print(f"\n--- h={h}, sorted by RATES-GATE value ---")
    print(df.to_string(index=False))
    gp = df["gate_pp"].dropna()
    if "XLRE" in set(df.loc[df["gate_pp"].notna(), "label"]):
        xg = float(df.loc[df["label"] == "XLRE", "gate_pp"].iloc[0])
        xe = float(df.loc[df["label"] == "XLRE", "excess_pp"].iloc[0])
        print(f"  XLRE gate {xg:+.3f}pp ranks {1+int((gp.values>xg).sum())} of "
              f"{len(gp)};  XLRE excess-over-drift {xe:+.3f}pp = "
              f"{xe*100/6:.1f}x a 6 bp round trip")
        print(f"  family mean gate {gp.mean():+.3f}pp, "
              f"{int((gp>0).sum())} of {len(gp)} positive, sd {gp.std():.3f}pp")

print("\n" + "=" * 78)
print("3. PERMUTATION: relocate the TLT gate in time.  Is the observed gate")
print("   value distinguishable from a randomly-placed conditioner?")
print("=" * 78)
rng = np.random.default_rng(11)
h = 3
bare_x, joint_x = rung("XLRE", Z_, R_, T_)
j, ej = stats("XLRE", joint_x, h)
b, eb = stats("XLRE", bare_x, h)
obs = 100 * (j.mean() - b.mean())
gate_raw = (tlt_r5 >= T_)
null = []
for _ in range(2000):
    sh = int(rng.integers(21, 500)) * (1 if rng.random() < 0.5 else -1)
    g = gate_raw.shift(sh)
    jj, _ = stats("XLRE", bare_x & g, h)
    if len(jj) >= 3:
        null.append(100 * (jj.mean() - b.mean()))
null = np.asarray(null)
print(f"  observed XLRE gate value at h=3 = {obs:+.3f}pp  (N_joint={len(j)})")
print(f"  shifted-gate null: mean {null.mean():+.3f}pp, sd {null.std():.3f}, "
      f"P(null >= observed) = {(null >= obs).mean():.3f}  ({len(null)} draws)")

print("\n" + "=" * 78)
print("4. MECHANISM: is XLRE actually the duration proxy the story needs?")
print("=" * 78)
dt = P["TLT"].pct_change()
rows = []
for t in SECT + ["IYR", "VNQ"]:
    ds = P[t].pct_change()
    ok = ds.notna() & dt.notna()
    beta = np.polyfit(dt[ok].values, ds[ok].values, 1)[0]
    corr = float(np.corrcoef(dt[ok].values, ds[ok].values)[0, 1])
    ok5 = ok & (P.index >= "2015-10-08")
    beta5 = np.polyfit(dt[ok5].values, ds[ok5].values, 1)[0]
    rows.append({"ticker": t, "beta_on_TLT": round(beta, 3),
                 "corr": round(corr, 3), "beta_2015+": round(beta5, 3),
                 "n": int(ok.sum())})
df = pd.DataFrame(rows).sort_values("beta_2015+", ascending=False)
print(df.to_string(index=False))
print("\n  If XLRE ranks first on duration beta but its duration GATE ranks")
print("  mid-pack or negative, the mechanism is falsified in its own window.")

print("\n" + "=" * 78)
print("5. TODAY'S READING under the two live z10 definitions")
print("=" * 78)
z_lab = zscore(P["XLRE"], 10).iloc[-1]
r10 = _valid_pct_change(P["XLRE"], 10)
vol21 = P["XLRE"].pct_change().rolling(21).std() * np.sqrt(10)
z_state = (r10 / vol21).iloc[-1]
print(f"  pitch_lab.zscore (10d ret / trailing-252 sd of 10d rets) = {z_lab:+.3f}")
print(f"  build_pitch_state _metrics_for style (10d ret / 21d vol x sqrt10) = "
      f"{z_state:+.3f}")
print(f"  the stated trigger is z10 >= 1.0.  Under the lab's own definition "
      f"today {'FIRES' if z_lab >= 1 else 'DOES NOT FIRE'}.")
