"""b1b — C3 round 2: reference-class permutation, vehicle neighbours, regime.

Round 1 already showed the gate is an inverter (fresh-high alone +0.691%,
thrust alone -0.068%, both -1.424%). This round asks the three questions the
registry says to ask BEFORE a development pass:

  A. Reference class. Run the IDENTICAL rule across every liquid single name
     with enough history and read where FCX sits, plus the permutation MAX out
     of pure noise (the IHI construction, 2026-08-13).
  B. Vehicles. Does the sign survive a change of instrument (COPX / XME /
     SCCO / HG=F / XLB), and does the "copper complex" trigger — any complex
     member thrusting to a fresh high — behave differently from FCX's own?
  C. Regime + midterm split, decluster-gap sensitivity, and the ONE variant
     that was not negative in round 1 (r5 >= 10%) priced against its own
     local control and against cost.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, cluster_note, declusters, era_split, load_prices,
    local_control, rolling_on_valid, show, sign_test, summarize, vehicle_ret,
)
from pitch_lab import _valid_pct_change as vpc  # noqa: E402
from strategy_config import LIQUID_PLUS_COMMODITIES  # noqa: E402

pd.set_option("display.width", 230)
H = 5
THRUSTS = (0.15, 0.10)

# ---------------------------------------------------------------------------
# A. reference class
# ---------------------------------------------------------------------------
print("=== A. reference-class permutation: the same rule on every liquid name ===")
mp = pd.read_parquet("data/master_prices.parquet",
                     columns=["ticker", "date", "Close"])
mp["date"] = pd.to_datetime(mp["date"])
pool = sorted(set(LIQUID_PLUS_COMMODITIES))
mp = mp[mp["ticker"].isin(pool)]

rows = {th: [] for th in THRUSTS}
per_name_trig = {}
for t, g in mp.groupby("ticker"):
    s = g.sort_values("date").set_index("date")["Close"]
    s = s[~s.index.duplicated(keep="last")]
    if len(s) < 3000:                       # ~12y minimum
        continue
    fwd = s.shift(-(1 + H)) / s.shift(-1) - 1.0
    hi = s.rolling(252).max()
    at_hi = s >= hi * (1 - 1e-9)
    r5 = s.pct_change(5)
    base = float(fwd.dropna().mean())
    for th in THRUSTS:
        m = (r5 >= th) & at_hi & fwd.notna()
        d = s.index[m.fillna(False).values]
        if len(d) == 0:
            continue
        epi = declusters(d, H, s.index)
        v = fwd.loc[epi].values
        rows[th].append({"ticker": t, "n_epi": len(epi), "n_days": len(d),
                         "mean_pct": 100 * float(np.nanmean(v)),
                         "base_pct": 100 * base,
                         "excess_pp": 100 * (float(np.nanmean(v)) - base)})
        if th == 0.15:
            per_name_trig[t] = (s, fwd, epi, base)

for th in THRUSTS:
    df = pd.DataFrame(rows[th])
    df = df[df["n_epi"] >= 5].sort_values("excess_pp", ascending=False)
    print(f"\n  --- thrust >= {th:.0%} & fresh 52w high, {len(df)} names with >=5 episodes ---")
    if df.empty:
        continue
    print(f"  cross-name excess: mean {df['excess_pp'].mean():+.3f}pp  "
          f"sd {df['excess_pp'].std():.3f}pp  median {df['excess_pp'].median():+.3f}pp")
    print(f"  names with a POSITIVE excess: {int((df['excess_pp'] > 0).sum())} "
          f"of {len(df)} ({100*(df['excess_pp'] > 0).mean():.0f}%)")
    if "FCX" in set(df["ticker"]):
        f = df[df["ticker"] == "FCX"].iloc[0]
        rank = int((df["excess_pp"] >= f["excess_pp"]).sum())
        print(f"  FCX: excess {f['excess_pp']:+.3f}pp on {int(f['n_epi'])} episodes "
              f"-> rank {rank} of {len(df)};  P(a random member >= FCX) = "
              f"{rank/len(df):.3f}")
    print("  best 6:", df.head(6)[["ticker", "n_epi", "excess_pp"]]
          .to_string(index=False).replace("\n", " | "))
    print("  worst 4:", df.tail(4)[["ticker", "n_epi", "excess_pp"]]
          .to_string(index=False).replace("\n", " | "))

# permutation: same estimator, random dates, what MAX does noise produce?
print("\n  --- permutation MAX out of pure noise (IHI construction) ---")
rng = np.random.default_rng(42)
df15 = pd.DataFrame(rows[0.15])
df15 = df15[df15["n_epi"] >= 5]
names = [t for t in df15["ticker"] if t in per_name_trig]
maxes = []
for _ in range(400):
    best = -1e9
    for t in names:
        s, fwd, epi, base = per_name_trig[t]
        valid = fwd.dropna().index
        k = len(epi)
        if k < 1 or len(valid) <= k:
            continue
        pick = rng.choice(len(valid), size=k, replace=False)
        v = fwd.loc[valid[pick]].values
        best = max(best, 100 * (float(np.nanmean(v)) - base))
    maxes.append(best)
maxes = np.array(maxes)
fcx_obs = float(df15.loc[df15["ticker"] == "FCX", "excess_pp"].iloc[0]) \
    if "FCX" in set(df15["ticker"]) else np.nan
print(f"  {len(names)} names, 400 draws. permutation max excess: mean "
      f"{maxes.mean():+.2f}pp, p95 {np.percentile(maxes, 95):+.2f}pp, "
      f"max {maxes.max():+.2f}pp")
print(f"  FCX observed excess {fcx_obs:+.3f}pp -> P(noise max >= FCX) = "
      f"{float((maxes >= fcx_obs).mean()):.3f}")

# ---------------------------------------------------------------------------
# B. vehicles + the "copper complex" trigger
# ---------------------------------------------------------------------------
print("\n=== B. vehicles: does the sign survive a change of instrument? ===")
EQ = ["FCX", "COPX", "XME", "XLB", "SCCO", "TECK", "SPY"]
px = close_panel(EQ)
hgraw = load_prices(["HG=F"])["HG=F"]["Close"]
px2 = px.join(hgraw.rename("HG=F"))
r5f = vpc(px["FCX"], 5)
hif = rolling_on_valid(px["FCX"], lambda x: x.rolling(252).max())
trig_fcx = ((r5f >= 0.15) & (px["FCX"] >= hif * (1 - 1e-9))).fillna(False)
dtrig = px.index[trig_fcx.values]

veh = []
for t in ["FCX", "COPX", "XME", "XLB", "SCCO", "TECK", "HG=F", "SPY"]:
    ret = vehicle_ret(px2, [(t, 1.0)], H)
    d = px.index[trig_fcx.values & ret.notna().reindex(px.index, fill_value=False).values]
    if len(d) < 3:
        veh.append({"label": f"{t} on the FCX trigger", "n": len(d)})
        continue
    epi = declusters(d, H, px.index)
    r = summarize(ret.loc[epi].values, f"{t} on the FCX trigger")
    r["own_all_days_pct"] = round(100 * float(ret.dropna().mean()), 3)
    r["excess_pp"] = round(r["mean_pct"] - r["own_all_days_pct"], 3)
    veh.append(r)
show(veh, "vehicle swap, episodes")

print("\n  --- the COMPLEX trigger: ANY of FCX/SCCO/TECK/COPX/XME at r5>=15% "
      "and a fresh 52w high ---")
anym = None
for t in ["FCX", "COPX", "XME", "SCCO", "TECK"]:
    rr = vpc(px[t], 5)
    hh = rolling_on_valid(px[t], lambda x: x.rolling(252).max())
    m = ((rr >= 0.15) & (px[t] >= hh * (1 - 1e-9))).fillna(False)
    anym = m if anym is None else (anym | m)
retX = vehicle_ret(px, [("XME", 1.0)], H)
retF = vehicle_ret(px, [("FCX", 1.0)], H)
for lbl, ret in (("XME leg", retX), ("FCX leg", retF)):
    d = px.index[anym.values & ret.notna().values]
    epi = declusters(d, H, px.index)
    show([summarize(ret.loc[epi].values, f"complex trigger -> {lbl} (episodes)"),
          summarize(ret.dropna().values, f"{lbl} all days")], "")
print(f"  complex-trigger days: {int(anym.sum())};  fires today: "
      f"{bool(anym.iloc[-1])}  (FCX and SCCO both at a fresh high with r5>=15%)")

# ---------------------------------------------------------------------------
# C. the one non-negative variant, priced honestly
# ---------------------------------------------------------------------------
print("\n=== C. the r5>=10% variant, its local control, midterm, decluster gap ===")
ret = vehicle_ret(px, [("FCX", 1.0)], H)
valid = ret.notna()
m10 = ((r5f >= 0.10) & (px["FCX"] >= hif * (1 - 1e-9))).fillna(False)
d10 = px.index[m10.values & valid.values]
e10 = declusters(d10, H, px.index)
loc = local_control(px.index[valid.values], d10)
show([summarize(ret.loc[e10].values, "r5>=10% & fresh high (episodes)"),
      summarize(ret.loc[d10].values, "  day level"),
      summarize(ret[valid].values, "CTRL-b FCX all days"),
      summarize(ret.loc[loc].values, "CTRL-c local +/-126td ex-trigger")],
     "the only non-negative variant")
e = ret.loc[e10].values
print(f"  excess over all-days: {100*(e.mean() - ret[valid].mean()):+.3f}pp;  "
      f"over local control: {100*(e.mean() - ret.loc[loc].mean()):+.3f}pp")
print(f"  cost: 10 bps round trip, episode mean {100*e.mean():.1f} bps -> "
      f"{100*e.mean()/10:.1f}x (need >=5x)")
print(f"  concentration: {cluster_note(e10, e)}")
wins = int((e > 0).sum())
print(f"  record {wins}-{len(e)-wins}, sign p = {sign_test(wins, len(e)):.4f}")

mid = pd.DatetimeIndex(e10).year % 4 == 2
show([summarize(e[mid], f"MIDTERM years (today's regime, N={int(mid.sum())})"),
      summarize(e[~mid], "non-midterm")], "midterm split, r5>=10% variant")

spy200 = rolling_on_valid(px["SPY"], lambda x: x.rolling(200).mean())
bull = (px["SPY"] > spy200).reindex(e10).values
show([summarize(e[bull], "SPY above its 200d (today)"),
      summarize(e[~bull], "SPY below")], "regime split")

print("\n  decluster-gap sensitivity (r5>=10% variant):")
for gap in (1, 5, 10, 21, 63):
    ee = declusters(d10, gap, px.index)
    v = ret.loc[ee].values
    print(f"   gap={gap:3d} td: N={len(ee):3d}  mean {100*v.mean():+.3f}%  "
          f"hit {100*(v > 0).mean():.1f}%")
