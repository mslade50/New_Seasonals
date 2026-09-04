"""b2 / C4: the dollar washout translated to the funding trade -- long EEM
on DX rank21 <= 2. Control legs EFA and SPY.

The candidate's whole defence is that the MECHANISM IS THE DOLLAR, not the
country, which is what keeps it outside the closed country-decoupling family
(EWZ x2, FXI, SMH/QQQ, EWJ). So the decisive tests are:
  (1) gate attribution -- does the dollar gate add anything over EEM's own
      state? If not, it is a country cell in disguise and dies with the family.
  (2) reference class across the international set with a permutation
      max-of-N null. Mid-pack = kill.
  (3) beta-neutral residual against SPY (registry: the long leg carries
      everything and the parent explains it).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change
import pandas as pd, numpy as np

pd.set_option("display.width", 220)

INTL = ["EEM", "EFA", "FXI", "EWJ", "EWZ", "INDA"]
TK = INTL + ["SPY", "DX-Y.NYB", "UUP"]
px = close_panel(TK)
px = px.dropna(subset=["DX-Y.NYB", "SPY"])

rk_dx = pct_rank(px["DX-Y.NYB"], 21, 252)
mask = rk_dx <= 2.0
print(f"TODAY DX rank21={rk_dx.iloc[-1]:.2f}; trigger days {int(mask.sum())}")
for t in INTL:
    s = px[t].dropna()
    print(f"  {t}: rank5={pct_rank(s,5,252).iloc[-1]:.1f} rank21={pct_rank(s,21,252).iloc[-1]:.1f} "
          f"rank63={pct_rank(s,63,252).iloc[-1]:.1f}")

# ---------- round 1 battery on EEM ----------
sub = px.dropna(subset=["EEM"])
rk_dx_s = pct_rank(sub["DX-Y.NYB"], 21, 252)
m_s = rk_dx_s <= 2.0
variants = {f"DX rank21<={k}": (rk_dx_s <= k) for k in (1, 2, 5, 10, 20, 50)}
battery(sub, m_s, [("EEM", 1.0)], 5, "C4 LONG EEM on DX rank21<=2", 6.0,
        variants=variants, event_kinds=("nfp", "cpi", "fomc_decision"))

print("\n=== horizon scan LONG EEM on the dollar washout ===")
trig = sub.index[m_s.values]
show(horizon_scan(sub, trig, [("EEM", 1.0)], hs=(1, 2, 3, 5, 10)))

# ---------- (2) reference class + permutation max-of-N ----------
print("\n=== REFERENCE CLASS: identical rule across the international set, h=5 episodes ===")
rows = []
per_name = {}
for t in INTL + ["SPY"]:
    s2 = px.dropna(subset=[t])
    r = pct_rank(s2["DX-Y.NYB"], 21, 252) <= 2.0
    ret = vehicle_ret(s2, [(t, 1.0)], 5, 1)
    valid = ret.dropna().index
    tt = s2.index[r.reindex(s2.index, fill_value=False).values].intersection(valid)
    epi = declusters(tt, 5, valid)
    d = summarize(ret.loc[epi].values, t)
    d["ctl_all"] = round(100 * ret.loc[valid].mean(), 3)
    d["edge"] = round(d["mean_pct"] - d["ctl_all"], 3)
    per_name[t] = d
    rows.append(d)
show(sorted(rows, key=lambda r: -r["edge"]))

# permutation max-of-N over the international set on a COMMON sample
common = px.dropna(subset=INTL + ["DX-Y.NYB"])
rkc = pct_rank(common["DX-Y.NYB"], 21, 252) <= 2.0
rets = {t: vehicle_ret(common, [(t, 1.0)], 5, 1) for t in INTL}
valid = pd.DataFrame(rets).dropna().index
tt = common.index[rkc.reindex(common.index, fill_value=False).values].intersection(valid)
epi = declusters(tt, 5, valid)
obs = {t: rets[t].loc[epi].mean() - rets[t].loc[valid].mean() for t in INTL}
print(f"\ncommon sample {valid[0].date()}..{valid[-1].date()}, {len(epi)} episodes")
print("  observed excess (episode mean - all-days mean):",
      {t: round(100 * v, 3) for t, v in obs.items()})
rank_eem = 1 + sum(1 for t in INTL if obs[t] > obs["EEM"])
print(f"  EEM ranks {rank_eem} of {len(INTL)} in its own family")

rng = np.random.default_rng(42)
allpos = np.arange(len(valid))
n_epi = len(epi)
maxes = []
obs_max = max(obs.values())
obs_eem = obs["EEM"]
for _ in range(2000):
    pick = valid[rng.choice(allpos, size=n_epi, replace=False)]
    ex = [rets[t].loc[pick].mean() - rets[t].loc[valid].mean() for t in INTL]
    maxes.append(max(ex))
maxes = np.asarray(maxes)
print(f"  permutation: P(max-of-{len(INTL)} random-date excess >= EEM's observed "
      f"{100*obs_eem:+.3f}%) = {(maxes >= obs_eem).mean():.3f}")
print(f"  permutation: P(max-of-{len(INTL)} >= family max {100*obs_max:+.3f}%) = "
      f"{(maxes >= obs_max).mean():.3f}")

# ---------- (1) gate attribution ----------
print("\n=== GATE ATTRIBUTION (h=5, episodes): does the dollar gate add anything? ===")
s2 = px.dropna(subset=["EEM"])
ret = vehicle_ret(s2, [("EEM", 1.0)], 5, 1)
valid = ret.dropna().index
dxg = (pct_rank(s2["DX-Y.NYB"], 21, 252) <= 2.0)
# EEM's own state today: rank63 very low (2.4) with a hot rank5 (79.8)
eem_r63 = pct_rank(s2["EEM"], 63, 252)
eem_r5 = pct_rank(s2["EEM"], 5, 252)
gates = {
    "ALL DAYS": pd.Series(True, index=s2.index),
    "DX rank21<=2 alone (pitched)": dxg,
    "EEM rank63<=5 alone": eem_r63 <= 5,
    "EEM rank63<=5 & rank5>=70 (today's shape)": (eem_r63 <= 5) & (eem_r5 >= 70),
    "JOINT DX<=2 & EEM r63<=5": dxg & (eem_r63 <= 5),
    "DX<=2 & EEM r63>5 (dollar w/o EEM washout)": dxg & (eem_r63 > 5),
}
rows = []
for lbl, g in gates.items():
    tt = s2.index[g.reindex(s2.index, fill_value=False).values].intersection(valid)
    epi = declusters(tt, 5, valid)
    d = summarize(ret.loc[epi].values, lbl)
    d["n_days"] = len(tt)
    rows.append(d)
show(rows)

# ---------- (3) beta-neutral residual vs SPY ----------
print("\n=== BETA-NEUTRAL RESIDUAL: EEM h=5 regressed on SPY h=5, same window ===")
s3 = px.dropna(subset=["EEM", "SPY"])
re = vehicle_ret(s3, [("EEM", 1.0)], 5, 1)
rs = vehicle_ret(s3, [("SPY", 1.0)], 5, 1)
both = pd.DataFrame({"e": re, "s": rs}).dropna()
beta = np.polyfit(both["s"], both["e"], 1)[0]
print(f"  full-sample beta(EEM h5 on SPY h5) = {beta:.3f}")
dxg3 = pct_rank(s3["DX-Y.NYB"], 21, 252) <= 2.0
tt = s3.index[dxg3.reindex(s3.index, fill_value=False).values].intersection(both.index)
epi = declusters(tt, 5, both.index)
resid = both["e"] - beta * both["s"]
show([summarize(both["e"].loc[epi].values, "EEM raw (episodes)"),
      summarize(both["s"].loc[epi].values, "SPY raw same days"),
      summarize(resid.loc[epi].values, f"EEM resid vs {beta:.2f}xSPY"),
      summarize(resid.loc[both.index].values, "resid all days"),
      summarize((both["e"] - both["s"]).loc[epi].values, "EEM-SPY equal dollar")])

# ---------- midterm split ----------
print("\n=== midterm split, EEM h=5 episodes ===")
yrs = pd.DatetimeIndex(epi).year
mid = (yrs % 4 == 2)
show([summarize(both["e"].loc[epi[mid]].values, "MIDTERM"),
      summarize(both["e"].loc[epi[~mid]].values, "non-midterm")])
