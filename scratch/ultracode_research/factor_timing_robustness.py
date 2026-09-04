"""Robustness of the fragility-timed de-risk overlay (SPY->BIL / SPY->TLT /
SPY->USMV, thr50 month-end rule). Episode attribution, LOYO, drop-one-episode,
significance of the active return.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN_DATE = pd.Timestamp("2026-07-02")

fac = pd.read_parquet(HERE / "factor_etf_prices.parquet")
fac.index = pd.to_datetime(fac.index).normalize()
fac = fac[fac.index < RUN_DATE]
mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet",
                     columns=["ticker", "date", "Close"])
mp["date"] = pd.to_datetime(mp["date"]).dt.normalize()
tlt = mp[mp["ticker"] == "TLT"].set_index("date")["Close"].rename("TLT").sort_index()
px = fac.join(tlt, how="outer").sort_index()
mret = px.resample("ME").last().pct_change()

frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
frag.index = pd.to_datetime(frag.index).normalize()
dial = frag["63d"].dropna().rolling(10, min_periods=1).mean()
dial_d = dial.reindex(pd.date_range(dial.index.min(), dial.index.max()),
                      method="ffill", limit=5)
dial_me = dial_d.resample("ME").last()
FRAG_START = pd.Period("2016-09", "M")

sig = (dial_me >= 50).shift(1).reindex(mret.index)  # defensive over month t
mm = mret.copy()
mm.index = mm.index.to_period("M")
sig.index = sig.index.to_period("M")
win = mm.index >= FRAG_START
mm, sig = mm[win], sig[mm.index[win].min():]
sig = sig.reindex(mm.index).fillna(0).astype(bool)

def_months = mm.index[sig]
print(f"defensive months (prior month-end dial>=50): N={len(def_months)}")
print(list(def_months.astype(str)))

# contiguous episodes
eps, cur = [], []
prev = None
for p in def_months:
    if prev is not None and (p - prev).n == 1:
        cur.append(p)
    else:
        if cur:
            eps.append(cur)
        cur = [p]
    prev = p
if cur:
    eps.append(cur)
print(f"\nepisodes: {len(eps)}")

COST = 0.001  # 10 bps round-turn per switch, charged once per episode entry+exit
for dfn in ["BIL", "TLT", "USMV"]:
    print(f"\n=== SPY->{dfn} thr50: active return attribution ===")
    act = (mm[dfn] - mm["SPY"])[sig]
    rows = []
    for ep in eps:
        a = act.loc[ep[0]:ep[-1]]
        spy_tot = 100 * ((1 + mm.loc[ep[0]:ep[-1], "SPY"]).prod() - 1)
        rows.append({"episode": f"{ep[0]}..{ep[-1]}", "n_mo": len(ep),
                     "SPY_tot%": spy_tot,
                     "active_tot%": 100 * ((1 + a).prod() - 1) - 100 * COST})
    tbl = pd.DataFrame(rows)
    print(tbl.round(2).to_string(index=False))
    tot = tbl["active_tot%"].sum()
    print(f"sum of episode actives (approx): {tot:+.1f}%")
    # drop-one-episode
    for i, ep in enumerate(eps):
        rest = tbl.drop(index=i)["active_tot%"].sum()
        print(f"  drop {tbl.loc[i,'episode']}: remaining active ~{rest:+.1f}%")
    # t-test on monthly active obs in defensive months (episode clustering: avg per episode)
    ep_means = [act.loc[ep[0]:ep[-1]].mean() for ep in eps]
    t1 = stats.ttest_1samp(act, 0)
    t2 = stats.ttest_1samp(ep_means, 0)
    print(f"  monthly active: mean={100*act.mean():+.2f}%/mo t={t1.statistic:+.2f} p={t1.pvalue:.3f} N={len(act)}")
    print(f"  episode-clustered: mean={100*np.mean(ep_means):+.2f}%/mo t={t2.statistic:+.2f} p={t2.pvalue:.3f} Neps={len(ep_means)}")

# LOYO on full-series Sharpe uplift for SPY->BIL thr50
print("\n=== LOYO: SPY->BIL thr50 vs SPY, Sharpe uplift ===")
rot = np.where(sig, mm["BIL"], mm["SPY"])
switch_cost = sig.astype(int).diff().abs().fillna(0) * COST
rot = pd.Series(rot, index=mm.index) - switch_cost
spy = mm["SPY"]

def sharpe(r):
    return r.mean() * 12 / (r.std() * np.sqrt(12))

print(f"all: rot Sharpe={sharpe(rot):.2f} SPY={sharpe(spy):.2f} "
      f"active tot={100*((1+rot).prod()/ (1+spy).prod()-1):+.1f}%")
for y in sorted(set(mm.index.year)):
    m = mm.index.year != y
    print(f"  drop {y}: rot={sharpe(rot[m]):.2f} spy={sharpe(spy[m]):.2f} "
          f"active tot={100*((1+rot[m]).prod()/(1+spy[m]).prod()-1):+.1f}%")

# what did the dial do 2016-2017 and 2023 (missed rallies)?
print("\nfraction of months defensive by year:")
print(sig.groupby(sig.index.year).mean().round(2).to_string())

# false-positive cost: defensive months where SPY was UP
up = mm.loc[sig & (mm["SPY"] > 0), "SPY"]
dn = mm.loc[sig & (mm["SPY"] <= 0), "SPY"]
print(f"\ndefensive months with SPY up: {len(up)} (avg {100*up.mean():+.2f}%) — missed gains")
print(f"defensive months with SPY dn: {len(dn)} (avg {100*dn.mean():+.2f}%) — avoided losses")
