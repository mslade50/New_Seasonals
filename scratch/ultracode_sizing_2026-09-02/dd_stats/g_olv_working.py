"""Refutation probe 1.4 (b): the shipped OLV depth rung counts FILLED legs plus WORKING entries (unfilled limits
inside their T+3 window). The study's n_open counts filled legs only. Measure how much the rung distribution
shifts when working entries are counted, using the raw candidate dump (flow_candidates.parquet) as the
staged-signal record and the ledger as the fill record.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
D = HERE.parent
pd.set_option("display.width", 250, "display.max_columns", 40, "display.float_format", "{:,.3f}".format)
OUT: dict = {}
feat = pd.read_parquet(D / "within_strategy_adds_features.parquet")
v = feat[feat.Strategy == "Oversold Low Volume"].copy().sort_values(["Signal Date", "trade_id"]).reset_index(drop=True)
cand = pd.read_parquet(D / "flow_candidates.parquet")
c = cand[cand.strategy == "Oversold Low Volume"][["signal_date", "ticker", "tier"]].drop_duplicates().sort_values("signal_date").reset_index(drop=True)
print(f"OLV fills {len(v)} ({v['Signal Date'].min().date()}..{v['Signal Date'].max().date()}); OLV raw candidates {len(c)} ({c.signal_date.min().date()}..{c.signal_date.max().date()})")
bd = pd.bdate_range("2002-01-01", "2026-12-31"); pos = pd.Series(np.arange(len(bd)), index=bd)
fills = v[["Ticker", "Signal Date", "Entry Date"]].rename(columns={"Ticker": "ticker", "Signal Date": "signal_date"})
c = c.merge(fills, on=["ticker", "signal_date"], how="left")   # Entry Date NaT = never filled
c["p"] = pos.reindex(c.signal_date).values
v["p"] = pos.reindex(v["Signal Date"]).values
work = np.zeros(len(v), dtype=int); sameday = np.zeros(len(v), dtype=int)
for i, (p, tk) in enumerate(zip(v.p, v.Ticker)):
    prior = c[(c.p >= p - 3) & (c.p <= p - 1)]
    # working at the close of day p: staged in the prior 3 sessions and not yet filled by day p (entry date > p or never)
    ep = pos.reindex(prior["Entry Date"]).values
    unfilled = np.isnan(ep) | (ep > p)
    work[i] = int(unfilled.sum())
    sameday[i] = int(((c.p == p) & (c.ticker != tk)).sum())
v["n_working"] = work; v["n_sameday_other"] = sameday
v["depth_filled"] = v.n_open
v["depth_live"] = v.n_open + v.n_working
rung = lambda d: np.select([d == 0, d <= 2], [0.5, 0.7], 1.0)
v["rung_filled"] = rung(v.depth_filled); v["rung_live"] = rung(v.depth_live)
v["rung_ship_filled"] = np.maximum(v.rung_ladder, v.rung_filled); v["rung_ship_live"] = np.maximum(v.rung_ladder, v.rung_live)
print("\nworking-entry count at signal close (prior 3 sessions, unfilled):", v.n_working.value_counts().sort_index().to_dict())
print("depth rung distribution  filled-only:", v.rung_filled.value_counts().sort_index().to_dict(), " filled+working:", v.rung_live.value_counts().sort_index().to_dict())
print("shipped max(recency, depth)  filled-only:", v.rung_ship_filled.value_counts().sort_index().to_dict(), " filled+working:", v.rung_ship_live.value_counts().sort_index().to_dict())
print(f"mean shipped rung: filled-only {v.rung_ship_filled.mean():.3f} vs filled+working {v.rung_ship_live.mean():.3f} (current ladder {v.rung_ladder.mean():.3f}); legs whose rung rises when working counted: {(v.rung_ship_live > v.rung_ship_filled).sum()}")
OUT["rung_dist"] = dict(filled=v.rung_ship_filled.value_counts().sort_index().to_dict(), live=v.rung_ship_live.value_counts().sort_index().to_dict(), mean_filled=float(v.rung_ship_filled.mean()), mean_live=float(v.rung_ship_live.mean()),
                        mean_current=float(v.rung_ladder.mean()), legs_raised=int((v.rung_ship_live > v.rung_ship_filled).sum()))
# edge of the legs that get raised only because of working entries
raised = v[v.rung_ship_live > v.rung_ship_filled]
print(f"legs raised by working count: N {len(raised)}, avgR {raised.R_Multiple.mean():+.3f}, years {sorted(raised.yr.unique())}; vs all OLV {v.R_Multiple.mean():+.3f}")
tab = v.groupby(pd.cut(v.depth_live, [-1, 0, 2, 5, 99], labels=["0", "1-2", "3-5", "6+"]), observed=True).agg(N=("R_Multiple", "size"), avgR=("R_Multiple", "mean"), n_working_mean=("n_working", "mean"))
print("by LIVE depth (filled+working):\n", tab.round(3).to_string())
tab2 = v.groupby(pd.cut(v.n_working, [-1, 0, 2, 99], labels=["0", "1-2", "3+"]), observed=True).agg(N=("R_Multiple", "size"), avgR=("R_Multiple", "mean"))
print("by working count alone:\n", tab2.round(3).to_string())
OUT["by_live_depth"] = tab.round(4).reset_index().astype(str).to_dict("records"); OUT["by_working"] = tab2.round(4).reset_index().astype(str).to_dict("records")
OUT["raised_legs"] = dict(N=int(len(raised)), avgR=float(raised.R_Multiple.mean()))
# equal-risk replay of the shipped rule under both depth definitions
def eval_rule(g, fac):
    risk = g.Risk_flat_750k.values; m = fac / ((fac * risk).sum() / risk.sum())
    flat = risk * g.R_Multiple.values; tier = risk * m * g.R_Multiple.values
    Yt = pd.DataFrame(dict(y=g.yr.values, f=flat, t=tier)).groupby("y").sum(); d = Yt.t - Yt.f
    return dict(gain_pct=float(d.sum() / abs(Yt.f.sum()) * 100), years_better=int((d > 0).sum()), years=len(Yt), raw_risk_ratio=float((fac * risk).sum() / risk.sum()))
for lab, r in [("filled-only depth", v.rung_ship_filled), ("filled+working depth", v.rung_ship_live)]:
    res = eval_rule(v, (r / v.rung_ladder).values); print(lab, res); OUT[f"replay_{lab}"] = res
json.dump(OUT, open(HERE / "g_olv_working.json", "w"), indent=1, default=float)
print("wrote g_olv_working.json")
