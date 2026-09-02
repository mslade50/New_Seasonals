"""Within-strategy adds, step 4: follow-up checks.
 1. OVS: path (P1/P2) x depth split; which cap binds on cluster days; replay of
    'P1 adds at depth>=6 x1.25' and 'P2 aggregate cap x2 on cluster days'.
 2. WCDS / LT Trend: same-day cluster sizes vs the 250 bps per-strategy daily cap
    if adds are up-sized; dial at cluster days (2016+).
 3. 52wh: the 6+ cell by episode/year; what the >=6 rule does per year.
 4. OLV: the 2+ same-sector cell by year/sector; same-ticker adds by depth.
 5. Episode-cluster bootstrap P(adds - solo <= 0) for the headline cells.
 6. Equal-risk reallocation variants (solo 0.75 / adds 1.25) for OLV, OVS, WCDS, LT.
Writes within_strategy_adds_checks.json.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
NAV = 750_000.0
RNG = np.random.default_rng(3)
pd.set_option("display.width", 250, "display.max_columns", 40, "display.float_format", "{:,.3f}".format)
df = pd.read_parquet(OUT / "within_strategy_adds_features.parquet")
M = pd.read_parquet(OUT / "within_strategy_adds_mtm.parquet")
RES: dict = {}

def metrics(g, factor, label):
    f = pd.Series(np.asarray(factor, float), index=g.index)
    pnl = g.PnL_flat_750k * f; risk = g.Risk_flat_750k * f
    ex = pnl.groupby(g["Exit Date"]).sum(); ex = ex.reindex(pd.bdate_range(ex.index.min(), ex.index.max())).fillna(0)
    eq = ex.cumsum()
    m = M[M.Strategy == g.Strategy.iloc[0]]
    mp = (m.pnl * m.idx.map(f)).groupby(m.date).sum(); mp = mp.reindex(pd.bdate_range(mp.index.min(), mp.index.max())).fillna(0)
    eqm = mp.cumsum()
    return dict(rule=label, total_pnl=float(pnl.sum()), risk_deployed=float(risk.sum()), pnl_per_risk=float(pnl.sum() / risk.sum()),
                worst21_exit=float(ex.rolling(21).sum().min()), maxdd_exit=float((eq - eq.cummax()).min()),
                worst21_mtm=float(mp.rolling(21).sum().min()), maxdd_mtm=float((eqm - eqm.cummax()).min()), legs_changed=int((np.abs(f - 1) > 1e-6).sum()))

def eq_risk(rows):
    cur = rows[0]
    for r in rows:
        k = cur["risk_deployed"] / r["risk_deployed"]
        r["pnl_at_equal_risk"] = r["total_pnl"] * k; r["d_pnl_equal_risk_vs_current"] = r["total_pnl"] * k - cur["total_pnl"]
        r["maxdd_mtm_at_equal_risk"] = r["maxdd_mtm"] * k
    return rows

# ---------------------------------------------------------------- 1. OVS path x depth
o = df[df.Strategy == "Overbot Vol Spike"].copy()
o["path"] = np.where(o.Size_Mult >= 0.7, "P1", "P2")
o["depth"] = pd.cut(o.n_open, [-1, 0, 2, 5, 12, 99], labels=["0", "1-2", "3-5", "6-12", "13+"])
t = o.groupby(["path", "depth"], observed=True).agg(N=("R_Multiple", "size"), avgR=("R_Multiple", "mean"), win=("R_Multiple", lambda s: (s > 0).mean()),
                                                    risk_bps=("Risk_flat_750k", lambda s: s.mean() / NAV * 1e4), pnl=("PnL_flat_750k", "sum"), size_mult=("Size_Mult", "mean"))
print("=== OVS path x open-leg depth ==="); print(t.round(3).to_string())
RES["ovs_path_x_depth"] = t.round(4).reset_index().to_dict("records")
# inspect the biggest day
big = o.groupby("Signal Date").size().sort_values().index[-1]
print(f"largest OVS day {big.date()}:"); print(o[o["Signal Date"] == big][["Ticker", "path", "Size_Mult", "Risk_flat_750k", "R_Multiple", "PnL_flat_750k", "unit_risk"]].head(12).to_string(index=False))
# which cap binds: per day P1 risk vs 250 bps, P2 risk vs 112.5 bps
day = o.groupby(["Signal Date", "path"]).Risk_flat_750k.sum().unstack().fillna(0) / NAV * 1e4
day["n"] = o.groupby("Signal Date").size()
day["P2_at_cap"] = day.get("P2", 0) >= 105; day["P1_at_cap"] = day.get("P1", 0) >= 235
c = day.groupby(pd.cut(day.n, [0, 1, 2, 5, 12, 99], labels=["1", "2", "3-5", "6-12", "13+"]), observed=True).agg(days=("n", "size"), P1_bps=("P1", "mean"), P2_bps=("P2", "mean"), P2_days_at_cap=("P2_at_cap", "sum"), P1_days_at_cap=("P1_at_cap", "sum"))
print("OVS per-day booked risk by path (bps) and cap-binding days:"); print(c.round(1).to_string())
RES["ovs_cap_binding"] = c.round(3).reset_index().rename(columns={"n": "fills"}).to_dict("records")
# replays
base = o.Size_Mult
rules = {"current": base,
         "P1_depth>=6_x1.25": base * np.where((o.path == "P1") & (o.n_open >= 6), 1.25, 1.0),
         "P1_depth>=3_x1.25": base * np.where((o.path == "P1") & (o.n_open >= 3), 1.25, 1.0),
         "P1_depth>=6_x1.5": base * np.where((o.path == "P1") & (o.n_open >= 6), 1.5, 1.0),
         "P2_x2_on_capped_days": base * np.where((o.path == "P2") & o["Signal Date"].map(day["P2_at_cap"]).fillna(False).values, 2.0, 1.0),
         "P2_x2_everywhere": base * np.where(o.path == "P2", 2.0, 1.0),
         "P1_solo_0.75_adds_1.25": base * np.where(o.path == "P1", np.where(o.n_open == 0, 0.75, 1.25), 1.0),
         "P1_solo_0.75_depth>=3_1.5": base * np.where(o.path == "P1", np.select([o.n_open == 0, o.n_open >= 3], [0.75, 1.5], 1.0), 1.0)}
rows = eq_risk([metrics(o, np.asarray(v) / np.asarray(base), k) for k, v in rules.items()])
print(pd.DataFrame(rows).to_string(index=False)); RES["ovs_replays"] = rows

# ---------------------------------------------------------------- 2. WCDS / LT cluster sizes vs cap
frag = pd.read_parquet(ROOT / "data/rd2_fragility.parquet")
dial = frag["63d"].rolling(10).mean().shift(1)
for s, bps in [("Weak Close Decent Sznls", 52.5), ("LT Trend ST OS", 45.0)]:
    g = df[df.Strategy == s].copy()
    cl = g.groupby("Signal Date").agg(n=("Ticker", "size"), risk_bps=("Risk_flat_750k", lambda x: x.sum() / NAV * 1e4), avgR=("R_Multiple", "mean"), pnl=("PnL_flat_750k", "sum"))
    cl["dial"] = dial.reindex(cl.index).values
    dist = cl.n.value_counts().sort_index().to_dict()
    binds = {f"x{m}": int((cl.n * bps * m > 250).sum()) for m in (1.0, 1.25, 1.5)}
    print(f"\n{s}: fills per signal day {dist}; days where cap binds at x1.0/x1.25/x1.5 on adds: {binds}; max day risk booked {cl.risk_bps.max():.0f} bps")
    d16 = cl[cl.index >= "2016-07-20"].dropna(subset=["dial"])
    hi = d16[d16.n >= 2]; lo = d16[d16.n == 1]
    print(f"  2016+: cluster days (n>=2) N={len(hi)} mean dial {hi.dial.mean():.1f}, share dial>=50 {(hi.dial >= 50).mean():.2f} | single days N={len(lo)} mean dial {lo.dial.mean():.1f}, share>=50 {(lo.dial >= 50).mean():.2f}")
    print("  cluster-size table:"); tt = cl.groupby(pd.cut(cl.n, [0, 1, 2, 3, 5, 99], labels=["1", "2", "3", "4-5", "6+"]), observed=True).agg(days=("n", "size"), avgR=("avgR", "mean"), pnl_day=("pnl", "mean"), pnl_tot=("pnl", "sum"), worst_day=("pnl", "min"))
    print(tt.round(2).to_string())
    RES[s + "_clusters"] = dict(fills_per_day=dist, cap_binds_days=binds, dial_cluster_mean=float(hi.dial.mean()) if len(hi) else None, dial_single_mean=float(lo.dial.mean()) if len(lo) else None,
                                share_cluster_dial_ge50=float((hi.dial >= 50).mean()) if len(hi) else None, table=tt.round(3).reset_index().to_dict("records"))
    base = g.Size_Mult
    rules = {"current": base, "adds_1.25": base * np.where(g.n_open >= 1, 1.25, 1.0), "solo_0.75_adds_1.25": base * np.where(g.n_open == 0, 0.75, 1.25),
             "sameday_cluster_1.5": base * np.where(g.same_day_prior >= 1, 1.5, 1.0), "solo_0.75_sameday_1.5": base * np.where(g.n_open == 0, 0.75, np.where(g.same_day_prior >= 1, 1.5, 1.0)),
             "adds_1.25_capped250": base * np.where(g.n_open >= 1, 1.25, 1.0) * np.minimum(1.0, 250 / (g.groupby("Signal Date").Ticker.transform("size") * bps * np.where(g.n_open >= 1, 1.25, 1.0)))}
    rows = eq_risk([metrics(g, np.asarray(v) / np.asarray(base), k) for k, v in rules.items()])
    print(pd.DataFrame(rows).to_string(index=False)); RES[s + "_replays"] = rows

# ---------------------------------------------------------------- 3. 52wh 6+ cell by episode/year
b = df[df.Strategy == "52wh Breakout"].copy()
deep = b[b.n_open >= 6]
print("\n=== 52wh legs entered with >=6 open, by year ===")
by = deep.groupby("yr").agg(N=("R_Multiple", "size"), avgR=("R_Multiple", "mean"), pnl=("PnL_flat_750k", "sum"), episodes=("episode", "nunique"), same_sec_mean=("n_same_sector", "mean"), rho=("rho63_mean", "mean"))
print(by.round(2).to_string()); RES["b52_ge6_by_year"] = {int(k): v for k, v in by.round(4).to_dict("index").items()}
# the >=6 rule per year: d_pnl by year (drop-best-year robustness)
f = np.where(b.n_open >= 6, 0.5, 1.0)
dy = (b.PnL_flat_750k * (f - 1)).groupby(b.yr).sum()
print("52wh open>=6 x0.5 rule, PnL change by year:", dy[dy != 0].round(0).to_dict(), "| total", round(dy.sum()), "| ex-2014", round(dy.drop(2014, errors='ignore').sum()))
RES["b52_ge6_rule_dpnl_by_year"] = {int(k): float(v) for k, v in dy[dy != 0].items()}
b10 = b[b.yr >= 2010]
print("52wh n_open exact 5 vs 6+ (2010+):", b10[b10.n_open == 5].R_Multiple.agg(["size", "mean"]).round(3).to_dict(), b10[b10.n_open >= 6].R_Multiple.agg(["size", "mean"]).round(3).to_dict())

# ---------------------------------------------------------------- 4. OLV same-sector and same-ticker cells
v = df[df.Strategy == "Oversold Low Volume"].copy()
ss = v[v.n_same_sector >= 2]
print("\n=== OLV legs with >=2 same-sector open legs, by year x sector ===")
tab = ss.groupby(["yr", "sector"]).agg(N=("R_Multiple", "size"), avgR=("R_Multiple", "mean"), pnl=("PnL_flat_750k", "sum"))
print(tab.round(2).to_string()); RES["olv_same_sector2_by_year_sector"] = [dict(yr=int(a), sector=b_, **r) for (a, b_), r in tab.round(4).to_dict("index").items()]
st = v[v.n_same_ticker >= 1].groupby(pd.cut(v[v.n_same_ticker >= 1].n_same_ticker, [0, 1, 2, 99], labels=["1", "2", "3+"]), observed=True).agg(N=("R_Multiple", "size"), avgR=("R_Multiple", "mean"), win=("R_Multiple", lambda s: (s > 0).mean()), pnl=("PnL_flat_750k", "sum"), rung=("rung_ladder", "mean"), yrs=("yr", "nunique"))
print("OLV same-ticker adds by # already-open legs in that ticker:"); print(st.round(3).to_string()); RES["olv_same_ticker_depth"] = st.round(4).reset_index().to_dict("records")
# what does the ladder do by depth: mean rung by n_open bucket
lad = v.groupby(pd.cut(v.n_open, [-1, 0, 2, 5, 99], labels=["0", "1-2", "3-5", "6+"]), observed=True).agg(N=("rung_ladder", "size"), rung_mean=("rung_ladder", "mean"), share_at_half=("rung_ladder", lambda s: (s == 0.5).mean()), avgR=("R_Multiple", "mean"))
print("OLV current ladder rung by sleeve depth:"); print(lad.round(3).to_string()); RES["olv_ladder_rung_by_depth"] = lad.round(4).reset_index().to_dict("records")

# ---------------------------------------------------------------- 5. episode-cluster bootstrap on headline cells
def boot(g, mask_a, mask_b, n=4000):
    ep = g.episode.values; R = g.R_Multiple.values; ua = np.unique(ep)
    out = []
    for _ in range(n):
        pick = RNG.choice(ua, len(ua), replace=True)
        idx = np.concatenate([np.flatnonzero(ep == e) for e in pick])
        a = R[idx][mask_a.values[idx]]; bb = R[idx][mask_b.values[idx]]
        if len(a) and len(bb):
            out.append(a.mean() - bb.mean())
    out = np.array(out)
    return dict(diff=float((R[mask_a.values].mean() - R[mask_b.values].mean())), p_le0=float((out <= 0).mean()), ci5=float(np.percentile(out, 5)), ci95=float(np.percentile(out, 95)), n_a=int(mask_a.sum()), n_b=int(mask_b.sum()), episodes=int(len(ua)))
cells = {
    "OLV adds(n_open>=1) vs solo": ("Oversold Low Volume", lambda g: g.n_open >= 1, lambda g: g.n_open == 0),
    "OLV depth>=3 vs solo": ("Oversold Low Volume", lambda g: g.n_open >= 3, lambda g: g.n_open == 0),
    "OLV same-ticker add vs other adds": ("Oversold Low Volume", lambda g: g.n_same_ticker >= 1, lambda g: (g.n_open >= 1) & (g.n_same_ticker == 0)),
    "OLV 2+ same-sector vs other adds": ("Oversold Low Volume", lambda g: g.n_same_sector >= 2, lambda g: (g.n_open >= 1) & (g.n_same_sector < 2)),
    "OVS depth>=6 vs solo": ("Overbot Vol Spike", lambda g: g.n_open >= 6, lambda g: g.n_open == 0),
    "OVS 2+ same-sector vs solo": ("Overbot Vol Spike", lambda g: g.n_same_sector >= 2, lambda g: g.n_open == 0),
    "LT depth>=3 vs solo": ("LT Trend ST OS", lambda g: g.n_open >= 3, lambda g: g.n_open == 0),
    "LT adds vs solo": ("LT Trend ST OS", lambda g: g.n_open >= 1, lambda g: g.n_open == 0),
    "WCDS same-day adds vs solo": ("Weak Close Decent Sznls", lambda g: g.same_day_prior >= 1, lambda g: g.n_open == 0),
    "WCDS adds vs solo": ("Weak Close Decent Sznls", lambda g: g.n_open >= 1, lambda g: g.n_open == 0),
    "52wh depth>=6 vs rest": ("52wh Breakout", lambda g: g.n_open >= 6, lambda g: g.n_open < 6),
    "52wh depth>=5 vs rest": ("52wh Breakout", lambda g: g.n_open >= 5, lambda g: g.n_open < 5),
    "3xETF fade adds vs solo": ("3x ETF Overbot Fade", lambda g: g.n_open >= 1, lambda g: g.n_open == 0),
    "3xBear fade adds vs solo": ("3x Bear ETF Overbot Fade", lambda g: g.n_open >= 1, lambda g: g.n_open == 0),
}
print("\n=== episode-cluster bootstrap (4000 resamples of episodes) ===")
bt = {}
for k, (s, fa, fb) in cells.items():
    g = df[df.Strategy == s]
    bt[k] = boot(g, fa(g), fb(g)); print(f"{k:38s} diff {bt[k]['diff']:+.3f}R  P(<=0)={bt[k]['p_le0']:.3f}  90% CI [{bt[k]['ci5']:+.2f},{bt[k]['ci95']:+.2f}]  N={bt[k]['n_a']}/{bt[k]['n_b']} episodes={bt[k]['episodes']}")
RES["episode_bootstrap"] = bt

# ---------------------------------------------------------------- 6. OLV equal-risk reallocations
base = v.Size_Mult; resid = base / v.rung_ladder
rules = {"current": base,
         "depth_ladder_OR_ticker": resid * np.maximum(v.rung_ladder, np.select([v.n_open == 0, v.n_open <= 2], [0.5, 0.7], 1.0)),
         "depth_ladder_OR_ticker_x_sameTk1.25": resid * np.maximum(v.rung_ladder, np.select([v.n_open == 0, v.n_open <= 2], [0.5, 0.7], 1.0)) * np.where(v.n_same_ticker >= 1, 1.25, 1.0),
         "solo0.5_1-2open0.7_3+open1.0_6+open1.25": resid * np.select([v.n_open == 0, v.n_open <= 2, v.n_open <= 5], [0.5, 0.7, 1.0], 1.25),
         "solo0.5_adds1.0_2+sameSec1.25": resid * np.where(v.n_open == 0, 0.5, np.where(v.n_same_sector >= 2, 1.25, 1.0))}
rows = eq_risk([metrics(v, np.asarray(r) / np.asarray(base), k) for k, r in rules.items()])
print("\n=== OLV reallocation variants ==="); print(pd.DataFrame(rows).to_string(index=False)); RES["olv_realloc"] = rows
# per-year robustness of the depth-OR-ticker ladder vs current (equal-risk not possible per year; show raw d_pnl and PnL/risk by year)
f = np.asarray(rules["depth_ladder_OR_ticker"]) / np.asarray(base)
yy = pd.DataFrame({"yr": v.yr, "cur": v.PnL_flat_750k, "new": v.PnL_flat_750k * f, "cur_risk": v.Risk_flat_750k, "new_risk": v.Risk_flat_750k * f}).groupby("yr").sum()
yy["cur_ppr"] = yy.cur / yy.cur_risk; yy["new_ppr"] = yy.new / yy.new_risk
print("OLV depth-OR-ticker ladder by year (PnL per $ risk):"); print(yy.round(2).to_string())
RES["olv_depth_or_ticker_by_year"] = {int(k): r for k, r in yy.round(3).to_dict("index").items()}
print("years new_ppr >= cur_ppr:", int((yy.new_ppr >= yy.cur_ppr).sum()), "of", len(yy))

json.dump(RES, open(OUT / "within_strategy_adds_checks.json", "w"), indent=1, default=float)
print("\nwrote within_strategy_adds_checks.json")
