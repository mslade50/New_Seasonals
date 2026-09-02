"""Within-strategy adds, step 5: tail checks and the combined package.
 1. 52wh: worst-21d MTM window under current; deep-stack (>=6 open) legs' MTM
    contribution inside it; pre-2010 vs 2010+ sign of the 6+ cell.
 2. OVS: worst single day / worst 21d under 'P1 depth>=6 x1.5' and x2; P1 depth>=6
    cell by year; booked risk per P1 fill by depth (the cap footprint).
 3. LT Trend ST OS: depth>=3 adds by lagged dial bucket (2016-07+, current-weights
    vintage) -- the exemption re-test interaction.
 4. OLV: 50%-NAV ticker-cap footprint (legs with a non-ladder, non-earnings residual).
 5. Combined package replay on the 8-strategy sleeve MTM: current vs package
    (OLV depth-OR-ticker ladder, OVS P1 depth>=6 x1.5, WCDS solo .75 / adds 1.25,
    LT solo .75 / adds 1.25; 52wh and 3x fades unchanged): PnL, worst day, worst 21d,
    maxDD, ann. Sharpe, by-year deltas, drop-best-year.
Writes within_strategy_adds_package.json.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
NAV = 750_000.0
pd.set_option("display.width", 250, "display.max_columns", 40, "display.float_format", "{:,.3f}".format)
df = pd.read_parquet(OUT / "within_strategy_adds_features.parquet")
M = pd.read_parquet(OUT / "within_strategy_adds_mtm.parquet")
RES: dict = {}

def sleeve_daily(s, factor):
    m = M[M.Strategy == s]
    return (m.pnl * m.idx.map(factor)).groupby(m.date).sum()

# ---------------------------------------------------------------- 1. 52wh worst window
b = df[df.Strategy == "52wh Breakout"]
cur = sleeve_daily("52wh Breakout", pd.Series(1.0, index=b.index))
cur = cur.reindex(pd.bdate_range(cur.index.min(), cur.index.max())).fillna(0)
r21 = cur.rolling(21).sum(); end = r21.idxmin(); start = end - pd.tseries.offsets.BDay(20)
m = M[M.Strategy == "52wh Breakout"]; win = m[(m.date >= start) & (m.date <= end)]
deep_idx = set(b[b.n_open >= 6].index)
contrib = win.groupby(win.idx.isin(deep_idx)).pnl.sum()
print(f"52wh worst 21d MTM window {start.date()}..{end.date()}: {r21.min():,.0f}; deep-stack legs' share {contrib.get(True, 0):,.0f}, other legs {contrib.get(False, 0):,.0f}; legs open in window: {win.idx.nunique()} (deep {len(set(win.idx) & deep_idx)})")
top = r21.nsmallest(60); seen = []; wins = []
for d, v in top.items():
    if any(abs((d - x).days) < 45 for x in seen):
        continue
    seen.append(d); w = m[(m.date > d - pd.tseries.offsets.BDay(21)) & (m.date <= d)]
    wins.append(dict(end=d.date().isoformat(), pnl21=float(v), deep_share=float(w[w.idx.isin(deep_idx)].pnl.sum() / v) if v else 0, legs=int(w.idx.nunique())))
    if len(wins) >= 5:
        break
print(pd.DataFrame(wins).to_string(index=False)); RES["b52_worst_windows"] = wins
e = b.assign(era=np.where(b.yr < 2010, "pre-2010", "2010+"))
t = e[e.n_open >= 6].groupby("era").agg(N=("R_Multiple", "size"), avgR=("R_Multiple", "mean"), win=("R_Multiple", lambda s: (s > 0).mean()), pnl=("PnL_flat_750k", "sum"), episodes=("episode", "nunique"))
print("52wh >=6 open by era:"); print(t.round(3).to_string()); RES["b52_ge6_by_era"] = t.round(4).reset_index().to_dict("records")

# ---------------------------------------------------------------- 2. OVS tails under the P1 cluster up-size
o = df[df.Strategy == "Overbot Vol Spike"].copy(); o["path"] = np.where(o.Size_Mult >= 0.7, "P1", "P2")
rows = []
for lab, f in [("current", np.ones(len(o))), ("P1 depth>=6 x1.5", np.where((o.path == "P1") & (o.n_open >= 6), 1.5, 1.0)), ("P1 depth>=6 x2", np.where((o.path == "P1") & (o.n_open >= 6), 2.0, 1.0)),
               ("P1 depth>=3 x1.5", np.where((o.path == "P1") & (o.n_open >= 3), 1.5, 1.0)), ("P1 solo .75 / depth>=3 x1.5", np.where(o.path == "P1", np.select([o.n_open == 0, o.n_open >= 3], [0.75, 1.5], 1.0), 1.0))]:
    d = sleeve_daily("Overbot Vol Spike", pd.Series(f, index=o.index)); d = d.reindex(pd.bdate_range(d.index.min(), d.index.max())).fillna(0)
    eq = d.cumsum(); act = d[d != 0]
    rows.append(dict(rule=lab, total=float(d.sum()), worst_day=float(d.min()), worst_day_date=d.idxmin().date().isoformat(), worst21=float(d.rolling(21).sum().min()), maxdd=float((eq - eq.cummax()).min()),
                     n_days_lt_10k=int((d < -10000).sum()), sharpe_active=float(act.mean() / act.std() * np.sqrt(252)), risk=float((o.Risk_flat_750k * f).sum())))
print("\nOVS tails:"); print(pd.DataFrame(rows).to_string(index=False)); RES["ovs_tails"] = rows
p1 = o[(o.path == "P1") & (o.n_open >= 6)]
by = p1.groupby("yr").agg(N=("R_Multiple", "size"), avgR=("R_Multiple", "mean"), pnl=("PnL_flat_750k", "sum"), risk_bps_per_fill=("Risk_flat_750k", lambda s: s.mean() / NAV * 1e4))
print("OVS P1 legs at depth>=6 by year:"); print(by.round(2).to_string()); RES["ovs_p1_deep_by_year"] = {int(k): v for k, v in by.round(4).to_dict("index").items()}
print("years positive:", int((by.avgR > 0).sum()), "of", len(by))

# ---------------------------------------------------------------- 3. LT Trend adds by dial
frag = pd.read_parquet(ROOT / "data/rd2_fragility.parquet"); dial = frag["63d"].rolling(10).mean().shift(1)
l = df[df.Strategy == "LT Trend ST OS"].copy(); l["dial"] = dial.reindex(l["Signal Date"]).values
l16 = l[l["Signal Date"] >= "2016-07-20"].dropna(subset=["dial"])
l16["db"] = pd.cut(l16.dial, [0, 50, 65, 101], labels=["<50", "50-65", "65+"], right=False)
l16["depth"] = np.select([l16.n_open == 0, l16.n_open >= 3], ["solo", "3+ open"], "1-2 open")
t = l16.pivot_table(index="depth", columns="db", values="R_Multiple", aggfunc=["size", "mean"], observed=True)
print("\nLT Trend ST OS adds x lagged dial (2016-07+, current-weights vintage):"); print(t.round(2).to_string())
RES["lt_depth_x_dial"] = {f"{a}|{b_}": dict(N=int(t[("size", b_)][a]), avgR=float(t[("mean", b_)][a])) for a in t.index for b_ in t.columns.levels[1] if ("size", b_) in t.columns and pd.notna(t[("size", b_)][a])}

# ---------------------------------------------------------------- 4. OLV ticker-cap footprint
v = df[df.Strategy == "Oversold Low Volume"].copy()
capclip = v[(np.abs(v.residual_mult - 1) > 0.02) & (np.abs(v.residual_mult - 15 / 52.5) > 0.02)]
print(f"\nOLV legs with a residual size not explained by ladder or earnings override (ticker-cap clips): N={len(capclip)}, mean residual {capclip.residual_mult.mean():.2f}, avgR {capclip.R_Multiple.mean():.2f}, booked PnL ${capclip.PnL_flat_750k.sum():,.0f}, PnL at full residual ${(capclip.PnL_flat_750k / capclip.residual_mult).sum():,.0f}")
print(capclip[["Entry Date", "Ticker", "Size_Mult", "rung_ladder", "residual_mult", "n_same_ticker", "R_Multiple", "PnL_flat_750k"]].to_string(index=False))
RES["olv_ticker_cap_clips"] = dict(N=int(len(capclip)), avgR=float(capclip.R_Multiple.mean()) if len(capclip) else None, pnl_booked=float(capclip.PnL_flat_750k.sum()), pnl_unclipped=float((capclip.PnL_flat_750k / capclip.residual_mult).sum()) if len(capclip) else 0.0,
                                   rows=capclip[["Entry Date", "Ticker", "Size_Mult", "rung_ladder", "residual_mult", "n_same_ticker", "R_Multiple", "PnL_flat_750k"]].assign(**{"Entry Date": lambda x: x["Entry Date"].dt.date.astype(str)}).round(3).to_dict("records"))

# ---------------------------------------------------------------- 5. combined package on the 8-strategy sleeve
def factors(pkg):
    f = pd.Series(1.0, index=df.index)
    if pkg:
        g = df[df.Strategy == "Oversold Low Volume"]
        new = np.maximum(g.rung_ladder, np.select([g.n_open == 0, g.n_open <= 2], [0.5, 0.7], 1.0))
        f[g.index] = new / g.rung_ladder
        g = df[df.Strategy == "Overbot Vol Spike"]
        f[g.index] = np.where((g.Size_Mult >= 0.7) & (g.n_open >= 6), 1.5, 1.0)
        for s in ["Weak Close Decent Sznls", "LT Trend ST OS"]:
            g = df[df.Strategy == s]; f[g.index] = np.where(g.n_open == 0, 0.75, 1.25)
    return f
res = {}
for lab, pkg in [("current", False), ("package", True)]:
    f = factors(pkg)
    d = (M.pnl * M.idx.map(f)).groupby(M.date).sum(); d = d.reindex(pd.bdate_range("2003-01-01", d.index.max())).fillna(0)
    d = d[d.index >= "2005-01-01"]
    eq = d.cumsum(); dd = eq - eq.cummax()
    per_s = {s: float((df.PnL_flat_750k * f)[df.Strategy == s].sum()) for s in df.Strategy.unique()}
    res[lab] = dict(total=float(d.sum()), risk=float((df.Risk_flat_750k * f).sum()), worst_day=float(d.min()), worst_day_date=d.idxmin().date().isoformat(), worst21=float(d.rolling(21).sum().min()),
                    worst21_end=d.rolling(21).sum().idxmin().date().isoformat(), maxdd=float(dd.min()), maxdd_date=dd.idxmin().date().isoformat(), ann_sharpe=float(d.mean() / d.std() * np.sqrt(252)),
                    ann_pnl=float(d.mean() * 252), ann_vol=float(d.std() * np.sqrt(252)), by_year={int(k): float(x) for k, x in d.groupby(d.index.year).sum().items()}, per_strategy=per_s)
    res[lab]["pnl_per_risk"] = res[lab]["total"] / res[lab]["risk"]
k = res["current"]["risk"] / res["package"]["risk"]
res["package"]["at_equal_risk"] = dict(total=res["package"]["total"] * k, worst_day=res["package"]["worst_day"] * k, worst21=res["package"]["worst21"] * k, maxdd=res["package"]["maxdd"] * k)
dy = pd.Series(res["package"]["by_year"]) - pd.Series(res["current"]["by_year"])
res["package"]["d_by_year"] = {int(a): float(x) for a, x in dy.items()}
res["package"]["years_better"] = int((dy > 0).sum()); res["package"]["years_total"] = int(len(dy)); res["package"]["d_total_ex_best_year"] = float(dy.sum() - dy.max()); res["package"]["best_year"] = int(dy.idxmax())
print("\n=== combined package, 8-strategy sleeve MTM (2005+) ===")
for lab in ["current", "package"]:
    r = res[lab]; print(f"{lab:8s} total {r['total']:,.0f} risk {r['risk']:,.0f} pnl/risk {r['pnl_per_risk']:.3f} worst day {r['worst_day']:,.0f} ({r['worst_day_date']}) worst21 {r['worst21']:,.0f} ({r['worst21_end']}) maxDD {r['maxdd']:,.0f} ({r['maxdd_date']}) sharpe {r['ann_sharpe']:.2f} annPnL {r['ann_pnl']:,.0f} vol {r['ann_vol']:,.0f}")
print("package at equal risk:", {a: round(x) for a, x in res["package"]["at_equal_risk"].items()})
print("delta by year:", {a: round(x) for a, x in res["package"]["d_by_year"].items()}); print("years better", res["package"]["years_better"], "of", res["package"]["years_total"], "; delta ex best year", round(res["package"]["d_total_ex_best_year"]), "(best:", res["package"]["best_year"], ")")
print("per-strategy PnL current -> package:", {s: (round(res["current"]["per_strategy"][s]), round(res["package"]["per_strategy"][s])) for s in res["current"]["per_strategy"]})
RES["package"] = res

json.dump(RES, open(OUT / "within_strategy_adds_package.json", "w"), indent=1, default=float)
print("\nwrote within_strategy_adds_package.json")
