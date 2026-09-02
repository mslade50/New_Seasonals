"""cross_strategy_regime step 1: correlation dynamics of the per-strategy daily
PnL and the market factor.

(A) regime-conditional correlation matrices (dial bucket, VIX bucket, SPY
    drawdown state, month): avg pairwise corr, effective N, book beta/R2 to SPY,
    the pairs that move most vs calm tape.
(B) rolling 126d avg pairwise corr / effective N / book beta, summarised by year
    and by the worst windows.
(C) which pairs become one bet: pair corr in stress vs calm, with day counts.
(D) betas of each strategy and the book to SPY, QQQ, IWM and a momentum
    factor built from master_prices (12-1 cross-sectional, top/bottom quintile,
    monthly rebalance, sector_map universe) by regime; multi-factor loadings
    (SPY, QQQ-SPY, IWM-SPY, MOM); downside vs upside beta by regime.

Basis: dist/data/strategy_daily.json (flat $750k, tiers collapsed, ends
2026-08-07). Dial = 10d MA of rd2_fragility 63d, lag-1 (CURRENT-weights
vintage; rows before 2026-07-02 are the recompute vintage). Writes
cross_strategy_regime_results_1_corr.json beside this file.
"""
from __future__ import annotations
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
NAV = 750_000.0
pd.set_option("display.width", 250, "display.max_columns", 40, "display.float_format", "{:,.3f}".format)
OUT: dict = {}

# ------------------------------------------------------------------ load
sd = json.load(open(ROOT / "dist/data/strategy_daily.json"))
dates = pd.to_datetime(sd["dates"])
S = pd.DataFrame(sd["series"], index=dates).fillna(0.0)
strat = S.T.groupby(S.columns.str.split("||", regex=False).str[0]).sum().T / NAV
book = pd.Series(sd["total_flat"], index=dates, dtype=float) / NAV
SHORT = {"3x Bear ETF Overbot Fade": "3xBear", "3x ETF Overbot Fade": "3xFade", "3x Leader Gap Fade": "3xLead",
         "52wh Breakout": "52wh", "ATR Extended Gap Up": "ATRGap", "Indices Oversold Bounce": "IOB", "LT Trend ST OS": "LTT",
         "Monday Dip": "MonDip", "Monthly Weak Close": "MWC", "Overbot Vol Spike": "OVS", "Oversold Low Volume": "OLV",
         "SPY QQQ MonFri Reversion": "MonFri", "Sector BO": "SecBO", "St OS Sznl": "StOS", "Weak Close Decent Sznls": "WCDS"}
strat.columns = [SHORT.get(c, c) for c in strat.columns]

frag = pd.read_parquet(ROOT / "data/rd2_fragility.parquet")
dial = frag["63d"].rolling(10).mean().shift(1)
dial.index = pd.to_datetime(dial.index).normalize()

TICK = ["SPY", "QQQ", "IWM", "^VIX", "TLT", "HYG"]
px = pq.read_table(ROOT / "data/master_prices.parquet", columns=["ticker", "date", "Close"],
                   filters=[("ticker", "in", TICK)]).to_pandas().pivot(index="date", columns="ticker", values="Close")
px.index = pd.to_datetime(px.index)
# strategy_daily carries exchange holidays as zero rows; keep only true sessions
keep = strat.index.intersection(px.index[px["SPY"].notna() & px["QQQ"].notna()])
strat = strat.loc[keep]; book = book.loc[keep]
pxk = px.loc[keep].ffill()
fac = pxk[["SPY", "QQQ", "IWM", "TLT", "HYG"]].pct_change(fill_method=None)
vix_lag = pxk["^VIX"].shift(1)
spy_dd_lag = (pxk["SPY"] / pxk["SPY"].rolling(252).max() - 1).shift(1)

# ------------------------------------------------------------------ momentum factor from master_prices
print("building momentum factor (12-1, quintiles, monthly) from sector_map universe ...")
sm = pd.read_parquet(ROOT / "data/sector_map.parquet")
uni = sorted(set(sm["ticker"]) - set(TICK))
mp = pq.read_table(ROOT / "data/master_prices.parquet", columns=["ticker", "date", "Close"],
                   filters=[("ticker", "in", uni)]).to_pandas()
close = mp.pivot(index="date", columns="ticker", values="Close").sort_index()
close.index = pd.to_datetime(close.index)
close = close.reindex(px.index[px["SPY"].notna()])          # equity sessions only (crypto/futures rows would insert weekend NaNs)
close = close[close.index >= "2002-01-01"]
mret = close.resample("ME").last()
mom = mret.shift(1) / mret.shift(12) - 1          # 12-1 momentum known at month end
daily_ret = close.pct_change(fill_method=None)
mom_days = []
month_ends = mret.index
for i in range(12, len(month_ends) - 1):
    me = month_ends[i]
    m = mom.loc[me].dropna()
    ok = mret.loc[me].reindex(m.index) > 5
    m = m[ok]
    if len(m) < 60:
        continue
    lo, hi = m.quantile(0.2), m.quantile(0.8)
    longs, shorts = m.index[m >= hi], m.index[m <= lo]
    nxt = daily_ret[(daily_ret.index > me) & (daily_ret.index <= month_ends[i + 1])]
    leg = nxt[longs].mean(axis=1) - nxt[shorts].mean(axis=1)
    mom_days.append(leg)
MOM = pd.concat(mom_days).sort_index().rename("MOM")
MOM = MOM[~MOM.index.duplicated()]
print(f"MOM factor {MOM.index.min().date()}..{MOM.index.max().date()} N={len(MOM)} ann mean {MOM.mean()*252*100:.1f}% vol {MOM.std()*np.sqrt(252)*100:.1f}% "
      f"corr(SPY) {MOM.corr(fac['SPY']):.2f}  universe {close.shape[1]} names")
OUT["mom_factor"] = dict(start=str(MOM.index.min().date()), end=str(MOM.index.max().date()), n=len(MOM), ann_mean_pct=MOM.mean() * 252 * 100,
                         ann_vol_pct=MOM.std() * np.sqrt(252) * 100, corr_spy=MOM.corr(fac["SPY"]), universe_names=int(close.shape[1]),
                         note="12-1 cross-sectional momentum, long top quintile / short bottom quintile, equal weight, monthly rebalance, price>5, "
                              "universe = data/sector_map.parquet tickers present in master_prices (survivorship: today's names only, so the "
                              "short leg is biased toward names that survived)")
MOM.to_frame().to_parquet(HERE / "cross_strategy_regime_mom_factor.parquet")

# ------------------------------------------------------------------ frame
F = pd.concat([fac, MOM], axis=1).reindex(strat.index)
F["QMS"] = F["QQQ"] - F["SPY"]; F["IMS"] = F["IWM"] - F["SPY"]
D = pd.DataFrame({"book": book, "dial": dial.reindex(strat.index), "vix": vix_lag.reindex(strat.index), "spydd": spy_dd_lag.reindex(strat.index)})
D["dial_b"] = pd.cut(D["dial"], [0, 30, 50, 65, 101], labels=["d<30", "d30-50", "d50-65", "d65+"], right=False)
D["vix_b"] = pd.cut(D["vix"], [0, 15, 20, 30, 200], labels=["v<15", "v15-20", "v20-30", "v30+"], right=False)
D["dd_b"] = pd.cut(D["spydd"], [-1, -0.10, -0.05, -0.02, 0.01], labels=["dd>10%", "dd5-10%", "dd2-5%", "dd<2%"])
D["month"] = D.index.month
WIN = D.index >= "2016-07-20"
D10 = D.index >= "2010-01-01"


def corr_stats(idx, min_active=0.10):
    sub = strat.reindex(idx)
    act = sub.loc[:, (sub != 0).mean() > min_active]
    if act.shape[1] < 2:
        return dict(n_active=int(act.shape[1]), avg_corr=np.nan, eff_n_vol=np.nan, eff_n_eq=np.nan, top_pairs=[])
    c = act.corr().fillna(0)
    off = c.values[np.triu_indices(len(c), 1)]
    w = act.std() / act.std().sum()
    eff_v = 1 / float(w.values @ c.values @ w.values)
    we = np.ones(len(c)) / len(c)
    eff_e = 1 / float(we @ c.values @ we)
    pairs = sorted([(c.columns[i], c.columns[j], float(c.iloc[i, j])) for i, j in zip(*np.triu_indices(len(c), 1))], key=lambda x: -x[2])[:5]
    return dict(n_active=int(act.shape[1]), avg_corr=float(np.mean(off)), med_corr=float(np.median(off)), frac_gt_02=float((off > 0.2).mean()),
                eff_n_vol=eff_v, eff_n_eq=eff_e, top_pairs=[f"{a}-{b} {r:.2f}" for a, b, r in pairs])


def beta_stats(idx):
    g = D.loc[idx]; s = F.loc[idx, "SPY"]
    m = pd.concat([g["book"], s], axis=1).dropna()
    if len(m) < 20:
        return dict(beta=np.nan, r2=np.nan)
    b = np.polyfit(m["SPY"], m["book"], 1)[0]
    r2 = b**2 * m["SPY"].var() / m["book"].var()
    dn = m[m["SPY"] < 0]; up = m[m["SPY"] > 0]
    bd = np.polyfit(dn["SPY"], dn["book"], 1)[0] if len(dn) > 15 else np.nan
    bu = np.polyfit(up["SPY"], up["book"], 1)[0] if len(up) > 15 else np.nan
    return dict(beta=float(b), r2=float(r2), beta_down=float(bd), beta_up=float(bu), spy_ann_pct=float(m["SPY"].mean() * 252 * 100),
                book_sharpe=float(m["book"].mean() / m["book"].std() * np.sqrt(252)), book_bps=float(m["book"].mean() * 1e4), book_sd_bps=float(m["book"].std() * 1e4))


# ------------------------------------------------------------------ A. regime-conditional
print("\n=== A. regime-conditional correlation structure (2016-07+ for dial; 2010+ for VIX / SPY-dd / month) ===")
OUT["regime"] = {}
for key, mask0 in [("dial_b", WIN), ("vix_b", D10), ("dd_b", D10), ("month", D10)]:
    rows = []
    for lab, g in D[mask0].groupby(key, observed=True):
        cs = corr_stats(g.index); bs = beta_stats(g.index)
        rows.append(dict(regime=str(lab), days=len(g), **cs, **bs))
    R = pd.DataFrame(rows)
    print(f"\n-- by {key} --")
    print(R.drop(columns=["top_pairs"]).to_string(index=False))
    for r in rows:
        print(f"   {r['regime']:>8s} top pairs: {r['top_pairs']}")
    OUT["regime"][key] = rows
# joint dial x VIX (is the dial adding anything beyond VIX for correlation?)
rows = []
for (a, b), g in D[WIN].groupby([pd.cut(D.loc[WIN, "dial"], [0, 50, 101], labels=["d<50", "d50+"], right=False),
                                 pd.cut(D.loc[WIN, "vix"], [0, 20, 200], labels=["v<20", "v20+"], right=False)], observed=True):
    if len(g) < 40:
        continue
    cs = corr_stats(g.index); bs = beta_stats(g.index)
    rows.append(dict(dial=str(a), vix=str(b), days=len(g), avg_corr=cs["avg_corr"], eff_n_vol=cs["eff_n_vol"], beta=bs["beta"], r2=bs["r2"],
                     beta_down=bs["beta_down"], book_sharpe=bs["book_sharpe"], spy_ann_pct=bs["spy_ann_pct"]))
print("\n-- dial x VIX joint --"); print(pd.DataFrame(rows).to_string(index=False))
OUT["regime"]["dial_x_vix"] = rows

# ------------------------------------------------------------------ B. rolling
print("\n=== B. rolling 126d avg pairwise corr / effective N / book beta(63d) by year ===")
roll = []
idx_all = D[D10].index
for i in range(126, len(idx_all), 5):
    win = idx_all[i - 126:i]
    cs = corr_stats(win, min_active=0.10)
    roll.append(dict(date=win[-1], avg_corr=cs["avg_corr"], eff_n=cs["eff_n_vol"], n_active=cs["n_active"]))
RB = pd.DataFrame(roll).set_index("date")
cov63 = D["book"].rolling(63).cov(F["SPY"]); var63 = F["SPY"].rolling(63).var()
beta63 = (cov63 / var63)
RB["beta63"] = beta63.reindex(RB.index)
RB["dial"] = D["dial"].reindex(RB.index)
yr = RB.groupby(RB.index.year).agg(avg_corr=("avg_corr", "mean"), corr_max=("avg_corr", "max"), eff_n=("eff_n", "mean"), eff_n_min=("eff_n", "min"),
                                   beta63=("beta63", "mean"), beta63_max=("beta63", "max"), n_active=("n_active", "mean"))
print(yr.round(2).to_string())
OUT["rolling_by_year"] = yr.round(3).reset_index().rename(columns={"date": "year"}).to_dict("records")
low = RB.sort_values("eff_n").head(40)
seen, lows = [], []
for d, r in low.iterrows():
    if any(abs((d - s).days) < 90 for s in seen):
        continue
    seen.append(d); lows.append(dict(date=str(d.date()), eff_n=r.eff_n, avg_corr=r.avg_corr, beta63=r.beta63, dial=r.dial))
    if len(lows) >= 8:
        break
print("lowest effective-N windows:"); print(pd.DataFrame(lows).round(2).to_string(index=False))
OUT["lowest_eff_n_windows"] = lows
# corr of rolling eff N with dial / VIX (does the dial track the collapse?)
RB["vix"] = D["vix"].reindex(RB.index)
m = RB[RB.index >= "2016-07-20"].dropna()
OUT["rolling_vs_state"] = dict(spearman_effn_dial=float(m["eff_n"].rank().corr(m["dial"].rank())), spearman_effn_vix=float(m["eff_n"].rank().corr(m["vix"].rank())),
                               spearman_beta_dial=float(m["beta63"].rank().corr(m["dial"].rank())), spearman_beta_vix=float(m["beta63"].rank().corr(m["vix"].rank())),
                               spearman_corr_dial=float(m["avg_corr"].rank().corr(m["dial"].rank())), spearman_corr_vix=float(m["avg_corr"].rank().corr(m["vix"].rank())))
print("rolling state vs dial/VIX (spearman):", {k: round(v, 3) for k, v in OUT["rolling_vs_state"].items()})

# ------------------------------------------------------------------ C. pairs that become one bet
print("\n=== C. pair correlations: calm vs stress (days where both strategies were active are what the corr sees) ===")
core = [c for c in strat.columns if (strat.loc[WIN, c] != 0).mean() > 0.08]
states = {"calm(d<50&v<20)": WIN & (D["dial"] < 50) & (D["vix"] < 20), "dial>=50": WIN & (D["dial"] >= 50), "dial>=65": WIN & (D["dial"] >= 65),
          "vix>=25": D10 & (D["vix"] >= 25), "spy_dd<-5%": D10 & (D["spydd"] < -0.05), "spy_dd<-10%": D10 & (D["spydd"] < -0.10)}
rows = []
for a, b in combinations(core, 2):
    rec = dict(pair=f"{a}-{b}")
    for lab, m in states.items():
        sub = strat.loc[m, [a, b]]
        both = sub[(sub != 0).all(axis=1)]
        rec[lab] = float(sub[a].corr(sub[b])) if len(sub) > 30 else np.nan
        rec[lab + "_nboth"] = int(len(both))
    rows.append(rec)
P = pd.DataFrame(rows)
P["delta_d50"] = P["dial>=50"] - P["calm(d<50&v<20)"]; P["delta_v25"] = P["vix>=25"] - P["calm(d<50&v<20)"]; P["delta_dd5"] = P["spy_dd<-5%"] - P["calm(d<50&v<20)"]
cols = ["pair", "calm(d<50&v<20)", "dial>=50", "dial>=65", "vix>=25", "spy_dd<-5%", "spy_dd<-10%", "delta_d50", "delta_v25", "delta_dd5"]
Ps = P.sort_values("dial>=50", ascending=False)
print(Ps[cols].head(15).round(2).to_string(index=False))
print("\nmost-increased in stress (delta vs calm, by dial>=50):"); print(P.sort_values("delta_d50", ascending=False)[cols].head(8).round(2).to_string(index=False))
OUT["pairs"] = P.round(3).to_dict("records")
onebet = P[(P["dial>=50"] > 0.2) | (P["vix>=25"] > 0.2) | (P["spy_dd<-5%"] > 0.2)][cols]
OUT["pairs_one_bet"] = onebet.round(3).to_dict("records")

# ------------------------------------------------------------------ D. betas / factor loadings by regime
print("\n=== D. per-strategy betas (per active day) to SPY / QQQ / IWM / MOM by regime, and multi-factor loadings ===")


def ols(y, X):
    X = np.column_stack([np.ones(len(y)), X]); b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b; dof = max(len(y) - X.shape[1], 1); s2 = res @ res / dof
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
    r2 = 1 - res.var() / y.var() if y.var() > 0 else np.nan
    return b, se, r2


rows = []
reg_masks = {"all16": WIN, "d<50": WIN & (D["dial"] < 50), "d50+": WIN & (D["dial"] >= 50), "d65+": WIN & (D["dial"] >= 65), "v<20": D10 & (D["vix"] < 20), "v25+": D10 & (D["vix"] >= 25),
             "dd<-5%": D10 & (D["spydd"] < -0.05)}
for name in list(strat.columns) + ["BOOK"]:
    y_all = book if name == "BOOK" else strat[name]
    for lab, m in reg_masks.items():
        y = y_all[m]; X = F.loc[m, ["SPY", "QQQ", "IWM", "MOM"]]
        ok = X.notna().all(axis=1) & y.notna()
        y = y[ok]; X = X[ok]
        act = (y != 0).mean() if name != "BOOK" else 1.0
        if len(y) < 40 or act < 0.03:
            continue
        rec = dict(strategy=name, regime=lab, days=len(y), active_share=float(act))
        for f in ["SPY", "QQQ", "IWM", "MOM"]:
            b, se, r2 = ols(y.values, X[[f]].values)
            rec[f"b_{f}"] = float(b[1] / act); rec[f"t_{f}"] = float(b[1] / se[1]); rec[f"r2_{f}"] = float(r2)
        b, se, r2 = ols(y.values, np.column_stack([X["SPY"], X["QQQ"] - X["SPY"], X["IWM"] - X["SPY"], X["MOM"]]))
        rec.update(mf_SPY=float(b[1] / act), mf_QMS=float(b[2] / act), mf_IMS=float(b[3] / act), mf_MOM=float(b[4] / act),
                   mf_t_SPY=float(b[1] / se[1]), mf_t_QMS=float(b[2] / se[2]), mf_t_IMS=float(b[3] / se[3]), mf_t_MOM=float(b[4] / se[4]), mf_r2=float(r2))
        # downside beta (SPY<0 days)
        dn = X["SPY"] < 0
        if dn.sum() > 20:
            bd, _, _ = ols(y[dn].values, X.loc[dn, ["SPY"]].values); rec["b_SPY_down"] = float(bd[1] / act)
        rows.append(rec)
BT = pd.DataFrame(rows)
show = ["strategy", "regime", "days", "active_share", "b_SPY", "b_QQQ", "b_IWM", "b_MOM", "b_SPY_down", "r2_SPY", "r2_QQQ", "r2_IWM", "mf_SPY", "mf_QMS", "mf_IMS", "mf_MOM", "mf_t_QMS", "mf_t_IMS", "mf_t_MOM", "mf_r2"]
print(BT[BT.regime.isin(["all16", "d<50", "d50+", "d65+"])][show].round(2).to_string(index=False))
print("\n-- VIX / drawdown regimes --")
print(BT[BT.regime.isin(["v<20", "v25+", "dd<-5%"])][show].round(2).to_string(index=False))
OUT["betas"] = BT.round(4).to_dict("records")

# which single factor explains the book best, by regime (R2) — the instrument question
print("\n-- which factor explains the BOOK best (R2) by regime --")
bk = BT[BT.strategy == "BOOK"][["regime", "days", "r2_SPY", "r2_QQQ", "r2_IWM", "r2_MOM", "mf_r2", "b_SPY", "b_QQQ", "b_IWM", "b_SPY_down"]]
print(bk.round(3).to_string(index=False))
OUT["book_factor_r2"] = bk.round(4).to_dict("records")

# sub-book (dip-buy family + OLV) loadings by regime
DIP = ["WCDS", "MonFri", "MonDip", "IOB", "MWC", "3xBear"]
subbook = strat[[c for c in DIP + ["OLV"] if c in strat.columns]].sum(axis=1)
rows = []
for lab, m in reg_masks.items():
    y = subbook[m]; X = F.loc[m, ["SPY", "QQQ", "IWM", "MOM"]]; ok = X.notna().all(axis=1); y = y[ok]; X = X[ok]
    rec = dict(regime=lab, days=len(y), sub_share_of_book_var=float(y.var() / book[m][ok].var()))
    for f in ["SPY", "QQQ", "IWM"]:
        b, se, r2 = ols(y.values, X[[f]].values); rec[f"b_{f}"] = float(b[1]); rec[f"r2_{f}"] = float(r2)
    rest = (book[m][ok] - y)
    b, se, r2 = ols(rest.values, X[["SPY"]].values); rec["rest_b_SPY"] = float(b[1]); rec["rest_r2_SPY"] = float(r2)
    rows.append(rec)
SB = pd.DataFrame(rows); print("\n-- dip-buy family + OLV sub-book vs rest of book --"); print(SB.round(3).to_string(index=False))
OUT["subbook_loadings"] = SB.round(4).to_dict("records")

# ------------------------------------------------------------------ E. seasonality of correlation / beta by month (2010+)
print("\n=== E. month-of-year: book beta, R2, avg corr, Sharpe (2010+) ===")
mo = [r for r in OUT["regime"]["month"]]
print(pd.DataFrame(mo)[["regime", "days", "avg_corr", "eff_n_vol", "beta", "r2", "beta_down", "book_sharpe", "book_bps"]].round(3).to_string(index=False))

json.dump(OUT, open(HERE / "cross_strategy_regime_results_1_corr.json", "w"), indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
print("\nwrote cross_strategy_regime_results_1_corr.json")
