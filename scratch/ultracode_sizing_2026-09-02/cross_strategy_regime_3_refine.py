"""cross_strategy_regime step 3: episode tables for the primary hedge specs,
two refined arming rules suggested by step 1's dial x VIX split (hedge the
complacent state: dial>=50 AND VIX<20 / VIX pctile<60; release on a VIX spike),
episode bootstrap on the PIT primary, correlation vs signal flow (open legs /
active strategies), per-strategy contribution to the hedge on armed days, and
per-instrument armed-day variance reduction.

The refined rules are IN-SAMPLE refinements found after looking at step 1 and
must be treated as candidates for a pre-registration, not results.
Writes cross_strategy_regime_results_3_refine.json beside this file.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
NAV = 750_000.0
RNG = np.random.default_rng(99)
pd.set_option("display.width", 260, "display.max_columns", 40, "display.float_format", "{:,.3f}".format)
OUT: dict = {}

# reuse step 2's machinery by importing it as a module would re-run the grid; re-implement the small pieces instead
sd = json.load(open(ROOT / "dist/data/strategy_daily.json"))
dates = pd.to_datetime(sd["dates"])
S = pd.DataFrame(sd["series"], index=dates).fillna(0.0)
strat = S.T.groupby(S.columns.str.split("||", regex=False).str[0]).sum().T / NAV
book = pd.Series(sd["total_flat"], index=dates, dtype=float) / NAV
px = pq.read_table(ROOT / "data/master_prices.parquet", columns=["ticker", "date", "Close"],
                   filters=[("ticker", "in", ["SPY", "QQQ", "IWM", "^VIX"])]).to_pandas().pivot(index="date", columns="ticker", values="Close")
px.index = pd.to_datetime(px.index)
keep = book.index.intersection(px.index[px["SPY"].notna() & px["QQQ"].notna()])
book = book.loc[keep]; strat = strat.loc[keep]
pxk = px.loc[keep].ffill()
fac = pxk[["SPY", "QQQ", "IWM"]].pct_change(fill_method=None)
vix_lag = pxk["^VIX"].shift(1); vix_pct_lag = pxk["^VIX"].rolling(252).rank(pct=True).shift(1)
spydd_lag = (pxk["SPY"] / pxk["SPY"].rolling(252).max() - 1).shift(1)
frag = pd.read_parquet(ROOT / "data/rd2_fragility.parquet")
dial_live = frag["63d"].rolling(10).mean(); dial_live.index = pd.to_datetime(dial_live.index).normalize()
dial_live_lag = dial_live.shift(1).reindex(keep)
dial_pit_lag = pd.read_parquet(HERE / "cross_strategy_regime_pit_dial.parquet")["pit"].shift(1).reindex(keep)
FRICTION_BPS = 2.0


def rolling_beta(y, x, w):
    return (y.rolling(w).cov(x) / x.rolling(w).var()).shift(1).clip(-1, 2)


def arm2(x: pd.Series, on: float, off: float, gate: pd.Series | None = None, gate_release: pd.Series | None = None) -> pd.Series:
    """hysteresis arming on x with an optional entry gate (must be True to arm) and a release trigger (True forces release)."""
    armed = np.zeros(len(x), bool); state = False
    xv = x.values; gv = gate.values if gate is not None else np.ones(len(x), bool); rv = gate_release.values if gate_release is not None else np.zeros(len(x), bool)
    for i, v in enumerate(xv):
        if np.isnan(v):
            armed[i] = state; continue
        if not state and v >= on and gv[i]:
            state = True
        elif state and (v < off or rv[i]):
            state = False
        armed[i] = state
    return pd.Series(armed, index=x.index)


def episodes(armed, gap=21):
    idx = np.where(armed.values)[0]
    if len(idx) == 0:
        return []
    eps = []; s = idx[0]; prev = idx[0]
    for i in idx[1:]:
        if i - prev > gap:
            eps.append((s, prev)); s = i
        prev = i
    eps.append((s, prev)); return eps


def sharpe(r):
    return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan


def maxdd(r):
    eq = r.cumsum(); return float((eq - eq.cummax()).min())


def run(idx, armed, beta, f, label, mult=1.0):
    m = pd.DataFrame({"r": book.reindex(idx), "f": f.reindex(idx), "b": beta.reindex(idx).fillna(0), "a": armed.reindex(idx).astype(float)}).dropna(subset=["r", "f"])
    h = -m.a * mult * m.b * m.f
    ev = (m.a.diff() > 0).astype(float); h = h - ev * FRICTION_BPS / 1e4 * mult * m.b.abs()
    eps = episodes(m.a > 0); rows = []
    for s, e in eps:
        seg = m.iloc[s:e + 1]; hs = h.iloc[s:e + 1]
        rows.append(dict(start=str(seg.index[0].date()), end=str(seg.index[-1].date()), armed_days=int(seg.a.sum()), hedge_usd=float(hs.sum() * NAV), book_usd=float(seg.r.sum() * NAV),
                         spy_ret_pct=float(((1 + seg.f).prod() - 1) * 100), beta_mean=float(seg.loc[seg.a > 0, "b"].mean()), vix_start=float(vix_lag.reindex(seg.index).iloc[0])))
    ep = np.array([x["hedge_usd"] for x in rows]); n = len(ep)
    t = float(ep.mean() / (ep.std(ddof=1) / np.sqrt(n))) if n > 2 and ep.std(ddof=1) > 0 else np.nan
    am = m.a > 0
    yrs = {}
    for y, g in m.groupby(m.index.year):
        if g.a.sum() > 0:
            yrs[int(y)] = dict(hedge_usd=float(h[g.index].sum() * NAV), sh_u=sharpe(g.r), sh_h=sharpe(g.r + h[g.index]))
    return dict(label=label, armed_days=int(am.sum()), n_episodes=n, hedge_usd=float(h.sum() * NAV), ep_mean=float(ep.mean()) if n else 0.0, ep_pos=float((ep > 0).mean()) if n else np.nan,
                t_clustered=t, sharpe_u=sharpe(m.r), sharpe_h=sharpe(m.r + h), maxdd_u=maxdd(m.r) * 100, maxdd_h=maxdd(m.r + h) * 100,
                armed_sh_u=sharpe(m.r[am]) if am.sum() > 20 else np.nan, armed_sh_h=sharpe((m.r + h)[am]) if am.sum() > 20 else np.nan,
                armed_spy_ann=float(m.f[am].mean() * 252 * 100) if am.any() else np.nan, false_alarms=int(sum(1 for x in rows if x["spy_ret_pct"] > 0)),
                loyo_years=len(yrs), loyo_not_worse=int(sum(1 for v in yrs.values() if v["sh_h"] >= v["sh_u"])), loyo_pnl_pos=int(sum(1 for v in yrs.values() if v["hedge_usd"] > 0)),
                episodes=rows, years=yrs)


beta63 = rolling_beta(book, fac["SPY"], 63); beta126 = rolling_beta(book, fac["SPY"], 126)
# shrunk beta: 0.5 x 63d estimate + 0.5 x expanding long-run beta (ex-ante)
lr = (book.expanding(252).cov(fac["SPY"]) / fac["SPY"].expanding(252).var()).shift(1)
beta_shr = (0.5 * beta63 + 0.5 * lr).clip(-1, 2)
wins = {"live": (dial_live_lag, book.index >= "2016-07-20"), "pit": (dial_pit_lag, (book.index >= "2018-01-02") & (book.index <= "2026-07-02"))}

# ------------------------------------------------------------------ 1. episode tables for the primary specs
print("=== 1. episode tables: book / SPY / 63d ===")
OUT["episodes"] = {}
for vn, (ds, w) in wins.items():
    idx = book.index[w]
    for an, (on, off) in {"dial50_h45": (50, 45), "dial65_h60": (65, 60)}.items():
        res = run(idx, arm2(ds.reindex(idx), on, off), beta63, fac["SPY"], f"{vn}|{an}")
        print(f"\n{vn} {an}: {res['n_episodes']} episodes, total ${res['hedge_usd']:,.0f}, t={res['t_clustered']:.2f}, LOYO not-worse {res['loyo_not_worse']}/{res['loyo_years']}")
        print(pd.DataFrame(res["episodes"]).round(2).to_string(index=False))
        print("  by year:", {y: round(v["hedge_usd"]) for y, v in res["years"].items()})
        OUT["episodes"][f"{vn}|{an}"] = res

# ------------------------------------------------------------------ 2. refined arming: hedge the complacent state, release on the spike
print("\n=== 2. refined arming (IN-SAMPLE candidates): dial>=50 gated on VIX<20 at arming, released when VIX>=25 or dial<45 ===")
rows = []
for vn, (ds, w) in wins.items():
    idx = book.index[w]; d = ds.reindex(idx); v = vix_lag.reindex(idx); vp = vix_pct_lag.reindex(idx); dd = spydd_lag.reindex(idx)
    variants = {
        "dial50_h45 (baseline)": arm2(d, 50, 45),
        "dial50_h45 & VIX<20 gate": arm2(d, 50, 45, gate=(v < 20)),
        "dial50_h45 & VIX<20 gate, release VIX>=25": arm2(d, 50, 45, gate=(v < 20), gate_release=(v >= 25)),
        "dial50_h45 & VIX<20 gate, release VIX>=30": arm2(d, 50, 45, gate=(v < 20), gate_release=(v >= 30)),
        "dial50_h45 & VIXpct<60 gate, release VIXpct>=90": arm2(d, 50, 45, gate=(vp < 0.6), gate_release=(vp >= 0.9)),
        "dial50_h45 & SPY within 3% of high gate": arm2(d, 50, 45, gate=(dd > -0.03)),
        "dial50_h45, release SPY dd<-7%": arm2(d, 50, 45, gate_release=(dd < -0.07)),
        "dial65_h60 (baseline)": arm2(d, 65, 60),
        "dial65_h60 & VIX<20 gate, release VIX>=25": arm2(d, 65, 60, gate=(v < 20), gate_release=(v >= 25)),
        "dial50_h45, beta126": None, "dial50_h45, beta shrunk": None,
    }
    for lab, armed in variants.items():
        if armed is None:
            b = beta126 if "126" in lab else beta_shr; armed = arm2(d, 50, 45)
        else:
            b = beta63
        res = run(idx, armed, b, fac["SPY"], f"{vn}|{lab}")
        rows.append({k: v_ for k, v_ in res.items() if k not in ("episodes", "years")} | dict(vintage=vn, rule=lab))
R = pd.DataFrame(rows)
cols = ["vintage", "rule", "armed_days", "n_episodes", "hedge_usd", "ep_mean", "ep_pos", "t_clustered", "sharpe_u", "sharpe_h", "maxdd_u", "maxdd_h", "armed_sh_u", "armed_sh_h", "armed_spy_ann", "false_alarms", "loyo_years", "loyo_not_worse", "loyo_pnl_pos"]
print(R[cols].round(2).to_string(index=False))
OUT["refined"] = R[cols].round(4).to_dict("records")

# ------------------------------------------------------------------ 3. episode bootstrap on the PIT primary (P(total <= 0)), and drop-best-episode
print("\n=== 3. episode bootstrap / drop-best on the primary specs ===")
OUT["bootstrap"] = {}
for key, res in OUT["episodes"].items():
    ep = np.array([x["hedge_usd"] for x in res["episodes"]])
    if len(ep) < 3:
        continue
    sims = np.array([RNG.choice(ep, len(ep), replace=True).sum() for _ in range(5000)])
    drop_best = ep.sum() - ep.max()
    ep2 = np.delete(ep, ep.argmax()); t2 = float(ep2.mean() / (ep2.std(ddof=1) / np.sqrt(len(ep2))))
    print(f"{key:18s}: n={len(ep)} total ${ep.sum():,.0f}  P(boot total<=0) {(sims <= 0).mean():.3f}  drop-best total ${drop_best:,.0f} t={t2:.2f}  best episode ${ep.max():,.0f} ({res['episodes'][ep.argmax()]['start']})")
    OUT["bootstrap"][key] = dict(n=len(ep), total=float(ep.sum()), p_boot_le0=float((sims <= 0).mean()), drop_best_total=float(drop_best), drop_best_t=t2, best_usd=float(ep.max()), best_start=res["episodes"][ep.argmax()]["start"])

# ------------------------------------------------------------------ 4. correlation vs signal flow: open legs / number of active strategies
print("\n=== 4. avg pairwise corr and book beta by signal-flow state (2010+): number of strategies active that day, trailing-5d signal count ===")
led = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet"); led = led[led["PnL_flat_750k"].notna()]
sig5 = led.groupby("Signal Date").size().reindex(pd.bdate_range("2003-01-01", "2026-09-01")).fillna(0).rolling(5).sum().shift(1).reindex(keep)
n_active = (strat != 0).sum(axis=1)
idx10 = book.index[book.index >= "2010-01-01"]


def corr_stats(idx):
    sub = strat.reindex(idx); act = sub.loc[:, (sub != 0).mean() > 0.10]
    if act.shape[1] < 2:
        return np.nan, np.nan
    c = act.corr().fillna(0); off = c.values[np.triu_indices(len(c), 1)]
    w = act.std() / act.std().sum(); return float(np.mean(off)), 1 / float(w.values @ c.values @ w.values)


rows = []
for lab, s, bins in [("n_active", n_active, [0, 2, 4, 6, 20]), ("sig5", sig5, [-1, 2, 6, 15, 999])]:
    for b, g in pd.Series(s.reindex(idx10)).groupby(pd.cut(s.reindex(idx10), bins), observed=True):
        ac, en = corr_stats(g.index); r = book.reindex(g.index); f = fac["SPY"].reindex(g.index)
        rows.append(dict(state=lab, bucket=str(b), days=len(g), avg_corr=ac, eff_n=en, beta=float(np.polyfit(f.fillna(0), r, 1)[0]), book_sharpe=sharpe(r), book_bps=float(r.mean() * 1e4), sd_bps=float(r.std() * 1e4)))
F4 = pd.DataFrame(rows); print(F4.round(3).to_string(index=False)); OUT["signal_flow"] = F4.round(4).to_dict("records")

# ------------------------------------------------------------------ 5. per-strategy contribution on armed days (live dial50_h45): who does the hedge actually hedge?
print("\n=== 5. per-strategy SPY covariance share on armed days (live dial50_h45) vs unarmed ===")
idx = book.index[wins["live"][1]]; armed = arm2(dial_live_lag.reindex(idx), 50, 45)
rows = []
for lab, m in [("armed", armed), ("unarmed", ~armed)]:
    sub = strat.reindex(idx)[m.values]; f = fac["SPY"].reindex(sub.index)
    cov = sub.apply(lambda c: c.cov(f)); tot = cov.sum()
    for s in sub.columns:
        if (sub[s] != 0).mean() > 0.03:
            rows.append(dict(state=lab, strategy=s, active=float((sub[s] != 0).mean()), beta_raw=float(cov[s] / f.var()), cov_share=float(cov[s] / tot), pnl_bps=float(sub[s].mean() * 1e4),
                             sharpe_active=sharpe(sub[s][sub[s] != 0])))
C5 = pd.DataFrame(rows).sort_values(["state", "cov_share"], ascending=[True, False]); print(C5.round(3).to_string(index=False)); OUT["contrib"] = C5.round(4).to_dict("records")

# ------------------------------------------------------------------ 6. instrument: armed-day variance reduction vs drift, live dial50_h45
print("\n=== 6. instrument anatomy on armed days (live dial50_h45, 63d beta): sd reduction vs drift ===")
rows = []
for f in ["SPY", "QQQ", "IWM"]:
    b = rolling_beta(book, fac[f], 63).reindex(idx).fillna(0); ff = fac[f].reindex(idx); r = book.reindex(idx); a = armed.values
    h = -b * ff; ra = r[a]; ha = h[a]
    rows.append(dict(instrument=f, beta_mean=float(b[a].mean()), factor_ann_armed=float(ff[a].mean() * 252 * 100), factor_ann_unarmed=float(ff[~a].mean() * 252 * 100),
                     hedge_usd=float(ha.sum() * NAV), drift_usd=float((-b[a] * ff[a].mean()).sum() * NAV), sd_book=float(ra.std() * 1e4), sd_hedged=float((ra + ha).std() * 1e4),
                     corr_book_factor_armed=float(ra.corr(ff[a])), r2_armed=float(ra.corr(ff[a]) ** 2)))
I6 = pd.DataFrame(rows); print(I6.round(3).to_string(index=False)); OUT["instrument_anatomy"] = I6.round(4).to_dict("records")

json.dump(OUT, open(HERE / "cross_strategy_regime_results_3_refine.json", "w"), indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
print("\nwrote cross_strategy_regime_results_3_refine.json")
