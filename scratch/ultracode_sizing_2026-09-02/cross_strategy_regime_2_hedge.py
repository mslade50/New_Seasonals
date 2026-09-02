"""cross_strategy_regime step 2: the hedge question, replayed on the daily book.

Grid: target {book, dip-buy family + OLV sub-book} x instrument {SPY, QQQ, IWM,
adaptive-by-R2} x beta window {42, 63, 126} x arming {dial>=50 rel 45,
dial>=65 rel 60, dial>=50 no hysteresis, VIX>=25 rel 20 (non-dial control),
VIX 252d-pctile>=80 rel <60 (control), SPY dd<-5% rel >-3% (control), always}
x dial vintage {live current-weights 2016-07+, PIT vintage-lagged 2018+,
live on the PIT window}. Hedge PnL = -armed x beta_hat x factor return on NAV,
minus 2 bps x |beta_hat| per arm event. Reports armed-episode PnL, clustered t
(episodes = armed runs separated by >= 21 unarmed sessions), full-sample Sharpe
hedged vs unhedged (equal-vol: Sharpe is scale-free, so the equal-vol annual
return is the hedged Sharpe x unhedged vol), maxDD, LOYO years, calm-tape carry,
a circular-shift placebo, and a drift-vs-variance decomposition of the hedge.

Basis: dist/data/strategy_daily.json total_flat (flat $750k, ends 2026-08-07).
Writes cross_strategy_regime_results_2_hedge.json beside this file.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
NAV = 750_000.0
FRICTION_BPS = 2.0
RNG = np.random.default_rng(2026)
pd.set_option("display.width", 260, "display.max_columns", 40, "display.float_format", "{:,.3f}".format)
OUT: dict = {}

sd = json.load(open(ROOT / "dist/data/strategy_daily.json"))
dates = pd.to_datetime(sd["dates"])
S = pd.DataFrame(sd["series"], index=dates).fillna(0.0)
strat = S.T.groupby(S.columns.str.split("||", regex=False).str[0]).sum().T / NAV
book = pd.Series(sd["total_flat"], index=dates, dtype=float) / NAV
DIP = ["Weak Close Decent Sznls", "SPY QQQ MonFri Reversion", "Monday Dip", "Indices Oversold Bounce", "Monthly Weak Close", "3x Bear ETF Overbot Fade", "Oversold Low Volume"]
subbook = strat[[c for c in DIP if c in strat.columns]].sum(axis=1)

px = pq.read_table(ROOT / "data/master_prices.parquet", columns=["ticker", "date", "Close"],
                   filters=[("ticker", "in", ["SPY", "QQQ", "IWM", "^VIX"])]).to_pandas().pivot(index="date", columns="ticker", values="Close")
px.index = pd.to_datetime(px.index)
keep = book.index.intersection(px.index[px["SPY"].notna() & px["QQQ"].notna()])   # exchange holidays carry a VIX-only row
book = book.loc[keep]; subbook = subbook.loc[keep]
pxk = px.loc[keep].ffill()
fac = pxk[["SPY", "QQQ", "IWM"]].pct_change(fill_method=None)
vix_lag = pxk["^VIX"].shift(1)
vix_pct_lag = pxk["^VIX"].rolling(252).rank(pct=True).shift(1)
spydd_lag = (pxk["SPY"] / pxk["SPY"].rolling(252).max() - 1).shift(1)

frag = pd.read_parquet(ROOT / "data/rd2_fragility.parquet")
dial_live = frag["63d"].rolling(10).mean(); dial_live.index = pd.to_datetime(dial_live.index).normalize()
pitf = pd.read_parquet(HERE / "cross_strategy_regime_pit_dial.parquet")
dial_pit = pitf["pit"]
dial_live_lag = dial_live.shift(1).reindex(keep)
dial_pit_lag = dial_pit.shift(1).reindex(keep)

# ------------------------------------------------------------------ helpers


def rolling_beta(y: pd.Series, x: pd.Series, w: int) -> pd.Series:
    cov = y.rolling(w).cov(x); var = x.rolling(w).var()
    return (cov / var).shift(1).clip(-1, 2)


def rolling_r2(y: pd.Series, x: pd.Series, w: int) -> pd.Series:
    return (y.rolling(w).corr(x) ** 2).shift(1)


def arm_hysteresis(x: pd.Series, on: float, off: float) -> pd.Series:
    """x is already lag-1 (known at the prior close). armed_t applies to day t."""
    armed = np.zeros(len(x), bool); state = False
    for i, v in enumerate(x.values):
        if np.isnan(v):
            armed[i] = state; continue
        if not state and v >= on:
            state = True
        elif state and v < off:
            state = False
        armed[i] = state
    return pd.Series(armed, index=x.index)


def arm_dd(x: pd.Series, on: float, off: float) -> pd.Series:
    armed = np.zeros(len(x), bool); state = False
    for i, v in enumerate(x.values):
        if np.isnan(v):
            armed[i] = state; continue
        if not state and v <= on:
            state = True
        elif state and v > off:
            state = False
        armed[i] = state
    return pd.Series(armed, index=x.index)


def episodes(armed: pd.Series, gap: int = 21) -> list[tuple[int, int]]:
    idx = np.where(armed.values)[0]
    if len(idx) == 0:
        return []
    eps = []; s = idx[0]; prev = idx[0]
    for i in idx[1:]:
        if i - prev > gap:
            eps.append((s, prev)); s = i
        prev = i
    eps.append((s, prev))
    return eps


def sharpe(r: pd.Series) -> float:
    return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan


def maxdd(r: pd.Series) -> float:
    eq = r.cumsum(); return float((eq - eq.cummax()).min())


def evaluate(r: pd.Series, target: pd.Series, f: pd.Series, beta: pd.Series, armed: pd.Series, mult: float = 1.0, label: str = "", detail: bool = False) -> dict:
    m = pd.DataFrame({"r": r, "f": f, "b": beta, "a": armed.astype(float)}).dropna(subset=["r", "f"])
    m["b"] = m["b"].fillna(0.0)
    h = -m["a"] * mult * m["b"] * m["f"]
    arm_events = ((m["a"].diff() > 0) | ((m["a"] == 1) & (m["a"].shift(1).isna()))).astype(float)
    fr = arm_events * (FRICTION_BPS / 1e4) * (mult * m["b"].abs())
    h = h - fr
    hedged = m["r"] + h
    eps = episodes(m["a"] > 0)
    ep_rows = []
    for s, e in eps:
        seg = m.iloc[s:e + 1]; hs = h.iloc[s:e + 1]
        ep_rows.append(dict(start=str(seg.index[0].date()), end=str(seg.index[-1].date()), days=int(len(seg)), armed_days=int(seg["a"].sum()),
                            hedge_usd=float(hs.sum() * NAV), book_usd=float(seg["r"].sum() * NAV), hedged_usd=float((seg["r"] + hs).sum() * NAV),
                            spy_ret_pct=float(((1 + seg["f"]).prod() - 1) * 100), beta_mean=float(seg.loc[seg["a"] > 0, "b"].mean()),
                            book_sd_bps=float(seg["r"].std() * 1e4), hedged_sd_bps=float((seg["r"] + hs).std() * 1e4)))
    ep_pnl = np.array([x["hedge_usd"] for x in ep_rows])
    n_ep = len(ep_pnl)
    t_cl = float(ep_pnl.mean() / (ep_pnl.std(ddof=1) / np.sqrt(n_ep))) if n_ep > 2 and ep_pnl.std(ddof=1) > 0 else np.nan
    armed_mask = m["a"] > 0
    yrs = []
    for y, g in m.groupby(m.index.year):
        if len(g) < 60:
            continue
        hy = h.loc[g.index]
        yrs.append(dict(year=int(y), armed_days=int(g["a"].sum()), hedge_usd=float(hy.sum() * NAV), sharpe_unhedged=sharpe(g["r"]), sharpe_hedged=sharpe(g["r"] + hy)))
    yrs_armed = [y for y in yrs if y["armed_days"] > 0]
    ru = m["r"]; rh = hedged
    calm = (~armed_mask)
    # calm-tape carry: what an ALWAYS hedge would have cost on the days this policy did not arm (bps/day), and false-alarm episodes
    always_h = -mult * m["b"] * m["f"]
    false_alarms = [x for x in ep_rows if x["spy_ret_pct"] > 0]
    res = dict(label=label, days=int(len(m)), armed_days=int(armed_mask.sum()), armed_share=float(armed_mask.mean()), n_episodes=n_ep,
               hedge_total_usd=float(h.sum() * NAV), hedge_bps_per_armed_day=float(h[armed_mask].mean() * 1e4) if armed_mask.any() else 0.0,
               friction_usd=float(fr.sum() * NAV), ep_mean_usd=float(ep_pnl.mean()) if n_ep else 0.0, ep_median_usd=float(np.median(ep_pnl)) if n_ep else 0.0,
               ep_pos_share=float((ep_pnl > 0).mean()) if n_ep else np.nan, t_clustered=t_cl,
               sharpe_unhedged=sharpe(ru), sharpe_hedged=sharpe(rh), ann_unhedged_pct=float(ru.mean() * 252 * 100), ann_hedged_pct=float(rh.mean() * 252 * 100),
               vol_unhedged_pct=float(ru.std() * np.sqrt(252) * 100), vol_hedged_pct=float(rh.std() * np.sqrt(252) * 100),
               ann_hedged_equal_vol_pct=float(sharpe(rh) * ru.std() * np.sqrt(252) * 100) if ru.std() > 0 else np.nan,
               maxdd_unhedged_pct=maxdd(ru) * 100, maxdd_hedged_pct=maxdd(rh) * 100, worst_unhedged_pct=float(ru.min() * 100), worst_hedged_pct=float(rh.min() * 100),
               armed_sharpe_unhedged=sharpe(ru[armed_mask]) if armed_mask.sum() > 20 else np.nan, armed_sharpe_hedged=sharpe(rh[armed_mask]) if armed_mask.sum() > 20 else np.nan,
               armed_sd_unhedged_bps=float(ru[armed_mask].std() * 1e4) if armed_mask.any() else np.nan, armed_sd_hedged_bps=float(rh[armed_mask].std() * 1e4) if armed_mask.any() else np.nan,
               armed_spy_ann_pct=float(m.loc[armed_mask, "f"].mean() * 252 * 100) if armed_mask.any() else np.nan,
               drift_component_usd=float((-mult * m.loc[armed_mask, "b"] * m.loc[armed_mask, "f"].mean()).sum() * NAV) if armed_mask.any() else 0.0,
               calm_carry_bps_per_day=float(always_h[calm].mean() * 1e4) if calm.any() else np.nan, calm_carry_usd_per_year=float(always_h[calm].mean() * 252 * NAV) if calm.any() else np.nan,
               false_alarm_episodes=len(false_alarms), false_alarm_usd=float(sum(x["hedge_usd"] for x in false_alarms)),
               loyo_years_with_arming=len(yrs_armed), loyo_years_hedged_not_worse=int(sum(1 for y in yrs_armed if y["sharpe_hedged"] >= y["sharpe_unhedged"] - 1e-9)),
               loyo_years_hedge_pnl_pos=int(sum(1 for y in yrs_armed if y["hedge_usd"] > 0)), beta_mean_armed=float(m.loc[armed_mask, "b"].mean()) if armed_mask.any() else np.nan)
    if detail:
        res["episodes"] = ep_rows; res["years"] = yrs
    return res


# ------------------------------------------------------------------ build the grid
targets = {"book": book, "subbook": subbook}
windows = [42, 63, 126]
betas = {(t, f, w): rolling_beta(targets[t], fac[f], w) for t in targets for f in fac.columns for w in windows}
r2s = {(t, f): rolling_r2(targets[t], fac[f], 126) for t in targets for f in fac.columns}


def adaptive(t: str, w: int):
    """instrument = argmax trailing-126d R2 of the target on {SPY,QQQ,IWM}; beta from window w on that instrument."""
    R = pd.DataFrame({f: r2s[(t, f)] for f in fac.columns}).reindex(keep)
    pick = R.fillna(-1.0).idxmax(axis=1)
    b = pd.Series([betas[(t, f, w)].get(d, np.nan) for d, f in zip(pick.index, pick.values)], index=pick.index)
    f = pd.Series([fac.at[d, f] for d, f in zip(pick.index, pick.values)], index=pick.index)
    return b, f, pick


vintages = {"live": (dial_live_lag, book.index >= "2016-07-20"),
            "pit": (dial_pit_lag, (book.index >= "2018-01-02") & (book.index <= "2026-07-02")),
            "live_pitwin": (dial_live_lag, (book.index >= "2018-01-02") & (book.index <= "2026-07-02"))}
rows = []
primary_detail = {}
for vn, (dseries, win) in vintages.items():
    idx = book.index[win]
    arms = {"dial50_h45": arm_hysteresis(dseries.reindex(idx), 50, 45), "dial65_h60": arm_hysteresis(dseries.reindex(idx), 65, 60),
            "dial50_nohyst": (dseries.reindex(idx) >= 50), "dial40_h35": arm_hysteresis(dseries.reindex(idx), 40, 35),
            "vix25_h20": arm_hysteresis(vix_lag.reindex(idx), 25, 20), "vixpct80_h60": arm_hysteresis(vix_pct_lag.reindex(idx), 0.80, 0.60),
            "spydd5_h3": arm_dd(spydd_lag.reindex(idx), -0.05, -0.03), "always": pd.Series(True, index=idx)}
    for t in targets:
        for f in list(fac.columns) + ["ADAPT"]:
            for w in windows:
                if f == "ADAPT":
                    b, ff, _ = adaptive(t, w); b = b.reindex(idx); ff = ff.reindex(idx)
                else:
                    b = betas[(t, f, w)].reindex(idx); ff = fac[f].reindex(idx)
                for an, armed in arms.items():
                    if vn != "live" and an in ("vix25_h20", "vixpct80_h60", "spydd5_h3", "always") and vn == "live_pitwin":
                        pass
                    lab = f"{vn}|{t}|{f}|{w}|{an}"
                    detail = (t == "book" and f == "SPY" and w == 63 and an in ("dial50_h45", "dial65_h60", "vix25_h20", "always"))
                    res = evaluate(book.reindex(idx), targets[t].reindex(idx), ff, b, armed, 1.0, lab, detail=detail)
                    res.update(vintage=vn, target=t, instrument=f, window=w, arming=an, mult=1.0)
                    if detail:
                        primary_detail[lab] = {k: res.pop(k) for k in ("episodes", "years")}
                    rows.append(res)
    # hedge-ratio multiples on the primary spec
    for mult in [0.5, 1.5, 2.0]:
        for an in ["dial50_h45", "dial65_h60"]:
            b = betas[("book", "SPY", 63)].reindex(idx)
            res = evaluate(book.reindex(idx), book.reindex(idx), fac["SPY"].reindex(idx), b, arms[an], mult, f"{vn}|book|SPY|63|{an}|x{mult}")
            res.update(vintage=vn, target="book", instrument="SPY", window=63, arming=an, mult=mult); rows.append(res)
G = pd.DataFrame(rows)
G.to_csv(HERE / "cross_strategy_regime_hedge_grid.csv", index=False)
show = ["vintage", "target", "instrument", "window", "arming", "mult", "armed_days", "n_episodes", "hedge_total_usd", "hedge_bps_per_armed_day", "ep_mean_usd", "ep_pos_share", "t_clustered",
        "sharpe_unhedged", "sharpe_hedged", "ann_hedged_equal_vol_pct", "maxdd_unhedged_pct", "maxdd_hedged_pct", "armed_sharpe_unhedged", "armed_sharpe_hedged",
        "armed_sd_unhedged_bps", "armed_sd_hedged_bps", "drift_component_usd", "calm_carry_bps_per_day", "false_alarm_episodes", "false_alarm_usd", "loyo_years_with_arming",
        "loyo_years_hedged_not_worse", "beta_mean_armed"]
print("=== primary spec: book, SPY, 63d, by arming rule and vintage ===")
P = G[(G.target == "book") & (G.instrument == "SPY") & (G.window == 63) & (G.mult == 1.0)]
print(P[show].round(2).to_string(index=False))
print("\n=== instrument comparison (book, 63d, dial50_h45) ===")
print(G[(G.target == "book") & (G.window == 63) & (G.arming == "dial50_h45") & (G.mult == 1.0)][show].round(2).to_string(index=False))
print("\n=== window comparison (book, SPY, dial50_h45 / dial65_h60) ===")
print(G[(G.target == "book") & (G.instrument == "SPY") & (G.arming.isin(["dial50_h45", "dial65_h60"])) & (G.mult == 1.0)][show].round(2).to_string(index=False))
print("\n=== sub-book (dip-buy family + OLV) hedge vs whole-book hedge (SPY, 63d) ===")
print(G[(G.instrument == "SPY") & (G.window == 63) & (G.arming.isin(["dial50_h45", "dial65_h60", "always"])) & (G.mult == 1.0)][show].round(2).to_string(index=False))
print("\n=== hedge-ratio multiples ===")
print(G[G.mult != 1.0][show].round(2).to_string(index=False))
OUT["grid"] = G.round(4).to_dict("records")
OUT["primary_detail"] = primary_detail

# ------------------------------------------------------------------ placebo: circular shift of the armed mask (preserves episode structure)
print("\n=== placebo: circular-shift the armed mask 500x (book, SPY, 63d) ===")
OUT["placebo"] = {}
for vn in ["live", "pit"]:
    dseries, win = vintages[vn]; idx = book.index[win]
    b = betas[("book", "SPY", 63)].reindex(idx).fillna(0); f = fac["SPY"].reindex(idx); r = book.reindex(idx)
    for an, (on, off) in {"dial50_h45": (50, 45), "dial65_h60": (65, 60)}.items():
        armed = arm_hysteresis(dseries.reindex(idx), on, off).values.astype(float)
        actual = float((-armed * b * f).sum() * NAV)
        sims = []
        for _ in range(500):
            k = int(RNG.integers(30, len(idx) - 30)); a2 = np.roll(armed, k)
            sims.append(float((-a2 * b * f).sum() * NAV))
        sims = np.array(sims)
        pct = float((sims < actual).mean())
        print(f"{vn:5s} {an:11s}: actual ${actual:,.0f}  placebo mean ${sims.mean():,.0f} sd ${sims.std():,.0f}  P(placebo<actual) {pct:.3f}  "
              f"(a random shift = unconditional short-beta carry over the same number of days)")
        OUT["placebo"][f"{vn}|{an}"] = dict(actual_usd=actual, placebo_mean_usd=float(sims.mean()), placebo_sd_usd=float(sims.std()), pct_rank=pct,
                                           placebo_p95_usd=float(np.percentile(sims, 95)))

# ------------------------------------------------------------------ armed-bucket anatomy on the live series: up-beta vs down-beta and the drift
print("\n=== anatomy: why does the hedge pay? (live, dial50_h45, book/SPY/63) ===")
idx = book.index[vintages["live"][1]]
armed = arm_hysteresis(dial_live_lag.reindex(idx), 50, 45)
m = pd.DataFrame({"r": book.reindex(idx), "f": fac["SPY"].reindex(idx), "b": betas[("book", "SPY", 63)].reindex(idx)}).dropna()
a = armed.reindex(m.index).fillna(False)
for lab, g in [("armed", m[a]), ("unarmed", m[~a])]:
    dn = g[g.f < 0]; up = g[g.f > 0]
    bd = np.polyfit(dn.f, dn.r, 1)[0]; bu = np.polyfit(up.f, up.r, 1)[0]; ball = np.polyfit(g.f, g.r, 1)[0]
    h = -g.b * g.f
    print(f"{lab:8s} days {len(g):4d}: realised beta {ball:.2f} (down-days {bd:.2f}, up-days {bu:.2f}); ex-ante beta_hat mean {g.b.mean():.2f}; SPY ann {g.f.mean()*252*100:+.1f}%; "
          f"hedge on SPY-down days ${h[g.f<0].sum()*NAV:,.0f}, on SPY-up days ${h[g.f>0].sum()*NAV:,.0f}; book sd {g.r.std()*1e4:.0f} -> hedged sd {(g.r+h).std()*1e4:.0f} bps")
    OUT.setdefault("anatomy", {})[lab] = dict(days=len(g), beta_realised=float(ball), beta_down=float(bd), beta_up=float(bu), beta_hat_mean=float(g.b.mean()),
                                              spy_ann_pct=float(g.f.mean() * 252 * 100), hedge_down_days_usd=float(h[g.f < 0].sum() * NAV), hedge_up_days_usd=float(h[g.f > 0].sum() * NAV),
                                              sd_book_bps=float(g.r.std() * 1e4), sd_hedged_bps=float((g.r + h).std() * 1e4))

# ------------------------------------------------------------------ the ex-ante beta problem: how well does beta_hat(63) predict realised beta in the next 21 days?
print("\n=== ex-ante beta quality: beta_hat(w) vs realised next-21d beta (live window) ===")
rows = []
r = book.reindex(idx); f = fac["SPY"].reindex(idx)
real = (r.rolling(21).cov(f) / f.rolling(21).var()).shift(-21)
for w in windows:
    bh = betas[("book", "SPY", w)].reindex(idx)
    mm = pd.DataFrame({"bh": bh, "real": real, "d": dial_live_lag.reindex(idx)}).dropna()
    rows.append(dict(window=w, corr_all=float(mm.bh.corr(mm.real)), corr_dial50=float(mm[mm.d >= 50].bh.corr(mm[mm.d >= 50].real)), mean_bh_dial50=float(mm[mm.d >= 50].bh.mean()),
                     mean_real_dial50=float(mm[mm.d >= 50].real.mean()), rmse_dial50=float(np.sqrt(((mm[mm.d >= 50].bh - mm[mm.d >= 50].real) ** 2).mean()))))
BQ = pd.DataFrame(rows); print(BQ.round(3).to_string(index=False)); OUT["beta_quality"] = BQ.round(4).to_dict("records")

# ------------------------------------------------------------------ 2026-08 episode as it stands in the daily payload (ends 2026-08-07)
last = m.index.max(); print(f"\ndaily payload ends {last.date()}; live dial on last day {dial_live_lag.reindex(m.index).iloc[-1]:.1f}")
json.dump(OUT, open(HERE / "cross_strategy_regime_results_2_hedge.json", "w"), indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
print("wrote cross_strategy_regime_results_2_hedge.json")
