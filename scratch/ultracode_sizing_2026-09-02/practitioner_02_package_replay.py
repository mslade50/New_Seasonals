"""Practitioner lens, check 2 (2026-09-02): the COMPOSED sizing package, replayed
on per-trade daily MTM with the per-strategy daily cap re-applied, plus the
gross-notional / margin tail the composition produces.

Every sibling study replayed ONE overlay at a time. A desk never runs one
overlay at a time: the L1 tilt, the OLV depth ladder, the flow up-size, the
cap relief, the OLV pullback tilt and the GRM step all compose multiplicatively
on the same leg on the same day. This script stacks them (using the sibling
studies' own per-trade feature files so the definitions are identical), and
reports book PnL / Sharpe / maxDD / worst day / worst 21d / PnL per unit risk,
the distribution of the composed per-leg multiplier, and the gross-notional
and stylised portfolio-margin tail (15% single stock, 8% broad index ETF,
45% 3x ETF), at GRM 1.5 and GRM 2.25 with the 250 bps cap fixed and scaled.

Layers (each can be switched off; see CONFIGS):
  tilt   L1 capped half-tilt, plan's 2025-fit multipliers (IN-SAMPLE when applied to history)
  olvdep OLV ladder re-keyed to max(recency rung, depth rung) [0.5 / 0.7 / 1.0 at 0 / 1-2 / 3+ open]
  adds   WCDS + LT Trend ST OS: solo 0.75x / adds 1.25x
  b52    52wh Breakout 0.5x at >= 6 open legs
  ovsx   OVS 0.5x when mean(rank_2d,5d,10d,21d) < 94
  flow   family 5d raw-candidate flow up-only 1.25x (dip_buy >= 6 & dial < 50; oversold_hold >= 7; short_fade >= 104)
  relief per-strategy cap x1.5 on the family's hi-flow days (same thresholds); OVS cap 375 always (ovscap)
  olvdd  OLV 1.25x when SPY is 3-10% off its 252d high at the signal close
  ltdial LT Trend ST OS 0.5x at dial >= 50 (current-weights proxy for the PIT rule; prereg candidate)
  hedge  dial-armed (50/45 hysteresis, lag-1 live dial) SPY beta hedge, 126d lag-1 OLS beta clipped [-1,2], 2 bps friction per arm
Output: practitioner_02_package_replay.json (+ .log via stdout redirect by the caller).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(r"C:/Users/McKinley Slade/dev/New_Seasonals")
D = ROOT / "scratch/ultracode_sizing_2026-09-02"
sys.path.insert(0, str(D))
from flow_conditional_lib import build_trade_mtm, FAMILY  # noqa: E402

NAV = 750_000.0
CAP_BPS = 250.0
OUT = D / "practitioner_02_package_replay.json"

TILT = {"52wh Breakout": 0.73, "Indices Oversold Bounce": 0.83, "Sector BO": 0.84, "Weak Close Decent Sznls": 0.84,
        "St OS Sznl": 0.92, "Monday Dip": 1.02, "ATR Extended Gap Up": 1.04, "3x ETF Overbot Fade": 1.15,
        "SPY QQQ MonFri Reversion": 1.16, "LT Trend ST OS": 1.19, "Oversold Low Volume": 1.29}
FLOW_THR = {"dip_buy": 6, "oversold_hold": 7, "short_fade": 104}
BROAD = {"SPY", "QQQ", "DIA", "IWM", "^GSPC", "^NDX", "VOO", "IVV", "VTI", "MDY", "IJH", "IJR", "RSP", "OEF"}
try:
    sys.path.insert(0, str(ROOT))
    from strategy_config import LEV3X_ALL
    LEV3X = set(LEV3X_ALL)
except Exception as e:  # pragma: no cover
    print("strategy_config import failed, using regex fallback:", e)
    LEV3X = set()

# ---------------------------------------------------------------- ledger + features
led = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
led = led[led["PnL_flat_750k"].notna()].copy()
led["ExitDate"] = led["Exit Date"]
led = led.drop(columns=["Shares"]).rename(columns={"PnL_flat_750k": "PnL", "Risk_flat_750k": "Risk", "Shares_flat": "Shares", "Entry Price": "EntryPrice"})
led["family"] = led["Strategy"].map(FAMILY)
k6 = ["Strategy", "Tier", "Ticker", "Signal Date", "Entry Date", "Direction"]
trade_risk = led.groupby(k6)["Risk"].transform("sum")          # OVS tranche rows sum to the trade's risk
nominal = led["Risk bps"] / 1e4 * NAV * led["Size_Mult"]
led["cap_scale"] = (trade_risk / nominal).clip(upper=1.0001)
led["eff_bps"] = led["Risk bps"] * led["Size_Mult"] * (led["Risk"] / trade_risk)   # row share of the trade's pre-cap effective bps
print(f"ledger rows {len(led)}, cap-bound rows {(led['cap_scale'] < 0.999).sum()} ({(led['cap_scale'] < 0.999).mean():.1%})")

fl = pd.read_parquet(D / "flow_trades_candidates.parquet")[k6 + ["f5", "f1", "nstrat1"]]
led = led.merge(fl, on=k6, how="left")
k5 = ["Strategy", "Tier", "Ticker", "Signal Date", "Entry Date"]
ad = pd.read_parquet(D / "within_strategy_adds_features.parquet")[k5 + ["n_open", "rung_ladder", "residual_mult"]]
led = led.merge(ad, on=k5, how="left")
k4 = ["Strategy", "Tier", "Ticker", "Signal Date"]
sq = pd.read_parquet(D / "signal_quality_features.parquet")[k4 + ["spy_hi252_dist", "rank_2d", "rank_5d", "rank_10d", "rank_21d"]]
led = led.merge(sq.drop_duplicates(k4), on=k4, how="left")
frag = pd.read_parquet(ROOT / "data/rd2_fragility.parquet")
dial = frag["63d"].rolling(10).mean()
led["dial"] = dial.reindex(led["Signal Date"]).values
print("join coverage: f5 %.1f%%, n_open %.1f%% (8 strats), spy_hi252 %.1f%%, dial %.1f%%" % (
    led["f5"].notna().mean() * 100, led["n_open"].notna().mean() * 100, led["spy_hi252_dist"].notna().mean() * 100, led["dial"].notna().mean() * 100))

# family hi-flow flag per row (flow counts start 2005; NaN -> not hi-flow)
thr = led["family"].map(FLOW_THR)
led["hi_flow"] = (led["f5"] >= thr) & thr.notna()
dial_ok = ~(led["dial"] >= 50)     # missing dial = pass (book convention)
led.loc[(led["family"] == "dip_buy") & ~dial_ok, "hi_flow"] = False

# ---------------------------------------------------------------- per-row multipliers by layer
def layer_mults(cfg: dict) -> pd.Series:
    m = pd.Series(1.0, index=led.index)
    if cfg.get("tilt"):
        m *= led["Strategy"].map(TILT).fillna(1.0)
    if cfg.get("olvdep"):
        o = led["Strategy"] == "Oversold Low Volume"
        depth_rung = np.select([led["n_open"] >= 3, led["n_open"] >= 1], [1.0, 0.7], 0.5)
        rung = led["rung_ladder"].fillna(1.0).clip(lower=0.5)
        new = np.maximum(rung, depth_rung)
        m[o & led["n_open"].notna()] *= (new / rung)[o & led["n_open"].notna()]
    if cfg.get("adds"):
        for s in ("Weak Close Decent Sznls", "LT Trend ST OS"):
            o = (led["Strategy"] == s) & led["n_open"].notna()
            m[o] *= np.where(led.loc[o, "n_open"] >= 1, 1.25, 0.75)
    if cfg.get("b52"):
        o = (led["Strategy"] == "52wh Breakout") & (led["n_open"] >= 6)
        m[o] *= 0.5
    if cfg.get("ovsx"):
        o = led["Strategy"] == "Overbot Vol Spike"
        ext = led[["rank_2d", "rank_5d", "rank_10d", "rank_21d"]].mean(axis=1)
        m[o & (ext < 94)] *= 0.5
    if cfg.get("flow"):
        m[led["hi_flow"]] *= 1.25
    if cfg.get("olvdd"):
        # spy_hi252_dist is in PERCENT in signal_quality_features.parquet
        o = (led["Strategy"] == "Oversold Low Volume") & (led["spy_hi252_dist"] < -3.0) & (led["spy_hi252_dist"] >= -10.0)
        m[o] *= 1.25
    if cfg.get("ltdial"):
        o = (led["Strategy"] == "LT Trend ST OS") & (led["dial"] >= 50)
        m[o] *= 0.5
    if cfg.get("olvclip"):
        # practitioner composition cap: OLV tilt 1.15 (not 1.29) and the OLV per-leg product clipped at 1.5x pre-GRM
        o = led["Strategy"] == "Oversold Low Volume"
        if cfg.get("tilt"):
            m[o] *= 1.15 / 1.29
        m[o] = m[o].clip(upper=1.5)
    if cfg.get("guard"):
        # margin-headroom guard proxy: no flow up-size on signal dates where the CURRENT book's open gross already exceeds 2x NAV
        m[led["hi_flow"] & (led["gross_at_signal"] > 2.0)] /= 1.25
    return m


def apply_cap(m: pd.Series, grm_mult: float, cap_fixed: bool, relief: bool, ovscap: bool) -> pd.Series:
    """Ratio of new booked risk to old booked risk per row, with the per-strategy daily cap re-applied
    per (Strategy, Signal Date) group. placed0 = cap/scale on bound days (exact), filled nominal otherwise."""
    g = led.groupby(["Strategy", "Signal Date"], sort=False)
    ratio = pd.Series(np.nan, index=led.index)
    for (strat, sd), idx in g.indices.items():
        rows = led.iloc[idx]
        s0 = float(rows["cap_scale"].min())
        seen0 = float(rows["eff_bps"].sum())
        placed0 = CAP_BPS / s0 if s0 < 0.999 else seen0
        unseen0 = max(placed0 - seen0, 0.0)
        mm = m.iloc[idx].values
        new_seen = float((rows["eff_bps"].values * mm).sum()) * grm_mult
        new_placed = new_seen + unseen0 * float(mm.mean()) * grm_mult
        cap1 = CAP_BPS if cap_fixed else CAP_BPS * grm_mult
        if ovscap and strat == "Overbot Vol Spike":
            cap1 *= 1.5
        if relief and bool(rows["hi_flow"].any()):
            cap1 *= 1.5
        s1 = min(1.0, cap1 / new_placed) if new_placed > 0 else 1.0
        ratio.iloc[idx] = mm * grm_mult * s1 / s0
    return ratio


# ---------------------------------------------------------------- MTM + notional matrices
days, MTM = build_trade_mtm(led)
day_pos = {d: i for i, d in enumerate(days)}
NOT = np.zeros_like(MTM)
for i, (e, x, sh, ep) in enumerate(zip(led["Entry Date"], led["ExitDate"], led["Shares"], led["EntryPrice"])):
    a, b = day_pos.get(e), day_pos.get(x)
    if a is None or b is None:
        continue
    NOT[i, a:b + 1] = sh * ep
cls_rate = np.where(led["Ticker"].isin(BROAD), 0.08, np.where(led["Ticker"].isin(LEV3X), 0.45, 0.15))
gross0 = pd.Series(NOT.sum(0), index=days) / NAV
led["gross_at_signal"] = gross0.reindex(led["Signal Date"]).fillna(0.0).values
print("baseline gross/NAV at signal date on hi-flow rows: p50 %.2f p90 %.2f max %.2f; share of hi-flow rows with gross>2: %.1f%%" % (
    led.loc[led.hi_flow, "gross_at_signal"].quantile(.5), led.loc[led.hi_flow, "gross_at_signal"].quantile(.9), led.loc[led.hi_flow, "gross_at_signal"].max(),
    (led.loc[led.hi_flow, "gross_at_signal"] > 2).mean() * 100))
print("MTM rebuilt: reconciliation residual max", float(np.abs(MTM.sum(1) - led["PnL"].values).max()))

spy = pq.read_table(ROOT / "data/master_prices.parquet", columns=["ticker", "date", "Close"],
                    filters=[("ticker", "=", "SPY")]).to_pandas().set_index("date")["Close"]
spy.index = pd.to_datetime(spy.index)
spy_ret = spy.pct_change().reindex(days).fillna(0.0)
dial_lag = dial.reindex(days).shift(1)


def hedge_series(book: pd.Series) -> tuple[pd.Series, dict]:
    r = book / NAV
    armed = np.zeros(len(days), dtype=bool)
    st = False
    for i, d in enumerate(dial_lag.values):
        if np.isnan(d):
            st = False
        elif st and d < 45:
            st = False
        elif (not st) and d >= 50:
            st = True
        armed[i] = st
    # 126d lag-1 OLS beta of book on SPY
    x, y = spy_ret.values, r.values
    beta = np.full(len(days), np.nan)
    for i in range(127, len(days)):
        xs, ys = x[i - 127:i - 1], y[i - 127:i - 1]
        vx = xs.var()
        beta[i] = np.clip(((xs - xs.mean()) * (ys - ys.mean())).mean() / vx, -1, 2) if vx > 0 else 0.0
    beta = np.nan_to_num(beta)
    h = -(armed.astype(float)) * beta * x * NAV
    arm_events = np.diff(armed.astype(int), prepend=0) == 1
    h = h - arm_events * 2e-4 * np.abs(beta) * NAV
    info = dict(armed_days=int(armed.sum()), arm_events=int(arm_events.sum()), hedge_pnl=float(h.sum()),
                mean_beta_armed=float(beta[armed].mean()) if armed.any() else np.nan)
    return pd.Series(h, index=days), info


def stats(book: pd.Series, gross: pd.Series, req: pd.Series, risk_total: float, win: tuple[str, str]) -> dict:
    b = book[(book.index >= win[0]) & (book.index <= win[1])]
    g = gross[(gross.index >= win[0]) & (gross.index <= win[1])] / NAV
    q = req[(req.index >= win[0]) & (req.index <= win[1])] / NAV
    eq = b.cumsum(); dd = eq - eq.cummax()
    yrs = (b.index[-1] - b.index[0]).days / 365.25
    return dict(total=float(b.sum()), ann_pnl_pct=float(b.sum() / yrs / NAV * 100), ann_vol_pct=float(b.std() * np.sqrt(252) / NAV * 100),
                sharpe=float(b.mean() / b.std() * np.sqrt(252)) if b.std() > 0 else np.nan,
                maxdd_pct=float(dd.min() / NAV * 100), worst_day_pct=float(b.min() / NAV * 100),
                worst21_pct=float(b.rolling(21).sum().min() / NAV * 100), worst63_pct=float(b.rolling(63).sum().min() / NAV * 100),
                cvar5_bps=float(b[b <= b.quantile(0.05)].mean() / NAV * 1e4),
                gross_nav_p95=float(g.quantile(0.95)), gross_nav_p99=float(g.quantile(0.99)), gross_nav_max=float(g.max()),
                gross_max_date=str(g.idxmax().date()),
                req_nav_p95=float(q.quantile(0.95)), req_nav_p99=float(q.quantile(0.99)), req_nav_max=float(q.max()),
                req_max_date=str(q.idxmax().date()), feas_mult_max=float(1 / q.max()), feas_mult_p99=float(1 / q.quantile(0.99)),
                risk_deployed=float(risk_total))


CONFIGS = {
    "baseline": {},
    "tilt_only": {"tilt": 1},
    "within_only": {"olvdep": 1, "adds": 1, "b52": 1, "ovsx": 1, "ovscap": 1},
    "flow_only": {"flow": 1, "relief": 1},
    "olvdd_only": {"olvdd": 1},
    "package_A_no_flow": {"tilt": 1, "olvdep": 1, "adds": 1, "b52": 1, "ovsx": 1, "ovscap": 1, "olvdd": 1},
    "package_B_full": {"tilt": 1, "olvdep": 1, "adds": 1, "b52": 1, "ovsx": 1, "ovscap": 1, "olvdd": 1, "flow": 1, "relief": 1},
    "package_B_plus_ltdial": {"tilt": 1, "olvdep": 1, "adds": 1, "b52": 1, "ovsx": 1, "ovscap": 1, "olvdd": 1, "flow": 1, "relief": 1, "ltdial": 1},
    "package_B_no_tilt": {"olvdep": 1, "adds": 1, "b52": 1, "ovsx": 1, "ovscap": 1, "olvdd": 1, "flow": 1, "relief": 1},
    "package_C_practitioner": {"tilt": 1, "olvdep": 1, "adds": 1, "b52": 1, "ovsx": 1, "ovscap": 1, "olvdd": 1, "flow": 1, "relief": 1, "olvclip": 1, "guard": 1},
    "package_C_no_relief": {"tilt": 1, "olvdep": 1, "adds": 1, "b52": 1, "ovsx": 1, "ovscap": 1, "olvdd": 1, "flow": 1, "olvclip": 1, "guard": 1},
}
WINDOWS = {"2005-2026": ("2005-01-01", "2026-08-28"), "2010-2026": ("2010-01-01", "2026-08-28"), "2016-07+": ("2016-07-20", "2026-08-28")}
GRMS = [(1.0, True, "grm1.5"), (1.25, True, "grm1.875_capfixed"), (1.5, True, "grm2.25_capfixed"), (1.5, False, "grm2.25_capscaled")]

results = {"configs": {}, "meta": {"n_rows": int(len(led)), "cap_bps": CAP_BPS, "tilt": TILT, "flow_thr": FLOW_THR,
                                    "notes": ["tilt multipliers are the plan's 2025 fit applied to all history (in-sample for levels; the "
                                              "point of this replay is composition/tails, not the tilt's own edge)",
                                              "flow counts from flow_trades_candidates.parquet (raw engine candidates, 2005+)",
                                              "depth from within_strategy_adds_features.parquet (filled legs only)",
                                              "dial = current-weights recompute vintage before 2026-07-02, value at signal date",
                                              "margin rates stylised: 15% single stock, 8% broad index ETF, 45% 3x ETF; TIMS no long/short offset",
                                              "OVS P2 aggregate cap, OLV ticker notional cap, cross-strategy clamp NOT re-applied"]}}
base_book = None
year_pnl = {}
books15 = {}
for name, cfg in CONFIGS.items():
    m = layer_mults(cfg)
    results["configs"][name] = {"layers": cfg, "grm": {}}
    for grm_mult, cap_fixed, glabel in GRMS:
        ratio = apply_cap(m, grm_mult, cap_fixed, relief=bool(cfg.get("relief")), ovscap=bool(cfg.get("ovscap")))
        rv = ratio.values.astype(np.float32)
        book = pd.Series((MTM * rv[:, None]).sum(0), index=days)
        gross = pd.Series((NOT * rv[:, None]).sum(0), index=days)
        req = pd.Series((NOT * (rv * cls_rate)[:, None]).sum(0), index=days)
        risk_total = float((led["Risk"] * ratio).sum())
        entry = {"windows": {w: stats(book, gross, req, risk_total, lim) for w, lim in WINDOWS.items()},
                 "mult_dist": {"p50": float(ratio.quantile(0.5)), "p90": float(ratio.quantile(0.9)), "p99": float(ratio.quantile(0.99)),
                               "max": float(ratio.max()), "share_gt_1.5x": float((ratio > 1.5 * grm_mult).mean()),
                               "share_lt_0.75x": float((ratio < 0.75 * grm_mult).mean())},
                 "pnl_per_risk": float(((MTM * rv[:, None]).sum()) / risk_total)}
        if glabel == "grm1.5":
            books15[name] = book.copy()
        if name == "baseline":
            year_pnl[glabel] = book.groupby(book.index.year).sum()
            if glabel == "grm1.5":
                base_book = book.copy()
        else:
            yp = book.groupby(book.index.year).sum(); byp = year_pnl[glabel]
            yy = yp.index[(yp.index >= 2005) & (yp.index <= 2026)]
            entry["years_better_vs_baseline_same_grm"] = f"{int((yp[yy] > byp[yy]).sum())}/{len(yy)}"
            d = (yp[yy] - byp[yy]) / byp[yy].abs()
            entry["worst_year_vs_baseline_pct_of_base"] = float(d.min() * 100)
        # hedge layer on top (2016-07+ window only is meaningful)
        h, hinfo = hedge_series(book)
        hb = book + h
        entry["hedged_2016-07+"] = {**stats(hb, gross, req, risk_total, WINDOWS["2016-07+"]), **hinfo}
        results["configs"][name]["grm"][glabel] = entry
        w = entry["windows"]["2010-2026"]; w16 = entry["windows"]["2016-07+"]; hh = entry["hedged_2016-07+"]
        print(f"{name:22s} {glabel:18s} 2010+: PnL {w['total']/1e6:5.2f}M ann {w['ann_pnl_pct']:5.1f}% vol {w['ann_vol_pct']:4.1f}% Sh {w['sharpe']:.2f} "
              f"maxDD {w['maxdd_pct']:6.1f}% worst {w['worst_day_pct']:5.1f}% w21 {w['worst21_pct']:6.1f}% PPR {entry['pnl_per_risk']:.3f} "
              f"| gross max {w['gross_nav_max']:.2f} p99 {w['gross_nav_p99']:.2f} req max {w['req_nav_max']:.2f} p99 {w['req_nav_p99']:.2f} "
              f"| 2016+ Sh {w16['sharpe']:.2f} maxDD {w16['maxdd_pct']:.1f} -> hedged Sh {hh['sharpe']:.2f} maxDD {hh['maxdd_pct']:.1f} (+{hh['hedge_pnl']/1e3:.0f}k, {hh['armed_days']}d)"
              + (f" | yrs {entry.get('years_better_vs_baseline_same_grm')}" if name != 'baseline' else ""))

# top composed-multiplier days (package_B_full, grm 2.25 cap fixed): where the stack fires together
m = layer_mults(CONFIGS["package_C_practitioner"]); ratio = apply_cap(m, 1.5, True, True, True)
led["_ratio"] = ratio; led["_new_risk"] = led["Risk"] * ratio
top = led.groupby("Signal Date").agg(new_risk=("_new_risk", "sum"), old_risk=("Risk", "sum"), n=("Strategy", "size"),
                                     strats=("Strategy", lambda s: ",".join(sorted(set(s))))).sort_values("new_risk", ascending=False).head(12)
top["new_bps"] = top["new_risk"] / NAV * 1e4; top["old_bps"] = top["old_risk"] / NAV * 1e4
print("\nTop staged-risk days under package_C_practitioner at GRM 2.25 (cap fixed):")
print(top[["n", "old_bps", "new_bps", "strats"]].round(0).to_string())
results["top_days_packageC_grm225"] = top.reset_index().assign(**{"Signal Date": lambda d: d["Signal Date"].astype(str)}).to_dict("records")
byS = led.groupby("Strategy").agg(old=("Risk", "sum"), new=("_new_risk", "sum"), pnl=("PnL", "sum"))
byS["risk_ratio"] = byS["new"] / byS["old"]
print("\nRisk deployed by strategy, package_C_practitioner GRM 2.25 vs current:")
print(byS[["old", "new", "risk_ratio"]].round(2).to_string())
results["risk_by_strategy_packageC_grm225"] = byS.round(3).to_dict("index")

# yearly PnL table at GRM 1.5 and drawdown-episode diagnostics (2016-07+)
Y = pd.DataFrame({k: v[v.index >= "2005-01-01"].groupby(v[v.index >= "2005-01-01"].index.year).sum() / 1e3 for k, v in books15.items()})
print("\nYearly PnL ($k, flat) at GRM 1.5:"); print(Y.round(0).to_string())
results["yearly_pnl_k_grm15"] = Y.round(1).to_dict()
def dd_episode(book):
    b = book[book.index >= "2016-07-20"]; eq = b.cumsum(); dd = eq - eq.cummax()
    t = dd.idxmin(); pk = eq[:t].idxmax()
    return pk, t, float(dd.min() / NAV * 100)
diag = {}
for nm in ("baseline", "package_A_no_flow", "package_B_full", "package_C_practitioner"):
    pk, t, d = dd_episode(books15[nm])
    m = layer_mults(CONFIGS[nm]); ratio = apply_cap(m, 1.0, True, bool(CONFIGS[nm].get("relief")), bool(CONFIGS[nm].get("ovscap"))).values.astype(np.float32)
    win = (days > pk) & (days <= t)
    contrib = pd.Series((MTM[:, win] * ratio[:, None]).sum(1), index=led.index).groupby(led["Strategy"]).sum().sort_values()
    diag[nm] = {"peak": str(pk.date()), "trough": str(t.date()), "maxdd_pct": d, "top_contrib_k": (contrib.head(4) / 1e3).round(1).to_dict()}
    print(f"{nm:22s} 2016+ maxDD {d:6.1f}% {pk.date()} -> {t.date()}  worst contributors: {(contrib.head(4)/1e3).round(1).to_dict()}")
results["dd_episode_2016plus_grm15"] = diag

# drawdown-probability frontier (stationary block bootstrap, mean block 10, 3000 one-year paths, flat % of NAV) for baseline and package C
rng = np.random.default_rng(7)
def boot_frontier(book: pd.Series, mults, haircuts, n_paths=3000, block=10, horizon=252):
    r = (book[book.index >= "2010-01-01"] / NAV).values
    n = len(r); out = {}
    for hc in haircuts:
        rr = r - hc * r.mean()
        paths = np.empty((n_paths, horizon))
        for k in range(n_paths):
            pos = 0; seq = []
            while pos < horizon:
                L = rng.geometric(1 / block); st = rng.integers(0, n)
                idx = (st + np.arange(L)) % n; seq.append(rr[idx]); pos += L
            paths[k] = np.concatenate(seq)[:horizon]
        for mlt in mults:
            eq = np.cumsum(paths * mlt, axis=1); dd = (eq - np.maximum.accumulate(eq, axis=1)).min(1)
            out[f"hc{int(hc*100)}_m{mlt}"] = {"grm": 1.5 * mlt, "median_maxdd_pct": float(-np.median(dd) * 100), "p95_maxdd_pct": float(-np.quantile(dd, 0.05) * 100),
                                              "P_dd_gt10": float((dd < -0.10).mean()), "P_dd_gt15": float((dd < -0.15).mean()), "P_dd_gt20": float((dd < -0.20).mean()),
                                              "P_dd_gt25": float((dd < -0.25).mean()), "median_1y_pnl_pct": float(np.median(eq[:, -1]) * 100)}
    return out
fr = {}
for nm in ("baseline", "package_C_practitioner"):
    fr[nm] = boot_frontier(books15[nm], mults=[1.0, 1.25, 1.5, 2.0], haircuts=[0.0, 0.25, 0.5])
    print(f"\nBootstrap 1y drawdown frontier, {nm} (2010+ series, flat % NAV):")
    for k, v in fr[nm].items():
        print(f"  {k:12s} GRM {v['grm']:.3f} medPnL {v['median_1y_pnl_pct']:5.1f}% medDD {v['median_maxdd_pct']:5.1f}% p95DD {v['p95_maxdd_pct']:5.1f}% P>10 {v['P_dd_gt10']:.2f} P>15 {v['P_dd_gt15']:.2f} P>20 {v['P_dd_gt20']:.3f} P>25 {v['P_dd_gt25']:.3f}")
results["dd_frontier_bootstrap"] = fr
json.dump(results, open(OUT, "w"), indent=1, default=float)
print("wrote", OUT)
