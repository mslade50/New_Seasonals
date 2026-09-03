"""Second pass: composition variants that could make the shipped package pass the WP5 gate,
the guard's trim cost, the hedge on the shipped book, yearly PnL, participation, and the
OLV earnings-override composition. Imports the first script (re-runs it, ~2 min)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

OUTD = Path(__file__).resolve().parent
sys.path.insert(0, str(OUTD))
import dd_tail_01_package_tails as T  # noqa: E402  (executes the base replay)

led, MTM, NOT, days, NAV = T.led, T.MTM, T.NOT, T.days, T.NAV
R2: dict = {}

def run(name, form, cfg, relief=True, clamp=True, ovscap=False, depth_max=None, olv_off=False, abs_clip=None):
    cfg = dict(cfg)
    grm_mult = 1.25 if cfg.get("grm", "ship") == "ship" else float(cfg["grm"])
    hi_col = "hi_brief" if form == "ship" else "hi_lib"
    fam_col = "fam_brief" if form == "ship" else "fam_lib"
    relief_fams = led[hi_col] & led[fam_col].isin(T.FLOW_FAMS if form == "ship" else set(T.FLOW_THR))
    relief_days = set(led.loc[relief_fams, "Signal Date"]) if relief else None
    L = T.levers(form, cfg)
    o = led.Strategy == "Oversold Low Volume"
    if depth_max is not None:            # depth rung capped (0 -> 0.5, 1+ -> depth_max)
        rung_old = led["rung_ladder"].fillna(1.0).clip(lower=0.5)
        depth_rung = np.where(led["depth_ship"] >= 1, depth_max, 0.5)
        rung_new = np.where(o & led["n_open"].notna(), np.maximum(rung_old, depth_rung), rung_old)
        L["ladder"] = np.where(o, rung_new / rung_old, 1.0)
        absP = L["tilt"] * pd.Series(rung_new, index=led.index) * L["flow"] * L["pullback"]
        L["clip"] = 1.0; L.loc[o, "clip"] = (absP.clip(upper=1.5) / absP)[o]
    if abs_clip is not None:
        rung_old = led["rung_ladder"].fillna(1.0).clip(lower=0.5)
        rung_new = rung_old * L["ladder"]
        absP = L["tilt"] * rung_new * L["flow"] * L["pullback"]
        L["clip"] = 1.0; L.loc[o, "clip"] = (absP.clip(upper=abs_clip) / absP)[o]
    if olv_off:
        L.loc[o, ["tilt", "ladder", "pullback", "clip"]] = 1.0
    L["total"] = L[["tilt", "ladder", "adds", "ovsx", "b52", "flow", "pullback", "p2cap", "iob_clone", "clip", "grm"]].prod(axis=1)
    r0 = T.apply_cap(L["total"], relief_days, relief_fams, ovscap)
    ginfo = None
    if form == "ship" and relief:
        gd, ginfo = T.guard_days_for(r0.values.astype(np.float32), T.NLV_LIVE)
        cfg["guard_days"] = gd
        L2 = T.levers(form, cfg); L["flow"] = L2["flow"]
        if depth_max is None and abs_clip is None:
            L["clip"] = L2["clip"]
        if olv_off:
            L.loc[o, ["tilt", "ladder", "pullback", "clip"]] = 1.0
        L["total"] = L[["tilt", "ladder", "adds", "ovsx", "b52", "flow", "pullback", "p2cap", "iob_clone", "clip", "grm"]].prod(axis=1)
        r0 = T.apply_cap(L["total"], relief_days - gd, relief_fams, ovscap)
    if clamp:
        r0 = T.clamp_ext(r0, grm_mult)
    rv = r0.values.astype(np.float32)
    book = pd.Series((MTM * rv[:, None]).sum(0), index=days)
    ent = {"windows": {w: T.stats(book, *lim) for w, lim in T.WIN.items()}, "dd_top5_2016": T.dd_episodes(book, rv, "2016-07-20", 5),
           "risk_deployed_ratio": round(float((led["Risk"] * r0).sum() / led["Risk"].sum()), 4), "pnl_per_risk": round(float((led["PnL"] * r0).sum() / (led["Risk"] * r0).sum()), 4),
           "olv_ratio_p95": round(float(r0[o].quantile(.95)), 3), "olv_ratio_max": round(float(r0[o].max()), 3), "guard": ginfo}
    w = ent["windows"]["2005-2026"]; w16 = ent["windows"]["2016-07+"]
    T.log(f"{name:26s} 2005+: ann {w['ann_pnl_pct']:5.1f} Sh {w['sharpe']:.2f} maxDD {w['maxdd_pct']:6.2f} worst {w['worst_day_pct']:5.2f} w21 {w['worst21_pct']:6.2f} | 2016+: ann {w16['ann_pnl_pct']:5.1f} Sh {w16['sharpe']:.2f} maxDD {w16['maxdd_pct']:6.2f} ({w16['maxdd_trough']}) w21 {w16['worst21_pct']:6.2f} | risk x{ent['risk_deployed_ratio']:.3f} PPR {ent['pnl_per_risk']:.3f} olv p95 {ent['olv_ratio_p95']} max {ent['olv_ratio_max']}")
    T.log("     top3 2016+:", [(e["peak"], e["trough"], e["depth_pct"], e["top3"][0]) for e in ent["dd_top5_2016"][:3]])
    R2[name] = ent
    return book, r0

books = {}
books["study_C_grm1.5"], _ = run("study_C_grm1.5 (replica)", "study", dict(grm=1.0), ovscap=True, clamp=False)
books["ship_ratioclip_1.5eq"], _ = run("ship_ratioclip_1.5eq", "ship", dict(grm=1.0, clip_mode="ratio"))
books["ship_depth07_1.875"], r_d07 = run("ship_depthmax0.7_1.875", "ship", dict(), depth_max=0.7)
books["ship_depth07_1.5eq"], _ = run("ship_depthmax0.7_1.5eq", "ship", dict(grm=1.0), depth_max=0.7)
books["ship_absclip1.0_1.875"], _ = run("ship_absclip1.0_1.875", "ship", dict(), abs_clip=1.0)
books["ship_absclip1.0_1.5eq"], _ = run("ship_absclip1.0_1.5eq", "ship", dict(grm=1.0), abs_clip=1.0)
books["ship_olvoff_1.875"], _ = run("ship_OLVlevers_off_1.875", "ship", dict(), olv_off=True)
books["ship_olvoff_1.5eq"], _ = run("ship_OLVlevers_off_1.5eq", "ship", dict(grm=1.0), olv_off=True)
books["ship_nodepth_1.875"], _ = run("ship_no_depth_rung_1.875", "ship", dict(olvdep=False))
books["ship_nodepth_1.5eq"], _ = run("ship_no_depth_rung_1.5eq", "ship", dict(grm=1.0, olvdep=False))
books["ship_notilt_1.5eq"], _ = run("ship_no_tilt_1.5eq", "ship", dict(grm=1.0, tilt=False))
books["ship_1.875"] = T.books["ship_grm1.875"]; books["today"] = T.books["today"]; books["today_1.875"] = T.books["today_grm1.875"]

# ------------------------------------------------------------------ guard >70% trim cost on the shipped book
rv = T.ratios["ship_grm1.875"].values.astype(np.float32)
req_open = T.req_series(rv)
staged_req = pd.Series(led["Shares"].values * led["EntryPrice"].values * rv * T.cls_rate, index=led.index).groupby(led["Signal Date"]).sum().reindex(days).fillna(0.0)
proj = (req_open.shift(1).fillna(0.0) + 1.10 * staged_req) / T.NLV_LIVE
trim = pd.Series(1.0, index=days)
over = proj > 0.70
trim[over] = ((0.70 * T.NLV_LIVE - req_open.shift(1).fillna(0.0)[over]) / (1.10 * staged_req[over])).clip(lower=0.0, upper=1.0)
row_trim = led["Signal Date"].map(trim).fillna(1.0).values
book_trim = pd.Series((MTM * (rv * row_trim)[:, None]).sum(0), index=days)
b0 = T.books["ship_grm1.875"]
gc = dict(trim_days=int(over.sum()), trim_days_per_year=round(float(over.sum() / 23.6), 2), mean_trim_factor=round(float(trim[over].mean()), 3),
          pnl_foregone_k=round(float((b0.sum() - book_trim.sum()) / 1e3), 1), pnl_foregone_pct_of_total=round(float((b0.sum() - book_trim.sum()) / b0.sum() * 100), 2),
          trimmed_trades=int((row_trim < 1).sum()), trimmed_trade_pnl_k=round(float(led.loc[row_trim < 1, "PnL"].mul(T.ratios["ship_grm1.875"][row_trim < 1]).sum() / 1e3), 1),
          trim_dates=[str(d.date()) for d in days[over]][:40], stats_after_trim=T.stats(book_trim, "2005-01-01", "2026-09-01"), stats_2016_after_trim=T.stats(book_trim, "2016-07-20", "2026-09-01"))
gb = dict(gc); gb.pop("trim_dates")
T.log("guard trim cost:", gb)
# same on the $750k base
proj_base = (req_open.shift(1).fillna(0.0) + 1.10 * staged_req) / NAV
gc["days_over_70_on_750k_base"] = int((proj_base > 0.70).sum()); gc["days_over_60_on_750k_base"] = int((proj_base > 0.60).sum())
R2["guard_trim"] = gc

# ------------------------------------------------------------------ dial-armed hedge on the shipped book (practitioner's hedge_series form)
spy_ret = T.spy_ret
dial_lag = T.dial_live.reindex(days).shift(1)
def hedge(book):
    r = book / NAV; armed = np.zeros(len(days), dtype=bool); st = False
    for i, d in enumerate(dial_lag.values):
        if np.isnan(d): st = False
        elif st and d < 45: st = False
        elif (not st) and d >= 50: st = True
        armed[i] = st
    x, y = spy_ret.values, r.values; beta = np.full(len(days), np.nan)
    for i in range(127, len(days)):
        xs, ys = x[i - 127:i - 1], y[i - 127:i - 1]; vx = xs.var()
        beta[i] = np.clip(((xs - xs.mean()) * (ys - ys.mean())).mean() / vx, -1, 2) if vx > 0 else 0.0
    beta = np.nan_to_num(beta)
    h = -(armed.astype(float)) * beta * x * NAV
    arm_events = np.diff(armed.astype(int), prepend=0) == 1
    h = h - arm_events * 2e-4 * np.abs(beta) * NAV
    return pd.Series(h, index=days), armed, beta
hd = {}
for nm in ("today", "ship_1.875"):
    h, armed, beta = hedge(books[nm]); hb = books[nm] + h
    s = T.stats(hb, "2016-07-20", "2026-09-01")
    eps = T.dd_episodes(hb, T.ratios["ship_grm1.875" if nm != "today" else "today"].values.astype(np.float32), "2016-07-20", 4)
    aug = (days >= pd.Timestamp("2026-08-26")) & (days <= pd.Timestamp("2026-09-01"))
    jun = (days >= pd.Timestamp("2026-06-12")) & (days <= pd.Timestamp("2026-07-01"))
    hd[nm] = dict(hedged_2016=s, hedge_pnl_k=round(float(h.loc["2016-07-20":].sum() / 1e3), 1), armed_days=int(armed[days >= pd.Timestamp("2016-07-20")].sum()),
                  june2026_armed_days=int(armed[jun].sum()), aug2026_armed_days=int(armed[aug].sum()), aug2026_hedge_pnl_k=round(float(h[aug].sum() / 1e3), 1),
                  aug2026_book_pnl_k=round(float(books[nm][aug].sum() / 1e3), 1), top_eps=[(e["peak"], e["trough"], e["depth_pct"]) for e in eps])
T.log("hedge on book:", json.dumps(hd, default=str))
R2["hedge"] = hd

# ------------------------------------------------------------------ yearly PnL today vs ship
Y = pd.DataFrame({k: v.groupby(v.index.year).sum() / 1e3 for k, v in books.items() if k in ("today", "today_1.875", "ship_1.875", "ship_depth07_1.875", "ship_ratioclip_1.5eq")})
Y = Y[Y.index >= 2005].round(0)
Y["ship_vs_todayGRM_k"] = (Y["ship_1.875"] - Y["today_1.875"]).round(0)
T.log("yearly PnL ($k):\n" + Y.to_string())
R2["yearly_k"] = Y.to_dict()
R2["years_ship_better_than_todayGRM"] = f"{int((Y['ship_1.875'] > Y['today_1.875']).sum())}/{len(Y)}"

# ------------------------------------------------------------------ participation under the brief's 1% / 0.4% rule
r = T.ratios["ship_grm1.875"]
part = (led["Shares"] * led["EntryPrice"] * r) / (led["dollar_vol_m"] * 1e6)
lim = led["Strategy"].map(lambda s: 0.004 if s in {"LT Trend ST OS", "St OS Sznl", "Weak Close Decent Sznls", "Indices Oversold Bounce"} else 0.01)
overp = part > lim
R2["participation_rule"] = dict(rows_over_limit=int(overp.sum()), by_strategy=led.loc[overp, "Strategy"].value_counts().to_dict(), by_tier=led.loc[overp, "Tier"].value_counts().to_dict(),
                                pnl_of_over_rows_k=round(float((led.loc[overp, "PnL"] * r[overp]).sum() / 1e3), 1), mean_haircut_if_trimmed=round(float((lim[overp] / part[overp]).mean()), 3),
                                rows_over_5pct_refusal=int((part > 0.05).sum()), refused_pnl_k=round(float((led.loc[part > 0.05, "PnL"] * r[part > 0.05]).sum() / 1e3), 1))
T.log("participation rule:", R2["participation_rule"])

# ------------------------------------------------------------------ OLV earnings override x new levers (absolute bps)
L = T.levtabs["ship_grm1.875"]
eo = (led.Strategy == "Oversold Low Volume") & led["Size_Mult"].round(3).isin([0.4, 0.2, 0.28, 0.286, 0.143])
rung_old = led["rung_ladder"].fillna(1.0).clip(lower=0.5)
abs_new_bps = 10 * 1.875 * L["tilt"] * rung_old * L["ladder"] * L["flow"] * L["pullback"] * L["clip"]
abs_old_bps = 10 * 1.5 * rung_old
R2["olv_earnings_override_composition"] = dict(rows=int(eo.sum()), old_eff_bps_mean=round(float(abs_old_bps[eo].mean()), 2), new_eff_bps_mean=round(float(abs_new_bps[eo].mean()), 2),
                                                new_eff_bps_max=round(float(abs_new_bps[eo].max()), 2), rows_gt_2x=int((abs_new_bps[eo] / abs_old_bps[eo] > 2).sum()),
                                                avgR_override_rows=round(float(led.loc[eo, "R_Multiple"].mean()), 3))
T.log("OLV earnings override composition:", R2["olv_earnings_override_composition"])

# ------------------------------------------------------------------ worst-20 book days: today vs ship membership overlap
w_t = set(books["today"].loc["2016-07-20":].nsmallest(20).index); w_s = set(books["ship_1.875"].loc["2016-07-20":].nsmallest(20).index)
R2["worst20_overlap"] = dict(common=len(w_t & w_s), new_in_ship=[str(d.date()) for d in sorted(w_s - w_t)], dropped=[str(d.date()) for d in sorted(w_t - w_s)])
T.log("worst20 overlap:", R2["worst20_overlap"])

json.dump(R2, open(OUTD / "dd_tail_results_2.json", "w"), indent=1, default=str)
T.log("wrote", OUTD / "dd_tail_results_2.json")
