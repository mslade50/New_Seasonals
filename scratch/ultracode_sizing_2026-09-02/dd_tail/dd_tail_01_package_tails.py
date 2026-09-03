"""Adversarial reviewer, composition/tails lens (2026-09-02).

Builds the per-trade multiplier table for the SHIPPED package (brief WP5-WP8 forms)
and for the practitioner's STUDY-form package C, scales each trade's daily MTM,
re-applies the per-strategy cap (250 / 375 relief), and reports tails, drawdown
episodes, lever stacking, the correlated-tail migration and the OLV footprint.

Writes dd_tail_results.json + a few CSVs next to this script. Reads only.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(r"C:/Users/McKinley Slade/dev/New_Seasonals")
D = ROOT / "scratch/ultracode_sizing_2026-09-02"
OUTD = D / "dd_tail"
sys.path.insert(0, str(D)); sys.path.insert(0, str(ROOT))
from flow_conditional_lib import build_trade_mtm, FAMILY as FAMILY_LIB, BDAYS  # noqa: E402
from strategy_config import LEV3X_ALL  # noqa: E402

NAV = 750_000.0
NLV_LIVE = 632_000.0
CAP = 250.0
LEV3X = set(LEV3X_ALL)
BROAD = {"SPY", "QQQ", "DIA", "IWM", "^GSPC", "^NDX", "VOO", "IVV", "VTI", "MDY", "IJH", "IJR", "RSP", "OEF"}
SPOT = {"^GSPC": "SPY", "^NDX": "QQQ"}

# brief WP6 shipping tilt
TILT_SHIP = {"52wh Breakout": 0.70, "Weak Close Decent Sznls": 0.75, "Sector BO": 0.87, "St OS Sznl": 0.88,
             "Indices Oversold Bounce": 0.89, "Overbot Vol Spike": 1.00, "LT Trend ST OS": 1.04, "Monday Dip": 1.09,
             "ATR Extended Gap Up": 1.10, "Oversold Low Volume": 1.17, "3x ETF Overbot Fade": 1.27, "SPY QQQ MonFri Reversion": 1.30}
# practitioner replay tilt (study form)
TILT_STUDY = {"52wh Breakout": 0.73, "Indices Oversold Bounce": 0.83, "Sector BO": 0.84, "Weak Close Decent Sznls": 0.84,
              "St OS Sznl": 0.92, "Monday Dip": 1.02, "ATR Extended Gap Up": 1.04, "3x ETF Overbot Fade": 1.15,
              "SPY QQQ MonFri Reversion": 1.16, "LT Trend ST OS": 1.19, "Oversold Low Volume": 1.29}
# brief WP8 family membership (differs from the lib's FAMILY used to fit the thresholds)
FAMILY_BRIEF = {"Weak Close Decent Sznls": "dip_buy", "SPY QQQ MonFri Reversion": "dip_buy", "Monday Dip": "dip_buy",
                "Indices Oversold Bounce": "dip_buy", "Monthly Weak Close": "dip_buy", "St OS Sznl": "dip_buy",
                "Oversold Low Volume": "oversold_hold", "LT Trend ST OS": "oversold_hold",
                "Overbot Vol Spike": "short_fade", "3x ETF Overbot Fade": "short_fade", "ATR Extended Gap Up": "short_fade",
                "3x Bear ETF Overbot Fade": "bear_etf_fade", "3x Leader Gap Fade": "bear_etf_fade",
                "52wh Breakout": "breakout", "Sector BO": "breakout"}
FLOW_THR = {"dip_buy": 6, "oversold_hold": 7, "short_fade": 104}
FLOW_FAMS = {"dip_buy", "oversold_hold", "short_fade"}
OVERFLOW_LONGS = {"Oversold Low Volume", "LT Trend ST OS", "St OS Sznl", "52wh Breakout"}
CLAMP_PAIRS = [("Indices Oversold Bounce", "SPY QQQ MonFri Reversion"),           # live today
               ("Monday Dip", "Weak Close Decent Sznls"), ("SPY QQQ MonFri Reversion", "Weak Close Decent Sznls"),
               ("Monthly Weak Close", "SPY QQQ MonFri Reversion"), ("Monthly Weak Close", "Indices Oversold Bounce"),
               ("Monday Dip", "Indices Oversold Bounce")]
CLAMP_BPS_NOMINAL = 20.0

R: dict = {"meta": {}}
def log(*a):
    print(*a, flush=True)

# ------------------------------------------------------------------ ledger assembly (mirrors practitioner_02)
led = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
led = led[led["PnL_flat_750k"].notna()].copy()
led["ExitDate"] = led["Exit Date"]
led = led.drop(columns=["Shares"]).rename(columns={"PnL_flat_750k": "PnL", "Risk_flat_750k": "Risk", "Shares_flat": "Shares", "Entry Price": "EntryPrice"})
led["fam_lib"] = led["Strategy"].map(FAMILY_LIB)
led["fam_brief"] = led["Strategy"].map(FAMILY_BRIEF)
k6 = ["Strategy", "Tier", "Ticker", "Signal Date", "Entry Date", "Direction"]
trade_risk = led.groupby(k6)["Risk"].transform("sum")
nominal = led["Risk bps"] / 1e4 * NAV * led["Size_Mult"]
led["cap_scale"] = (trade_risk / nominal).clip(upper=1.0001)
led["eff_bps"] = led["Risk bps"] * led["Size_Mult"] * (led["Risk"] / trade_risk)
fl = pd.read_parquet(D / "flow_trades_candidates.parquet")[k6 + ["f5"]]
led = led.merge(fl, on=k6, how="left")
k5 = ["Strategy", "Tier", "Ticker", "Signal Date", "Entry Date"]
ad = pd.read_parquet(D / "within_strategy_adds_features.parquet")[k5 + ["n_open", "rung_ladder", "residual_mult"]]
led = led.merge(ad, on=k5, how="left")
k4 = ["Strategy", "Tier", "Ticker", "Signal Date"]
sq = pd.read_parquet(D / "signal_quality_features.parquet")[k4 + ["spy_hi252_dist", "rank_2d", "rank_5d", "rank_10d", "rank_21d", "dial_pit", "dollar_vol_m"]]
led = led.merge(sq.drop_duplicates(k4), on=k4, how="left")
frag = pd.read_parquet(ROOT / "data/rd2_fragility.parquet")
dial_live = frag["63d"].rolling(10).mean()
led["dial"] = dial_live.reindex(led["Signal Date"]).values          # value at signal date (replay convention)
led["ext"] = led[["rank_2d", "rank_5d", "rank_10d", "rank_21d"]].mean(axis=1)
log(f"rows {len(led)}; cap-bound {(led.cap_scale < 0.999).mean():.1%}; f5 cov {led.f5.notna().mean():.1%}; n_open cov {led.n_open.notna().mean():.1%}")

# flow counts under the BRIEF's family membership (recomputed from the raw candidate dump)
cand = pd.read_parquet(D / "flow_candidates.parquet")
cand["fam_brief"] = cand["strategy"].map(FAMILY_BRIEF)
cand["fam_lib"] = cand["strategy"].map(FAMILY_LIB)
def f5_of(fam_col):
    ct = cand.groupby(["signal_date", fam_col]).size().unstack(fam_col).reindex(BDAYS).fillna(0.0)
    return ct.rolling(5, min_periods=1).sum()
f5b, f5l = f5_of("fam_brief"), f5_of("fam_lib")
led["f5_brief"] = [f5b.at[d, f] if (f in f5b.columns and d in f5b.index) else np.nan for d, f in zip(led["Signal Date"], led["fam_brief"])]
led["f5_lib2"] = [f5l.at[d, f] if (f in f5l.columns and d in f5l.index) else np.nan for d, f in zip(led["Signal Date"], led["fam_lib"])]
chk = led.dropna(subset=["f5"]); log("f5 (study file) vs recomputed lib-family f5: max abs diff", float((chk.f5 - chk.f5_lib2).abs().max()))
dial_ok = ~(led["dial"] >= 50)
def hi_flow(fam_col, f5_col):
    thr = led[fam_col].map(FLOW_THR)
    h = (led[f5_col] >= thr) & thr.notna()
    h[(led[fam_col] == "dip_buy") & ~dial_ok] = False
    return h
led["hi_lib"] = hi_flow("fam_lib", "f5")
led["hi_brief"] = hi_flow("fam_brief", "f5_brief")
flip = led[led.hi_lib != led.hi_brief]
R["flow_family_membership"] = {
    "rows_hi_lib": int(led.hi_lib.sum()), "rows_hi_brief": int(led.hi_brief.sum()), "rows_flipped": int(len(flip)),
    "flipped_by_strategy": flip.groupby("Strategy").size().to_dict(),
    "note": "thresholds 6/7/104 were fit on FAMILY_LIB (St OS Sznl in oversold_hold, 3x Bear in dip_buy, 3x Leader in short_fade); the brief re-assigns them"}
log("hi-flow rows lib vs brief:", R["flow_family_membership"])

# OLV working entries at signal time (later-filled limits inside their T+3 window; unfilled ones are invisible)
olv = led[led.Strategy == "Oversold Low Volume"]
sd, ed = olv["Signal Date"].values, olv["Entry Date"].values
work = np.array([((sd < d) & (ed > d)).sum() for d in sd])
led.loc[olv.index, "n_working"] = work
for s in ("Weak Close Decent Sznls", "LT Trend ST OS"):
    sub = led[led.Strategy == s]; sd2, ed2 = sub["Signal Date"].values, sub["Entry Date"].values
    led.loc[sub.index, "n_working"] = [((sd2 < d) & (ed2 > d)).sum() for d in sd2]
led["n_working"] = led["n_working"].fillna(0)
led["depth_ship"] = led["n_open"].fillna(0) + led["n_working"]

# IOB clone days and the clamp pairs (same date, same tradeable ticker)
led["tradeable"] = led["Ticker"].map(lambda t: SPOT.get(t, t))
iob = led[led.Strategy == "Indices Oversold Bounce"]
both = iob.groupby("Signal Date")["tradeable"].nunique()
led["iob_clone"] = (led.Strategy == "Indices Oversold Bounce") & led["Signal Date"].map(both).fillna(0).ge(2).values
pairs_new = set()
grp = led.groupby(["Signal Date", "tradeable"])["Strategy"].agg(set)
for (dte, tk), ss in grp.items():
    for a, b in CLAMP_PAIRS[1:]:
        if a in ss and b in ss:
            pairs_new.update(led.index[(led["Signal Date"] == dte) & (led["tradeable"] == tk) & led["Strategy"].isin([a, b])])
led["clamp_new"] = led.index.isin(pairs_new)
log("IOB clone rows", int(led.iob_clone.sum()), "new clamp-pair rows", int(led.clamp_new.sum()))

# GRM ratio per row: overflow longs excluded from the step
led["grm_ratio_ship"] = np.where((led.Tier == "Overflow") & led.Strategy.isin(OVERFLOW_LONGS), 1.0, 1.25)

# ------------------------------------------------------------------ MTM + notional matrices
days, MTM = build_trade_mtm(led)
day_pos = {d: i for i, d in enumerate(days)}
NOT = np.zeros_like(MTM)
for i, (e, x, sh, ep) in enumerate(zip(led["Entry Date"], led["ExitDate"], led["Shares"], led["EntryPrice"])):
    a, b = day_pos.get(e), day_pos.get(x)
    if a is None or b is None:
        continue
    NOT[i, a:b + 1] = sh * ep
sign = np.where(led["Direction"].values == "Long", 1.0, -1.0).astype(np.float32)
cls_rate = np.where(led["Ticker"].isin(BROAD), 0.08, np.where(led["Ticker"].isin(LEV3X), 0.45, 0.15)).astype(np.float32)
gross0 = pd.Series(NOT.sum(0), index=days) / NAV
led["gross_at_signal"] = gross0.reindex(led["Signal Date"]).fillna(0.0).values
log("MTM reconciliation residual max", float(np.abs(MTM.sum(1) - led["PnL"].values).max()))

spy = pq.read_table(ROOT / "data/master_prices.parquet", columns=["ticker", "date", "Close"], filters=[("ticker", "=", "SPY")]).to_pandas().set_index("date")["Close"]
spy.index = pd.to_datetime(spy.index)
spy_ret = spy.pct_change().reindex(days).fillna(0.0)

# ------------------------------------------------------------------ lever table
def levers(form: str, cfg: dict) -> pd.DataFrame:
    """Per-row multipliers RELATIVE TO TODAY (today's Size_Mult already holds the live overlays).
    form='ship' = brief WP5-8 forms; form='study' = practitioner package C forms."""
    L = pd.DataFrame(index=led.index)
    S = led["Strategy"]
    if form == "study":
        L["tilt"] = S.map(TILT_STUDY).fillna(1.0)
        hi = led["hi_lib"]; flow_m, adds_hi, adds_lo, ext_m, pb_m = 1.25, 1.25, 0.75, 0.5, 1.25
        depth_rung = np.select([led["n_open"] >= 3, led["n_open"] >= 1], [1.0, 0.7], 0.5)
        has_depth = led["n_open"].notna()
        adds_on = led["n_open"] >= 1
    else:
        L["tilt"] = S.map(TILT_SHIP).fillna(1.0)
        hi = led["hi_brief"] if cfg.get("fam", "brief") == "brief" else led["hi_lib"]
        flow_m, adds_hi, adds_lo, ext_m, pb_m = 1.2, 1.2, 0.8, 0.7, 1.15
        depth_rung = np.select([led["depth_ship"] >= 3, led["depth_ship"] >= 1], [1.0, 0.7], 0.5)
        has_depth = led["n_open"].notna()
        adds_on = (led["n_open"] >= 1) | (led["n_working"] >= 1)
    if not cfg.get("tilt", True):
        L["tilt"] = 1.0
    o = S == "Oversold Low Volume"
    rung_old = led["rung_ladder"].fillna(1.0).clip(lower=0.5)
    rung_new = np.where(o & has_depth, np.maximum(rung_old, depth_rung), rung_old)
    L["ladder"] = np.where(o, rung_new / rung_old, 1.0) if cfg.get("olvdep", True) else 1.0
    L["adds"] = 1.0
    for s in ("Weak Close Decent Sznls", "LT Trend ST OS"):
        m = (S == s) & has_depth
        L.loc[m, "adds"] = np.where(adds_on[m], adds_hi, adds_lo)
    if not cfg.get("adds", True):
        L["adds"] = 1.0
    L["ovsx"] = np.where((S == "Overbot Vol Spike") & (led["ext"] < 94), ext_m, 1.0) if cfg.get("ovsx", True) else 1.0
    # study form also carried b52 0.5x at >=6 open (withdrawn in the brief)
    L["b52"] = np.where((S == "52wh Breakout") & (led["n_open"] >= 6), 0.5, 1.0) if form == "study" else 1.0
    fam = led["fam_brief"] if form == "ship" else led["fam_lib"]
    flow_rows = hi & (fam.isin(FLOW_FAMS) if form == "ship" else hi)
    L["flow"] = np.where(flow_rows, flow_m, 1.0) if cfg.get("flow", True) else 1.0
    pb = o & (led["spy_hi252_dist"] < -3.0) & (led["spy_hi252_dist"] >= -10.0)
    L["pullback"] = np.where(pb, pb_m, 1.0) if cfg.get("olvdd", True) else 1.0
    # OVS P2 cap 0.75 -> 1.0: P2 rows below the 0.2 (12/60) path-2 mult were cap-scaled; relax by up to 4/3
    p2c = (S == "Overbot Vol Spike") & (led["Size_Mult"] < 0.199)
    L["p2cap"] = np.where(p2c, np.minimum(4 / 3, 0.2 / led["Size_Mult"]), 1.0) if (form == "ship" and cfg.get("p2cap", True)) else 1.0
    L["iob_clone"] = np.where(led["iob_clone"], 0.5, 1.0) if (form == "ship" and cfg.get("clone", True)) else 1.0
    # guard proxies: study = no flow up-size when baseline gross > 2 NAV at signal; ship = req_proj/NLV > 0.60 (computed below, passed in cfg)
    if cfg.get("guard_days") is not None:
        gd = led["Signal Date"].isin(cfg["guard_days"])
        L.loc[gd, "flow"] = 1.0
    elif form == "study":
        L.loc[hi & (led["gross_at_signal"] > 2.0), "flow"] = 1.0
    # OLV composite clip
    L["clip"] = 1.0
    if cfg.get("clip", True):
        if form == "study" or cfg.get("clip_mode") == "ratio":
            # practitioner: OLV tilt 1.15 and the RATIO-to-today of the product clipped at 1.5
            prod = L["tilt"] * L["ladder"] * L["flow"] * L["pullback"]
            if form == "study":
                prod = prod * (1.15 / 1.29)
            L.loc[o, "clip"] = (prod.clip(upper=1.5) / prod)[o]
        else:
            # brief WP7: ABSOLUTE product tilt x ladder(new rung) x pullback x flow <= 1.5 pre-GRM
            absP = L["tilt"] * pd.Series(rung_new, index=led.index) * L["flow"] * L["pullback"]
            L.loc[o, "clip"] = (absP.clip(upper=1.5) / absP)[o]
    L["grm"] = led["grm_ratio_ship"] if cfg.get("grm", "ship") == "ship" else float(cfg["grm"])
    L["total"] = L[["tilt", "ladder", "adds", "ovsx", "b52", "flow", "pullback", "p2cap", "iob_clone", "clip", "grm"]].prod(axis=1)
    return L


def apply_cap(m: pd.Series, relief_days: set | None, relief_fams: pd.Series | None, ovscap: bool, cap_scaled_grm: float | None = None) -> pd.Series:
    """Ratio of new booked risk to old booked risk per row after re-applying the per-strategy daily cap
    (the practitioner's placed/unseen reconstruction). m already includes the GRM ratio."""
    ratio = pd.Series(np.nan, index=led.index)
    for (strat, sdate), idx in led.groupby(["Strategy", "Signal Date"], sort=False).indices.items():
        rows = led.iloc[idx]
        s0 = float(rows["cap_scale"].min())
        seen0 = float(rows["eff_bps"].sum())
        placed0 = CAP / s0 if s0 < 0.999 else seen0
        unseen0 = max(placed0 - seen0, 0.0)
        mm = m.iloc[idx].values
        new_seen = float((rows["eff_bps"].values * mm).sum())
        new_placed = new_seen + unseen0 * float(mm.mean())
        cap1 = CAP if cap_scaled_grm is None else CAP * cap_scaled_grm
        if ovscap and strat == "Overbot Vol Spike":
            cap1 = max(cap1, 375.0)
        if relief_days is not None and sdate in relief_days and (relief_fams is None or bool(relief_fams.iloc[idx].any())):
            cap1 = max(cap1, 375.0)          # max-not-product
        s1 = min(1.0, cap1 / new_placed) if new_placed > 0 else 1.0
        ratio.iloc[idx] = mm * s1 / s0
    return ratio


def clamp_ext(ratio: pd.Series, grm_mult: float) -> pd.Series:
    """Cross-strategy clamp extension: new pairs clamped to 20 bps nominal x GRM on the row's new risk."""
    cap_risk = CLAMP_BPS_NOMINAL * 1.5 * grm_mult / 1e4 * NAV
    new_risk = led["Risk"] * ratio
    over = led["clamp_new"] & (new_risk > cap_risk)
    r = ratio.copy(); r[over] = cap_risk / led.loc[over, "Risk"]
    return r


def stats(book: pd.Series, lo: str, hi: str) -> dict:
    b = book[(book.index >= lo) & (book.index <= hi)]
    eq = b.cumsum(); dd = eq - eq.cummax()
    yrs = (b.index[-1] - b.index[0]).days / 365.25
    return dict(ann_pnl_pct=round(float(b.sum() / yrs / NAV * 100), 2), ann_vol_pct=round(float(b.std() * np.sqrt(252) / NAV * 100), 2),
                sharpe=round(float(b.mean() / b.std() * np.sqrt(252)), 3), maxdd_pct=round(float(dd.min() / NAV * 100), 2),
                maxdd_trough=str(dd.idxmin().date()), worst_day_pct=round(float(b.min() / NAV * 100), 2), worst_day=str(b.idxmin().date()),
                worst21_pct=round(float(b.rolling(21).sum().min() / NAV * 100), 2), worst21_end=str(b.rolling(21).sum().idxmin().date()),
                worst63_pct=round(float(b.rolling(63).sum().min() / NAV * 100), 2),
                cvar1_pct=round(float(b[b <= b.quantile(0.01)].mean() / NAV * 100), 3))


def dd_episodes(book: pd.Series, rv: np.ndarray, lo: str, n: int = 10) -> list[dict]:
    b = book[book.index >= lo]; eq = b.cumsum(); hwm = eq.cummax(); dd = eq - hwm
    under = dd < 0
    ep_id = (under != under.shift()).cumsum()
    out = []
    for g, seg in dd[under].groupby(ep_id[under]):
        t = seg.idxmin(); pk_pos = eq.index.get_loc(seg.index[0]) - 1
        pk = eq.index[max(pk_pos, 0)]
        rec_candidates = dd.index[(dd.index > t) & (dd >= 0)]
        rec = str(rec_candidates[0].date()) if len(rec_candidates) else "open"
        win = (days > pk) & (days <= t)
        contrib = pd.Series((MTM[:, win] * rv[:, None]).sum(1), index=led.index).groupby(led["Strategy"]).sum().sort_values()
        out.append(dict(peak=str(pk.date()), trough=str(t.date()), recovery=rec, depth_pct=round(float(seg.min() / NAV * 100), 2),
                        days_to_trough=int(((days > pk) & (days <= t)).sum()),
                        top3=[(k, round(v / 1e3, 1)) for k, v in contrib.head(3).items()]))
    return sorted(out, key=lambda e: e["depth_pct"])[:n]


# ------------------------------------------------------------------ guard proxy (req_proj / NLV) for the ship form
def req_series(rv: np.ndarray) -> pd.Series:
    return pd.Series((NOT * (rv * cls_rate)[:, None]).sum(0), index=days)

def guard_days_for(rv: np.ndarray, nlv: float, line: float = 0.60) -> tuple[set, dict]:
    """Days where open-book requirement (prior close) + 1.10 x that day's staged entries (notional at full size) > line x NLV.
    Staged = trades whose Signal Date is d (only FILLED ones are in the ledger: lower bound)."""
    req_open = req_series(rv)
    staged = pd.Series(led["Shares"].values * led["EntryPrice"].values * rv * cls_rate, index=led.index).groupby(led["Signal Date"]).sum().reindex(days).fillna(0.0)
    proj = (req_open.shift(1).fillna(0.0) + 1.10 * staged) / nlv
    gd = set(days[proj > line])
    return gd, dict(days_over_60=int((proj > 0.60).sum()), days_over_70=int((proj > 0.70).sum()), days_over_85=int((proj > 0.85).sum()),
                    max=round(float(proj.max()), 3), max_date=str(proj.idxmax().date()), p99=round(float(proj.quantile(0.99)), 3))


# ------------------------------------------------------------------ configurations
CFGS = {
    "today": dict(form="ship", cfg=dict(tilt=False, olvdep=False, adds=False, ovsx=False, flow=False, olvdd=False, p2cap=False, clone=False, clip=False, grm=1.0), relief=False, clamp=False),
    "today_grm1.875": dict(form="ship", cfg=dict(tilt=False, olvdep=False, adds=False, ovsx=False, flow=False, olvdd=False, p2cap=False, clone=False, clip=False, grm=1.25), relief=False, clamp=False),
    "study_C_grm1.875": dict(form="study", cfg=dict(grm=1.25), relief=True, ovscap=True, clamp=False),
    "ship_grm1.875": dict(form="ship", cfg=dict(), relief=True, clamp=True),
    "ship_grm1.5eq": dict(form="ship", cfg=dict(grm=1.0), relief=True, clamp=True),
    "ship_noclip": dict(form="ship", cfg=dict(clip=False), relief=True, clamp=True),
    "ship_noflow": dict(form="ship", cfg=dict(flow=False), relief=True, clamp=True),
    "ship_noflow_norelief": dict(form="ship", cfg=dict(flow=False), relief=False, clamp=True),
    "ship_ratioclip": dict(form="ship", cfg=dict(clip_mode="ratio"), relief=True, clamp=True),
    "ship_libfam": dict(form="ship", cfg=dict(fam="lib"), relief=True, clamp=True),
    "ship_noguard": dict(form="ship", cfg=dict(), relief=True, clamp=True, noguard=True),
}
WIN = {"2005-2026": ("2005-01-01", "2026-09-01"), "2010-2026": ("2010-01-01", "2026-09-01"), "2016-07+": ("2016-07-20", "2026-09-01")}
books, ratios, levtabs = {}, {}, {}
for name, spec in CFGS.items():
    cfg = dict(spec["cfg"])
    form = spec["form"]
    grm_mult = 1.25 if cfg.get("grm", "ship") == "ship" else float(cfg["grm"])
    hi_col = "hi_brief" if (form == "ship" and cfg.get("fam", "brief") == "brief") else "hi_lib"
    fam_col = "fam_brief" if form == "ship" else "fam_lib"
    relief_fams = led[hi_col] & led[fam_col].isin(FLOW_FAMS if form == "ship" else set(FLOW_THR))
    relief_days = set(led.loc[relief_fams, "Signal Date"]) if spec.get("relief") else None
    # first pass to get the guard days (ship form, req on live NLV)
    L = levers(form, cfg)
    r0 = apply_cap(L["total"], relief_days, relief_fams, spec.get("ovscap", False))
    ginfo = None
    if form == "ship" and not spec.get("noguard") and spec.get("relief"):
        gd, ginfo = guard_days_for(r0.values.astype(np.float32), NLV_LIVE)
        cfg["guard_days"] = gd
        L = levers(form, cfg)
        relief_days = relief_days - gd if relief_days is not None else None
        r0 = apply_cap(L["total"], relief_days, relief_fams, spec.get("ovscap", False))
    if spec.get("clamp"):
        r0 = clamp_ext(r0, grm_mult)
    rv = r0.values.astype(np.float32)
    book = pd.Series((MTM * rv[:, None]).sum(0), index=days)
    books[name], ratios[name], levtabs[name] = book, r0, L
    ent = {"windows": {w: stats(book, *lim) for w, lim in WIN.items()},
           "dd_top10_2005": dd_episodes(book, rv, "2005-01-01"), "dd_top10_2016": dd_episodes(book, rv, "2016-07-20", 6),
           "risk_deployed_ratio": round(float((led["Risk"] * r0).sum() / led["Risk"].sum()), 4),
           "pnl_per_risk": round(float((led["PnL"] * r0).sum() / (led["Risk"] * r0).sum()), 4),
           "ratio_dist": {q: round(float(r0.quantile(q)), 3) for q in (0.05, 0.25, 0.5, 0.75, 0.95, 0.99)}, "ratio_max": round(float(r0.max()), 3),
           "guard": ginfo, "relief_days": len(relief_days) if relief_days else 0}
    olv_rows = led.Strategy == "Oversold Low Volume"
    ent["olv_ratio_dist"] = {q: round(float(r0[olv_rows].quantile(q)), 3) for q in (0.05, 0.5, 0.95)} | {"max": round(float(r0[olv_rows].max()), 3),
                             "share_gt_2x": round(float((r0[olv_rows] > 2.0).mean()), 3), "share_gt_1.5x": round(float((r0[olv_rows] > 1.5).mean()), 3)}
    R.setdefault("configs", {})[name] = ent
    w = ent["windows"]["2005-2026"]; w16 = ent["windows"]["2016-07+"]
    log(f"{name:22s} 2005+: ann {w['ann_pnl_pct']:5.1f} Sh {w['sharpe']:.2f} maxDD {w['maxdd_pct']:6.2f} ({w['maxdd_trough']}) worst {w['worst_day_pct']:5.2f} ({w['worst_day']}) w21 {w['worst21_pct']:6.2f} "
        f"| 2016+: Sh {w16['sharpe']:.2f} maxDD {w16['maxdd_pct']:6.2f} ({w16['maxdd_trough']}) worst {w16['worst_day_pct']:5.2f} w21 {w16['worst21_pct']:6.2f} | risk x{ent['risk_deployed_ratio']:.3f} PPR {ent['pnl_per_risk']:.3f} guard {ginfo}")
    log("    top DD 2016+:", [(e["peak"], e["trough"], e["depth_pct"], e["top3"][0]) for e in ent["dd_top10_2016"][:3]])

# ------------------------------------------------------------------ (a) June 2026 OLV stack detail
win = (days >= pd.Timestamp("2026-06-12")) & (days <= pd.Timestamp("2026-07-01"))
olv_idx = led.index[led.Strategy == "Oversold Low Volume"]
jun = {}
for name in ("today", "study_C_grm1.875", "ship_grm1.875", "ship_noclip", "ship_ratioclip", "ship_noflow"):
    rv = ratios[name].values.astype(np.float32)
    jun[name] = dict(olv_window_pnl_k=round(float((MTM[olv_idx][:, win] * rv[olv_idx, None]).sum() / 1e3), 1),
                     book_window_pnl_k=round(float((MTM[:, win] * rv[:, None]).sum() / 1e3), 1),
                     olv_window_exit_pnl_k=round(float((led.loc[olv_idx, "PnL"] * ratios[name][olv_idx])[(led.loc[olv_idx, "ExitDate"] >= "2026-06-12") & (led.loc[olv_idx, "Entry Date"] <= "2026-07-01")].sum() / 1e3), 1))
R["june_2026_olv_window"] = jun
log("June-2026 OLV window (06-12..07-01) MTM by config:", jun)
# per-leg table for the legs open in the window
legs = led.loc[olv_idx][(led.loc[olv_idx, "Entry Date"] <= "2026-07-01") & (led.loc[olv_idx, "ExitDate"] >= "2026-06-12")].copy()
for name in ("study_C_grm1.875", "ship_grm1.875", "ship_noclip"):
    legs[name] = ratios[name][legs.index].round(2)
    legs[name + "_absmult"] = (legs[name] * legs["Size_Mult"] / led.loc[legs.index, "grm_ratio_ship"]).round(2)
Lship = levtabs["ship_grm1.875"]
legs["tilt"] = Lship.loc[legs.index, "tilt"]; legs["ladder"] = Lship.loc[legs.index, "ladder"]; legs["flow"] = Lship.loc[legs.index, "flow"]; legs["pullback"] = Lship.loc[legs.index, "pullback"]; legs["clip"] = Lship.loc[legs.index, "clip"].round(3)
cols = ["Ticker", "Tier", "Signal Date", "Entry Date", "ExitDate", "Size_Mult", "rung_ladder", "n_open", "n_working", "f5_brief", "spy_hi252_dist", "tilt", "ladder", "flow", "pullback", "clip", "study_C_grm1.875", "ship_grm1.875", "ship_noclip", "Risk", "PnL"]
legs[cols].to_csv(OUTD / "june2026_olv_legs.csv", index=False)
log(legs[cols].to_string())

# ------------------------------------------------------------------ (b) lever stacking
L = levtabs["ship_grm1.875"]
up = pd.DataFrame({"grm": L["grm"] > 1.0, "tilt": L["tilt"] > 1.0, "ladder": L["ladder"] > 1.0, "adds": L["adds"] > 1.0, "flow": L["flow"] > 1.0,
                   "pullback": L["pullback"] > 1.0, "p2cap": L["p2cap"] > 1.0, "fear_boost": led["Size_Mult"].round(3).isin([1.25, 0.625, 1.125, 1.875])})
n_up = up.sum(axis=1)
led["n_up"] = n_up
relief_days_ship = set(led.loc[led.hi_brief & led.fam_brief.isin(FLOW_FAMS), "Signal Date"])
led["relief_day"] = led["Signal Date"].isin(relief_days_ship)
r = ratios["ship_grm1.875"]
led["new_risk"] = led["Risk"] * r
st = {"rows_by_n_up": n_up.value_counts().sort_index().to_dict(), "rows_3plus": int((n_up >= 3).sum()), "rows_4plus": int((n_up >= 4).sum()),
      "ratio_dist_3plus": {q: round(float(r[n_up >= 3].quantile(q)), 2) for q in (0.5, 0.9, 1.0)},
      "lever_combos_3plus": up[n_up >= 3].apply(lambda s: "+".join(s.index[s]), axis=1).value_counts().head(12).to_dict(),
      "strategies_3plus": led.loc[n_up >= 3, "Strategy"].value_counts().to_dict()}
# staged risk per (strategy, day) pre-cap under ship vs cap
Lt = L["total"]
pre = (led["eff_bps"] * Lt).groupby([led["Strategy"], led["Signal Date"]]).sum()
pre0 = led["eff_bps"].groupby([led["Strategy"], led["Signal Date"]]).sum()
st["strategy_days_over_250_today"] = int((pre0 > 250).sum()); st["strategy_days_over_250_ship"] = int((pre > 250).sum()); st["strategy_days_over_375_ship"] = int((pre > 375).sum())
dd3 = led[n_up >= 3].groupby("Signal Date").agg(rows=("Strategy", "size"), strats=("Strategy", lambda s: ",".join(sorted(set(s)))), old_bps=("Risk", lambda s: s.sum() / NAV * 1e4), new_bps=("new_risk", lambda s: s.sum() / NAV * 1e4))
book_day = pd.Series(led.groupby("Signal Date")["new_risk"].sum() / NAV * 1e4)
dd3["book_new_bps"] = book_day.reindex(dd3.index).values
dd3 = dd3.sort_values("book_new_bps", ascending=False).head(15)
st["top_3plus_days"] = dd3.round(0).reset_index().assign(**{"Signal Date": lambda d: d["Signal Date"].astype(str)}).to_dict("records")
# max-not-product audit: where does the design take a max?
prod_all = L[["tilt", "ladder", "adds", "flow", "pullback", "p2cap"]].prod(axis=1)
st["composed_product_preGRM_dist"] = {q: round(float(prod_all.quantile(q)), 3) for q in (0.5, 0.9, 0.99)} | {"max": round(float(prod_all.max()), 3)}
st["rows_product_gt_1.5_preGRM"] = int((prod_all > 1.5).sum()); st["rows_product_gt_1.5_preGRM_nonOLV"] = int(((prod_all > 1.5) & (led.Strategy != "Oversold Low Volume")).sum())
# worst days: how many levers on the trades open that day
worst_days = books["ship_grm1.875"].nsmallest(20).index
open_mat = (NOT > 0)
wd = []
for d in worst_days:
    j = day_pos[d]; rows = led.index[open_mat[:, j]]
    wd.append(dict(date=str(d.date()), book_ship_k=round(float(books["ship_grm1.875"][d] / 1e3), 1), book_today_k=round(float(books["today"][d] / 1e3), 1),
                   open_rows=int(len(rows)), rows_3plus=int((n_up[rows] >= 3).sum()), mean_ratio=round(float(r[rows].mean()), 2), dial=round(float(dial_live.get(d, np.nan)), 1) if d in dial_live.index else None))
st["worst20_ship_days_levers"] = wd
R["stacking"] = st
log("stacking:", {k: v for k, v in st.items() if k not in ("top_3plus_days", "worst20_ship_days_levers")})

# ------------------------------------------------------------------ (c) correlated tail
net_today = pd.Series((NOT * (ratios["today"].values.astype(np.float32) * sign)[:, None]).sum(0), index=days) / NAV
net_ship = pd.Series((NOT * (ratios["ship_grm1.875"].values.astype(np.float32) * sign)[:, None]).sum(0), index=days) / NAV
gross_ship = pd.Series((NOT * ratios["ship_grm1.875"].values.astype(np.float32)[:, None]).sum(0), index=days) / NAV
dial_lag = dial_live.reindex(days).shift(1)
pit = pd.read_parquet(D / "cross_strategy_regime_pit_dial.parquet")["pit"].rolling(10).mean().reindex(days).shift(1)
spy10 = spy_ret[spy_ret.index >= "2010-01-01"]
w20 = spy10.nsmallest(20).index
tab = pd.DataFrame({"spy_ret_pct": (spy_ret[w20] * 100).round(2), "book_today_pct": (books["today"][w20] / NAV * 100).round(2), "book_ship_pct": (books["ship_grm1.875"][w20] / NAV * 100).round(2),
                    "book_todayGRM_pct": (books["today_grm1.875"][w20] / NAV * 100).round(2),
                    "net_today": net_today[w20].round(2), "net_ship": net_ship[w20].round(2), "gross_ship": gross_ship[w20].round(2), "dial_lag": dial_lag[w20].round(1)}).sort_values("spy_ret_pct")
R["worst20_spy_days"] = tab.reset_index().assign(index=lambda d: d["index"].astype(str)).to_dict("records")
R["worst20_spy_summary"] = dict(sum_today_pct=round(float(tab.book_today_pct.sum()), 2), sum_ship_pct=round(float(tab.book_ship_pct.sum()), 2), sum_todayGRM_pct=round(float(tab.book_todayGRM_pct.sum()), 2),
                               mean_net_today=round(float(tab.net_today.mean()), 3), mean_net_ship=round(float(tab.net_ship.mean()), 3))
log("worst-20 SPY days:", R["worst20_spy_summary"]); log(tab.to_string())

def cond_beta(book: pd.Series, dial_s: pd.Series, lo="2016-07-20") -> dict:
    m = pd.DataFrame({"r": book / NAV, "f": spy_ret, "d": dial_s}).loc[lo:].dropna()
    out = {}
    for lab, mask in (("dial>=50", m.d >= 50), ("dial<50", m.d < 50), ("all", m.d.notna())):
        g = m[mask]
        if len(g) < 30:
            continue
        b = np.cov(g.r, g.f)[0, 1] / g.f.var(); r2 = np.corrcoef(g.r, g.f)[0, 1] ** 2
        dn = g[g.f < 0]; bd = np.cov(dn.r, dn.f)[0, 1] / dn.f.var() if len(dn) > 20 else np.nan
        out[lab] = dict(days=int(len(g)), beta=round(float(b), 3), beta_down=round(float(bd), 3), r2=round(float(r2), 3), sd_bps=round(float(g.r.std() * 1e4), 1))
    return out
cb = {}
for name in ("today", "today_grm1.875", "ship_grm1.875", "ship_noflow", "study_C_grm1.875"):
    cb[name] = {"live_dial": cond_beta(books[name], dial_lag), "pit_dial_2018+": cond_beta(books[name], pit, "2018-01-01")}
R["conditional_beta"] = cb
log("conditional beta:", json.dumps(cb, indent=0))
# worst-20 book days: migration toward high dial
mig = {}
for name in ("today", "ship_grm1.875", "study_C_grm1.875"):
    b = books[name].loc["2016-07-20":]; w = b.nsmallest(20)
    d = dial_lag[w.index]
    mig[name] = dict(share_dial_ge50=round(float((d >= 50).mean()), 2), mean_dial=round(float(d.mean()), 1), dates_ge50=[str(x.date()) for x in w.index[d >= 50]],
                     sum_pct=round(float(w.sum() / NAV * 100), 2), n_2026=int((w.index.year == 2026).sum()))
R["worst20_book_days_migration"] = mig
log("worst-20 book-day migration:", mig)
# net exposure at dial >= 50, today vs ship
ne = {}
for lab, s in (("today", net_today), ("ship", net_ship)):
    x = pd.DataFrame({"n": s, "d": dial_lag}).loc["2016-07-20":].dropna()
    ne[lab] = dict(mean_net_ge50=round(float(x[x.d >= 50].n.mean()), 3), mean_net_lt50=round(float(x[x.d < 50].n.mean()), 3), p95_net_ge50=round(float(x[x.d >= 50].n.quantile(.95)), 3))
R["net_exposure_by_dial"] = ne
log("net exposure by dial:", ne)

# ------------------------------------------------------------------ OLV footprint (book cap disabled 2026-08-25)
olv_mask = (led.Strategy == "Oversold Low Volume").values
def olv_not(rv):
    return pd.Series((NOT[olv_mask] * rv[olv_mask, None]).sum(0), index=days) / NAV
fo = {}
for name in ("today", "today_grm1.875", "ship_grm1.875", "ship_noclip", "study_C_grm1.875"):
    s = olv_not(ratios[name].values.astype(np.float32)).loc["2016-07-20":]
    fo[name] = dict(max=round(float(s.max()), 2), max_date=str(s.idxmax().date()), days_gt_100=int((s > 1.0).sum()), days_gt_50=int((s > 0.5).sum()), p99=round(float(s.quantile(.99)), 2))
R["olv_notional_nav"] = fo
log("OLV notional/NAV:", fo)
# fill-size feasibility: participation of new notional vs 21d dollar ADV (signal_quality dollar_vol_m)
part = (led["Shares"] * led["EntryPrice"] * ratios["ship_grm1.875"]) / (led["dollar_vol_m"] * 1e6)
part0 = (led["Shares"] * led["EntryPrice"]) / (led["dollar_vol_m"] * 1e6)
R["participation"] = dict(rows_with_adv=int(part.notna().sum()), today_gt_1pct=int((part0 > 0.01).sum()), ship_gt_1pct=int((part > 0.01).sum()), ship_gt_5pct=int((part > 0.05).sum()),
                          ship_gt_1pct_by_strategy=led.loc[part > 0.01, "Strategy"].value_counts().to_dict(), ship_gt_1pct_by_tier=led.loc[part > 0.01, "Tier"].value_counts().to_dict())
log("participation:", R["participation"])

# ------------------------------------------------------------------ (d) live-overlay carriers x new levers
S = led["Strategy"]; SM = led["Size_Mult"].round(3)
FAM4 = {"Weak Close Decent Sznls", "SPY QQQ MonFri Reversion", "Monday Dip", "Indices Oversold Bounce"}
BAND = FAM4 | {"3x Bear ETF Overbot Fade", "Monthly Weak Close"}
live = pd.DataFrame(index=led.index)
live["frag_band_dial50"] = S.isin(BAND) & (led["dial"] >= 50)
live["pc_fear_boost"] = S.isin(BAND) & SM.isin([1.25, 0.625, 1.125, 1.875])
live["olv_recency_ladder"] = (S == "Oversold Low Volume") & (led["rung_ladder"] < 1.0)
live["earnings_override"] = ((S == "Oversold Low Volume") & SM.isin([0.4, 0.2, 0.28, 0.286, 0.143])) | ((S == "St OS Sznl") & (SM == 0.15))
live["cycle_tilt_ovs"] = (S == "Overbot Vol Spike") & (led["Signal Date"].dt.year % 4 == 2)
live["same_day_derate"] = (S == "3x Bear ETF Overbot Fade") & SM.isin([0.9, 0.8, 0.7, 1.125])
live["gap_derate"] = S.isin(["Monday Dip", "SPY QQQ MonFri Reversion"]) & SM.isin([0.5, 0.625, 0.286])
live["overlap_clamp_live"] = S.isin(["Indices Oversold Bounce", "SPY QQQ MonFri Reversion"]) & SM.isin([0.571, 0.286])
live["ticker_cap_olv"] = (S == "Oversold Low Volume") & (led["residual_mult"] < 0.999) & ~live["earnings_override"]
live["adv_cap"] = pd.Series(False, index=led.index)   # not recoverable from the ledger
live["per_strat_cap_bound"] = led["cap_scale"] < 0.999
live["ovs_p2"] = (S == "Overbot Vol Spike") & (SM <= 0.2)
live["ovs_p2_capped"] = (S == "Overbot Vol Spike") & (SM < 0.199)
live["ovs_scaleout"] = led["Tranche"].isin(["near", "far"])
live["wcds_legacy_sznl_mult"] = (S == "Weak Close Decent Sznls") & SM.isin([1.5, 0.66, 0.825, 1.875])
new = pd.DataFrame(index=led.index)
new["grm_step"] = L["grm"] > 1
new["tilt_ne1"] = L["tilt"] != 1
new["olv_ladder_up"] = L["ladder"] > 1
new["adds"] = L["adds"] != 1
new["ovs_ext"] = L["ovsx"] < 1
new["flow_up"] = L["flow"] > 1
new["relief_day"] = led["relief_day"]
new["pullback"] = L["pullback"] > 1
new["p2cap_relax"] = L["p2cap"] > 1
new["clamp_ext"] = led["clamp_new"]
new["iob_clone"] = led["iob_clone"]
mat = pd.DataFrame({lc: {nc: int((live[lc] & new[nc]).sum()) for nc in new.columns} for lc in live.columns}).T
mat.insert(0, "rows_live", live.sum().astype(int))
mat.to_csv(OUTD / "interaction_matrix.csv")
R["interaction_matrix"] = mat.to_dict("index")
log(mat.to_string())
# specific composition cells
cells = {}
mf = (S == "SPY QQQ MonFri Reversion")
cells["monfri_fear_boost_x_flow"] = int((live.pc_fear_boost & mf & new.flow_up).sum())
cells["monfri_fear_boost_rows"] = int((live.pc_fear_boost & mf).sum())
cells["fam4_dial50_present_rows"] = int(live.frag_band_dial50.sum())
cells["fam4_dial50_x_flow"] = int((live.frag_band_dial50 & new.flow_up).sum())
cells["olv_earnings_override_x_ladder_up"] = int((live.earnings_override & new.olv_ladder_up).sum())
cells["olv_earnings_override_x_pullback"] = int((live.earnings_override & new.pullback).sum())
cells["olv_earnings_override_x_flow"] = int((live.earnings_override & new.flow_up).sum())
cells["olv_tickercap_x_ladder_up"] = int((live.ticker_cap_olv & new.olv_ladder_up).sum())
cells["ovs_cycle_x_ext"] = int((live.cycle_tilt_ovs & new.ovs_ext).sum())
cells["ovs_cycle_x_flow"] = int((live.cycle_tilt_ovs & new.flow_up).sum())
cells["ovs_p2_x_flow"] = int((live.ovs_p2 & new.flow_up).sum())
cells["bear_derate_rows_flow_excluded"] = int((live.same_day_derate).sum())
cells["gap_derate_x_flow"] = int((live.gap_derate & new.flow_up).sum())
cells["gap_derate_x_tilt"] = int((live.gap_derate & new.tilt_ne1).sum())
cells["capbound_x_flow_relief"] = int((live.per_strat_cap_bound & new.relief_day).sum())
cells["wcds_legacy_x_adds"] = int((live.wcds_legacy_sznl_mult & new.adds).sum())
cells["stos_flow_rows_brief_family"] = int(((S == "St OS Sznl") & new.flow_up).sum())
cells["dip_buy_hiflow_rows_dial_pit_vs_live_flip"] = int(((led.fam_brief == "dip_buy") & (led.f5_brief >= 6) & ((led.dial >= 50) != (led.dial_pit.rolling(1).mean() >= 50)) & led.dial_pit.notna()).sum())
R["interaction_cells"] = cells
log("cells:", cells)
# composed absolute multiplier of nominal (not ratio) for MonFri fear-boost hi-flow rows
mfb = live.pc_fear_boost & mf & new.flow_up
R["monfri_fear_flow_abs_mult"] = dict(rows=int(mfb.sum()), abs_mult_of_nominal_today=1.25 * 1.5, abs_mult_ship=round(1.25 * 1.30 * 1.2 * 1.875, 3))

R["meta"] = dict(nav=NAV, nlv_live=NLV_LIVE, ledger_rows=int(len(led)), ledger_end=str(led["Signal Date"].max().date()),
                 notes=["ratios are RELATIVE TO TODAY's booked risk; Size_Mult already holds the live overlays",
                        "depth counts filled legs + LATER-FILLED working entries (unfilled limits are not in the ledger)",
                        "OVS P2 cap relax approximated as min(4/3, 0.2/Size_Mult) on cap-scaled P2 rows",
                        "guard proxy: (open req at prior close + 1.10 x staged FILLED entries) / $632k live NLV > 0.60 turns flow+relief off; stylised 15/8/45 rates",
                        "OLV ticker cap, OVS P2 cap at the row level and fills are NOT re-simulated (same bound as the practitioner replay)",
                        "dial = live 10d-MA 63d at signal date (recompute vintage pre-2026-07-02); PIT variant reported where used"])
json.dump(R, open(OUTD / "dd_tail_results.json", "w"), indent=1, default=str)
pd.DataFrame({k: v for k, v in ratios.items()}).assign(Strategy=led.Strategy, Ticker=led.Ticker, SignalDate=led["Signal Date"], Tier=led.Tier, Size_Mult=led.Size_Mult, n_up=led.n_up).to_csv(OUTD / "per_trade_ratios.csv", index=False)
log("wrote", OUTD / "dd_tail_results.json")
