"""Unconstrained growth, part 1b: (a) a Student-t fit that does not collapse on
the book's flat days (mixture: P(flat) + t on active days), and (b) a
CAP-AWARE growth curve.  Scaling the daily series by m assumes the per-strategy
250 bps/day cap never binds harder; it does.  Rebuild the daily MTM series
from dist/data/trade_mtm.json (per-trade daily PnL vectors, flat $750k, the
2026-08-07 ledger vintage that matches strategy_daily.json) with each trade
scaled by s_i(m) = min(m, 250 / B_sd) where B_sd is the strategy's filled risk
in effective bps on that signal date at m=1.  Also the cap-scales-with-m
variant (cap = 250 m) for comparison.  Writes unconstrained_growth_01b_capaware.json.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(r"C:/Users/McKinley Slade/dev/New_Seasonals")
HERE = Path(__file__).resolve().parent
NAV = 750_000.0
GRM_NOW = 1.5
CAP_BPS = 250.0
M_GRID = np.array([0.5, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0])
M_FINE = np.round(np.arange(0.25, 30.01, 0.25), 2)
OUT: dict = {"m_grid": M_GRID.tolist()}

sd = json.load(open(ROOT / "dist/data/strategy_daily.json"))
dates = pd.to_datetime(sd["dates"])
tot = pd.Series(sd["total_flat"], index=dates, dtype=float)
r_all = tot / NAV

# ------------------------------------------------------------ a. t-fit on active days + flat-day mass
print("=== a. Student-t on ACTIVE days (mixture with flat-day mass) ===")
OUT["t_mixture"] = {}
for wname, r in [("2003+", r_all), ("2016+", r_all[r_all.index >= "2016-01-01"])]:
    rv = r.values; act = rv[rv != 0]; p_flat = 1 - len(act) / len(rv)
    df_t, loc_t, sc_t = stats.t.fit(act)
    xs = np.linspace(-0.6, 0.6, 240_001); pdf = stats.t.pdf(xs, df_t, loc_t, sc_t); dx = xs[1] - xs[0]
    curve = {}
    for m in M_FINE:
        ok = xs > -1.0 / m + 1e-9
        g = 252 * (1 - p_flat) * np.sum(np.log1p(m * xs[ok]) * pdf[ok]) * dx
        pr = (1 - p_flat) * float(stats.t.cdf(-1.0 / m, df_t, loc_t, sc_t))
        curve[float(m)] = (float(g), pr, 1 - (1 - pr) ** 252)
    gv = np.array([curve[float(m)][0] for m in M_FINE]); i = int(np.argmax(gv))
    OUT["t_mixture"][wname] = dict(p_flat=float(p_flat), df=float(df_t), loc=float(loc_t), scale=float(sc_t), active_days=int(len(act)),
                                  m_star_trunc=float(M_FINE[i]), g_star_trunc=float(gv[i]),
                                  curve={f"{m:g}": dict(g_trunc=curve[float(m)][0], p_ruin_day=curve[float(m)][1], p_ruin_year=curve[float(m)][2]) for m in M_GRID})
    print(f"{wname}: flat-day share {p_flat:.1%}, active t df={df_t:.2f} loc={loc_t*1e4:.1f}bps scale={sc_t*1e4:.1f}bps; truncated-growth m*={M_FINE[i]:.2f} g*={gv[i]:.1%}")
    print("   annual P(a day with m*r<-1):", {k: f"{v['p_ruin_year']:.2%}" for k, v in OUT["t_mixture"][wname]["curve"].items() if v["p_ruin_year"] > 1e-3})

# ------------------------------------------------------------ b. cap-aware daily series from per-trade MTM vectors
tm = json.load(open(ROOT / "dist/data/trade_mtm.json"))
tj = json.load(open(ROOT / "dist/data/trades.json"))["columns"]
tr = pd.DataFrame({k: tj[k] for k in ["trade_id", "Strategy", "Tier", "Ticker", "Signal_Date", "Entry_Date", "Exit_Date", "Risk_flat", "Risk_bps", "PnL_flat"]})
tr["Signal_Date"] = pd.to_datetime(tr["Signal_Date"])
mtm_dates = pd.to_datetime(tm["dates"]); T = len(mtm_dates)
main = tm["main"]; tid = np.array(main["trade_id"]); start = np.array(main["start"]); vecs = main["pnl"]
# per-(strategy, signal date) filled risk in bps at m=1
tr["bps"] = tr["Risk_flat"] / NAV * 1e4
B = tr.groupby(["Strategy", "Signal_Date"])["bps"].transform("sum")
tr["B_sd"] = B
byid = tr.set_index("trade_id")
print(f"\n=== b. cap-aware series: {len(tid)} MTM vectors, {len(tr)} trades; signal-day strategy risk at m=1: p50 {B.quantile(.5):.0f} p90 {B.quantile(.9):.0f} p99 {B.quantile(.99):.0f} max {B.max():.0f} bps ===")
OUT["signal_day_risk_bps_at_1x"] = dict(p50=float(B.quantile(.5)), p90=float(B.quantile(.9)), p99=float(B.quantile(.99)), max=float(B.max()),
                                       share_days_over_250_at_m={f"{m:g}": float((tr.drop_duplicates(["Strategy", "Signal_Date"])["B_sd"] * m > CAP_BPS).mean()) for m in M_GRID})
print("share of strategy-signal-days where m*B > 250:", {k: f"{v:.0%}" for k, v in OUT["signal_day_risk_bps_at_1x"]["share_days_over_250_at_m"].items()})

# build a dense matrix trade x day would be 4.7k x 6.1k floats = 230 MB; instead accumulate per m
B_id = byid["B_sd"].reindex(tid).values
print(f"MTM vectors without a trades.json row (uncapped): {np.isnan(B_id).sum()}")
B_id = np.where(np.isnan(B_id), 1e-9, B_id)   # unknown -> treated as uncapped (cap/B huge)
RISK_ID = np.nan_to_num(byid["Risk_flat"].reindex(tid).values, nan=0.0)
def series_at(m: float, cap: float | None) -> np.ndarray:
    s = np.full(len(tid), m) if cap is None else np.minimum(m, cap / B_id)
    out = np.zeros(T)
    for k in range(len(tid)):
        v = vecs[k]; a = start[k]
        out[a:a + len(v)] += s[k] * np.asarray(v)
    return out
base = series_at(1.0, None)
chk = pd.Series(base, index=mtm_dates).reindex(tot.index).fillna(0)
print(f"reconcile m=1 rebuilt vs strategy_daily total: corr {np.corrcoef(chk.values, tot.values)[0,1]:.4f}, sum {chk.sum():,.0f} vs {tot.sum():,.0f}")

def gcurve(r: np.ndarray) -> tuple[float, float, float]:
    x = 1 + r
    if (x <= 0).any(): return -np.inf, float(r.min()), float(r.std() * np.sqrt(252))
    return float(252 * np.mean(np.log(x))), float(r.min()), float(r.std() * np.sqrt(252))
OUT["cap_aware"] = {}
for wname, start_d in [("2016+", "2016-01-01"), ("2003+", "2003-01-01")]:
    mask = np.asarray(mtm_dates >= start_d)
    rows = {}
    for m in M_GRID:
        lin = m * base[mask] / NAV
        fixed = series_at(m, CAP_BPS)[mask] / NAV
        scaled = series_at(m, CAP_BPS * m)[mask] / NAV
        g_lin, w_lin, v_lin = gcurve(lin); g_fix, w_fix, v_fix = gcurve(fixed); g_sc, w_sc, v_sc = gcurve(scaled)
        # effective multiplier realised under the fixed cap: mean deployed risk / mean deployed risk at m=1
        s_fix = np.minimum(m, CAP_BPS / B_id); eff = float(np.average(s_fix, weights=RISK_ID))
        eq = np.cumprod(1 + fixed) if g_fix > -np.inf else None
        dd = float((1 - eq / np.maximum.accumulate(eq)).max()) if eq is not None else None
        rows[f"{m:g}"] = dict(g_linear=g_lin, g_cap_fixed=g_fix, g_cap_scaled=g_sc, eff_mult_fixed_cap=eff,
                              ann_ret_fixed=float(fixed.mean() * 252), ann_vol_fixed=v_fix, worst_day_fixed=w_fix, maxdd_fixed=dd,
                              ann_ret_linear=float(lin.mean() * 252), ann_vol_linear=v_lin, worst_day_linear=w_lin)
        print(f"{wname} m={m:4g}: g linear {g_lin:7.1%} | cap fixed {g_fix:7.1%} (eff mult {eff:.2f}, ret {fixed.mean()*252:.0%}, vol {v_fix:.0%}, DD {dd if dd is None else round(dd,3)}) | cap scaled {g_sc:7.1%}")
    OUT["cap_aware"][wname] = rows
    gf = {float(k): v["g_cap_fixed"] for k, v in rows.items()}
    OUT["cap_aware"][wname + "_summary"] = dict(m_star_fixed_cap=max(gf, key=gf.get), g_star_fixed_cap=max(gf.values()))

json.dump(OUT, open(HERE / "unconstrained_growth_01b_capaware.json", "w"), indent=1, default=float)
print("\nwrote", HERE / "unconstrained_growth_01b_capaware.json")
