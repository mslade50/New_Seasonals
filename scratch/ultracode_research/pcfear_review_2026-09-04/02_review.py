"""P/C-fear family band post-ship review, part 1 (prereg gates 1a/1b/1c,
leg B non-inferiority, gate 3 LOYO, Aug-2026 leg-C scoring, live status).

Population: data/backtest_trades_pcfear_shadow.parquet (pc_fear disabled,
incumbent 0.25x table; local build 2026-08-07, last signal 2026-07-29) plus
shadow_2026_pcfear_off.parquet rows after that date (this folder's own
family-only engine re-run, 01_shadow_2026.py). Tranche rows are collapsed
to positions; position R = sum PnL / sum risk.

Dial vintages: A = dd_pit/pit_dial_extended.parquet 'pit' (expanding-window
vintage weights, 10d-MA basis); B = data/rd2_fragility.parquet 63d ->
rolling(10) (the live sizing series). Both looked up exactly the engine's
way (daily grid, ffill limit 5). Fear state: pc_fear.fear_state_asof
(lag-1 by data date), pct re-thresholded for the 80/90 grid cells.

Writes trades_scored.csv, results.json, checks.json here. Reads only."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
import pc_fear  # noqa: E402

FAMILY = ["Weak Close Decent Sznls", "SPY QQQ MonFri Reversion", "Monday Dip",
          "Indices Oversold Bounce", "3x Bear ETF Overbot Fade", "Monthly Weak Close"]
SHADOW = os.path.join(ROOT, "data", "backtest_trades_pcfear_shadow.parquet")
RERUN_OFF = os.path.join(HERE, "shadow_2026_pcfear_off.parquet")
RERUN_ON = os.path.join(HERE, "shadow_2026_pcfear_on.parquet")
RERUN_NB = os.path.join(HERE, "shadow_2026_bands_off.parquet")
VINT_A = os.path.join(ROOT, "scratch", "ultracode_sizing_2026-09-02", "dd_pit", "pit_dial_extended.parquet")
VINT_B = os.path.join(ROOT, "data", "rd2_fragility.parquet")
POP_START = pd.Timestamp("2016-06-01")
SHIP = pd.Timestamp("2026-08-05")
ARM_LIVE = pd.Timestamp("2026-07-30")
KEY = ["Strategy", "Ticker", "Signal Date", "Entry Date"]
pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 500)
log = open(os.path.join(HERE, "02_review.log"), "w", encoding="utf-8")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log.write(s + "\n")


# ---------------------------------------------------------------- population
sh = pd.read_parquet(SHADOW)
sh["Signal Date"] = pd.to_datetime(sh["Signal Date"])
sh = sh[sh["Strategy"].isin(FAMILY)]
rr = pd.read_parquet(RERUN_OFF)
rr["Signal Date"] = pd.to_datetime(rr["Signal Date"])
rr_new = rr[rr["Signal Date"] > sh["Signal Date"].max()]
P(f"shadow rows {len(sh)} (last signal {sh['Signal Date'].max().date()}); re-run rows appended after that: {len(rr_new)}")
raw = pd.concat([sh, rr_new], ignore_index=True)
raw["Entry Date"] = pd.to_datetime(raw["Entry Date"])
raw["Exit Date"] = pd.to_datetime(raw["Exit Date"])
pos = (raw.groupby(KEY)
       .agg(pnl=("PnL_flat_750k", "sum"), risk=("Risk_flat_750k", "sum"),
            exit_date=("Exit Date", "max"), exit_type=("Exit Type", "first"),
            rows=("PnL_flat_750k", "size"), direction=("Direction", "first"))
       .reset_index())
pos["R"] = pos.pnl / pos.risk
pos = pos[pos["Signal Date"] >= POP_START].reset_index(drop=True)
P(f"positions 2016-06+: {len(pos)} (from {int(pos.rows.sum())} rows; tranche collapse changed {int((pos.rows > 1).sum())} positions)")

# ---------------------------------------------------------------- dials
def engine_series(s: pd.Series) -> pd.Series:
    s = s.dropna().copy()
    s.index = pd.to_datetime(s.index).normalize()
    grid = pd.date_range(s.index.min(), s.index.max(), freq="D")
    return s.reindex(grid).ffill(limit=5)


pit = pd.read_parquet(VINT_A)
dial_a = engine_series(pit["pit"])
frag = pd.read_parquet(VINT_B)
dial_b = engine_series(frag["63d"].dropna().rolling(10, min_periods=1).mean())
P(f"vintage A: {VINT_A}  non-null {dial_a.dropna().index.min().date()} .. {dial_a.dropna().index.max().date()}")
P(f"vintage B: {VINT_B}  {dial_b.dropna().index.min().date()} .. {dial_b.dropna().index.max().date()} (frozen_through meta: {frag.attrs})")
pos["dial_A"] = pos["Signal Date"].map(dial_a)
pos["dial_B"] = pos["Signal Date"].map(dial_b)

# ---------------------------------------------------------------- fear (module, lag-1)
fs = {d: pc_fear.fear_state_asof(d) for d in pos["Signal Date"].unique()}
pos["fear_pct"] = pos["Signal Date"].map(lambda d: fs[d]["pct"])
pos["fear_state"] = pos["Signal Date"].map(lambda d: fs[d]["state"])
pos["fear_data_date"] = pos["Signal Date"].map(lambda d: fs[d]["data_date"])
P("fear state counts:", pos["fear_state"].value_counts().to_dict())
pos["scored_A"] = pos["dial_A"].notna()
pos["scored_B"] = pos["dial_B"].notna()
pos["year"] = pos["Signal Date"].dt.year
P(f"scored on A: {int(pos.scored_A.sum())} (first A-scored signal {pos[pos.scored_A]['Signal Date'].min().date()}); scored on B: {int(pos.scored_B.sum())}")
P(f"unscored on B: {int((~pos.scored_B).sum())} -> {pos[~pos.scored_B]['Signal Date'].dt.date.unique().tolist()}")
pos.sort_values(["Signal Date", "Strategy", "Ticker"]).to_csv(os.path.join(HERE, "trades_scored.csv"), index=False)


# ---------------------------------------------------------------- stats
def cell(df: pd.DataFrame) -> dict:
    n = len(df)
    if n == 0:
        return {"n": 0, "dates": 0, "avgR": None, "win": None, "date_avgR": None, "t_date": None, "t_cluster": None}
    dm = df.groupby("Signal Date")["R"].mean()
    g = len(dm)
    t_date = float(dm.mean() / (dm.std(ddof=1) / np.sqrt(g))) if g > 1 and dm.std(ddof=1) > 0 else None
    e = df["R"] - df["R"].mean()
    ssum = e.groupby(df["Signal Date"]).sum()
    var = (ssum ** 2).sum() / n ** 2 * (g / (g - 1) if g > 1 else 1.0)
    t_cl = float(df["R"].mean() / np.sqrt(var)) if var > 0 else None
    return {"n": int(n), "dates": int(g), "avgR": float(df["R"].mean()), "win": float((df["R"] > 0).mean() * 100),
            "date_avgR": float(dm.mean()), "t_date": t_date, "t_cluster": t_cl,
            "se_cluster": float(np.sqrt(var)) if var > 0 else None, "years": sorted(df["year"].unique().tolist())}


def diff(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """a minus b on date-mean series (Welch) + Mann-Whitney + cluster-robust difference."""
    da = a.groupby("Signal Date")["R"].mean()
    db = b.groupby("Signal Date")["R"].mean()
    out = {"diff_avgR": float(a["R"].mean() - b["R"].mean()) if len(a) and len(b) else None,
           "diff_date_avgR": float(da.mean() - db.mean()) if len(da) and len(db) else None}
    if len(da) > 1 and len(db) > 1:
        t, p = sps.ttest_ind(da, db, equal_var=False)
        mw = sps.mannwhitneyu(da, db, alternative="two-sided")
        out.update(welch_t=float(t), welch_p=float(p), mw_U=float(mw.statistic), mw_p=float(mw.pvalue))
        ca, cb = cell(a), cell(b)
        if ca["se_cluster"] and cb["se_cluster"]:
            out["t_cluster_diff"] = out["diff_avgR"] / np.sqrt(ca["se_cluster"] ** 2 + cb["se_cluster"] ** 2)
    return out


def bucket(df: pd.DataFrame, dcol: str, dial_thr: float, fear_thr: float):
    d = df[df[dcol].notna() & df["fear_pct"].notna() & (df["fear_state"] != "stale")]
    hi = d[dcol] >= dial_thr
    on = d["fear_pct"] > fear_thr
    return {"lo_off": d[~hi & ~on], "hi_off": d[hi & ~on], "lo_on": d[~hi & on], "hi_on": d[hi & on]}


def gates(df, dcol, dial_thr=50.0, fear_thr=85.0):
    b = bucket(df, dcol, dial_thr, fear_thr)
    c = {k: cell(v) for k, v in b.items()}
    g1a = diff(b["hi_off"], b["lo_off"])          # deficit: hi minus lo, no fear
    g1b = c["hi_on"]
    legb = diff(b["lo_on"], b["lo_off"])          # fear-ON lo vs no-fear lo
    fear_split_hi = diff(b["hi_on"], b["hi_off"])  # the prereg's own contrast
    return {"cells": c, "gate_1a": g1a, "gate_1b": g1b, "legB": legb, "hi_fear_vs_nofear": fear_split_hi}


def loyo(df, dcol, dial_thr=50.0, fear_thr=85.0):
    b = bucket(df, dcol, dial_thr, fear_thr)["hi_on"]
    out = {}
    for y in sorted(b["year"].unique()):
        rest = b[b["year"] != y]
        out[int(y)] = {"dropped_n": int((b["year"] == y).sum()), "rest_n": int(len(rest)),
                       "rest_avgR": float(rest["R"].mean()) if len(rest) else None}
    by_year = b.groupby("year")["R"].agg(["size", "mean"]).rename(columns={"size": "n", "mean": "avgR"})
    return {"by_year": {int(k): {"n": int(v.n), "avgR": float(v.avgR)} for k, v in by_year.iterrows()}, "drop_one": out}


results = {"vintage_a_path": VINT_A, "vintage_a_last_date": str(dial_a.dropna().index.max().date()),
           "vintage_b_path": VINT_B, "population": {"positions": int(len(pos)), "scored_A": int(pos.scored_A.sum()),
                                                    "scored_B": int(pos.scored_B.sum()),
                                                    "first_signal": str(pos["Signal Date"].min().date()),
                                                    "last_signal": str(pos["Signal Date"].max().date())}}


def fmt(c):
    if c["n"] == 0:
        return "n=0"
    return (f"n={c['n']:3d} dates={c['dates']:3d} avgR={c['avgR']:+.3f} win={c['win']:.0f}% "
            f"dateAvgR={c['date_avgR']:+.3f} t_date={c['t_date'] if c['t_date'] is None else round(c['t_date'], 2)} "
            f"t_cl={c['t_cluster'] if c['t_cluster'] is None else round(c['t_cluster'], 2)} years={c['years']}")


for tag, dcol in [("A", "dial_A"), ("B", "dial_B")]:
    P(f"\n================ VINTAGE {tag} ({dcol}) : 2x2 at dial 50 / fear 85 ================")
    g = gates(pos, dcol)
    for k in ["lo_off", "hi_off", "lo_on", "hi_on"]:
        P(f"  {k:7s} {fmt(g['cells'][k])}")
    P(f"  gate 1a (hi_off - lo_off): diff avgR {g['gate_1a']['diff_avgR']:+.3f}, date-mean diff {g['gate_1a']['diff_date_avgR']:+.3f}, "
      f"Welch t {g['gate_1a'].get('welch_t', float('nan')):+.2f} (p={g['gate_1a'].get('welch_p', float('nan')):.3f}), "
      f"cluster-robust t {g['gate_1a'].get('t_cluster_diff', float('nan')):+.2f}, MW p={g['gate_1a'].get('mw_p', float('nan')):.3f}")
    P(f"  gate 1b (hi_on): avgR {g['gate_1b']['avgR']:+.3f}, t_date {g['gate_1b']['t_date']}, t_cluster {g['gate_1b']['t_cluster']}")
    P(f"  leg B (lo_on - lo_off): diff avgR {g['legB']['diff_avgR']:+.3f}, date-mean diff {g['legB']['diff_date_avgR']:+.3f}, "
      f"Welch t {g['legB'].get('welch_t', float('nan')):+.2f}, MW p={g['legB'].get('mw_p', float('nan')):.3f}")
    P(f"  hi_on - hi_off (prereg contrast): diff {g['hi_fear_vs_nofear']['diff_avgR']:+.3f}, Welch t {g['hi_fear_vs_nofear'].get('welch_t', float('nan')):+.2f} "
      f"(p={g['hi_fear_vs_nofear'].get('welch_p', float('nan')):.3f}), MW p={g['hi_fear_vs_nofear'].get('mw_p', float('nan')):.3f}")
    lo = loyo(pos, dcol)
    P(f"  gate 3 LOYO hi_on by year: {lo['by_year']}")
    P(f"  drop-one: {lo['drop_one']}")
    grid = {}
    P("  gate 1c grid (fear thr x dial thr): 1a Welch t | 1b avgR (n) | pass")
    for ft in [80, 85, 90]:
        for dt in [45, 50, 55]:
            gg = gates(pos, dcol, dial_thr=dt, fear_thr=ft)
            t1a = gg["gate_1a"].get("welch_t")
            r1b = gg["gate_1b"]["avgR"]
            ok = (t1a is not None and t1a <= -1.5) and (r1b is not None and r1b >= 0.3)
            grid[f"fear{ft}_dial{dt}"] = {"gate_1a_t": t1a, "gate_1a_n_hi_off": gg["cells"]["hi_off"]["n"],
                                          "gate_1b_avgR": r1b, "gate_1b_n": gg["gate_1b"]["n"],
                                          "gate_1b_t_date": gg["gate_1b"]["t_date"], "pass_both": bool(ok)}
            P(f"    fear>{ft} dial>={dt}: 1a t={t1a if t1a is None else round(t1a, 2)} (hi_off n={gg['cells']['hi_off']['n']}) | "
              f"1b avgR={r1b if r1b is None else round(r1b, 3)} (n={gg['gate_1b']['n']}) | {'PASS' if ok else 'FAIL'}")
    results[f"vintage_{tag}"] = {"gates_50_85": g, "loyo": lo, "grid": grid,
                                 "grid_cells_passing_both": int(sum(v["pass_both"] for v in grid.values()))}

# ---------------------------------------------------------------- Aug-2026 leg-C scoring (report line, one episode)
on = pd.read_parquet(RERUN_ON); on["Signal Date"] = pd.to_datetime(on["Signal Date"])
nb = pd.read_parquet(RERUN_NB); nb["Signal Date"] = pd.to_datetime(nb["Signal Date"])
k3 = ["Strategy", "Ticker", "Signal Date"]
zer = rr.merge(on[k3].drop_duplicates(), on=k3, how="left", indicator=True)
zer = zer[(zer["_merge"] == "left_only") & (zer["Signal Date"] >= ARM_LIVE)].drop(columns="_merge")
nb_pos = nb.groupby(k3).agg(pnl_100=("PnL_flat_750k", "sum"), risk_100=("Risk_flat_750k", "sum")).reset_index()
zer = zer.merge(nb_pos, on=k3, how="left")
zer["dial_B"] = zer["Signal Date"].map(dial_b)
zer["dial_A"] = zer["Signal Date"].map(dial_a)
zer["fear_pct"] = zer["Signal Date"].map(lambda d: pc_fear.fear_state_asof(d)["pct"])
zer["fear_state"] = zer["Signal Date"].map(lambda d: pc_fear.fear_state_asof(d)["state"])
last_bar = pd.Timestamp("2026-09-03")
zer["open_at_last_bar"] = pd.to_datetime(zer["Exit Date"]) >= last_bar
zer["pnl_025"] = zer["PnL_flat_750k"]
zer = zer.sort_values("Signal Date")
P("\n================ Aug-2026 out-of-sample, leg C (ONE episode; report line, not a gate) ================")
P("rows zeroed by the live table since 2026-07-30 (present in the pc_fear-OFF re-run, absent from the production-rule re-run):")
P(zer[["Strategy", "Ticker", "Signal Date", "Entry Date", "Exit Date", "Exit Type", "dial_B", "dial_A", "fear_pct", "fear_state",
       "R_Multiple", "pnl_025", "pnl_100", "open_at_last_bar"]].round(3).to_string(index=False))
z85 = zer[zer["Signal Date"] >= SHIP]
zpre = zer[zer["Signal Date"] < SHIP]
aug = {"zeroed_since_0805_n": int(len(z85)), "zeroed_0730_to_0804_n": int(len(zpre)),
       "sum_R": float(z85["R_Multiple"].sum()), "avg_R": float(z85["R_Multiple"].mean()) if len(z85) else None,
       "win": float((z85["R_Multiple"] > 0).mean() * 100) if len(z85) else None,
       "usd_at_025": float(z85["pnl_025"].sum()), "usd_at_100": float(z85["pnl_100"].sum()),
       "open_at_last_bar": z85[z85.open_at_last_bar][["Strategy", "Ticker"]].astype(str).agg(" ".join, axis=1).tolist(),
       "rows": z85[["Strategy", "Ticker", "Signal Date", "Exit Date", "Exit Type", "dial_B", "dial_A", "fear_pct", "R_Multiple", "pnl_025", "pnl_100", "open_at_last_bar"]]
       .assign(**{"Signal Date": lambda d: d["Signal Date"].dt.strftime("%Y-%m-%d"), "Exit Date": lambda d: pd.to_datetime(d["Exit Date"]).dt.strftime("%Y-%m-%d")})
       .round(3).to_dict(orient="records")}
P(f"since 2026-08-05: n={aug['zeroed_since_0805_n']}, sum R {aug['sum_R']:+.3f}, avg R {aug['avg_R']:+.3f}, win {aug['win']:.0f}%, "
  f"$ at 0.25x {aug['usd_at_025']:+,.0f}, $ at 1.0x {aug['usd_at_100']:+,.0f}; still open at the 2026-09-03 bar: {aug['open_at_last_bar']}")
P(f"(2026-07-30 .. 08-04, before the rule shipped: n={aug['zeroed_0730_to_0804_n']})")
results["aug2026"] = aug

# ---------------------------------------------------------------- live status since 2026-07-30
sess = frag.index[frag.index >= ARM_LIVE]
rows = []
for d in sess:
    st = pc_fear.fear_state_asof(d)
    b = dial_b.get(pd.Timestamp(d).normalize())
    a = dial_a.get(pd.Timestamp(d).normalize())
    off = bool(b is not None and not pd.isna(b) and b >= 50 and st["state"] == "off")
    rows.append({"session": str(pd.Timestamp(d).date()), "dial_B_live": None if pd.isna(b) else round(float(b), 1),
                 "dial_A_pit": None if a is None or pd.isna(a) else round(float(a), 1),
                 "fear_pct_lag1": None if st["pct"] is None else round(st["pct"], 1), "fear_state": st["state"],
                 "fear_data_date": str(st["data_date"]), "family_zeroed": off})
status = pd.DataFrame(rows)
P("\n================ live-regime status since 2026-07-30 (dial = 10d-MA 63d; fear = lag-1 pct252) ================")
P(status.to_string(index=False))
n_off = int(status["family_zeroed"].sum())
P(f"sessions with the family zeroed (live dial >= 50 and fear OFF): {n_off} of {len(status)}; "
  f"latest: dial {status.iloc[-1]['dial_B_live']}, fear pct {status.iloc[-1]['fear_pct_lag1']} ({status.iloc[-1]['fear_state']})")
results["live_status"] = rows
results["family_off_sessions_since_0730"] = n_off

# ---------------------------------------------------------------- checks.json
A = results["vintage_A"]; B = results["vintage_B"]
la = A["loyo"]["drop_one"]
checks = {
    "vintage_a_path": os.path.relpath(VINT_A, ROOT).replace("\\", "/"),
    "vintage_a_last_date": results["vintage_a_last_date"],
    "trades_scored": int(pos.scored_B.sum()),
    "gate_1a_sigma_A": round(A["gates_50_85"]["gate_1a"]["welch_t"], 3),
    "gate_1a_sigma_B": round(B["gates_50_85"]["gate_1a"]["welch_t"], 3),
    "gate_1b_avgR_A": round(A["gates_50_85"]["gate_1b"]["avgR"], 3),
    "gate_1b_t_A": round(A["gates_50_85"]["gate_1b"]["t_date"], 3),
    "gate_1b_n": int(A["gates_50_85"]["gate_1b"]["n"]),
    "grid_cells_passing_both": int(A["grid_cells_passing_both"]),
    "grid_cells_total": 9,
    "legB_diff_R": round(A["gates_50_85"]["legB"]["diff_avgR"], 3),
    "loyo_min_avgR": round(min(v["rest_avgR"] for v in la.values()), 3),
    "loyo_all_positive": bool(all(v["rest_avgR"] > 0 for v in la.values())),
    "aug2026_zeroed_n": aug["zeroed_since_0805_n"],
    "aug2026_at_025_R": round(aug["sum_R"], 3),
    "aug2026_at_100_R": round(aug["sum_R"], 3),
    "aug2026_at_100_usd": round(aug["usd_at_100"], 0),
    "family_off_sessions_since_0730": n_off,
}
json.dump(checks, open(os.path.join(HERE, "checks.json"), "w"), indent=1)


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, pd.DataFrame):
        return None
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


json.dump(_clean(results), open(os.path.join(HERE, "results.json"), "w"), indent=1, default=str)
P("\nchecks.json:", json.dumps(checks, indent=1))
log.close()
