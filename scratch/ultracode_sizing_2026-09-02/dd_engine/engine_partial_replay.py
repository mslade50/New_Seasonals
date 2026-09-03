"""
engine_partial_replay.py -- GAP 2 partial engine confirmation (WP5 step 4)
using ONLY levers the existing process_signals_fast can express.

Scenarios (flat $750k basis, per-strategy cap 250 eff, ticker cap, P2 cap,
overlap clamp -- exactly the build_trade_ledger.py flat pass):
  A   baseline reproduction (GRM 1.5, prod book)
  B   GRM 1.875 alone (every bps x1.25, overflow OLV takes the step)
  C   B + overflow-long exclusion (OLV 25->20, LTT 30->24, StOS 40->32, 52wh 35->28 nominal)
  D   C + WP6 keep-adjusted tilt (risk_bps and earnings override x tilt)
  E   D + OVS path2_daily_cap_pct 0.75->1.0 + 5 new overlap-clamp pairs @20 nominal
  L   levers only at GRM 1.5 (tilt + P2 cap + clamp pairs; no step, no exclusion)

Each scenario is run three ways: main, cap_bps=0 (250-cap absorption) and
ticker_notional_cap stripped (ticker-cap absorption).

Nothing in the repo is modified: the book is deep-copied and the engine's
module-level OVERFLOW_RISK_OVERRIDES / GLOBAL_RISK_MULTIPLIER plus
strategy_config.CROSS_STRATEGY_OVERLAP_OVERRIDES are monkeypatched per run.
"""
import copy
import json
import os
import sys
import time

import numpy as np
import pandas as pd

# streamlit's generated protos vs the installed protobuf (same workaround the
# sibling growthmax_* studies use; no package changes).
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import strategy_config as sc                      # noqa: E402
import pages.strat_backtester as sb               # noqa: E402
import build_trade_ledger as btl                  # noqa: E402

ACCOUNT = float(sc.ACCOUNT_VALUE)
PROD_GRM = float(sc.GLOBAL_RISK_MULTIPLIER)       # 1.5
NEW_GRM = 1.875
STEP = NEW_GRM / PROD_GRM                         # 1.25
LEDGER = os.path.join(_ROOT, "data", "backtest_trades_full.parquet")
OUT_JSON = os.path.join(_HERE, "engine_partial_replay.json")
OUT_TRADES_DIR = os.path.join(_HERE, "trades")

OVERFLOW_NOMINAL_TODAY = {"Oversold Low Volume": 25}
OVERFLOW_NOMINAL_EXCL = {
    "Oversold Low Volume": 20,
    "LT Trend ST OS": 24,
    "St OS Sznl": 32,
    "52wh Breakout": 28,
}
TILT = {
    "52wh Breakout": 0.70,
    "Weak Close Decent Sznls": 0.75,
    "Sector BO": 0.87,
    "St OS Sznl": 0.88,
    "Indices Oversold Bounce": 0.89,
    "Overbot Vol Spike": 1.00,
    "LT Trend ST OS": 1.04,
    "Monday Dip": 1.09,
    "ATR Extended Gap Up": 1.10,
    "Oversold Low Volume": 1.17,
    "3x ETF Overbot Fade": 1.27,
    "SPY QQQ MonFri Reversion": 1.30,
}
NEW_CLAMP_PAIRS = [
    ("Monday Dip", "Weak Close Decent Sznls"),
    ("SPY QQQ MonFri Reversion", "Weak Close Decent Sznls"),
    ("Monthly Weak Close", "SPY QQQ MonFri Reversion"),
    ("Monthly Weak Close", "Indices Oversold Bounce"),
    ("Monday Dip", "Indices Oversold Bounce"),
]
CLAMP_NOMINAL = 20.0
WINDOWS = {"2010+": "2010-01-01", "2016-07+": "2016-07-01"}
BPS_KEYS = ("risk_bps", "path1_bps", "path2_bps")


def scenario_book(base_book, grm, overflow_nominal, tilt, p2cap_nominal, clamp_pairs):
    """Deep-copy the prod (GRM-1.5-scaled) book and re-express it at `grm`
    with the levers applied. Returns (book, overflow_dict_for_engine, clamp_list)."""
    g = grm / PROD_GRM
    book = copy.deepcopy(base_book)
    for s in book:
        e = s["execution"]
        t = tilt.get(s["name"], 1.0)
        for k in BPS_KEYS:
            if k in e:
                e[k] = e[k] * g * t
        if "path2_daily_cap_pct" in e:
            e["path2_daily_cap_pct"] = (p2cap_nominal * grm if p2cap_nominal is not None
                                        else e["path2_daily_cap_pct"] * g)
        eo = e.get("earnings_size_override")
        if eo and "risk_bps" in eo:
            eo["risk_bps"] = eo["risk_bps"] * g * t
    # engine overflow path: risk_bps = nominal * GRM (clobbers the book value
    # for tickers outside LIQUID_PLUS_COMMODITIES), so fold the tilt in here.
    ovf = {k: v * tilt.get(k, 1.0) for k, v in overflow_nominal.items()}
    clamp = [{"strategies": ("Indices Oversold Bounce", "SPY QQQ MonFri Reversion"),
              "risk_bps_when_overlapping": CLAMP_NOMINAL * grm}]
    for pair in clamp_pairs:
        clamp.append({"strategies": pair, "risk_bps_when_overlapping": CLAMP_NOMINAL * grm})
    return book, ovf, clamp


def strip_ticker_cap(book):
    nb = copy.deepcopy(book)
    for s in nb:
        s["execution"].pop("ticker_notional_cap", None)
    return nb


def run_engine(candidates, signal_data, processed, book, grm, ovf, clamp, cap_bps=250):
    sb.GLOBAL_RISK_MULTIPLIER = grm
    sb.OVERFLOW_RISK_OVERRIDES = dict(ovf)
    sc.CROSS_STRATEGY_OVERLAP_OVERRIDES = [dict(c) for c in clamp]
    t0 = time.time()
    sig = sb.process_signals_fast(
        list(candidates), signal_data, processed, book, ACCOUNT,
        cap_bps=cap_bps, overflow_active=True, flat_sizing=True,
        max_long_risk_bps=btl.POOLED_LONG_CAP_BPS,
        max_short_risk_bps=btl.POOLED_SHORT_CAP_BPS,
    )
    print(f"      engine: {len(sig)} rows in {time.time()-t0:.1f}s", flush=True)
    return sig.reset_index(drop=True)


def trade_key(df):
    return (df["Strategy"].astype(str) + "|" + df["Ticker"].astype(str) + "|"
            + pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d") + "|"
            + pd.to_datetime(df["Entry Date"]).dt.strftime("%Y-%m-%d") + "|"
            + df["Price"].round(4).astype(str) + "|" + df["Tranche"].astype(str))


def cap_binding(main, alt):
    """Per-trade risk ratio main/alt -> per-strategy share of signal-days bound,
    dollars of staged risk removed and PnL foregone by the cap."""
    m = main.assign(_k=trade_key(main))
    a = alt.assign(_k=trade_key(alt))
    gm = m.groupby("_k").agg(Risk_m=("Risk $", "sum"), PnL_m=("PnL", "sum"),
                             Strategy=("Strategy", "first"), Date=("Date", "first"))
    ga = a.groupby("_k").agg(Risk_a=("Risk $", "sum"), PnL_a=("PnL", "sum"))
    j = gm.join(ga, how="inner")
    j["ratio"] = j["Risk_m"] / j["Risk_a"].replace(0, np.nan)
    j["bound"] = j["ratio"] < 0.999
    out = {}
    for strat, g in j.groupby("Strategy"):
        days = g.groupby("Date")["bound"].any()
        out[strat] = {
            "signal_days": int(len(days)),
            "bound_days": int(days.sum()),
            "bound_share": float(days.mean()) if len(days) else 0.0,
            "trades": int(len(g)),
            "bound_trades": int(g["bound"].sum()),
            "risk_removed": float((g["Risk_a"] - g["Risk_m"]).sum()),
            "pnl_foregone": float((g["PnL_a"] - g["PnL_m"]).sum()),
            "mean_scale_when_bound": float(g.loc[g["bound"], "ratio"].mean()) if g["bound"].any() else 1.0,
        }
    days_all = j.groupby(["Strategy", "Date"])["bound"].any()
    out["_book"] = {
        "signal_days": int(len(days_all)),
        "bound_days": int(days_all.sum()),
        "bound_share": float(days_all.mean()) if len(days_all) else 0.0,
        "risk_removed": float((j["Risk_a"] - j["Risk_m"]).sum()),
        "pnl_foregone": float((j["PnL_a"] - j["PnL_m"]).sum()),
        "unmatched_main": int(len(m) - len(j)),
        "unmatched_alt": int(len(a) - len(j)),
    }
    return out


def daily_by_strategy(sig, md, index):
    out = {}
    for strat, g in sig.groupby("Strategy"):
        s = sb.get_daily_mtm_series(g, md, start_date=btl.BT_START)
        out[strat] = s.reindex(index).fillna(0.0)
    return out


def window_metrics(book_series, per_strat, start):
    s = book_series[book_series.index >= pd.Timestamp(start)]
    n_years = len(s) / 252.0
    cum = s.cumsum()
    dd = cum - cum.cummax()
    trough = dd.idxmin()
    peak = cum.loc[:trough].idxmax()
    r21 = s.rolling(21).sum()
    w21_end = r21.idxmin()
    w21_start = s.index[max(0, s.index.get_loc(w21_end) - 20)]
    contrib_dd = {k: float(v.loc[peak:trough].sum()) for k, v in per_strat.items()}
    contrib_21 = {k: float(v.loc[w21_start:w21_end].sum()) for k, v in per_strat.items()}
    top = lambda d: sorted(d.items(), key=lambda kv: kv[1])[:4]
    return {
        "start": str(pd.Timestamp(start).date()), "end": str(s.index[-1].date()),
        "years": round(n_years, 2),
        "total_pnl": float(s.sum()),
        "annual_pnl_pct": float(s.sum() / n_years / ACCOUNT * 100),
        "sharpe": float(s.mean() / s.std() * np.sqrt(252)) if s.std() > 0 else 0.0,
        "ann_vol_pct": float(s.std() * np.sqrt(252) / ACCOUNT * 100),
        "maxdd_pct": float(dd.min() / ACCOUNT * 100),
        "maxdd_peak": str(peak.date()), "maxdd_trough": str(trough.date()),
        "maxdd_top_contrib": [(k, round(v)) for k, v in top(contrib_dd)],
        "worst_day_pct": float(s.min() / ACCOUNT * 100), "worst_day": str(s.idxmin().date()),
        "worst_21d_pct": float(r21.min() / ACCOUNT * 100),
        "worst_21d_window": f"{w21_start.date()}..{w21_end.date()}",
        "worst_21d_top_contrib": [(k, round(v)) for k, v in top(contrib_21)],
        "annual_pnl_by_year": {int(y): float(v) for y, v in s.groupby(s.index.year).sum().items()},
    }


def trade_stats(sig, start):
    g = sig[pd.to_datetime(sig["Date"]) >= pd.Timestamp(start)]
    r = g["PnL"] / g["Risk $"].replace(0, np.nan)
    per = {}
    for strat, gg in g.groupby("Strategy"):
        rr = gg["PnL"] / gg["Risk $"].replace(0, np.nan)
        per[strat] = {"trades": int(len(gg)), "pnl": float(gg["PnL"].sum()),
                      "risk": float(gg["Risk $"].sum()), "totR": float(rr.sum()),
                      "avgR": float(rr.mean()) if len(rr) else 0.0}
    return {"trades": int(len(g)), "total_pnl": float(g["PnL"].sum()),
            "total_R": float(r.sum()), "avg_R": float(r.mean()),
            "per_strategy": per}


def main():
    t_all = time.time()
    print("=" * 70)
    print("GAP 2 partial engine replay -- existing levers only")
    print("=" * 70, flush=True)

    full_book = btl.build_full_strategy_book()
    print(f"  book: {len(full_book)} passes; overflow tier = {len(btl.OVERFLOW_TICKERS)} tickers")
    for s in full_book[len(sc.STRATEGY_BOOK):]:
        print(f"    overflow variant {s['name']:<24} risk_bps={s['execution']['risk_bps']} "
              f"universe={len(s['universe_tickers'])}")

    sznl_map = sb.load_seasonal_map()
    atr_sznl_map = sb.load_atr_seasonal_map()
    all_tickers = set()
    for s in full_book:
        all_tickers.update(s["universe_tickers"])
    all_tickers.update(["SPY", "^VIX"])
    md = btl.load_data(all_tickers)
    vix_df = md.get("^VIX")
    vix_series = None
    if vix_df is not None and not vix_df.empty:
        vd = vix_df.copy()
        if isinstance(vd.columns, pd.MultiIndex):
            vd.columns = vd.columns.get_level_values(0)
        vd.columns = [c.capitalize() for c in vd.columns]
        vix_series = vd["Close"]

    t0 = time.time()
    print("\n  precompute_all_indicators ...", flush=True)
    processed = sb.precompute_all_indicators(md, full_book, sznl_map, vix_series, atr_sznl_map)
    print(f"    {time.time()-t0:.0f}s", flush=True)
    t0 = time.time()
    candidates, signal_data = sb.generate_candidates_fast(processed, full_book, sznl_map, btl.BT_START)
    print(f"  {len(candidates)} candidates in {time.time()-t0:.0f}s", flush=True)

    scenarios = {
        "A": dict(grm=PROD_GRM, overflow=OVERFLOW_NOMINAL_TODAY, tilt={}, p2cap=None, clamp=[]),
        "B": dict(grm=NEW_GRM, overflow=OVERFLOW_NOMINAL_TODAY, tilt={}, p2cap=None, clamp=[]),
        "C": dict(grm=NEW_GRM, overflow=OVERFLOW_NOMINAL_EXCL, tilt={}, p2cap=None, clamp=[]),
        "D": dict(grm=NEW_GRM, overflow=OVERFLOW_NOMINAL_EXCL, tilt=TILT, p2cap=None, clamp=[]),
        "E": dict(grm=NEW_GRM, overflow=OVERFLOW_NOMINAL_EXCL, tilt=TILT, p2cap=1.0, clamp=NEW_CLAMP_PAIRS),
        "L": dict(grm=PROD_GRM, overflow=OVERFLOW_NOMINAL_TODAY, tilt=TILT, p2cap=1.0, clamp=NEW_CLAMP_PAIRS),
    }
    labels = {
        "A": "baseline (GRM 1.5, prod book)",
        "B": "GRM 1.875 alone",
        "C": "B + overflow-long exclusion",
        "D": "C + WP6 tilt",
        "E": "D + P2 cap 1.0 + clamp pairs",
        "L": "levers only at GRM 1.5 (tilt + P2 cap + clamp)",
    }

    index = pd.date_range(btl.BT_START, pd.Timestamp.today().normalize(), freq="B")
    results = {"meta": {
        "generated": pd.Timestamp.now().isoformat(), "account": ACCOUNT,
        "prod_grm": PROD_GRM, "new_grm": NEW_GRM, "candidates": len(candidates),
        "overflow_tier_tickers": len(btl.OVERFLOW_TICKERS),
        "master_last_date": str(max(v.index.max() for v in md.values() if v is not None and not v.empty).date()),
        "tilt": TILT, "overflow_nominal_excl": OVERFLOW_NOMINAL_EXCL,
        "new_clamp_pairs": NEW_CLAMP_PAIRS, "windows": WINDOWS,
        "scenario_labels": labels,
    }, "scenarios": {}}
    os.makedirs(OUT_TRADES_DIR, exist_ok=True)

    sigs = {}
    for code, cfg in scenarios.items():
        print(f"\n  [{code}] {labels[code]}", flush=True)
        book, ovf, clamp = scenario_book(full_book, cfg["grm"], cfg["overflow"], cfg["tilt"],
                                         cfg["p2cap"], cfg["clamp"])
        ovs = next(s for s in book if s["name"] == "Overbot Vol Spike")["execution"]
        print(f"      GRM={cfg['grm']} ovf_eff={ {k: round(v*cfg['grm'],2) for k,v in ovf.items()} } "
              f"OVS p1={ovs['path1_bps']:.2f} p2={ovs['path2_bps']:.2f} p2cap={ovs['path2_daily_cap_pct']:.3f} "
              f"clamp_eff={clamp[0]['risk_bps_when_overlapping']} pairs={len(clamp)}")
        main_sig = run_engine(candidates, signal_data, processed, book, cfg["grm"], ovf, clamp, cap_bps=250)
        nocap_sig = run_engine(candidates, signal_data, processed, book, cfg["grm"], ovf, clamp, cap_bps=0)
        notick_sig = run_engine(candidates, signal_data, processed, strip_ticker_cap(book),
                                cfg["grm"], ovf, clamp, cap_bps=250)
        sigs[code] = main_sig
        main_sig.to_parquet(os.path.join(OUT_TRADES_DIR, f"{code}_main.parquet"), index=False)
        nocap_sig.to_parquet(os.path.join(OUT_TRADES_DIR, f"{code}_nocap.parquet"), index=False)

        per_strat = daily_by_strategy(main_sig, md, index)
        book_series = sum(per_strat.values())
        pd.DataFrame({"pnl_flat": book_series}).to_parquet(
            os.path.join(OUT_TRADES_DIR, f"{code}_daily.parquet"))
        rec = {"label": labels[code], "config": {
                   "grm": cfg["grm"], "overflow_nominal": ovf, "tilt": cfg["tilt"],
                   "p2cap_nominal": cfg["p2cap"] if cfg["p2cap"] else 0.75,
                   "clamp_pairs": len(clamp)},
               "rows": int(len(main_sig)),
               "total_pnl_all": float(main_sig["PnL"].sum()),
               "daily_sum_check": float(book_series.sum()),
               "windows": {}, "cap250": cap_binding(main_sig, nocap_sig),
               "ticker_cap": cap_binding(main_sig, notick_sig)}
        for wname, wstart in WINDOWS.items():
            wm = window_metrics(book_series, per_strat, wstart)
            wm.update(trade_stats(main_sig, wstart))
            # cap absorption inside the window
            for nm, alt in (("cap250", nocap_sig), ("ticker_cap", notick_sig)):
                cb = cap_binding(main_sig[pd.to_datetime(main_sig["Date"]) >= wstart],
                                 alt[pd.to_datetime(alt["Date"]) >= wstart])
                wm[f"{nm}_book"] = cb["_book"]
                wm[f"{nm}_by_strategy"] = {k: v for k, v in cb.items() if k != "_book"}
            rec["windows"][wname] = wm
            print(f"      {wname:<9} pnl/yr {wm['annual_pnl_pct']:6.2f}%  sharpe {wm['sharpe']:.2f}  "
                  f"maxDD {wm['maxdd_pct']:6.2f}%  worst day {wm['worst_day_pct']:6.2f}%  "
                  f"worst21 {wm['worst_21d_pct']:6.2f}%  trades {wm['trades']}  totR {wm['total_R']:.0f}  "
                  f"cap250 bound {wm['cap250_book']['bound_share']*100:.1f}% days, "
                  f"pnl foregone {wm['cap250_book']['pnl_foregone']:,.0f}", flush=True)
        results["scenarios"][code] = rec

    # ---- baseline reproduction vs the committed ledger ----
    led = pd.read_parquet(LEDGER)
    a = sigs["A"]
    led_cut = led["Signal Date"].max()
    a_cut = a[pd.to_datetime(a["Date"]) <= led_cut]
    a_R = (a_cut["PnL"] / a_cut["Risk $"].replace(0, np.nan)).sum()
    rep = {
        "ledger_rows": int(len(led)), "ledger_pnl_flat": float(led["PnL_flat_750k"].sum()),
        "ledger_totR": float(led["R_Multiple"].sum()), "ledger_last_signal": str(led_cut.date()),
        "ledger_meta": {k.decode(): v.decode() for k, v in
                        (__import__("pyarrow.parquet").parquet.read_schema(LEDGER).metadata or {}).items()
                        if k.startswith(b"ledger")},
        "A_rows_all": int(len(a)), "A_pnl_all": float(a["PnL"].sum()),
        "A_rows_to_ledger_cut": int(len(a_cut)), "A_pnl_to_ledger_cut": float(a_cut["PnL"].sum()),
        "A_totR_to_ledger_cut": float(a_R),
    }
    # per-strategy row/PnL diff
    lg = led.groupby("Strategy").agg(rows=("trade_id", "size"), pnl=("PnL_flat_750k", "sum"))
    ag = a_cut.groupby("Strategy").agg(rows=("PnL", "size"), pnl=("PnL", "sum"))
    cmp_ = lg.join(ag, lsuffix="_ledger", rsuffix="_A", how="outer").fillna(0)
    rep["per_strategy"] = {k: {c: float(v[c]) for c in cmp_.columns} for k, v in cmp_.iterrows()}
    # trade-key diff
    lk = (led["Strategy"] + "|" + led["Ticker"] + "|" + led["Signal Date"].dt.strftime("%Y-%m-%d")
          + "|" + led["Tranche"].astype(str))
    ak = (a_cut["Strategy"] + "|" + a_cut["Ticker"] + "|" + pd.to_datetime(a_cut["Date"]).dt.strftime("%Y-%m-%d")
          + "|" + a_cut["Tranche"].astype(str))
    rep["keys_only_in_ledger"] = sorted(set(lk) - set(ak))[:40]
    rep["keys_only_in_A"] = sorted(set(ak) - set(lk))[:40]
    rep["n_keys_only_in_ledger"] = int(len(set(lk) - set(ak)))
    rep["n_keys_only_in_A"] = int(len(set(ak) - set(lk)))
    results["baseline_reproduction"] = rep
    print("\n  Baseline reproduction vs ledger:")
    print(f"    ledger {rep['ledger_rows']} rows / ${rep['ledger_pnl_flat']:,.0f} / {rep['ledger_totR']:.1f}R "
          f"(signals <= {rep['ledger_last_signal']})")
    print(f"    A      {rep['A_rows_to_ledger_cut']} rows / ${rep['A_pnl_to_ledger_cut']:,.0f} / "
          f"{rep['A_totR_to_ledger_cut']:.1f}R to that cut; {rep['A_rows_all']} rows / "
          f"${rep['A_pnl_all']:,.0f} incl. tail")
    print(f"    key diff: {rep['n_keys_only_in_ledger']} only in ledger, {rep['n_keys_only_in_A']} only in A")

    # ---- GRM absorption: B vs 1.25 x A ----
    absorb = {}
    for wname, wstart in WINDOWS.items():
        A_w, B_w = results["scenarios"]["A"]["windows"][wname], results["scenarios"]["B"]["windows"][wname]
        per = {}
        for strat in A_w["per_strategy"]:
            pa = A_w["per_strategy"][strat]["pnl"]
            pb = B_w["per_strategy"].get(strat, {}).get("pnl", 0.0)
            ra = A_w["per_strategy"][strat]["risk"]
            rb = B_w["per_strategy"].get(strat, {}).get("risk", 0.0)
            per[strat] = {"pnl_A": pa, "pnl_B": pb, "pnl_B_over_1.25A": (pb / (STEP * pa)) if pa else None,
                          "risk_B_over_1.25A": (rb / (STEP * ra)) if ra else None,
                          "cap250_bound_share_A": A_w["cap250_by_strategy"].get(strat, {}).get("bound_share"),
                          "cap250_bound_share_B": B_w["cap250_by_strategy"].get(strat, {}).get("bound_share"),
                          "tick_bound_share_A": A_w["ticker_cap_by_strategy"].get(strat, {}).get("bound_share"),
                          "tick_bound_share_B": B_w["ticker_cap_by_strategy"].get(strat, {}).get("bound_share")}
        absorb[wname] = {
            "pnl_A": A_w["total_pnl"], "pnl_B": B_w["total_pnl"], "expected_B": STEP * A_w["total_pnl"],
            "realised_step": B_w["total_pnl"] / A_w["total_pnl"] if A_w["total_pnl"] else None,
            "absorbed_share_of_step": 1.0 - (B_w["total_pnl"] - A_w["total_pnl"]) / ((STEP - 1) * A_w["total_pnl"])
                                      if A_w["total_pnl"] else None,
            "risk_A": sum(v["risk"] for v in A_w["per_strategy"].values()),
            "risk_B": sum(v["risk"] for v in B_w["per_strategy"].values()),
            "per_strategy": per,
        }
        print(f"\n  GRM absorption {wname}: A ${A_w['total_pnl']:,.0f} -> B ${B_w['total_pnl']:,.0f} "
              f"(1.25x A = ${STEP*A_w['total_pnl']:,.0f}); realised step {absorb[wname]['realised_step']:.3f}, "
              f"absorbed {absorb[wname]['absorbed_share_of_step']*100:.1f}% of the step")
    results["grm_absorption"] = absorb

    # ---- WP5 step-4 criteria on E vs A ----
    crit = {}
    for wname in WINDOWS:
        A_w, E_w = results["scenarios"]["A"]["windows"][wname], results["scenarios"]["E"]["windows"][wname]
        L_w = results["scenarios"]["L"]["windows"][wname]
        d_ann = E_w["annual_pnl_pct"] - A_w["annual_pnl_pct"]
        dd_eq = E_w["maxdd_pct"] / STEP
        w21_eq = E_w["worst_21d_pct"] / STEP
        crit[wname] = {
            "annual_gain_pts": d_ann, "annual_gain_pass": d_ann >= 4.0,
            "maxdd_A": A_w["maxdd_pct"], "maxdd_E": E_w["maxdd_pct"], "maxdd_E_grm15_equiv": dd_eq,
            "maxdd_L_levers_only": L_w["maxdd_pct"],
            "maxdd_pass_equiv": dd_eq >= A_w["maxdd_pct"] - 1.0,
            "maxdd_pass_levers_only": L_w["maxdd_pct"] >= A_w["maxdd_pct"] - 1.0,
            "worst21_A": A_w["worst_21d_pct"], "worst21_E": E_w["worst_21d_pct"], "worst21_E_grm15_equiv": w21_eq,
            "worst21_L_levers_only": L_w["worst_21d_pct"],
            "worst21_pass_equiv": w21_eq >= A_w["worst_21d_pct"] * 1.10,
            "worst21_pass_levers_only": L_w["worst_21d_pct"] >= A_w["worst_21d_pct"] * 1.10,
            "worst21_pass_raw": E_w["worst_21d_pct"] >= A_w["worst_21d_pct"] * 1.10,
            "E_maxdd_episode": f"{E_w['maxdd_peak']}..{E_w['maxdd_trough']}",
            "E_maxdd_top_contrib": E_w["maxdd_top_contrib"],
            "E_worst21_window": E_w["worst_21d_window"], "E_worst21_top_contrib": E_w["worst_21d_top_contrib"],
            "A_maxdd_episode": f"{A_w['maxdd_peak']}..{A_w['maxdd_trough']}",
            "A_maxdd_top_contrib": A_w["maxdd_top_contrib"],
        }
    results["wp5_step4_criteria"] = crit

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"\n  wrote {OUT_JSON}  ({time.time()-t_all:.0f}s total)")


if __name__ == "__main__":
    main()
