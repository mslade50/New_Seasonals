"""OLV-only engine replay, with and without the pivot entry policy.

Arm A: strategy_config as loaded (policy enabled).
Arm B: deepcopy with execution['pivot_entry_policy']['enabled'] = False
       (the documented one-switch rollback; resolver still audits the band).

Both arms: flat $ACCOUNT_VALUE sizing, cap_bps=250, overflow_active=True,
pooled caps None -- the production ledger's flat pass. Candidates generated
from 2003-01-01 (ledger BT_START) so the signal-recency ladder warm-up is
identical to the ledger; reporting windows start 2010-01-01.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import data_provider  # noqa: E402
from strategy_config import ACCOUNT_VALUE  # noqa: E402
from pages.strat_backtester import (  # noqa: E402
    load_seasonal_map,
    load_atr_seasonal_map,
    precompute_all_indicators,
    generate_candidates_fast,
    process_signals_fast,
    get_daily_mtm_series,
)
from daily_portfolio_report import build_full_strategy_book, OVERFLOW_TICKERS  # noqa: E402
from olv_pivot_entry import resolve_olv_pivot_entry_from_row  # noqa: E402

STRAT = "Oversold Low Volume"
GEN_START = dt.date(2003, 1, 1)
REPORT_END = pd.Timestamp("2026-08-31")
ERAS = {"2010": pd.Timestamp("2010-01-01"),
        "2016H2": pd.Timestamp("2016-07-01"),
        "2024": pd.Timestamp("2024-01-01")}


def record_tree_state() -> None:
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout
    lines = [f"run_utc: {dt.datetime.utcnow().isoformat()}Z",
             f"HEAD: {git('rev-parse', 'HEAD').strip()}",
             "git diff --stat:", git("diff", "--stat"),
             "git status --short (money-path files):",
             git("status", "--short", "--", "strategy_config.py", "daily_scan.py",
                 "pages/strat_backtester.py", "olv_pivot_entry.py", "indicators.py", "filters.py")]
    import strategy_config as sc
    olv = next(s for s in sc.STRATEGY_BOOK if s["name"] == STRAT)
    lines += ["OLV execution pivot_entry_policy as loaded:",
              json.dumps(olv["execution"]["pivot_entry_policy"], indent=1),
              f"OLV risk_bps (GRM-scaled): {olv['execution']['risk_bps']}",
              f"OLV signal_recency_ladder: {olv['execution'].get('signal_recency_ladder')}",
              f"OLV fill_window_days: {olv['execution'].get('fill_window_days')}",
              f"OLV stop_mode: {olv['execution'].get('stop_mode')}",
              f"OLV ticker_notional_cap: {olv['execution'].get('ticker_notional_cap')}",
              f"OLV frag_risk_bands: {olv['execution'].get('frag_risk_bands')}",
              f"GLOBAL_RISK_MULTIPLIER: {sc.GLOBAL_RISK_MULTIPLIER}",
              f"ACCOUNT_VALUE: {sc.ACCOUNT_VALUE}",
              f"master_prices mtime: {dt.datetime.utcfromtimestamp(os.path.getmtime(ROOT / 'data' / 'master_prices.parquet')).isoformat()}Z"]
    (HERE / "tree_state.txt").write_text("\n".join(lines))


def r_col(df: pd.DataFrame) -> pd.Series:
    return df["PnL"] / df["Risk $"].replace(0, np.nan)


def censored_mask(df: pd.DataFrame) -> pd.Series:
    return df["Exit Type"].eq("Time") & df["Time Stop"].notna() & df["Exit Date"].lt(df["Time Stop"])


def clustered_t(d: pd.Series, clusters: pd.Series) -> float:
    if len(d) == 0:
        return float("nan")
    s = d.groupby(clusters).sum()
    denom = float(np.sqrt((s ** 2).sum()))
    return float(d.sum() / denom) if denom > 0 else float("nan")


def era_stats(trades: pd.DataFrame, cands: pd.DataFrame, era: pd.Timestamp, md: dict, label: str) -> dict:
    c = cands[(cands["signal_date"] >= era) & (cands["signal_date"] <= REPORT_END)]
    t = trades[(trades["Date"] >= era) & (trades["Date"] <= REPORT_END)].copy()
    cen = censored_mask(t)
    tc = t[~cen]
    r = r_col(tc)
    out = {
        "signals_total": int(len(c)),
        "signals_staged": int((c["staged_" + label]).sum()),
        "fills": int(len(t)),
        "fills_completed": int(len(tc)),
        "censored": int(cen.sum()),
        "avgR": float(r.mean()) if len(r) else float("nan"),
        "medianR": float(r.median()) if len(r) else float("nan"),
        "win_rate": float((r > 0).mean()) if len(r) else float("nan"),
        "pnl_per_unit_risk": float(tc["PnL"].sum() / tc["Risk $"].sum()) if tc["Risk $"].sum() else float("nan"),
        "total_flat_pnl": float(tc["PnL"].sum()),
        "total_risk": float(tc["Risk $"].sum()),
        "total_R": float(r.sum()),
        "stop_rate": float(tc["Exit Type"].eq("Stop").mean()) if len(tc) else float("nan"),
        "target_rate": float(tc["Exit Type"].eq("Target").mean()) if len(tc) else float("nan"),
    }
    if len(t):
        mtm = get_daily_mtm_series(t, md, start_date=era)
        mtm = mtm[mtm.index <= pd.Timestamp("2026-09-03")]
        cum = mtm.cumsum()
        out["worst_21d_flat_pnl"] = float(mtm.rolling(21).sum().min())
        out["max_dd_flat_pnl"] = float((cum - cum.cummax()).min())
        out["worst_day_flat_pnl"] = float(mtm.min())
    return out


def main() -> None:
    record_tree_state()
    full_book = build_full_strategy_book()
    olv_book = [s for s in full_book if s["name"] == STRAT]
    assert len(olv_book) == 2, [s["name"] for s in olv_book]
    liquid_set = set(olv_book[0]["universe_tickers"])
    overflow_set = set(olv_book[1]["universe_tickers"])
    assert not (liquid_set & overflow_set)
    print(f"OLV liquid universe {len(liquid_set)}, overflow universe {len(overflow_set)} "
          f"(OVERFLOW_TICKERS {len(OVERFLOW_TICKERS)})")
    print("liquid risk_bps", olv_book[0]["execution"]["risk_bps"], "overflow risk_bps", olv_book[1]["execution"]["risk_bps"])

    tickers = liquid_set | overflow_set | {"SPY", "^VIX"}
    md = data_provider.get_history(list(tickers), start="2000-01-01")
    print(f"loaded {len(md)} tickers from master_prices")
    vix_df = md.get("^VIX")
    vix_series = None
    if vix_df is not None and not vix_df.empty:
        vd = vix_df.copy()
        vd.columns = [c.capitalize() for c in vd.columns]
        vix_series = vd["Close"]
    sznl_map = load_seasonal_map()
    atr_sznl_map = load_atr_seasonal_map()
    assert atr_sznl_map, "atr_seasonal_ranks.parquet missing -- OLV would under-fire"

    processed = precompute_all_indicators(md, olv_book, sznl_map, vix_series, atr_sznl_map)
    print(f"processed {len(processed)} tickers")
    candidates, signal_data = generate_candidates_fast(processed, olv_book, sznl_map, GEN_START)
    print(f"{len(candidates)} OLV candidates from {GEN_START}")

    # Candidate audit: band each signal would be assigned (policy-independent).
    policy = olv_book[0]["execution"]["pivot_entry_policy"]
    rows = []
    for signal_ts, ticker, t_clean, strat_idx, signal_idx in candidates:
        df = processed[t_clean]
        row = df.iloc[signal_idx]
        atr = signal_data[(t_clean, signal_idx)]["atr"]
        res = resolve_olv_pivot_entry_from_row(row, atr, policy)
        rows.append({
            "signal_date": pd.Timestamp(signal_ts), "ticker": ticker, "t_clean": t_clean,
            "tier": "Liquid" if strat_idx == 0 else "Overflow", "close": float(row["Close"]),
            "atr": float(atr) if pd.notna(atr) else np.nan,
            "nearest_type": res["nearest_type"], "nearest_level": res["nearest_level"],
            "nearest_date": res["nearest_date"], "nearest_source_age_bars": res["nearest_source_age_bars"],
            "high_age": res["pivot_high_source_age_bars"], "low_age": res["pivot_low_source_age_bars"],
            "high_expired": res["pivot_high_expired"], "low_expired": res["pivot_low_expired"],
            "distance_atr": res["distance_atr"], "matched_rule": res["matched_rule"],
            "proposed_action": res["proposed_action"], "proposed_offset_atr": res["proposed_offset_atr"],
        })
    cands = pd.DataFrame(rows)
    cands["affected"] = cands["matched_rule"].ne("default")
    cands["staged_with"] = cands["proposed_action"].ne("skip")
    cands["staged_without"] = True
    cands.to_csv(HERE / "candidates_pivot_audit.csv", index=False)

    book_off = copy.deepcopy(olv_book)
    for s in book_off:
        s["execution"]["pivot_entry_policy"]["enabled"] = False

    common = dict(cap_bps=250, overflow_active=True, flat_sizing=True)
    sig_with = process_signals_fast(list(candidates), signal_data, processed, olv_book, ACCOUNT_VALUE, **common)
    sig_without = process_signals_fast(list(candidates), signal_data, processed, book_off, ACCOUNT_VALUE, **common)
    for name, sig in (("with", sig_with), ("without", sig_without)):
        sig["R"] = r_col(sig)
        sig["Tier"] = np.where(sig["Ticker"].isin(overflow_set), "Overflow", "Liquid")
        sig["censored"] = censored_mask(sig)
        sig.to_csv(HERE / f"replay_{name}_policy.csv", index=False)
    print(f"with-policy trades {len(sig_with)}, without-policy trades {len(sig_without)}")

    # Ledger parity (with-policy arm vs production ledger OLV rows).
    ledger = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
    lo = ledger[ledger["Strategy"] == STRAT].copy()
    key = ["Ticker", "Signal Date", "Entry Date", "Exit Date"]
    lw = sig_with.rename(columns={"Date": "Signal Date"})
    a = lo[key + ["Entry Price", "PnL_flat_750k", "Risk_flat_750k"]].copy()
    b = lw[key + ["Price", "PnL", "Risk $"]].copy()
    m = a.merge(b, on=key, how="outer", indicator=True)
    both = m[m["_merge"] == "both"]
    parity = {
        "ledger_olv_rows": int(len(lo)), "replay_with_rows": int(len(sig_with)),
        "matched_on_ticker_signal_entry_exit": int(len(both)),
        "ledger_only": int((m["_merge"] == "left_only").sum()),
        "replay_only": int((m["_merge"] == "right_only").sum()),
        "max_abs_entry_price_diff": float((both["Entry Price"] - both["Price"]).abs().max()) if len(both) else None,
        "max_abs_pnl_flat_diff": float((both["PnL_flat_750k"] - both["PnL"]).abs().max()) if len(both) else None,
        "max_abs_risk_flat_diff": float((both["Risk_flat_750k"] - both["Risk $"]).abs().max()) if len(both) else None,
        "ledger_only_rows": m[m["_merge"] == "left_only"][key].astype(str).to_dict("records")[:20],
        "replay_only_rows": m[m["_merge"] == "right_only"][key].astype(str).to_dict("records")[:20],
    }
    (HERE / "ledger_parity.json").write_text(json.dumps(parity, indent=2, default=str))
    print("parity", {k: v for k, v in parity.items() if not k.endswith("_rows")})

    # Per-era stats + affected-signal differences.
    summary = {"eras": {}, "affected": {}, "skipped": {}}
    for era_name, era in ERAS.items():
        summary["eras"][era_name] = {
            "with_policy": era_stats(sig_with, cands, era, md, "with"),
            "without_policy": era_stats(sig_without, cands, era, md, "without"),
        }

    # Affected signals: per-signal R with vs without (unfilled/skipped -> 0).
    def per_signal_R(sig: pd.DataFrame) -> pd.DataFrame:
        s = sig[~censored_mask(sig)]
        g = s.groupby(["Ticker", "Date"]).agg(R=("R", "sum"), PnL=("PnL", "sum"), Risk=("Risk $", "sum"),
                                              filled=("R", "size"))
        return g
    with_R = per_signal_R(sig_with)
    without_R = per_signal_R(sig_without)
    cen_keys = set(map(tuple, sig_with.loc[censored_mask(sig_with), ["Ticker", "Date"]].values)) | \
        set(map(tuple, sig_without.loc[censored_mask(sig_without), ["Ticker", "Date"]].values))
    aff = cands[cands["affected"] & (cands["signal_date"] <= REPORT_END)].copy()
    aff["key"] = list(zip(aff["ticker"], aff["signal_date"]))
    aff["censored_either"] = aff["key"].isin(cen_keys)
    aff["R_with"] = [float(with_R["R"].get(k, 0.0)) for k in aff["key"]]
    aff["R_without"] = [float(without_R["R"].get(k, 0.0)) for k in aff["key"]]
    aff["filled_with"] = [int(with_R["filled"].get(k, 0)) for k in aff["key"]]
    aff["filled_without"] = [int(without_R["filled"].get(k, 0)) for k in aff["key"]]
    aff["PnL_with"] = [float(with_R["PnL"].get(k, 0.0)) for k in aff["key"]]
    aff["PnL_without"] = [float(without_R["PnL"].get(k, 0.0)) for k in aff["key"]]
    aff["diff_R"] = aff["R_with"] - aff["R_without"]
    aff.drop(columns=["key"]).to_csv(HERE / "affected_signals.csv", index=False)

    for era_name, era in ERAS.items():
        a = aff[(aff["signal_date"] >= era) & ~aff["censored_either"]]
        by_rule = {}
        for rule, g in a.groupby("matched_rule"):
            by_rule[rule] = {"n": int(len(g)), "filled_with": int((g["filled_with"] > 0).sum()),
                             "filled_without": int((g["filled_without"] > 0).sum()),
                             "sum_R_with": float(g["R_with"].sum()), "sum_R_without": float(g["R_without"].sum()),
                             "sum_diff_R": float(g["diff_R"].sum()), "mean_diff_R": float(g["diff_R"].mean()),
                             "t_clustered": clustered_t(g["diff_R"], g["signal_date"])}
        summary["affected"][era_name] = {
            "n": int(len(a)), "n_censored_excluded": int(((aff["signal_date"] >= era) & aff["censored_either"]).sum()),
            "sum_R_with": float(a["R_with"].sum()), "sum_R_without": float(a["R_without"].sum()),
            "sum_diff_R": float(a["diff_R"].sum()), "mean_diff_R": float(a["diff_R"].mean()) if len(a) else float("nan"),
            "t_clustered_signal_date": clustered_t(a["diff_R"], a["signal_date"]),
            "n_clusters": int(a["signal_date"].nunique()),
            "sum_PnL_with": float(a["PnL_with"].sum()), "sum_PnL_without": float(a["PnL_without"].sum()),
            "by_rule": by_rule,
        }
        sk = a[a["proposed_action"] == "skip"]
        summary["skipped"][era_name] = {
            "n_signals": int(len(sk)), "n_filled_without": int((sk["filled_without"] > 0).sum()),
            "wouldbe_R_without": float(sk["R_without"].sum()),
            "wouldbe_avgR_per_fill": float(sk.loc[sk["filled_without"] > 0, "R_without"].mean()) if (sk["filled_without"] > 0).any() else float("nan"),
            "wouldbe_flat_pnl": float(sk["PnL_without"].sum()),
            "t_clustered_wouldbe_R_vs_zero": clustered_t(sk["R_without"], sk["signal_date"]),
        }
    summary["generation_start"] = str(GEN_START)
    summary["report_end"] = str(REPORT_END.date())
    summary["policy_off_value"] = "execution['pivot_entry_policy']['enabled'] = False"
    summary["n_candidates"] = int(len(cands))
    summary["n_affected_all"] = int(cands["affected"].sum())
    (HERE / "replay_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary["affected"]["2010"], indent=1, default=str))
    print(json.dumps(summary["eras"]["2010"], indent=1, default=str))


if __name__ == "__main__":
    main()
