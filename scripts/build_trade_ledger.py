"""
build_trade_ledger.py — one-shot full-history trade ledger for the whole book.

Runs every strategy in STRATEGY_BOOK against its configured universe (liquid
pass) PLUS the overflow pass for the 6 overflow-eligible strategies, exactly the
way daily_portfolio_report.run_12month_backtest builds the book — but over FULL
history (2003 -> today) instead of the trailing 12 months. Every executed trade
is written to a parquet so downstream questions need no re-run.

Faithful-to-production knobs (imported, not reinvented):
  - build_full_strategy_book()  -> liquid + overflow variants (OLV 35->25 bps)
  - cap_bps=250, overflow_active=True   (matches the live portfolio report)

Stored per trade:
  - identity: trade_id, Strategy, Tier (Liquid/Overflow), Ticker, Direction
  - dates:    Signal Date, Entry Date, Exit Date, Exit Type, Time Stop
  - prices:   Entry Price, Exit Price, Signal Close, T+1 Open, ATR, stop/tgt ATR, Range %
  - sizing-invariant: Return_Pct (signed), R_Multiple, hold_days_target
  - dollars (two bases): PnL_flat_750k / Risk_flat_750k  AND
                         PnL_compounded / Risk_compounded / Equity_at_Signal
  - Risk_bps, Entry Criteria

R_Multiple and Return_Pct do not depend on the sizing basis; only the dollar
columns do. PnL_flat_750k sizes every trade off a fixed $750k (era-comparable);
PnL_compounded follows the realistic growing-equity path the live report uses.
"""
import argparse
import copy
import datetime
import os
import socket
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import data_provider
from strategy_config import STRATEGY_BOOK, ACCOUNT_VALUE
from pages.strat_backtester import (
    download_historical_data,
    load_seasonal_map,
    load_atr_seasonal_map,
    precompute_all_indicators,
    generate_candidates_fast,
    process_signals_fast,
    get_daily_mtm_series,
)
# Pull the production book-builder + overflow definitions so we stay faithful.
from daily_portfolio_report import (
    build_full_strategy_book,
    OVERFLOW_TICKERS,
    OVERFLOW_ELIGIBLE,
)

OUT_PARQUET = os.path.join(_ROOT, "data", "backtest_trades_full.parquet")
OUT_NOGATE = os.path.join(_ROOT, "data", "backtest_trades_nogate.parquet")
OUT_DAILY = os.path.join(_ROOT, "data", "backtest_daily_pnl.parquet")
OUT_SUMMARY = os.path.join(_HERE, "trade_ledger_summary.csv")
DATA_START = datetime.date(2000, 1, 1)   # history for percentile/SMA warmup
BT_START = datetime.date(2003, 1, 1)     # first eligible signal date
DIFF_WINDOW_TD = 15                      # vintage-diff lookback (business days)


def _provenance_meta(n_rows):
    """Build metadata embedded in the parquet schema. daily_scan prints this
    at gate time so a stale or non-GHA ledger vintage is visible in scan logs
    (2026-07-06: a weekend vintage of unknown origin drove live gate blocks)."""
    if os.environ.get("GITHUB_ACTIONS"):
        source = f"gha:{os.environ.get('GITHUB_RUN_ID', 'unknown-run')}"
    else:
        source = f"local:{socket.gethostname()}"
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_ROOT,
            capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        sha = ""
    return {
        "ledger_build_utc": datetime.datetime.now(datetime.timezone.utc)
                            .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ledger_source": source,
        "ledger_git_sha": sha or "unknown",
        "ledger_rows": str(n_rows),
    }


def _write_ledger_with_meta(df, path, meta):
    table = pa.Table.from_pandas(df, preserve_index=False)
    schema_meta = dict(table.schema.metadata or {})
    schema_meta.update({k.encode(): str(v).encode() for k, v in meta.items()})
    pq.write_table(table.replace_schema_metadata(schema_meta), path)


def _prior_vintage_path():
    """Previous ledger vintage for the churn diff: the local file if present
    (about to be overwritten), else a best-effort R2 pull (GHA deploy job
    starts from a clean checkout)."""
    if os.path.exists(OUT_PARQUET):
        return OUT_PARQUET
    try:
        from cache_io import download_to_local, is_configured
        if is_configured():
            tmp = OUT_PARQUET + ".prior"
            if download_to_local("backtest_trades_full.parquet", tmp):
                return tmp
    except Exception:
        pass
    return None


def _diff_vs_prior(new_df, prior_path):
    """Print trades that appeared/disappeared/rebooked in the trailing
    DIFF_WINDOW_TD business days vs the prior vintage. Marginal limit fills
    flicker as yfinance revises recent bars; this churn silently moves the
    live sector_loss_gate (the 2026-07-06 TS/USO false block), so make it
    visible in every build log. Best effort — never fails the build."""
    try:
        cols = ["Strategy", "Tier", "Ticker", "Signal Date", "Entry Date",
                "Exit Date", "Exit Type", "R_Multiple"]
        prior = pd.read_parquet(prior_path)
        prior = prior[[c for c in cols if c in prior.columns]].copy()
        new = new_df[[c for c in cols if c in new_df.columns]].copy()
        for d in (prior, new):
            for c in ("Signal Date", "Exit Date"):
                d[c] = pd.to_datetime(d[c])
        try:
            pmeta = pq.read_schema(prior_path).metadata or {}
            built = (pmeta.get(b"ledger_build_utc") or b"?").decode()
            src = (pmeta.get(b"ledger_source") or b"?").decode()
            prov = f"built {built} by {src}"
        except Exception:
            prov = "no provenance metadata"
        cutoff = pd.Timestamp.today().normalize() - pd.tseries.offsets.BDay(DIFF_WINDOW_TD)
        recent = lambda d: d[(d["Signal Date"] >= cutoff) | (d["Exit Date"] >= cutoff)]
        p, n = recent(prior), recent(new)
        key = lambda d: (d["Strategy"].astype(str) + "|" + d["Ticker"].astype(str)
                         + "|" + d["Signal Date"].dt.strftime("%Y-%m-%d"))
        p, n = p.assign(_k=key(p)).set_index("_k"), n.assign(_k=key(n)).set_index("_k")
        p, n = p[~p.index.duplicated()], n[~n.index.duplicated()]
        added = n.index.difference(p.index)
        removed = p.index.difference(n.index)
        common = n.index.intersection(p.index)
        print(f"\n  Vintage diff vs prior ledger ({prov}), trades touching last {DIFF_WINDOW_TD}td:")
        for k in added:
            r = n.loc[k]
            print(f"    + NEW      {r['Ticker']:<6} {r['Strategy']:<24} sig {r['Signal Date'].date()} "
                  f"exit {r['Exit Date'].date() if pd.notna(r['Exit Date']) else 'open'} "
                  f"{r['R_Multiple']:+.2f}R {r.get('Exit Type', '')}")
        for k in removed:
            r = p.loc[k]
            print(f"    - GONE     {r['Ticker']:<6} {r['Strategy']:<24} sig {r['Signal Date'].date()} "
                  f"was exit {r['Exit Date'].date() if pd.notna(r['Exit Date']) else 'open'} "
                  f"{r['R_Multiple']:+.2f}R {r.get('Exit Type', '')}")
        n_rebooked = 0
        for k in common:
            a, b = p.loc[k], n.loc[k]
            if (abs(float(a["R_Multiple"]) - float(b["R_Multiple"])) > 0.005
                    or a["Exit Date"] != b["Exit Date"]
                    or str(a.get("Exit Type")) != str(b.get("Exit Type"))):
                n_rebooked += 1
                print(f"    ~ REBOOKED {b['Ticker']:<6} {b['Strategy']:<24} sig {b['Signal Date'].date()} "
                      f"{a['R_Multiple']:+.2f}R {a.get('Exit Type', '')} ({a['Exit Date'].date() if pd.notna(a['Exit Date']) else 'open'})"
                      f" -> {b['R_Multiple']:+.2f}R {b.get('Exit Type', '')} ({b['Exit Date'].date() if pd.notna(b['Exit Date']) else 'open'})")
        if not (len(added) or len(removed) or n_rebooked):
            print("    (no churn — recent-window trades identical)")
        else:
            print(f"    churn: {len(added)} new, {len(removed)} gone, {n_rebooked} rebooked "
                  f"(recent-window trades: {len(p)} -> {len(n)})")
    except Exception as e:
        print(f"  (vintage diff skipped: {e})")


def gated_strategy_names(book):
    """Strategies carrying execution['sector_loss_gate'] (OLV today)."""
    return sorted({s["name"] for s in book
                   if (s.get("execution") or {}).get("sector_loss_gate")})


def strip_sector_gate(book):
    nb = copy.deepcopy(book)
    for s in nb:
        (s.get("execution") or {}).pop("sector_loss_gate", None)
    return nb


def shape_flat_trades(sig):
    """process_signals_fast flat-sizing output -> ledger-style flat columns
    (same names build_site.py reads, minus the compounded-basis pair)."""
    df = sig.copy().reset_index(drop=True)
    df = df.rename(columns={
        "Date": "Signal Date",
        "Price": "Entry Price",
        "PnL": "PnL_flat_750k",
        "Risk $": "Risk_flat_750k",
    })
    df["Shares_flat"] = df["Shares"]
    df["Direction"] = np.where(
        df["Action"].astype(str).str.upper().str.contains("SHORT"), "Short", "Long")
    _sign = np.where(df["Direction"] == "Short", -1.0, 1.0)
    df["Return_Pct"] = _sign * (df["Exit Price"] - df["Entry Price"]) / df["Entry Price"] * 100.0
    df["R_Multiple"] = df["PnL_flat_750k"] / df["Risk_flat_750k"].replace(0, np.nan)
    _of = set(OVERFLOW_TICKERS)
    df["Tier"] = np.where(
        df["Strategy"].isin(OVERFLOW_ELIGIBLE) & df["Ticker"].isin(_of),
        "Overflow", "Liquid")
    for c in ["Signal Date", "Entry Date", "Exit Date", "Time Stop"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    return df


def build_nogate_counterfactual(candidates, signal_data, processed, full_book,
                                starting_equity):
    """Counterfactual pass with execution['sector_loss_gate'] stripped from a
    deep-copied book (flat sizing, same candidates). Written to OUT_NOGATE
    restricted to the gated strategies; build_site.py diffs it against the
    main ledger to surface the gate-blocked trades (gate_lab.json). NOT a
    pure baseline+blocked union: an unblocked fill moves OLV's open-position
    count (ladder rungs) and the 250bps/day cap, so a few kept trades can
    resize between the runs — the with/without comparison is still coherent."""
    gated = gated_strategy_names(full_book)
    if not gated:
        print("  No sector_loss_gate strategies in the book — skipping nogate pass.")
        return
    print(f"\n  Processing trades [no sector gate, flat sizing] for {gated} ...")
    sig_ng = process_signals_fast(
        candidates, signal_data, processed, strip_sector_gate(full_book),
        starting_equity, cap_bps=250, overflow_active=True, flat_sizing=True,
    )
    ng = shape_flat_trades(sig_ng)
    ng = ng[ng["Strategy"].isin(gated)].reset_index(drop=True)
    _write_ledger_with_meta(ng, OUT_NOGATE, _provenance_meta(len(ng)))
    print(f"    {len(ng)} nogate trades ({'/'.join(gated)}) -> {OUT_NOGATE}")


def load_data(tickers):
    if data_provider.has_master():
        print(f"  Loading {len(tickers)} tickers from master_prices.parquet ...")
        md = data_provider.get_history(list(tickers), start=DATA_START.strftime("%Y-%m-%d"))
        missing = [t for t in tickers if t not in md or md[t] is None or md[t].empty]
        if missing:
            print(f"  {len(missing)} missing from master (skipping yfinance backfill): "
                  f"{missing[:15]}{'...' if len(missing) > 15 else ''}")
        return md
    print("  No master_prices.parquet — falling back to yfinance ...")
    return download_historical_data(list(tickers), start_date=DATA_START.strftime("%Y-%m-%d"))


def main(upload=False):
    starting_equity = ACCOUNT_VALUE
    print("=" * 74)
    print("FULL-BOOK TRADE LEDGER — all strategies, full history")
    print(f"  Backtest range: {BT_START} -> today | start equity ${starting_equity:,.0f}")
    print("=" * 74)

    full_book = build_full_strategy_book()
    n_liquid = len(STRATEGY_BOOK)
    n_overflow = len(full_book) - n_liquid
    print(f"  Book: {n_liquid} liquid passes + {n_overflow} overflow passes "
          f"(overflow tier = {len(OVERFLOW_TICKERS)} tickers)")

    sznl_map = load_seasonal_map()
    atr_sznl_map = load_atr_seasonal_map()
    if not atr_sznl_map:
        print("  WARNING: atr_seasonal_ranks.parquet missing — ATR-seasonal "
              "strategies (OLV/St OS Sznl/52wh/OVS) will under-fire.")

    all_tickers = set()
    for s in full_book:
        all_tickers.update(s["universe_tickers"])
    all_tickers.update(["SPY", "^VIX"])
    md = load_data(all_tickers)
    if not md:
        print("FAILED to load data")
        return

    vix_df = md.get("^VIX")
    vix_series = None
    if vix_df is not None and not vix_df.empty:
        vd = vix_df.copy()
        if isinstance(vd.columns, pd.MultiIndex):
            vd.columns = vd.columns.get_level_values(0)
        vd.columns = [c.capitalize() for c in vd.columns]
        vix_series = vd["Close"]

    print("\n  Precomputing indicators (full book x full universe — slow part) ...")
    processed = precompute_all_indicators(md, full_book, sznl_map, vix_series, atr_sznl_map)

    print(f"\n  Generating candidates from {BT_START} ...")
    candidates, signal_data = generate_candidates_fast(processed, full_book, sznl_map, BT_START)
    print(f"  {len(candidates)} candidate signal-dates")
    if not candidates:
        print("No signals fired.")
        return

    # --- two sizing passes on identical candidates (process is the cheap part) ---
    print("\n  Processing trades [compounded sizing] ...")
    sig_comp = process_signals_fast(
        candidates, signal_data, processed, full_book, starting_equity,
        cap_bps=250, overflow_active=True,
    )
    print(f"    {len(sig_comp)} trades")
    print("  Processing trades [flat $750k sizing] ...")
    sig_flat = process_signals_fast(
        candidates, signal_data, processed, full_book, starting_equity,
        cap_bps=250, overflow_active=True, flat_sizing=True,
    )
    print(f"    {len(sig_flat)} trades")

    if sig_comp.empty:
        print("No trades executed.")
        return

    # --- sector-loss-gate counterfactual (best effort, never fails the build) ---
    try:
        build_nogate_counterfactual(candidates, signal_data, processed,
                                    full_book, starting_equity)
    except Exception as e:
        print(f"  (nogate counterfactual skipped: {e})")

    df = sig_comp.copy().reset_index(drop=True)

    # Attach flat-sizing dollars. Trade set/order is identical across sizing
    # passes (fill depends on price, cap only scales size), but verify before
    # aligning positionally; fall back to a key-merge if anything drifted.
    key = ["Strategy", "Ticker", "Date", "Entry Date", "Price"]
    aligned = (
        len(sig_flat) == len(df)
        and df[key].reset_index(drop=True).round({"Price": 4}).astype(str).equals(
            sig_flat[key].reset_index(drop=True).round({"Price": 4}).astype(str))
    )
    if aligned:
        df["PnL_flat_750k"] = sig_flat["PnL"].values
        df["Risk_flat_750k"] = sig_flat["Risk $"].values
        df["Shares_flat"] = sig_flat["Shares"].values
        # Size_Mult from the FLAT pass (equity-basis-clean; the earnings override
        # uses starting_equity so the compounded ratio drifts with book growth).
        df["Size_Mult"] = sig_flat["Size_Mult"].values
    else:
        print("  NOTE: sizing passes not positionally aligned — merging on key.")
        fl = sig_flat[key + ["PnL", "Risk $", "Shares", "Size_Mult"]].copy()
        fl["_k"] = fl[key].round({"Price": 4}).astype(str).agg("|".join, axis=1)
        df["_k"] = df[key].round({"Price": 4}).astype(str).agg("|".join, axis=1)
        fl_dedup = fl.drop_duplicates("_k").set_index("_k")
        df["PnL_flat_750k"] = df["_k"].map(fl_dedup["PnL"]).values
        df["Risk_flat_750k"] = df["_k"].map(fl_dedup["Risk $"]).values
        df["Shares_flat"] = df["_k"].map(fl_dedup["Shares"]).values
        df["Size_Mult"] = df["_k"].map(fl_dedup["Size_Mult"]).values
        df.drop(columns="_k", inplace=True)

    # --- derive analysis columns ---
    df = df.rename(columns={
        "Date": "Signal Date",
        "Price": "Entry Price",
        "PnL": "PnL_compounded",
        "Risk $": "Risk_compounded",
        "Equity at Signal": "Equity_at_Signal",
    })
    df["Direction"] = np.where(
        df["Action"].astype(str).str.upper().str.contains("SHORT"), "Short", "Long")
    _sign = np.where(df["Direction"] == "Short", -1.0, 1.0)
    df["Return_Pct"] = _sign * (df["Exit Price"] - df["Entry Price"]) / df["Entry Price"] * 100.0
    df["R_Multiple"] = df["PnL_compounded"] / df["Risk_compounded"].replace(0, np.nan)

    # Tier: a trade is from the overflow pass iff its strategy is overflow-
    # eligible AND its ticker lives in the (disjoint) overflow tier.
    _of = set(OVERFLOW_TICKERS)
    df["Tier"] = np.where(
        df["Strategy"].isin(OVERFLOW_ELIGIBLE) & df["Ticker"].isin(_of),
        "Overflow", "Liquid")

    # target hold days per strategy (reference)
    _hold = {s["name"]: s["execution"].get("hold_days") for s in full_book}
    df["hold_days_target"] = df["Strategy"].map(_hold)

    df.insert(0, "trade_id", np.arange(len(df)))

    col_order = [
        "trade_id", "Strategy", "Tier", "Ticker", "Direction",
        "Signal Date", "Entry Date", "Exit Date", "Exit Type", "Time Stop",
        "Entry Price", "Exit Price", "Signal Close", "T+1 Open",
        "Return_Pct", "R_Multiple",
        "PnL_flat_750k", "Risk_flat_750k",
        "PnL_compounded", "Risk_compounded", "Equity_at_Signal",
        "Risk bps", "Entry Criteria", "ATR", "stop_atr", "tgt_atr",
        "Range %", "Shares", "hold_days_target",
    ]
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order + [c for c in df.columns if c not in col_order]]

    for c in ["Signal Date", "Entry Date", "Exit Date", "Time Stop"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])

    os.makedirs(os.path.dirname(OUT_PARQUET), exist_ok=True)

    # Vintage churn diff BEFORE overwriting the prior copy — recent-window
    # trades that flicker between rebuilds steer the live sector_loss_gate.
    prior = _prior_vintage_path()
    if prior:
        _diff_vs_prior(df, prior)
        if prior.endswith(".prior"):
            try:
                os.remove(prior)
            except OSError:
                pass

    _write_ledger_with_meta(df, OUT_PARQUET, _provenance_meta(len(df)))
    print(f"\n  Wrote {len(df)} trades -> {OUT_PARQUET}")

    # Mirror the ledger to R2 so daily_scan's sector_loss_gate (which reads
    # recent closed trades at scan time) sees exits through yesterday in the
    # GHA screener job. Gated behind --upload (deploy_site.yml passes it):
    # this key sizes/gates LIVE orders, so local runs (refresh_view etc.)
    # must not silently replace it. Graceful no-op without R2 creds.
    if upload:
        try:
            from cache_io import upload_from_local
            upload_from_local(OUT_PARQUET, "backtest_trades_full.parquet")
        except Exception as e:
            print(f"  (R2 mirror skipped: {e})")
    else:
        print("  (R2 mirror skipped: run with --upload to overwrite the prod ledger key)")

    # --- daily portfolio MTM series (both sizing bases) for equity/DD figs ---
    # Uses the raw process_signals_fast frames (Price/PnL/Shares column names).
    print("  Computing daily portfolio MTM series ...")
    pnl_comp = get_daily_mtm_series(sig_comp, md, start_date=BT_START)
    pnl_flat = get_daily_mtm_series(sig_flat, md, start_date=BT_START)
    daily = pd.DataFrame({"pnl_compounded": pnl_comp, "pnl_flat": pnl_flat}).fillna(0.0)
    daily.index.name = "date"
    daily["equity_compounded"] = starting_equity + daily["pnl_compounded"].cumsum()
    daily["equity_flat"] = starting_equity + daily["pnl_flat"].cumsum()
    daily.reset_index().to_parquet(OUT_DAILY, index=False)
    print(f"  Wrote {len(daily)} daily rows -> {OUT_DAILY}")

    # ---- summary ----
    print("\n" + "=" * 74)
    print("LEDGER SUMMARY")
    print("=" * 74)
    print(f"  {len(df)} trades | {df['Ticker'].nunique()} tickers | "
          f"{df['Signal Date'].min().date()} -> {df['Signal Date'].max().date()}")
    print(f"  Tier: " + ", ".join(f"{k}={v}" for k, v in df['Tier'].value_counts().items()))

    rows = []
    for (strat, tier), g in df.groupby(["Strategy", "Tier"]):
        rows.append({
            "Strategy": strat, "Tier": tier, "Trades": len(g),
            "Win%": round((g["PnL_compounded"] > 0).mean() * 100, 1),
            "Tot_R": round(g["R_Multiple"].sum(), 1),
            "Avg_R": round(g["R_Multiple"].mean(), 3),
            "PnL_flat_750k": round(g["PnL_flat_750k"].sum()),
            "AvgRet%": round(g["Return_Pct"].mean(), 2),
        })
    summ = pd.DataFrame(rows).sort_values(["Strategy", "Tier"]).reset_index(drop=True)
    summ.to_csv(OUT_SUMMARY, index=False)
    print()
    print(summ.to_string(index=False))
    print(f"\n  Wrote per-strategy summary -> {OUT_SUMMARY}")
    print("\nDone.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true",
                    help="mirror the ledger to the prod R2 key (deploy pipeline only)")
    main(upload=ap.parse_args().upload)
