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
import json
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
OUT_OVERLAY_FREE = os.path.join(_ROOT, "data", "backtest_trades_overlay_free.parquet")
OUT_NOGATE = os.path.join(_ROOT, "data", "backtest_trades_nogate.parquet")
OUT_OVSEXT = os.path.join(_ROOT, "data", "backtest_trades_ovsext.parquet")
OUT_PCSHADOW = os.path.join(_ROOT, "data", "backtest_trades_pcfear_shadow.parquet")
OUT_DAILY = os.path.join(_ROOT, "data", "backtest_daily_pnl.parquet")
OUT_OVERLAY_FREE_DAILY = os.path.join(
    _ROOT, "data", "backtest_daily_pnl_overlay_free.parquet")
OUT_OVERLAY_LAB = os.path.join(_ROOT, "data", "backtest_overlay_lab.json")
OUT_SUMMARY = os.path.join(_HERE, "trade_ledger_summary.csv")
DATA_START = datetime.date(2000, 1, 1)   # history for percentile/SMA warmup
BT_START = datetime.date(2003, 1, 1)     # first eligible signal date
DIFF_WINDOW_TD = 15                      # vintage-diff lookback (business days)

# Pooled per-direction daily risk caps: REMOVED from prod 2026-07-16
# (McKinley). The cap-impact study (scratch/cap_impact_study.py) showed the
# pooled layer bound on the same net-positive cluster days as the
# per-strategy 250 and cost ~$125k/23y with IDENTICAL maxDD and worst day —
# redundant with the per-strategy cap, which stays. None disables in the
# engine. Change together with order_staging.py (pooled stage removed there
# same day) + daily_portfolio_report call site + strat_backtester UI
# defaults. Engine machinery retained for counterfactuals
# (tests/test_pooled_cap_sequential.py still guards it).
POOLED_LONG_CAP_BPS = None
POOLED_SHORT_CAP_BPS = None

# Cross-cutting filters and sizing controls layered on top of the strategies'
# core signal/entry/exit definitions.  The private site's overlay-free book
# removes exactly this allow-list; adding a new production overlay therefore
# requires an explicit decision here rather than silently contaminating the
# counterfactual.
OVERLAY_EXECUTION_KEYS = frozenset({
    "cycle_risk_mults",
    "earnings_blackout_td",
    "earnings_size_override",
    "frag_risk_bands",
    "gap_size_derate",
    "ladder_multipliers",
    "pc_fear_bands",
    "same_day_derate_floor",
    "same_day_signal_derate",
    "sector_loss_gate",
    "signal_recency_ladder",
    "ticker_notional_cap",
})

# Overlay Lab controls.  Each row becomes one exact standalone engine replay
# against the all-off book, plus one checkbox in the private site.  The client
# may add standalone deltas together for fast multi-checkbox attribution, but
# it always preserves the actual production curve alongside that estimate.
OVERLAY_LAB_SPECS = (
    {
        "id": "risk_dial_gates",
        "label": "Risk-dial signal gates",
        "description": "Block 52wh Breakout and St OS Sznl signals when their production risk-dial thresholds fail.",
        "settings_keys": ("dial_filters",),
        "regenerate_candidates": True,
    },
    {
        "id": "t1_gap_kill",
        "label": "T+1 gap-kill filter",
        "description": "Drop the configured Friday SPY/QQQ reversion signals after an adverse T+1 opening gap.",
        "settings_keys": ("use_t1_gap_kill",),
        "regenerate_candidates": True,
    },
    {
        "id": "earnings_blackouts",
        "label": "Earnings blackouts",
        "description": "Suppress configured OVS and LT Trend ST OS signals near earnings dates.",
        "execution_keys": ("earnings_blackout_td",),
    },
    {
        "id": "fragility_pc_sizing",
        "label": "Fragility + put/call sizing",
        "description": "Apply the production fragility bands and their put/call fear-state table selection.",
        "execution_keys": ("frag_risk_bands", "pc_fear_bands"),
    },
    {
        "id": "earnings_size_overrides",
        "label": "Earnings size overrides",
        "description": "Replace normal risk with the configured reduced size near earnings.",
        "execution_keys": ("earnings_size_override",),
    },
    {
        "id": "cycle_year_sizing",
        "label": "Cycle-year sizing",
        "description": "Apply the OVS midterm-year risk reduction.",
        "execution_keys": ("cycle_risk_mults",),
    },
    {
        "id": "signal_recency_sizing",
        "label": "Signal-recency ladder",
        "description": "Scale OLV risk by its recent same-ticker signal count.",
        "execution_keys": ("signal_recency_ladder", "ladder_multipliers"),
    },
    {
        "id": "same_day_signal_sizing",
        "label": "Same-day signal de-rate",
        "description": "Reduce risk when a configured strategy produces several signals on the same day.",
        "execution_keys": ("same_day_signal_derate", "same_day_derate_floor"),
    },
    {
        "id": "gap_size_derates",
        "label": "Opening-gap size de-rates",
        "description": "Reduce configured trade sizes after a large T+1 opening gap.",
        "execution_keys": ("gap_size_derate",),
    },
    {
        "id": "ticker_notional_caps",
        "label": "Per-ticker notional cap",
        "description": "Limit stacked OLV exposure in a single non-exempt ticker.",
        "execution_keys": ("ticker_notional_cap",),
    },
    # "wcds_seasonal_sizing" was removed 2026-09-04 (plan D3.1): the engine no
    # longer carries the Weak Close seasonal-rank size tiers, so the control
    # replayed as a zero-delta pass.
    {
        "id": "ovs_path2_sizing",
        "label": "OVS mild-gap path sizing",
        "description": "Downsize OVS mild-gap entries and enforce their aggregate daily path-2 cap.",
        "engine_overlay": "ovs_path2_sizing",
        "strategy_names": ("Overbot Vol Spike",),
    },
    {
        "id": "ovs_atr_extended_precedence",
        "label": "ATR Gap / OVS precedence",
        "description": "Drop OVS when ATR Extended Gap Up fires on the same symbol and date.",
        "engine_overlay": "ovs_atr_extended_precedence",
        "strategy_names": ("Overbot Vol Spike", "ATR Extended Gap Up"),
        "require_all_strategy_names": True,
    },
    {
        "id": "cross_strategy_overlap",
        "label": "Cross-strategy overlap clamp",
        "description": "Clamp paired strategies when both fire on the same tradeable ticker and date.",
        "engine_overlay": "cross_strategy_overlap",
        "strategy_names": ("Indices Oversold Bounce", "SPY QQQ MonFri Reversion"),
        "require_all_strategy_names": True,
    },
    {
        "id": "overflow_risk_override",
        "label": "Overflow risk override",
        "description": "Apply the production OLV risk rate to overflow-universe signals.",
        "engine_overlay": "overflow_risk_override",
        "strategy_names": ("Oversold Low Volume",),
        "overflow_active": True,
    },
    {
        "id": "per_strategy_daily_cap",
        "label": "250 bps daily strategy cap",
        "description": "Pro-rata cap each strategy's staged risk on a signal date at 250 bps of equity.",
        "cap_bps": 250,
    },
)


def strip_portfolio_overlays(book):
    """Deep-copy ``book`` and retain only core strategy mechanics.

    Core technical/seasonal conditions, universes, order types, stops,
    targets, and holding periods stay intact.  Regime gates and portfolio
    sizing/risk overlays are removed.  Overflow passes also return to the
    strategy's native liquid ``risk_bps`` so the comparison has one base size.
    Hard-coded cross-strategy/OVS/WCDS overlays are disabled at the engine call
    site with ``portfolio_overlays_enabled=False``.
    """
    clean = copy.deepcopy(book)
    native_risk = {
        s["name"]: (s.get("execution") or {}).get("risk_bps")
        for s in STRATEGY_BOOK
    }
    for strat in clean:
        settings = strat.setdefault("settings", {})
        settings["dial_filters"] = []
        if "use_t1_gap_kill" in settings:
            settings["use_t1_gap_kill"] = False

        execution = strat.setdefault("execution", {})
        for key in OVERLAY_EXECUTION_KEYS:
            execution.pop(key, None)
        base_risk = native_risk.get(strat.get("name"))
        if base_risk is not None:
            execution["risk_bps"] = base_risk
    return clean


def _overlay_spec_active(spec, full_book):
    """Return whether a lab control has a live carrier in this book."""
    settings_keys = spec.get("settings_keys", ())
    execution_keys = spec.get("execution_keys", ())
    if settings_keys and not any(
            any((s.get("settings") or {}).get(k) for k in settings_keys)
            for s in full_book):
        return False
    if execution_keys and not any(
            any((s.get("execution") or {}).get(k) for k in execution_keys)
            for s in full_book):
        return False

    required_names = set(spec.get("strategy_names", ()))
    if required_names:
        present_names = {s.get("name") for s in full_book}
        if spec.get("require_all_strategy_names"):
            if not required_names.issubset(present_names):
                return False
        elif not (required_names & present_names):
            return False

    if spec.get("overflow_active") and len(full_book) <= len(STRATEGY_BOOK):
        return False
    return True


def _book_with_single_overlay(clean_book, full_book, spec):
    """Restore one config-backed overlay onto the clean book by pass index."""
    variant = copy.deepcopy(clean_book)
    if len(variant) != len(full_book):
        raise ValueError("overlay lab book variants are not index-aligned")

    for clean_strat, prod_strat in zip(variant, full_book):
        for key in spec.get("settings_keys", ()):
            prod_settings = prod_strat.get("settings") or {}
            if key in prod_settings:
                clean_strat.setdefault("settings", {})[key] = copy.deepcopy(
                    prod_settings[key])
        for key in spec.get("execution_keys", ()):
            prod_execution = prod_strat.get("execution") or {}
            if key in prod_execution:
                clean_strat.setdefault("execution", {})[key] = copy.deepcopy(
                    prod_execution[key])
    return variant


def _json_pnl_values(series, dates):
    values = series.reindex(dates).fillna(0.0).to_numpy(dtype=float)
    return [round(float(v), 2) if np.isfinite(v) else 0.0 for v in values]


def build_overlay_lab(processed, full_book, clean_book, sznl_map,
                      starting_equity, md, clean_candidates,
                      clean_signal_data, clean_sig_flat,
                      production_sig_flat):
    """Build exact standalone flat-PnL replays for every active overlay.

    A browser cannot replay path-dependent portfolio logic from one static
    ledger.  The cloud build therefore runs each overlay exactly against the
    all-off book.  The site can display any single switch exactly and combine
    several standalone deltas for fast attribution, while the actual fully
    interacted production curve remains visible as the truth benchmark.
    """
    base_pnl = get_daily_mtm_series(clean_sig_flat, md, start_date=BT_START)
    production_pnl = get_daily_mtm_series(
        production_sig_flat, md, start_date=BT_START)
    overlay_runs = []

    for spec in OVERLAY_LAB_SPECS:
        if not _overlay_spec_active(spec, full_book):
            continue
        variant_book = _book_with_single_overlay(clean_book, full_book, spec)

        if spec.get("regenerate_candidates"):
            print(f"  Generating candidates [overlay lab: {spec['label']}] ...")
            variant_candidates, variant_signal_data = generate_candidates_fast(
                processed, variant_book, sznl_map, BT_START)
        else:
            variant_candidates = clean_candidates.copy()
            variant_signal_data = clean_signal_data

        engine_overlay = spec.get("engine_overlay")
        enabled_names = (engine_overlay,) if engine_overlay else ()
        print(f"  Processing trades [overlay lab: {spec['label']}] ...")
        sig_flat = process_signals_fast(
            variant_candidates.copy(), variant_signal_data, processed,
            variant_book, starting_equity,
            cap_bps=spec.get("cap_bps", 0),
            flat_sizing=True,
            overflow_active=bool(spec.get("overflow_active")),
            max_long_risk_bps=None,
            max_short_risk_bps=None,
            portfolio_overlays_enabled=False,
            portfolio_overlay_names=enabled_names,
        )
        if sig_flat.empty:
            raise RuntimeError(
                f"overlay lab run {spec['id']} produced no executed trades")
        pnl = get_daily_mtm_series(sig_flat, md, start_date=BT_START)
        overlay_runs.append((spec, sig_flat, pnl))
        print(f"    {len(sig_flat)} trades")

    if not overlay_runs:
        raise RuntimeError("overlay lab found no active production overlays")

    dates = base_pnl.index.union(production_pnl.index)
    for _, _, pnl in overlay_runs:
        dates = dates.union(pnl.index)
    dates = dates.sort_values()

    provenance = _provenance_meta(len(production_sig_flat))
    payload = {
        "schema_version": 1,
        "generated_utc": provenance["ledger_build_utc"],
        "ledger_source": provenance["ledger_source"],
        "ledger_git_sha": provenance["ledger_git_sha"],
        "basis": "flat_750k",
        "starting_equity": float(starting_equity),
        "display_start_equity": 10000.0,
        "combination_method": "additive_standalone_deltas",
        "dates": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates],
        "all_off_pnl": _json_pnl_values(base_pnl, dates),
        "production_pnl": _json_pnl_values(production_pnl, dates),
        "all_off_trade_count": int(len(clean_sig_flat)),
        "production_trade_count": int(len(production_sig_flat)),
        "overlays": [],
    }
    for spec, sig_flat, pnl in overlay_runs:
        payload["overlays"].append({
            "id": spec["id"],
            "label": spec["label"],
            "description": spec["description"],
            "trade_count": int(len(sig_flat)),
            "pnl": _json_pnl_values(pnl, dates),
        })

    os.makedirs(os.path.dirname(OUT_OVERLAY_LAB), exist_ok=True)
    with open(OUT_OVERLAY_LAB, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"), ensure_ascii=False)
    print(f"  Wrote {len(payload['overlays'])} overlay-lab curves -> "
          f"{OUT_OVERLAY_LAB}")
    return payload


def _provenance_meta(n_rows):
    """Build metadata embedded in the parquet schema. daily_scan prints this
    at gate time so a stale or non-GHA ledger vintage is visible in scan logs
    (2026-07-06: a weekend vintage of unknown origin drove live gate blocks)."""
    if os.environ.get("GITHUB_ACTIONS"):
        source = f"gha:{os.environ.get('GITHUB_RUN_ID', 'unknown-run')}"
    else:
        source = f"local:{socket.gethostname()}"
    return {
        "ledger_build_utc": datetime.datetime.now(datetime.timezone.utc)
                            .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ledger_source": source,
        "ledger_git_sha": _git_sha(),
        "ledger_rows": str(n_rows),
    }


def _git_sha():
    """Commit the ledger was built from: GITHUB_SHA first, then the working
    tree's HEAD, else 'unknown'. GITHUB_SHA comes first because the deploy
    builds inside `generator/`, a git-ls-files copy with NO .git directory,
    so `git rev-parse` there fails and every CI vintage carried 'unknown'
    (2026-09-04 parity recon)."""
    sha = (os.environ.get("GITHUB_SHA") or "").strip()
    if sha:
        return sha
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT,
            capture_output=True, text=True, timeout=10)
        sha = (out.stdout or "").strip() if out.returncode == 0 else ""
    except Exception:
        sha = ""
    return sha or "unknown"


def _write_ledger_with_meta(df, path, meta):
    table = pa.Table.from_pandas(df, preserve_index=False)
    schema_meta = dict(table.schema.metadata or {})
    schema_meta.update({k.encode(): str(v).encode() for k, v in meta.items()})
    pq.write_table(table.replace_schema_metadata(schema_meta), path)


def _prior_vintage_path():
    """Previous ledger vintage for the churn diff: the local file if present
    (about to be overwritten), else a best-effort R2 pull (GHA deploy job
    starts from a clean checkout). The temporary prior is left in the
    ephemeral cloud workspace; production code never performs cleanup or
    deletion as part of a build."""
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


def combine_sizing_passes(sig_comp, sig_flat, book):
    """Join compounded and flat engine passes into the canonical ledger shape."""
    df = sig_comp.copy().reset_index(drop=True)
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

    df = df.rename(columns={
        "Date": "Signal Date",
        "Price": "Entry Price",
        "PnL": "PnL_compounded",
        "Risk $": "Risk_compounded",
        "Equity at Signal": "Equity_at_Signal",
    })
    df["Direction"] = np.where(
        df["Action"].astype(str).str.upper().str.contains("SHORT"), "Short", "Long")
    sign = np.where(df["Direction"] == "Short", -1.0, 1.0)
    df["Return_Pct"] = sign * (
        df["Exit Price"] - df["Entry Price"]) / df["Entry Price"] * 100.0
    df["R_Multiple"] = (
        df["PnL_compounded"] / df["Risk_compounded"].replace(0, np.nan))

    overflow = set(OVERFLOW_TICKERS)
    df["Tier"] = np.where(
        df["Strategy"].isin(OVERFLOW_ELIGIBLE) & df["Ticker"].isin(overflow),
        "Overflow", "Liquid")
    holds = {s["name"]: s["execution"].get("hold_days") for s in book}
    df["hold_days_target"] = df["Strategy"].map(holds)
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
    for col in ["Signal Date", "Entry Date", "Exit Date", "Time Stop"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
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
        max_long_risk_bps=POOLED_LONG_CAP_BPS,
        max_short_risk_bps=POOLED_SHORT_CAP_BPS,
    )
    ng = shape_flat_trades(sig_ng)
    ng = ng[ng["Strategy"].isin(gated)].reset_index(drop=True)
    _write_ledger_with_meta(ng, OUT_NOGATE, _provenance_meta(len(ng)))
    print(f"    {len(ng)} nogate trades ({'/'.join(gated)}) -> {OUT_NOGATE}")


def build_pcfear_shadow(candidates, signal_data, processed, full_book,
                        starting_equity):
    """P/C-fear counterfactual pass (2026-08-05, mandatory per the prereg's
    leg-C shadow-tracking requirement): the fear-conditioned band tables
    ZERO family trades at dial>=50 without P/C washout, which would freeze
    that cell's evidence at n=70 forever. This pass re-runs the engine with
    pc_fear_enabled=False (incumbent 0.25x tables everywhere) and writes the
    family strategies' trades to OUT_PCSHADOW — the would-have-been record
    the "+20 hi-frag family trades" re-exam reads. Same flat-sizing caveats
    as the nogate pass."""
    fam = sorted({s["name"] for s in full_book
                  if s.get("execution", {}).get("pc_fear_bands")})
    if not fam:
        print("  No pc_fear_bands strategies in the book — skipping pcfear shadow pass.")
        return
    print(f"\n  Processing trades [pc_fear disabled, flat sizing] for {fam} ...")
    sig_sh = process_signals_fast(
        candidates, signal_data, processed, full_book,
        starting_equity, cap_bps=250, overflow_active=True, flat_sizing=True,
        max_long_risk_bps=POOLED_LONG_CAP_BPS,
        max_short_risk_bps=POOLED_SHORT_CAP_BPS,
        pc_fear_enabled=False,
    )
    sh = shape_flat_trades(sig_sh)
    sh = sh[sh["Strategy"].isin(fam)].reset_index(drop=True)
    _write_ledger_with_meta(sh, OUT_PCSHADOW, _provenance_meta(len(sh)))
    print(f"    {len(sh)} pcfear-shadow trades -> {OUT_PCSHADOW}")


def _norm_ohlc(frame, ticker):
    """Normalize a price frame to capitalized single-level OHLC columns
    (yfinance MultiIndex rule — see CLAUDE.md)."""
    f = frame
    if isinstance(f.columns, pd.MultiIndex):
        try:
            f = f.xs(ticker, level="Ticker", axis=1)
        except Exception:
            f = f.copy()
            f.columns = f.columns.get_level_values(0)
    f = f.copy()
    f.columns = [str(c).capitalize() for c in f.columns]
    return f


def build_ovsext_counterfactual(df, md):
    """OVS hold-extension counterfactual (what-if lab, 2026-07-11): a trade
    still LOSING at its T+2 time exit holds 3 more sessions (to T+5) with the
    2-ATR target live; exit at target if touched, else the T+5 close. Pure
    post-pass on the finished ledger — no engine rerun. Exact for OVS because
    nothing downstream keys off its exits (max_one_pos=False, no ladder, no
    sector gate, caps are staged-risk based). Writes ONLY the rebooked rows to
    OUT_OVSEXT; build_site.py diffs them against the main ledger (ext_lab.json)
    and the site swaps them in behind a toggle. NOT a live rule.
    Study: scratch/ovs_hold_extension_*.py (evidence + Sharpe/DD tradeoff)."""
    strat_name = "Overbot Vol Spike"
    extra_days = 3
    mask = (
        (df["Strategy"] == strat_name)
        & (df["Exit Type"] == "Time")
        & (df["R_Multiple"] < 0)
        & df["Exit Date"].notna()
    )
    if not mask.any():
        print("  No losing OVS time exits — skipping ovsext pass.")
        return
    rows = []
    n_censored = 0
    for idx, row in df[mask].iterrows():
        frame = md.get(row["Ticker"])
        if frame is None or frame.empty:
            n_censored += 1
            continue
        f = _norm_ohlc(frame, row["Ticker"])
        if row["Exit Date"] not in f.index:
            n_censored += 1
            continue
        pos = f.index.get_loc(row["Exit Date"])
        ext = f.iloc[pos + 1 : pos + 1 + extra_days]
        if len(ext) < extra_days:
            n_censored += 1  # too recent to have a full T+5 window
            continue
        tgt = row["Entry Price"] - row["tgt_atr"] * row["ATR"]
        new_date, new_price, new_type = ext.index[-1], ext["Close"].iloc[-1], "Time5"
        for d, day in ext.iterrows():
            if day["Low"] <= tgt:
                new_date, new_price, new_type = d, tgt, "Target"
                break
        delta_r = (row["Exit Price"] - new_price) / row["ATR"]  # OVS is short
        new = row.copy()
        new["Exit Date"] = new_date
        new["Exit Price"] = new_price
        new["Exit Type"] = new_type
        new["R_Multiple"] = row["R_Multiple"] + delta_r
        new["Return_Pct"] = (row["Entry Price"] - new_price) / row["Entry Price"] * 100.0
        new["PnL_flat_750k"] = row["PnL_flat_750k"] + delta_r * row["Risk_flat_750k"]
        new["PnL_compounded"] = row["PnL_compounded"] + delta_r * row["Risk_compounded"]
        rows.append(new)
    if not rows:
        print("  ovsext pass: nothing rebookable — skipping.")
        return
    out = pd.DataFrame(rows)
    _write_ledger_with_meta(out, OUT_OVSEXT, _provenance_meta(len(out)))
    print(f"  ovsext pass: {len(out)} losing T+2 exits rebooked to T+5 "
          f"({n_censored} censored) -> {OUT_OVSEXT}")


def build_overlay_free_counterfactual(processed, full_book, sznl_map,
                                      starting_equity, md,
                                      production_sig_flat=None):
    """Build the private site's full-book, all-portfolio-overlays-off ledger."""
    clean_book = strip_portfolio_overlays(full_book)
    print("\n  Generating candidates [all portfolio overlays off] ...")
    candidates, signal_data = generate_candidates_fast(
        processed, clean_book, sznl_map, BT_START)
    print(f"    {len(candidates)} overlay-free candidate signal-dates")
    if not candidates:
        raise RuntimeError("overlay-free book produced no candidates")

    common = {
        "cap_bps": 0,
        "overflow_active": False,
        "max_long_risk_bps": None,
        "max_short_risk_bps": None,
        "portfolio_overlays_enabled": False,
    }
    print("  Processing trades [all overlays off, compounded sizing] ...")
    sig_comp = process_signals_fast(
        candidates.copy(), signal_data, processed, clean_book,
        starting_equity, **common)
    print(f"    {len(sig_comp)} trades")
    print("  Processing trades [all overlays off, flat $750k sizing] ...")
    sig_flat = process_signals_fast(
        candidates.copy(), signal_data, processed, clean_book,
        starting_equity, flat_sizing=True, **common)
    print(f"    {len(sig_flat)} trades")
    if sig_comp.empty or sig_flat.empty:
        raise RuntimeError("overlay-free book produced no executed trades")

    df = combine_sizing_passes(sig_comp, sig_flat, clean_book)
    meta = _provenance_meta(len(df))
    meta.update({
        "portfolio_variant": "overlay_free",
        "removed_execution_overlays": ",".join(sorted(OVERLAY_EXECUTION_KEYS)),
        "removed_signal_overlays": "dial_filters,use_t1_gap_kill",
        "removed_portfolio_overlays": (
            "cross_strategy_overlap,overflow_risk_override,"
            "per_strategy_daily_cap,ovs_path2_downsize,ovs_path2_daily_cap,"
            "ovs_atr_extended_precedence,wcds_seasonal_size_tier"),
    })
    _write_ledger_with_meta(df, OUT_OVERLAY_FREE, meta)
    print(f"  Wrote {len(df)} overlay-free trades -> {OUT_OVERLAY_FREE}")

    pnl_comp = get_daily_mtm_series(sig_comp, md, start_date=BT_START)
    pnl_flat = get_daily_mtm_series(sig_flat, md, start_date=BT_START)
    daily = pd.DataFrame({
        "pnl_compounded": pnl_comp,
        "pnl_flat": pnl_flat,
    }).fillna(0.0)
    daily.index.name = "date"
    daily["equity_compounded"] = starting_equity + daily["pnl_compounded"].cumsum()
    daily["equity_flat"] = starting_equity + daily["pnl_flat"].cumsum()
    daily.reset_index().to_parquet(OUT_OVERLAY_FREE_DAILY, index=False)
    print(f"  Wrote {len(daily)} overlay-free daily rows -> {OUT_OVERLAY_FREE_DAILY}")

    if production_sig_flat is not None:
        build_overlay_lab(
            processed, full_book, clean_book, sznl_map, starting_equity, md,
            candidates, signal_data, sig_flat, production_sig_flat)
    return df


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
        max_long_risk_bps=POOLED_LONG_CAP_BPS,
        max_short_risk_bps=POOLED_SHORT_CAP_BPS,
    )
    print(f"    {len(sig_comp)} trades")
    print("  Processing trades [flat $750k sizing] ...")
    sig_flat = process_signals_fast(
        candidates, signal_data, processed, full_book, starting_equity,
        cap_bps=250, overflow_active=True, flat_sizing=True,
        max_long_risk_bps=POOLED_LONG_CAP_BPS,
        max_short_risk_bps=POOLED_SHORT_CAP_BPS,
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

    # --- P/C-fear shadow pass (best effort; leg-C evidence accrual) ---
    try:
        build_pcfear_shadow(candidates, signal_data, processed,
                            full_book, starting_equity)
    except Exception as e:
        print(f"  (pcfear shadow pass skipped: {e})")

    df = combine_sizing_passes(sig_comp, sig_flat, full_book)

    os.makedirs(os.path.dirname(OUT_PARQUET), exist_ok=True)

    # Vintage churn diff BEFORE overwriting the prior copy — recent-window
    # trades that flicker between rebuilds steer the live sector_loss_gate.
    prior = _prior_vintage_path()
    if prior:
        _diff_vs_prior(df, prior)

    _write_ledger_with_meta(df, OUT_PARQUET, _provenance_meta(len(df)))
    print(f"\n  Wrote {len(df)} trades -> {OUT_PARQUET}")

    # --- OVS hold-extension counterfactual (best effort, never fails the build) ---
    try:
        build_ovsext_counterfactual(df, md)
    except Exception as e:
        print(f"  (ovsext counterfactual skipped: {e})")

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

    # Full second ledger for the private site's one-click comparison.  This is
    # intentionally required (not best effort): a production deploy must never
    # advertise the overlay-free mode while serving an absent/stale bundle.
    overlay_free_df = build_overlay_free_counterfactual(
        processed, full_book, sznl_map, starting_equity, md,
        production_sig_flat=sig_flat)
    print(f"  Overlay-free comparison: {len(overlay_free_df)} trades")

    # Mirror the production ledger only after every required comparison
    # artifact has been built successfully.  This key sizes/gates LIVE orders,
    # so a failed site-counterfactual build must not advance it by itself.
    # Gated behind --upload (deploy_site.yml passes it); local runs never write
    # this production R2 key.
    if upload:
        try:
            from cache_io import upload_from_local
            if not upload_from_local(OUT_PARQUET, "backtest_trades_full.parquet"):
                raise RuntimeError("R2 ledger upload returned false")
        except Exception as e:
            raise RuntimeError(f"R2 ledger upload failed: {e}") from e
    else:
        print("  (R2 mirror skipped: run with --upload to overwrite the prod ledger key)")

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
