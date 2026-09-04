"""Build the current-book Kelly research dataset without touching live data.

Outputs are confined to scratch/:
  - kelly_current_trades_2026-08-05.parquet
  - kelly_current_signals_2026-08-05.parquet
  - kelly_current_daily_components_2026-08-05.parquet
  - kelly_current_daily_tier_components_2026-08-05.parquet
  - kelly_current_replay_summary_2026-08-05.json

The script calls the production engine in memory with the current strategy
book, flat $750k sizing, the live 250 bps per-strategy cap, and removed pooled
caps left off. It never calls build_trade_ledger.main() and never writes data/.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.build_trade_ledger as btl
from strategy_config import ACCOUNT_VALUE, GLOBAL_RISK_MULTIPLIER, STRATEGY_BOOK


STAMP = "2026-08-05"
OUT = ROOT / "scratch"
TRADES_OUT = OUT / f"kelly_current_trades_{STAMP}.parquet"
SIGNALS_OUT = OUT / f"kelly_current_signals_{STAMP}.parquet"
DAILY_OUT = OUT / f"kelly_current_daily_components_{STAMP}.parquet"
DAILY_TIER_OUT = OUT / f"kelly_current_daily_tier_components_{STAMP}.parquet"
SUMMARY_OUT = OUT / f"kelly_current_replay_summary_{STAMP}.json"


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "-c", f"safe.directory={ROOT.as_posix()}",
             "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, check=True,
            timeout=10,
        ).stdout.strip()
    except Exception:
        return "unknown"


def normalize_vix(md: dict[str, pd.DataFrame]) -> pd.Series | None:
    frame = md.get("^VIX")
    if frame is None or frame.empty:
        return None
    out = frame.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    out.columns = [str(c).capitalize() for c in out.columns]
    return out["Close"]


def classify_components(sig: pd.DataFrame) -> pd.Series:
    comp = sig["Strategy"].astype(str).copy()
    ovs = sig["Strategy"].eq("Overbot Vol Spike")
    gap_atr = (
        (pd.to_numeric(sig["T+1 Open"], errors="coerce")
         - pd.to_numeric(sig["Signal Close"], errors="coerce"))
        / pd.to_numeric(sig["ATR"], errors="coerce").replace(0, np.nan)
    )
    comp.loc[ovs & gap_atr.gt(0.25)] = "Overbot Vol Spike P1"
    comp.loc[ovs & ~gap_atr.gt(0.25)] = "Overbot Vol Spike P2"
    return comp


def tier_for(sig: pd.DataFrame) -> pd.Series:
    overflow = set(btl.OVERFLOW_TICKERS)
    return pd.Series(
        np.where(
            sig["Strategy"].isin(btl.OVERFLOW_ELIGIBLE)
            & sig["Ticker"].isin(overflow),
            "Overflow", "Liquid",
        ),
        index=sig.index,
    )


def collapse_signals(sig: pd.DataFrame) -> pd.DataFrame:
    """Collapse OVS near/far tranches back to one filled signal.

    The same grouping is harmless for all one-row strategies. Risk and PnL are
    additive, so R is recomputed from the aggregated dollars.
    """
    work = sig.copy()
    work["Component"] = classify_components(work)
    work["Tier"] = tier_for(work)
    keys = [
        "Component", "Strategy", "Tier", "Ticker", "Date", "Entry Date",
        "Price",
    ]
    first_cols = [
        "Exit Date", "Direction", "Action", "ATR", "Risk bps", "Size_Mult",
        "Signal Close", "T+1 Open", "stop_atr", "tgt_atr",
    ]
    agg: dict[str, object] = {
        "PnL": "sum",
        "Risk $": "sum",
        "Shares": "sum",
        "Exit Date": "max",
        "Tranche": "size",
    }
    for col in first_cols:
        if col in work.columns and col not in agg:
            agg[col] = "first"
    out = work.groupby(keys, dropna=False, as_index=False).agg(agg)
    out = out.rename(columns={"Tranche": "Ledger_Rows"})
    out["R"] = out["PnL"] / out["Risk $"].replace(0, np.nan)
    out["Risk_Fraction"] = out["Risk $"] / float(ACCOUNT_VALUE)
    out["PnL_Return"] = out["PnL"] / float(ACCOUNT_VALUE)
    out = out.sort_values(["Date", "Component", "Ticker"]).reset_index(drop=True)
    out.insert(0, "signal_id", np.arange(len(out), dtype=int))
    return out


def daily_component_frames(sig: pd.DataFrame, md: dict[str, pd.DataFrame]):
    shaped = sig.copy()
    shaped["Component"] = classify_components(shaped)
    shaped["Tier"] = tier_for(shaped)

    comp = {}
    for name, group in shaped.groupby("Component", sort=True):
        print(f"  daily MTM component: {name} ({len(group)} rows)")
        comp[name] = btl.get_daily_mtm_series(
            group, md, start_date=btl.BT_START
        )
    daily = pd.DataFrame(comp).fillna(0.0).sort_index()
    daily.index.name = "date"

    tier_comp = {}
    for (name, tier), group in shaped.groupby(["Component", "Tier"], sort=True):
        key = f"{name}||{tier}"
        print(f"  daily MTM tier/component: {key} ({len(group)} rows)")
        tier_comp[key] = btl.get_daily_mtm_series(
            group, md, start_date=btl.BT_START
        )
    daily_tier = pd.DataFrame(tier_comp).fillna(0.0).sort_index()
    daily_tier.index.name = "date"
    return daily, daily_tier


def pnl_metrics(daily: pd.DataFrame) -> dict[str, float]:
    pnl = daily.sum(axis=1).fillna(0.0)
    equity = ACCOUNT_VALUE + pnl.cumsum()
    dd = equity - equity.cummax()
    ann_pnl = pnl.mean() * 252.0
    ann_vol = pnl.std(ddof=1) * np.sqrt(252.0)
    return {
        "total_pnl": float(pnl.sum()),
        "annual_pnl": float(ann_pnl),
        "annual_vol": float(ann_vol),
        "sharpe": float(ann_pnl / ann_vol) if ann_vol > 0 else None,
        "max_dd_dollars": float(dd.min()),
        "max_dd_pct_start_nav": float(dd.min() / ACCOUNT_VALUE),
        "worst_day": float(pnl.min()),
        "best_day": float(pnl.max()),
        "n_days": int(len(pnl)),
        "date_min": pnl.index.min().strftime("%Y-%m-%d"),
        "date_max": pnl.index.max().strftime("%Y-%m-%d"),
    }


def main() -> int:
    print("KELLY CURRENT-BOOK SCRATCH REPLAY")
    print(f"  repo={ROOT}")
    print(f"  git={git_sha()} | GRM={GLOBAL_RISK_MULTIPLIER} | NAV=${ACCOUNT_VALUE:,.0f}")
    print("  cap=250 bps/strategy/day | pooled caps=off | flat sizing")

    full_book = btl.build_full_strategy_book()
    print(f"  {len(STRATEGY_BOOK)} liquid strategies + "
          f"{len(full_book) - len(STRATEGY_BOOK)} overflow passes")

    sznl_map = btl.load_seasonal_map()
    atr_sznl_map = btl.load_atr_seasonal_map()
    tickers = {"SPY", "^VIX"}
    for strat in full_book:
        tickers.update(strat["universe_tickers"])
    md = btl.load_data(tickers)
    if not md:
        raise RuntimeError("No market data loaded")

    print("  precomputing current indicators...")
    processed = btl.precompute_all_indicators(
        md, full_book, sznl_map, normalize_vix(md), atr_sznl_map
    )
    print("  generating current candidates...")
    candidates, signal_data = btl.generate_candidates_fast(
        processed, full_book, sznl_map, btl.BT_START
    )
    print(f"  candidates={len(candidates):,}")
    if not candidates:
        raise RuntimeError("No candidates generated")

    print("  processing current flat-size trades...")
    sig = btl.process_signals_fast(
        candidates, signal_data, processed, full_book, ACCOUNT_VALUE,
        cap_bps=250, flat_sizing=True, overflow_active=True,
        max_long_risk_bps=None, max_short_risk_bps=None,
    )
    if sig.empty:
        raise RuntimeError("No filled trades")
    sig = sig.reset_index(drop=True)
    sig.insert(0, "trade_id", np.arange(len(sig), dtype=int))
    sig["Component"] = classify_components(sig)
    sig["Tier"] = tier_for(sig)
    sig.to_parquet(TRADES_OUT, index=False)
    print(f"  wrote {len(sig):,} engine rows -> {TRADES_OUT}")

    signals = collapse_signals(sig)
    signals.to_parquet(SIGNALS_OUT, index=False)
    print(f"  wrote {len(signals):,} collapsed signals -> {SIGNALS_OUT}")

    daily, daily_tier = daily_component_frames(sig, md)
    daily.reset_index().to_parquet(DAILY_OUT, index=False)
    daily_tier.reset_index().to_parquet(DAILY_TIER_OUT, index=False)
    print(f"  wrote {len(daily):,} daily rows x {daily.shape[1]} components -> {DAILY_OUT}")
    print(f"  wrote {len(daily_tier):,} daily rows x {daily_tier.shape[1]} tier components -> {DAILY_TIER_OUT}")

    summary = {
        "study": "kelly_current_book_replay",
        "study_date": STAMP,
        "git_sha": git_sha(),
        "account_value": float(ACCOUNT_VALUE),
        "global_risk_multiplier": float(GLOBAL_RISK_MULTIPLIER),
        "per_strategy_cap_bps": 250.0,
        "pooled_long_cap_bps": None,
        "pooled_short_cap_bps": None,
        "flat_sizing": True,
        "overflow_universe_active_env": os.environ.get("OVERFLOW_UNIVERSE_ACTIVE", "0"),
        "liquid_strategy_count": len(STRATEGY_BOOK),
        "overflow_pass_count": len(full_book) - len(STRATEGY_BOOK),
        "candidate_count": len(candidates),
        "engine_row_count": len(sig),
        "collapsed_signal_count": len(signals),
        "components": list(daily.columns),
        "engine_rows_by_component": {
            str(k): int(v) for k, v in sig.groupby("Component").size().items()
        },
        "signals_by_component": {
            str(k): int(v) for k, v in signals.groupby("Component").size().items()
        },
        "signals_by_component_tier": {
            f"{k[0]}||{k[1]}": int(v)
            for k, v in signals.groupby(["Component", "Tier"]).size().items()
        },
        "metrics": pnl_metrics(daily),
        "outputs": {
            "trades": str(TRADES_OUT.relative_to(ROOT)),
            "signals": str(SIGNALS_OUT.relative_to(ROOT)),
            "daily_components": str(DAILY_OUT.relative_to(ROOT)),
            "daily_tier_components": str(DAILY_TIER_OUT.relative_to(ROOT)),
        },
    }
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["metrics"], indent=2))
    print(f"  wrote summary -> {SUMMARY_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

