"""Engine validation of the prespecified strongest robust Kelly rotation.

Selection rule (fixed before this replay): choose the smallest liquid-supported
static rotation in the direction that survived every allocation sensitivity.
That direction is more liquid-tier OLV risk and less WCDS risk. Overflow OLV
is frozen because the overflow history is survivorship-flattered.

Proposal:
  - OLV liquid base: 35 -> 40 nominal bps (+14.2857%)
  - OLV overflow base: unchanged at 25 nominal bps
  - WCDS base: scaled down just enough to preserve the baseline annual filled-
    risk budget to first order (about 35 -> 34.25 nominal bps)
  - all overlays, pilots, caps, and other strategies unchanged

The script rebuilds baseline and proposal from identical current candidates in
memory. Outputs only scratch/ artifacts.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.build_trade_ledger as btl
from strategy_config import ACCOUNT_VALUE, GLOBAL_RISK_MULTIPLIER, STRATEGY_BOOK


STAMP = "2026-08-05"
SCRATCH = ROOT / "scratch"
SIGNALS_BASE = SCRATCH / f"kelly_current_signals_{STAMP}.parquet"
OUT_JSON = SCRATCH / f"kelly_engine_replay_results_{STAMP}.json"
OUT_DAILY = SCRATCH / f"kelly_engine_replay_daily_{STAMP}.parquet"
OUT_COMPONENT = SCRATCH / f"kelly_engine_replay_components_{STAMP}.csv"

OLV_LIQ_NOMINAL_OLD = 35.0
OLV_LIQ_NOMINAL_NEW = 40.0
WCDS_NOMINAL_OLD = 35.0


def vix_series(md):
    frame = md.get("^VIX")
    if frame is None or frame.empty:
        return None
    f = frame.copy()
    if isinstance(f.columns, pd.MultiIndex):
        f.columns = f.columns.get_level_values(0)
    f.columns = [str(c).capitalize() for c in f.columns]
    return f["Close"]


def component(sig: pd.DataFrame) -> pd.Series:
    out = sig["Strategy"].astype(str).copy()
    ovs = sig["Strategy"].eq("Overbot Vol Spike")
    gap = ((sig["T+1 Open"] - sig["Signal Close"])
           / sig["ATR"].replace(0, np.nan))
    out.loc[ovs & gap.gt(0.25)] = "Overbot Vol Spike P1"
    out.loc[ovs & ~gap.gt(0.25)] = "Overbot Vol Spike P2"
    return out


def tier(sig: pd.DataFrame) -> pd.Series:
    overflow = set(btl.OVERFLOW_TICKERS)
    return pd.Series(np.where(
        sig["Strategy"].isin(btl.OVERFLOW_ELIGIBLE)
        & sig["Ticker"].isin(overflow), "Overflow", "Liquid"), index=sig.index)


def metrics(sig: pd.DataFrame, md) -> tuple[dict, pd.Series]:
    daily = btl.get_daily_mtm_series(sig, md, start_date=btl.BT_START).fillna(0.0)
    equity = ACCOUNT_VALUE + daily.cumsum()
    dd = equity - equity.cummax()
    ann = float(daily.mean() * 252.0)
    vol = float(daily.std(ddof=1) * np.sqrt(252.0))
    downside = float(np.sqrt(np.mean(np.minimum(daily.to_numpy(), 0.0) ** 2))
                     * np.sqrt(252.0))
    result = {
        "engine_rows": int(len(sig)),
        "total_pnl": float(daily.sum()),
        "annual_pnl": ann,
        "annual_vol": vol,
        "sharpe": ann / vol if vol > 0 else None,
        "sortino": ann / downside if downside > 0 else None,
        "max_dd": float(dd.min()),
        "max_dd_pct_nav": float(dd.min() / ACCOUNT_VALUE),
        "worst_day": float(daily.min()),
        "best_day": float(daily.max()),
        "filled_risk_total": float(sig["Risk $"].sum()),
        "annual_filled_risk_fraction": float(
            sig["Risk $"].sum() / ACCOUNT_VALUE * 252.0 / len(daily)
        ),
    }
    return result, daily


def component_table(base: pd.DataFrame, prop: pd.DataFrame) -> pd.DataFrame:
    def one(sig, suffix):
        s = sig.copy()
        s["Component"] = component(s)
        s["Tier"] = tier(s)
        return s.groupby(["Component", "Tier"]).agg(
            **{
                f"Rows_{suffix}": ("PnL", "size"),
                f"PnL_{suffix}": ("PnL", "sum"),
                f"Risk_{suffix}": ("Risk $", "sum"),
            }
        )
    out = one(base, "Base").join(one(prop, "Proposal"), how="outer").fillna(0.0)
    out["Delta_PnL"] = out["PnL_Proposal"] - out["PnL_Base"]
    out["Delta_Risk"] = out["Risk_Proposal"] - out["Risk_Base"]
    return out.reset_index()


def proposal_book(full_book: list[dict], wcds_mult: float) -> list[dict]:
    book = copy.deepcopy(full_book)
    n_liquid = len(STRATEGY_BOOK)
    for idx, strat in enumerate(book):
        name = strat["name"]
        if idx < n_liquid and name == "Oversold Low Volume":
            strat["execution"]["risk_bps"] = (
                OLV_LIQ_NOMINAL_NEW * GLOBAL_RISK_MULTIPLIER
            )
        elif name == "Weak Close Decent Sznls":
            strat["execution"]["risk_bps"] *= wcds_mult
    return book


def main() -> int:
    prior = pd.read_parquet(SIGNALS_BASE)
    n_days = 6156  # frozen replay matrix length from the current baseline
    risk_annual = (
        prior.groupby(["Component", "Tier"])["Risk_Fraction"].sum()
        * 252.0 / n_days
    )
    olv_liq_budget = float(risk_annual.loc[("Oversold Low Volume", "Liquid")])
    wcds_budget = float(risk_annual.loc[("Weak Close Decent Sznls", "Liquid")])
    olv_mult = OLV_LIQ_NOMINAL_NEW / OLV_LIQ_NOMINAL_OLD
    wcds_mult = 1.0 - olv_liq_budget * (olv_mult - 1.0) / wcds_budget
    wcds_nominal_new = WCDS_NOMINAL_OLD * wcds_mult
    print("KELLY ENGINE REPLAY")
    print(f"  OLV liquid {OLV_LIQ_NOMINAL_OLD:g}->{OLV_LIQ_NOMINAL_NEW:g} nominal")
    print(f"  WCDS {WCDS_NOMINAL_OLD:g}->{wcds_nominal_new:.4f} nominal")
    print("  OLV overflow 25 unchanged; cap=250; pooled caps off")

    full_book = btl.build_full_strategy_book()
    prop_book = proposal_book(full_book, wcds_mult)
    sznl_map = btl.load_seasonal_map()
    atr_sznl_map = btl.load_atr_seasonal_map()
    tickers = {"SPY", "^VIX"}
    for strat in full_book:
        tickers.update(strat["universe_tickers"])
    md = btl.load_data(tickers)
    print("  precomputing indicators once...")
    processed = btl.precompute_all_indicators(
        md, full_book, sznl_map, vix_series(md), atr_sznl_map
    )
    candidates, signal_data = btl.generate_candidates_fast(
        processed, full_book, sznl_map, btl.BT_START
    )
    print(f"  candidates={len(candidates):,}; running baseline + proposal")
    common = dict(
        starting_equity=ACCOUNT_VALUE, cap_bps=250, flat_sizing=True,
        overflow_active=True, max_long_risk_bps=None, max_short_risk_bps=None,
    )
    base = btl.process_signals_fast(
        list(candidates), signal_data, processed, full_book, **common
    ).reset_index(drop=True)
    prop = btl.process_signals_fast(
        list(candidates), signal_data, processed, prop_book, **common
    ).reset_index(drop=True)

    base_metrics, base_daily = metrics(base, md)
    prop_metrics, prop_daily = metrics(prop, md)
    daily = pd.DataFrame({"date": base_daily.index,
                          "baseline_pnl": base_daily.values,
                          "proposal_pnl": prop_daily.reindex(base_daily.index).fillna(0.0).values})
    daily["delta_pnl"] = daily["proposal_pnl"] - daily["baseline_pnl"]
    daily.to_parquet(OUT_DAILY, index=False)
    comp = component_table(base, prop)
    comp.to_csv(OUT_COMPONENT, index=False)

    delta = {k: (prop_metrics[k] - base_metrics[k])
             for k in base_metrics if isinstance(base_metrics[k], (int, float))
             and base_metrics[k] is not None and prop_metrics[k] is not None}
    result = {
        "study_date": STAMP,
        "selection_rule": "smallest liquid-supported rotation in strongest robust Kelly direction",
        "proposal": {
            "olv_liquid_nominal_bps_old": OLV_LIQ_NOMINAL_OLD,
            "olv_liquid_nominal_bps_new": OLV_LIQ_NOMINAL_NEW,
            "olv_overflow_nominal_bps": 25.0,
            "wcds_nominal_bps_old": WCDS_NOMINAL_OLD,
            "wcds_nominal_bps_new": wcds_nominal_new,
            "wcds_multiplier": wcds_mult,
            "per_strategy_cap_bps": 250.0,
            "pooled_caps": False,
        },
        "baseline": base_metrics,
        "proposal_metrics": prop_metrics,
        "delta": delta,
        "outputs": {
            "daily": str(OUT_DAILY.relative_to(ROOT)),
            "components": str(OUT_COMPONENT.relative_to(ROOT)),
        },
    }
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

