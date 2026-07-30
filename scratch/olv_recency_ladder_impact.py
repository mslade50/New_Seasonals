"""OLV recency-ladder impact: PnL over time + time-in-market Sharpe.

Engine A/B/C on IDENTICAL candidates (masks don't depend on sizing):
  flat      no ladder (pre-2026-07-29 config, base 35/25 bps)
  opencount ladder_multipliers [0.5, 1, 1] (2026-07-29 form)
  recency   signal_recency_ladder {21td, [0.5, 0.7, 1.0]} (2026-07-30 prod)

OLV-only book (liquid + overflow passes), flat $750k sizing, per-strategy
250 bps cap, full history 2003+. Reports totR/avgR/win/PnL, maxDD on the
flat equity curve, Sharpe on ALL days vs IN-MARKET days (>=1 open OLV
position), % days in market, and per-year flat PnL side by side.
"""
from __future__ import annotations

import copy
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))


class _NoOp:
    def __getattr__(self, name):
        def f(*a, **k):
            return self
        return f
    def __call__(self, *a, **k): return self
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def cache_data(self, *a, **k):
        if len(a) == 1 and callable(a[0]) and not k:   # bare @st.cache_data
            return a[0]
        def deco(fn): return fn
        return deco
    cache_resource = cache_data


sys.modules["streamlit"] = _NoOp()

from build_trade_ledger import BT_START, load_data
from daily_portfolio_report import build_full_strategy_book
from pages.strat_backtester import (
    get_daily_mtm_series,
    generate_candidates_fast,
    load_atr_seasonal_map,
    load_seasonal_map,
    precompute_all_indicators,
    process_signals_fast,
)

NAV = 750_000.0


def olv_book(variant: str):
    book = [copy.deepcopy(s) for s in build_full_strategy_book()
            if s["name"] == "Oversold Low Volume"]
    assert len(book) == 2, f"expected liquid+overflow OLV passes, got {len(book)}"
    for s in book:
        ex = s["execution"]
        ex.pop("signal_recency_ladder", None)
        ex.pop("ladder_multipliers", None)
        if variant == "opencount":
            ex["ladder_multipliers"] = [0.5, 1.0, 1.0]
        elif variant == "recency":
            ex["signal_recency_ladder"] = {"window_td": 21, "mults": [0.5, 0.7, 1.0]}
        elif variant != "flat":
            raise ValueError(variant)
    return book


def metrics(sig, md, label):
    daily = get_daily_mtm_series(sig, md, start_date=BT_START).fillna(0.0)
    eq = NAV + daily.cumsum()
    dd = (eq - eq.cummax())
    ret = daily / NAV

    # in-market days: any open OLV position (entry..exit inclusive)
    entry = pd.to_datetime(sig["Entry Date"])
    exit_ = pd.to_datetime(sig["Exit Date"])
    open_any = pd.Series(0, index=daily.index)
    for e, x in zip(entry, exit_):
        open_any.loc[e:x] = 1
    in_mkt = open_any.astype(bool)

    sharpe_all = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else np.nan
    r_in = ret[in_mkt]
    sharpe_in = r_in.mean() / r_in.std() * np.sqrt(252) if len(r_in) > 1 and r_in.std() > 0 else np.nan

    r = sig["PnL"] / sig["Risk $"]
    return {
        "variant": label,
        "trades": len(sig),
        "totR": float(r.sum()),
        "avgR": float(r.mean()),
        "win%": float((sig["PnL"] > 0).mean() * 100),
        "PnL_flat": float(sig["PnL"].sum()),
        "maxDD_$": float(dd.min()),
        "maxDD_%NAV": float(dd.min() / NAV * 100),
        "Sharpe_all": float(sharpe_all),
        "Sharpe_inmkt": float(sharpe_in),
        "days_inmkt%": float(in_mkt.mean() * 100),
        "_daily": daily,
    }


def main():
    base_book = olv_book("flat")
    tickers = set()
    for s in base_book:
        tickers.update(t.replace(".", "-") for t in s["universe_tickers"])
    print(f"Loading data for {len(tickers)} tickers ...")
    md = load_data(tickers)

    sznl_map = load_seasonal_map()
    atr_sznl_map = load_atr_seasonal_map()
    vix_series = None
    vd = md.get("^VIX")
    if vd is not None:
        vix_series = vd["Close"]

    print("Precomputing indicators (OLV universes) ...")
    processed = precompute_all_indicators(md, base_book, sznl_map, vix_series, atr_sznl_map)
    print(f"Generating candidates from {BT_START} ...")
    candidates, signal_data = generate_candidates_fast(processed, base_book, sznl_map, BT_START)
    print(f"{len(candidates)} candidates")

    rows, dailies = [], {}
    for variant in ("flat", "opencount", "recency"):
        book = olv_book(variant)
        sig = process_signals_fast(candidates, signal_data, processed, book,
                                   NAV, cap_bps=250, overflow_active=True,
                                   flat_sizing=True)
        m = metrics(sig, md, variant)
        dailies[variant] = m.pop("_daily")
        rows.append(m)
        print(f"  {variant}: {m['trades']} trades, PnL ${m['PnL_flat']:,.0f}")

    out = pd.DataFrame(rows).set_index("variant")
    pd.set_option("display.width", 160)
    print("\n=== SUMMARY (flat $750k basis) ===")
    print(out.round(3).to_string())

    yearly = pd.DataFrame({v: d.groupby(d.index.year).sum() for v, d in dailies.items()})
    yearly["recency-opencount"] = yearly["recency"] - yearly["opencount"]
    yearly["recency-flat"] = yearly["recency"] - yearly["flat"]
    print("\n=== PnL BY YEAR ===")
    print(yearly.round(0).astype(int).to_string())

    out.to_csv(os.path.join(_HERE, "olv_recency_ladder_impact.csv"))
    yearly.to_csv(os.path.join(_HERE, "olv_recency_ladder_impact_yearly.csv"))
    print("\nWrote olv_recency_ladder_impact{,_yearly}.csv")


if __name__ == "__main__":
    main()
