"""
monday_dip_split_tp.py — RESEARCH: Monday Dip split take-profit what-if.

Question (2026-07-09): instead of taking the whole position off at 2R, take
half at 1R and half at 2R. A split exit with a shared entry/stop/time-exit is
EXACTLY the average of two independent runs (half A = full position with
tgt_atr=1.0, half B = prod tgt_atr=2.0), so no engine changes: run the engine
twice on identical candidates and blend 0.5/0.5 per signal.

Conventions inherited from the engine (identical in both runs, but they bind
the 1R half more often): entry-day targets are never credited; stops arm on
day 2. Same-bar stop+target ambiguity resolves per the engine's existing
order of checks.

Run:  python scratch/monday_dip_split_tp.py   (~1-2 min, Monday Dip universe only)
"""
import copy
import datetime
import os
import sys
import time

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("OVERFLOW_UNIVERSE_ACTIVE", "0")

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import data_provider
from strategy_config import ACCOUNT_VALUE
from pages.strat_backtester import (
    load_seasonal_map,
    load_atr_seasonal_map,
    precompute_all_indicators,
    generate_candidates_fast,
    process_signals_fast,
)
from daily_portfolio_report import build_full_strategy_book

NAME = "Monday Dip"
DATA_START = "2000-01-01"
BT_START = datetime.date(2003, 1, 1)
EQ = float(ACCOUNT_VALUE)
ERAS = [
    ("2003-2009", "2003-01-01", "2009-12-31"),
    ("2010-2014", "2010-01-01", "2014-12-31"),
    ("2015-2019", "2015-01-01", "2019-12-31"),
    ("2020-2022", "2020-01-01", "2022-12-31"),
    ("2023-now", "2023-01-01", "2099-01-01"),
]


def keyed(sig):
    d = sig[sig["Strategy"] == NAME].copy()
    d["_k"] = (d["Ticker"].astype(str) + "|"
               + pd.to_datetime(d["Date"]).dt.strftime("%Y-%m-%d"))
    dup = d["_k"].duplicated()
    if dup.any():
        raise RuntimeError(f"duplicate signal keys: {d.loc[dup, '_k'].tolist()[:5]}")
    return d.set_index("_k").sort_index()


def stats(pnl, r, label):
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    return {
        "variant": label,
        "n": len(pnl),
        "win_pct": round(float((pnl > 0).mean()) * 100, 1),
        "tot_r": round(float(r.sum()), 1),
        "avg_r": round(float(r.mean()), 3),
        "pf": round(float(wins.sum() / abs(losses.sum())), 2) if len(losses) else None,
        "expectancy": round(float(pnl.mean()), 0),
        "tot_pnl": round(float(pnl.sum()), 0),
    }


def daily_metrics(pnl_by_exit, calendar):
    s = pnl_by_exit.reindex(calendar).fillna(0.0)
    rets = s.values / EQ
    n = len(rets)
    m = rets.mean()
    sd = rets.std(ddof=1)
    downside = rets[rets < 0]
    dsd = np.sqrt((downside ** 2).sum() / len(downside)) if len(downside) > 1 else 0.0
    cum = np.cumsum(s.values)
    runmax = np.maximum.accumulate(cum)
    return {
        "sharpe": round(float(m / sd * np.sqrt(252)), 2) if sd else None,
        "sortino": round(float(m / dsd * np.sqrt(252)), 2) if dsd else None,
        "ann_ret_pct": round(float(m * 252 * 100), 2),
        "max_dd_flat": round(float((runmax - cum).max()), 0),
    }


def main():
    t0 = time.time()
    book_all = build_full_strategy_book()
    book = [copy.deepcopy(s) for s in book_all if s["name"] == NAME]
    if len(book) != 1:
        raise RuntimeError(f"expected 1 Monday Dip pass, got {len(book)}")
    print(f"Monday Dip: stop_atr={book[0]['execution']['stop_atr']}, "
          f"tgt_atr={book[0]['execution']['tgt_atr']}, "
          f"hold_days={book[0]['execution']['hold_days']}, "
          f"universe={len(book[0]['universe_tickers'])} tickers")

    tickers = set(book[0]["universe_tickers"]) | {"SPY", "^VIX"}
    md = data_provider.get_history(sorted(tickers), start=DATA_START)
    vix_df = md.get("^VIX")
    vix_series = None
    if vix_df is not None and not vix_df.empty:
        vd = vix_df.copy()
        if isinstance(vd.columns, pd.MultiIndex):
            vd.columns = vd.columns.get_level_values(0)
        vd.columns = [c.capitalize() for c in vd.columns]
        vix_series = vd["Close"]

    sznl_map = load_seasonal_map()
    atr_sznl_map = load_atr_seasonal_map()
    processed = precompute_all_indicators(md, book, sznl_map, vix_series, atr_sznl_map)
    candidates, signal_data = generate_candidates_fast(processed, book, sznl_map, BT_START)
    print(f"{len(candidates)} candidates in {time.time() - t0:.0f}s")

    def run(tgt):
        bk = copy.deepcopy(book)
        bk[0]["execution"]["tgt_atr"] = float(tgt)
        sig = process_signals_fast(candidates, signal_data, processed, bk, EQ,
                                   cap_bps=250, overflow_active=False,
                                   flat_sizing=True)
        print(f"  tgt_atr={tgt}: {len(sig)} trades")
        return keyed(sig)

    base = run(2.0)   # prod: all off at 2R
    one = run(1.0)    # the 1R half

    if not base.index.equals(one.index):
        only_b = base.index.difference(one.index)
        only_o = one.index.difference(base.index)
        raise RuntimeError(f"trade sets differ: {len(only_b)} base-only, "
                           f"{len(only_o)} 1R-only — blend invalid")

    risk = base["Risk $"].replace(0, np.nan)
    if not np.allclose(risk.fillna(0), one["Risk $"].fillna(0)):
        raise RuntimeError("risk differs between runs — sizing not identical")

    pnl_split = 0.5 * base["PnL"] + 0.5 * one["PnL"]
    r_base = base["PnL"] / risk
    r_one = one["PnL"] / risk
    r_split = 0.5 * r_base + 0.5 * r_one

    rows = [stats(base["PnL"], r_base, "prod (all at 2R)"),
            stats(one["PnL"], r_one, "all at 1R (the new half)"),
            stats(pnl_split, r_split, "SPLIT 1/2 at 1R + 1/2 at 2R")]

    # realized-at-exit daily streams on the SPY calendar
    cal = pd.to_datetime(md["SPY"].index).normalize()
    cal = cal[cal >= pd.Timestamp(BT_START)]
    exd_b = pd.to_datetime(base["Exit Date"]).dt.normalize()
    exd_o = pd.to_datetime(one["Exit Date"]).dt.normalize()
    day_b = base["PnL"].groupby(exd_b).sum()
    day_o = one["PnL"].groupby(exd_o).sum()
    day_split = (0.5 * day_b).add(0.5 * day_o, fill_value=0.0)
    for row, day in zip(rows, (day_b, day_o, day_split)):
        row.update(daily_metrics(day, cal))

    print("\n" + pd.DataFrame(rows).to_string(index=False))

    # how often does the 1R half bank a target the 2R run never reaches?
    et_b = base["Exit Type"].astype(str)
    et_o = one["Exit Type"].astype(str)
    tab = pd.crosstab(et_o, et_b)
    tab.index.name = "1R-half exit"
    tab.columns.name = "2R-run exit"
    print("\nExit-type transition (rows: 1R half, cols: prod 2R run):")
    print(tab.to_string())

    # era split on avg R
    sig_dates = pd.to_datetime(base["Date"])
    print("\nEra split (avg R):")
    hdr = f"{'era':<10} {'n':>5} {'prod 2R':>9} {'split':>9} {'delta':>8}"
    print(hdr)
    for era, a, b in ERAS:
        m = (sig_dates >= a) & (sig_dates <= b)
        if not m.any():
            continue
        print(f"{era:<10} {int(m.sum()):>5} {r_base[m].mean():>9.3f} "
              f"{r_split[m].mean():>9.3f} {(r_split[m] - r_base[m]).mean():>+8.3f}")

    print(f"\nTotal runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
