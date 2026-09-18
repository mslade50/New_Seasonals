"""Pre-FOMC drift (Lucca-Moench) split by cycle year -- descriptive only.

Tonight is 2026-09-08. Next scheduled FOMC decision: 2026-09-16 (14:00 ET),
5 trading sessions ahead. 2026 is a midterm year (year %% 4 == 2).

Three windows, all lag-0 close-to-close off the anchor close:
  RUNUP    close[p-5] -> close[p-1]   (the 4-session run-up into the decision)
  DECISION close[p-1] -> close[p]     (the decision session itself)
  FULL     close[p-5] -> close[p]     (5 sessions)
where p is the trading-day position of the scheduled decision date.

Splits: all years / midterm (year%%4==2) / other three cycle years.
Controls: non-overlapping k-session blocks over the same index, labelled.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pitch_lab import (  # noqa: E402
    anchor_positions,
    load_events,
    load_prices,
    sign_test,
    summarize,
)

TICKER = "^GSPC"
FALLBACK = "SPY"
PRE_TD = 5  # anchor is the close 5 sessions before the decision session


def line(ch: str = "-", n: int = 78) -> None:
    print(ch * n)


def load_close() -> tuple[pd.Series, str]:
    px = load_prices([TICKER, FALLBACK])
    if TICKER in px and len(px[TICKER]) > 100:
        return px[TICKER]["Close"].dropna(), TICKER
    print(f"NOTE: {TICKER} unusable, falling back to {FALLBACK}")
    return px[FALLBACK]["Close"].dropna(), FALLBACK


def build_table(close: pd.Series) -> pd.DataFrame:
    idx = pd.DatetimeIndex(close.index)
    c = close.values
    ev = load_events(["fomc_decision"])["date"]
    ev = pd.DatetimeIndex(ev)
    positions, kept = anchor_positions(idx, ev, offset=0)

    rows = []
    dropped_offcal = 0
    for p, d in zip(positions, kept):
        if idx[p] != d:
            # decision date not itself a trading session in this index
            dropped_offcal += 1
            continue
        if p - PRE_TD < 0 or p >= len(idx):
            continue
        anchor = c[p - PRE_TD]
        rows.append({
            "decision": d,
            "year": d.year,
            "midterm": (d.year % 4 == 2),
            "anchor_date": idx[p - PRE_TD],
            "runup": c[p - 1] / anchor - 1.0,
            "decision_day": c[p] / c[p - 1] - 1.0,
            "full": c[p] / anchor - 1.0,
        })
    if dropped_offcal:
        print(f"NOTE: {dropped_offcal} decision date(s) not on the price "
              f"calendar, dropped")
    return pd.DataFrame(rows)


def block_control(close: pd.Series, k: int) -> np.ndarray:
    """Non-overlapping k-session blocks over the whole index (simple + honest:
    every block counted once, no overlap inflation)."""
    c = close.values
    starts = np.arange(0, len(c) - k, k)
    return c[starts + k] / c[starts] - 1.0


def report(sub: pd.DataFrame, col: str, label: str) -> None:
    v = sub[col].values.astype(float)
    n = len(v)
    if n == 0:
        print(f"{label:<34} n=0")
        return
    wins = int((v > 0).sum())
    s = summarize(v, label)
    p = sign_test(wins, n)
    i_w, i_b = int(np.argmin(v)), int(np.argmax(v))
    print(f"{label:<34} n={n:<4} record {wins}-{n - wins}  "
          f"mean {100 * v.mean():+.3f}%  med {100 * np.median(v):+.3f}%  "
          f"hit {s['hit']:.1f}%  sign p={p:.4f}")
    print(f"{'':<34} worst {100 * v[i_w]:+.2f}% "
          f"({sub['decision'].iloc[i_w].date()})   "
          f"best {100 * v[i_b]:+.2f}% ({sub['decision'].iloc[i_b].date()})")


def era(sub: pd.DataFrame, col: str, cut: str, label: str) -> None:
    d = pd.DatetimeIndex(sub["decision"])
    m = d < pd.Timestamp(cut)
    for part, mask in ((f"pre-{cut[:4]}", m), (f"{cut[:4]}+", ~m)):
        v = sub.loc[mask, col].values.astype(float)
        if len(v) == 0:
            print(f"    {label} {part:<10} n=0")
            continue
        w = int((v > 0).sum())
        print(f"    {label} {part:<10} n={len(v):<4} record {w}-{len(v) - w}  "
              f"mean {100 * v.mean():+.3f}%  med {100 * np.median(v):+.3f}%  "
              f"sign p={sign_test(w, len(v)):.4f}")


def main() -> None:
    close, used = load_close()
    tbl = build_table(close)
    idx = pd.DatetimeIndex(close.index)

    line("=")
    print("PRE-FOMC DRIFT BY CYCLE YEAR -- descriptive, lag-0 close-to-close")
    line("=")
    print(f"instrument            : {used}")
    print(f"price history         : {idx[0].date()} .. {idx[-1].date()} "
          f"({len(idx)} sessions)")
    print(f"scheduled decisions    : {len(tbl)} measured "
          f"({tbl['decision'].min().date()} .. {tbl['decision'].max().date()})")
    print("event kind             : fomc_decision (data/macro_events.csv); "
          "fomc_intermeeting EXCLUDED")
    print("windows                : RUNUP = close[p-5]->close[p-1] (4 sessions)")
    print("                         DECISION = close[p-1]->close[p] (1 session)")
    print("                         FULL  = close[p-5]->close[p] (5 sessions)")
    print(f"next decision          : 2026-09-16 (not measured, in the future)")

    splits = [
        ("ALL YEARS", tbl),
        ("MIDTERM (year%4==2)", tbl[tbl["midterm"]]),
        ("OTHER 3 CYCLE YEARS", tbl[~tbl["midterm"]]),
    ]
    windows = [("runup", "RUNUP 4-session"),
               ("decision_day", "DECISION session"),
               ("full", "FULL 5-session")]

    for name, sub in splits:
        line("=")
        yrs = sorted(sub["year"].unique())
        print(f"SPLIT: {name}   ({len(yrs)} years: {yrs[0]}..{yrs[-1]})")
        line()
        for col, wlab in windows:
            report(sub, col, wlab)
            era(sub, col, "2013-01-01", wlab[:8])
            era(sub, col, "2018-01-01", wlab[:8])
            print()

    # ------------------------------------------------------------------
    line("=")
    print("CONTROL: non-overlapping blocks over the same index, all sessions")
    print("(each block counted once; no overlap inflation, no event condition)")
    line()
    for k, wlab in ((4, "RUNUP-equivalent 4-session"),
                    (1, "DECISION-equivalent 1-session"),
                    (5, "FULL-equivalent 5-session")):
        v = block_control(close, k)
        w = int((v > 0).sum())
        print(f"{wlab:<32} n={len(v):<5} record {w}-{len(v) - w}  "
              f"mean {100 * v.mean():+.3f}%  med {100 * np.median(v):+.3f}%  "
              f"hit {100 * (v > 0).mean():.1f}%")
    print("\nedge vs control (mean, percentage points):")
    ctl = {c: block_control(close, k).mean()
           for c, k in (("runup", 4), ("decision_day", 1), ("full", 5))}
    for name, sub in splits:
        parts = [f"{lab.split()[0]} {100 * (sub[c].mean() - ctl[c]):+.3f}pp"
                 for c, lab in windows]
        print(f"  {name:<22} " + "   ".join(parts))

    # ------------------------------------------------------------------
    line("=")
    print("MIDTERM PER-MEETING DETAIL (concentration check)")
    line()
    mt = tbl[tbl["midterm"]].sort_values("decision")
    print(f"{'decision':<12}{'anchor':<12}{'runup%':>9}{'decision%':>11}"
          f"{'full%':>9}")
    for _, r in mt.iterrows():
        print(f"{str(r['decision'].date()):<12}"
              f"{str(r['anchor_date'].date()):<12}"
              f"{100 * r['runup']:>9.2f}{100 * r['decision_day']:>11.2f}"
              f"{100 * r['full']:>9.2f}")

    print()
    for col, wlab in windows:
        v = mt[col].values.astype(float)
        tot = v.sum()
        order = np.argsort(-np.abs(v))[:2]
        top2 = v[order].sum()
        share = 100 * top2 / tot if tot != 0 else np.nan
        dts = [str(mt["decision"].iloc[i].date()) for i in order]
        print(f"{wlab:<20} total {100 * tot:+.2f}pp | top-2 by |x| {dts} "
              f"= {100 * top2:+.2f}pp ({share:.0f}% of total)")
        rest = np.delete(v, order)
        wr = int((rest > 0).sum())
        print(f"{'':<20} ex-top-2: n={len(rest)} record {wr}-{len(rest) - wr} "
              f"mean {100 * rest.mean():+.3f}%  med {100 * np.median(rest):+.3f}%"
              f"  sign p={sign_test(wr, len(rest)):.4f}")

    print()
    print("midterm RUNUP by year:")
    by_yr = mt.groupby("year")["runup"].agg(["count", "mean", "sum"])
    for y, r in by_yr.iterrows():
        print(f"  {y}  n={int(r['count'])}  mean {100 * r['mean']:+.3f}%  "
              f"sum {100 * r['sum']:+.2f}pp")

    line("=")
    print("done")


if __name__ == "__main__":
    main()
