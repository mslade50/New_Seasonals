"""Regime qualifier + LOYO/bootstrap validation for the h21 weak-close variant.

Variant under test: SPY+QQQ, close <= 15% of monthly range, buy the signal
close, hold 21 td, no target. Qualifier: month-end close vs 10-month SMA of
monthly closes (trend-sleeve convention).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from monthly_weak_close_mr import load_data, month_signals, run_trades

RNG = np.random.default_rng(42)


def sma10_map(df: pd.DataFrame) -> pd.Series:
    """Month-end close vs 10-month SMA (incl. current month), keyed by period."""
    mc = df["Close"].groupby(df.index.to_period("M")).last()
    return mc > mc.rolling(10).mean()


def trades_df(data: dict, tickers: list[str], thresh: float = 0.15,
              hold: int = 21) -> pd.DataFrame:
    rows = []
    for tk in tickers:
        df = data[tk]
        above = sma10_map(df)
        for x in run_trades(df, month_signals(df, thresh), "close", hold, None):
            per = pd.Timestamp(x["sig_day"]).to_period("M")
            x["ticker"] = tk
            x["above_ma"] = bool(above.get(per, False))
            rows.append(x)
    t = pd.DataFrame(rows)
    t["ym"] = pd.to_datetime(t["sig_day"]).dt.to_period("M")
    t["yr"] = pd.to_datetime(t["sig_day"]).dt.year
    return t


def stats(t: pd.DataFrame, label: str) -> dict:
    cl = t.groupby("ym")["ret"].mean()
    tstat = cl.mean() / (cl.std(ddof=1) / np.sqrt(len(cl))) if len(cl) > 2 else np.nan
    return {
        "variant": label, "N": len(t), "clusters": len(cl),
        "win%": round(100 * (t.ret > 0).mean(), 1),
        "avg%": round(100 * t.ret.mean(), 2),
        "tot%": round(100 * t.ret.sum(), 1),
        "worst%": round(100 * t.ret.min(), 2),
        "t_cl": round(float(tstat), 2),
    }


def main() -> None:
    data = load_data()
    t = trades_df(data, ["SPY", "QQQ"])

    print("=== 10-month MA qualifier (SPY+QQQ, close entry, h21, no tgt) ===")
    rows = [stats(t, "all signals"),
            stats(t[t.above_ma], "above 10m MA only"),
            stats(t[~t.above_ma], "below 10m MA (skipped)")]
    print(pd.DataFrame(rows).to_string(index=False))

    print("\nWorst 5 trades still in the above-MA set:")
    keep = t[t.above_ma].sort_values("ret")
    print(keep[["ticker", "sig_day", "ret"]].head(5).to_string(index=False))

    print("\nYearly total ret% above-MA vs all:")
    yr = pd.DataFrame({
        "all": 100 * t.groupby("yr").ret.sum(),
        "above_ma": 100 * t[t.above_ma].groupby("yr").ret.sum(),
    }).round(1).fillna(0)
    print(yr.to_string())

    print("\n=== LOYO (leave-one-year-out), above-MA set ===")
    loyo = []
    for yr_ in sorted(keep.yr.unique()):
        sub = keep[keep.yr != yr_]
        cl = sub.groupby("ym")["ret"].mean()
        loyo.append({"drop_yr": yr_,
                     "avg%": round(100 * sub.ret.mean(), 2),
                     "t_cl": round(float(cl.mean() / (cl.std(ddof=1) / np.sqrt(len(cl)))), 2)})
    lo = pd.DataFrame(loyo)
    print(f"avg% range [{lo['avg%'].min()}, {lo['avg%'].max()}], "
          f"t_cl range [{lo['t_cl'].min()}, {lo['t_cl'].max()}]")
    print("weakest 3:")
    print(lo.sort_values("t_cl").head(3).to_string(index=False))

    print("\n=== Cluster bootstrap (10k resamples of month-clusters) ===")
    for label, sub in [("all", t), ("above_ma", t[t.above_ma])]:
        cl = sub.groupby("ym")["ret"].mean().values
        means = np.array([RNG.choice(cl, len(cl), replace=True).mean()
                          for _ in range(10_000)])
        print(f"{label}: mean {100*cl.mean():+.2f}%, P(mean<=0) = {(means <= 0).mean():.4f}")

    print("\n=== Same qualifier on the 5d/2ATR spec (for reference) ===")
    t5 = trades_df(data, ["SPY", "QQQ"], hold=5)
    # rebuild with target: reuse run_trades directly
    rows5 = []
    for tk in ["SPY", "QQQ"]:
        df = data[tk]
        above = sma10_map(df)
        for x in run_trades(df, month_signals(df, 0.15), "close", 5, 2.0):
            x["above_ma"] = bool(above.get(pd.Timestamp(x["sig_day"]).to_period("M"), False))
            rows5.append(x)
    t5 = pd.DataFrame(rows5)
    t5["ym"] = pd.to_datetime(t5["sig_day"]).dt.to_period("M")
    print(pd.DataFrame([stats(t5, "5d all"),
                        stats(t5[t5.above_ma], "5d above MA")]).to_string(index=False))


if __name__ == "__main__":
    main()
