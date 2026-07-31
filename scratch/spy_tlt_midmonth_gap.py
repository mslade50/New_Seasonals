"""SPY/TLT mid-month divergence fade.

At the k-th trading day of each month, measure each ticker's month-to-date
move in units of its own ATR (Wilder-14, frozen at the prior month-end
anchor). If the two have diverged by more than `thresh` ATRs, go long the
LAGGARD at that day's close and hold to the month's last trading-day close.

Example: TLT +2 ATR MTD, SPY -2 ATR MTD -> gap 4 -> long SPY.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from monthly_weak_close_mr import load_data as _load, wilder_atr


def month_frames(data: dict) -> pd.DataFrame:
    """Align SPY/TLT on common dates."""
    spy, tlt = data["SPY"], data["TLT"]
    idx = spy.index.intersection(tlt.index)
    return spy.loc[idx], tlt.loc[idx]


def run(spy: pd.DataFrame, tlt: pd.DataFrame, k: int = 10, thresh: float = 4.0,
        hold_mode: str = "eom") -> pd.DataFrame:
    """k = signal on the k-th trading day of the month (1-based)."""
    per = spy.index.to_period("M")
    trades = []
    for p in per.unique():
        days = spy.index[per == p]
        if len(days) < k + 2:
            continue
        prev_days = spy.index[per == (p - 1)]
        if not len(prev_days):
            continue
        anchor = prev_days[-1]
        sig_day = days[k - 1]
        rets = {}
        for name, df in [("SPY", spy), ("TLT", tlt)]:
            atr0 = df["ATR"].loc[anchor]
            if np.isnan(atr0) or atr0 <= 0:
                rets = None
                break
            rets[name] = (df["Close"].loc[sig_day] - df["Close"].loc[anchor]) / atr0
        if rets is None:
            continue
        gap = rets["SPY"] - rets["TLT"]
        if abs(gap) < thresh:
            continue
        lag = "SPY" if gap < 0 else "TLT"
        df = spy if lag == "SPY" else tlt
        entry = df["Close"].loc[sig_day]
        if hold_mode == "eom":
            exit_day = days[-1]
        else:  # fixed 5td
            j = df.index.get_loc(sig_day)
            exit_day = df.index[min(j + 5, len(df) - 1)]
        exitp = df["Close"].loc[exit_day]
        trades.append({
            "month": str(p), "sig_day": sig_day, "exit_day": exit_day,
            "lag": lag, "gap": gap, "spy_atr": rets["SPY"], "tlt_atr": rets["TLT"],
            "ret": exitp / entry - 1,
            "hold_td": len(days) - k if hold_mode == "eom" else 5,
        })
    return pd.DataFrame(trades)


def summarize(t: pd.DataFrame, label: str) -> dict:
    if not len(t):
        return {"variant": label, "N": 0}
    tstat = t.ret.mean() / (t.ret.std(ddof=1) / np.sqrt(len(t))) if len(t) > 2 else np.nan
    up = t.loc[t.ret > 0, "ret"].sum()
    dn = -t.loc[t.ret < 0, "ret"].sum()
    return {"variant": label, "N": len(t),
            "win%": round(100 * (t.ret > 0).mean(), 1),
            "avg%": round(100 * t.ret.mean(), 2),
            "med%": round(100 * t.ret.median(), 2),
            "tot%": round(100 * t.ret.sum(), 1),
            "PF": round(up / dn, 2) if dn > 0 else np.inf,
            "worst%": round(100 * t.ret.min(), 2),
            "t": round(float(tstat), 2)}


def baseline(spy, tlt, k=10):
    """Unconditional: hold each ticker day-k close -> month-end close."""
    per = spy.index.to_period("M")
    rows = []
    for p in per.unique():
        days = spy.index[per == p]
        if len(days) < k + 2:
            continue
        for name, df in [("SPY", spy), ("TLT", tlt)]:
            rows.append(df["Close"].loc[days[-1]] / df["Close"].loc[days[k - 1]] - 1)
    return float(np.mean(rows))


def main() -> None:
    global TICKERS
    data = _load()
    spy, tlt = month_frames(data)
    print(f"common history: {spy.index[0].date()} -> {spy.index[-1].date()}")

    t = run(spy, tlt)
    print("\n=== Base: k=10, thresh=4 ATR, hold to EOM ===")
    rows = [summarize(t, "all"),
            summarize(t[t.lag == "SPY"], "long SPY (lagging)"),
            summarize(t[t.lag == "TLT"], "long TLT (lagging)")]
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"baseline (uncond. day10->EOM, both tickers): {100 * baseline(spy, tlt):+.2f}%")

    if len(t):
        print("\ntrades:")
        cols = ["month", "sig_day", "lag", "gap", "spy_atr", "tlt_atr", "ret", "hold_td"]
        tt = t[cols].copy()
        tt["ret"] = (100 * tt.ret).round(2)
        for c in ["gap", "spy_atr", "tlt_atr"]:
            tt[c] = tt[c].round(2)
        print(tt.to_string(index=False))

    print("\n=== Sensitivity: thresh x signal day (avg% / N, hold EOM) ===")
    grid = {}
    for th in [2.0, 3.0, 4.0, 5.0]:
        row = {}
        for k in [8, 10, 12, 14]:
            s = run(spy, tlt, k=k, thresh=th)
            row[f"k{k}"] = f"{100 * s.ret.mean():+.2f} ({len(s)})" if len(s) else "-"
        grid[f">={th:.0f} ATR"] = row
    print(pd.DataFrame(grid).T.to_string())

    print("\n=== By leg at thresh=3 (more N) ===")
    s = run(spy, tlt, k=10, thresh=3.0)
    print(pd.DataFrame([summarize(s[s.lag == "SPY"], "long SPY"),
                        summarize(s[s.lag == "TLT"], "long TLT")]).to_string(index=False))

    print("\n=== Per-year (k=10, thresh=4) ===")
    if len(t):
        t2 = t.copy()
        t2["yr"] = pd.to_datetime(t2.sig_day).dt.year
        yr = t2.groupby("yr").agg(n=("ret", "count"), avg=("ret", "mean"),
                                  tot=("ret", "sum"))
        yr[["avg", "tot"]] = (100 * yr[["avg", "tot"]]).round(2)
        print(yr.to_string())


if __name__ == "__main__":
    main()
