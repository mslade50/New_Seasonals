"""RMW seasonality ETF confirmation — runs the PRE-REGISTERED test only.

See scratch/prereg_rmw_seasonality.md (written first). H1: QUAL-SPY spread
negative in January. H2: positive in July. Loading gate: beta of the
spread on RMW >= 0.20 else UNINFORMATIVE.
"""
import os
import sys

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def monthly_returns(tickers, start="2005-01-01"):
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    px = raw["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(tickers[0])
    m = px.resample("ME").last()
    return m.pct_change() * 100.0


def cell(s: pd.Series, month: int):
    x = s[s.index.month == month].dropna()
    n = len(x)
    t = x.mean() / (x.std(ddof=1) / np.sqrt(n)) if n > 2 else np.nan
    return x.mean(), t, n


def main():
    ff = pd.read_parquet(os.path.join(ROOT, "data", "factor_returns_monthly.parquet"))
    rmw = ff["RMW"].dropna()

    rets = monthly_returns(["QUAL", "SPHQ", "SPY"])
    spread_qual = (rets["QUAL"] - rets["SPY"]).dropna()
    spread_sphq = (rets["SPHQ"] - rets["SPY"]).dropna()
    spread_sphq = spread_sphq[spread_sphq.index.year >= 2011]  # provenance caveat

    for name, sp in [("QUAL-SPY", spread_qual), ("SPHQ-SPY (2011+)", spread_sphq)]:
        sp.index = sp.index + pd.offsets.MonthEnd(0)
        both = pd.concat([sp, rmw], axis=1, join="inner").dropna()
        both.columns = ["spread", "rmw"]
        beta = np.polyfit(both["rmw"], both["spread"], 1)[0]
        corr = both["spread"].corr(both["rmw"])
        print(f"\n=== {name}: {both.index.min().date()} -> {both.index.max().date()}, "
              f"n={len(both)} ===")
        print(f"  loading: beta={beta:+.2f}, corr={corr:+.2f} "
              f"(gate: beta >= 0.20 -> {'MET' if beta >= 0.20 else 'NOT MET — uninformative'})")
        for label, month, want_sign, ff_cell in [("H1 January", 1, -1, -0.96),
                                                 ("H2 July", 7, +1, +0.66)]:
            mean, t, n = cell(sp, month)
            pred = beta * ff_cell
            band_ok = (np.sign(mean) == np.sign(pred)
                       and 0.25 * abs(pred) <= abs(mean) <= 4 * abs(pred)) if pred != 0 else False
            t_ok = (t <= -1.0) if want_sign < 0 else (t >= 1.0)
            print(f"  {label}: spread {mean:+.2f}%/mo (t={t:+.2f}, n={n}) | "
                  f"beta x FF-cell predicts {pred:+.2f} | "
                  f"t-gate {'PASS' if t_ok else 'FAIL'} | magnitude-band "
                  f"{'PASS' if band_ok else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
