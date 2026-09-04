"""Candidate config for the long-only seasonal book (McKinley 2026-08-05):
wide sqrt-time stop (stop = entry - ATR*sqrt(h/5), same unit as sizing) PLUS a
0.8x size multiplier on trades with an earnings print inside the hold window
(ex-ante knowable: forward dates are in the calendar). Macro/no-data: 1.0x.

Compares: nostop_sqrt | sqrt_stop | sqrt_stop + 0.8x earnings (FINAL).
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = r"C:\Users\McKinley Slade\dev\New_Seasonals"
sys.path.insert(0, ROOT)
from earnings_filter import load_earnings_dates_map
from scripts.seasonal_sharpe import ratios

TRADES = os.path.join(ROOT, "scratch", "seasonal_nostop_trades.parquet")
EARN_MULT = 0.8


def flag_earnings_inside(df: pd.DataFrame) -> pd.Series:
    emap = load_earnings_dates_map()
    win_end = df[df.variant == "nostop_sqrt"].set_index(["asof", "ticker"])["exit_date"]
    flags = {}
    for r in df[["asof", "ticker", "entry_date", "channel"]].drop_duplicates(["asof", "ticker"]).itertuples():
        k = (r.asof, r.ticker)
        arr = emap.get(str(r.ticker).upper()) if r.channel == "detect_seasonal" else None
        if arr is None or len(arr) == 0:
            flags[k] = False
            continue
        lo = np.datetime64(pd.Timestamp(r.entry_date).date())
        hi = np.datetime64(pd.Timestamp(win_end.get(k)).date())
        flags[k] = bool(((arr >= lo) & (arr <= hi)).any())
    return pd.Series([flags[(a, t)] for a, t in zip(df["asof"], df["ticker"])], index=df.index)


def summarize(name, b, full):
    d = b["dollars"].astype(float)
    pf = d[d > 0].sum() / abs(d[d < 0].sum())
    daily = b.groupby(b["exit_date"].dt.normalize())["dollars"].sum().reindex(full, fill_value=0.0)
    monthly = daily.resample("ME").sum()
    sh, so = ratios(monthly, 12)
    eq = daily.cumsum()
    maxdd = float((eq - eq.cummax()).min())
    print(f"{name:26s} N{len(b):5d} win%{100*(d>0).mean():5.1f} PF{pf:5.2f} "
          f"$/tr{d.mean():7.1f} tot$k{d.sum()/1e3:6.0f} Sharpe{sh:5.2f} Sortino{so:5.2f} "
          f"maxDD$k{maxdd/1e3:7.1f} worst${d.min():7.0f} worstMo$k{monthly.min()/1e3:6.1f}")
    return daily


def main():
    df = pd.read_parquet(TRADES)
    df["e_inside"] = flag_earnings_inside(df)

    books = {
        "nostop_sqrt": df[df.variant == "nostop_sqrt"].copy(),
        "sqrt_stop": df[df.variant == "sqrt_stop"].copy(),
    }
    final = df[df.variant == "sqrt_stop"].copy()
    final["dollars"] = final["dollars"] * np.where(final["e_inside"], EARN_MULT, 1.0)
    books[f"FINAL (stop + {EARN_MULT}x earn)"] = final
    print(f"trades/variant: {len(final)} | earnings-inside-hold (0.8x): "
          f"{int(final.e_inside.sum())} ({100*final.e_inside.mean():.1f}%)")

    full = pd.date_range(df["exit_date"].min().normalize(), df["exit_date"].max().normalize(), freq="B")
    print()
    dailies = {}
    for name, b in books.items():
        dailies[name] = summarize(name, b, full)
    print("\n--- ex-midterm ---")
    for name, b in books.items():
        summarize(name, b[b.cycle != 2], full)

    fb = books[f"FINAL (stop + {EARN_MULT}x earn)"]
    print("\n--- FINAL by segment ---")
    fb2 = fb.copy()
    fb2["asset"] = np.where(fb2["channel"] == "detect_seasonal", "stock", "macro")
    for a in ["stock", "macro"]:
        summarize(f"  {a}", fb2[fb2.asset == a], full)
    for h in [5, 10, 21]:
        summarize(f"  h={h}d", fb[fb.h == h], full)
    summarize("  earnings-inside (0.8x'd)", fb[fb.e_inside], full)
    summarize("  clean", fb[~fb.e_inside], full)

    print("\n--- FINAL annual ($k) ---")
    ann = fb.groupby(fb["asof"].dt.year)["dollars"].sum() / 1e3
    for y, v in ann.items():
        tag = " (midterm)" if y % 4 == 2 else ""
        print(f"  {y}: {v:7.1f}{tag}")

    out = os.path.join(ROOT, "scratch", "seasonal_final_config_trades.parquet")
    fb.to_parquet(out)
    print(f"\nfinal-config trades -> {out}")


if __name__ == "__main__":
    main()
