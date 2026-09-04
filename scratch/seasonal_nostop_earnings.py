"""Does the long-only seasonal edge hold around earnings?

Splits scratch/seasonal_nostop_trades.parquet (stock-channel longs) by earnings
proximity, two ways:
  1. Does an earnings print land INSIDE the hold window (entry..time-stop]?
  2. Signed trading-day offset of the SIGNAL date to the nearest print
     (earnings_filter conventions: negative = signal before earnings).
Reports the no-stop sqrt-time book and the incumbent ticket book side by side.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = r"C:\Users\McKinley Slade\dev\New_Seasonals"
sys.path.insert(0, ROOT)
from earnings_filter import load_earnings_dates_map, signed_offset

TRADES = os.path.join(ROOT, "scratch", "seasonal_nostop_trades.parquet")


def line(tag, b):
    if b.empty:
        print(f"  {tag:28s} N    0")
        return
    R = b["R"].astype(float)
    d = b["dollars"].astype(float)
    pf = d[d > 0].sum() / abs(d[d < 0].sum()) if (d < 0).any() else np.inf
    print(f"  {tag:28s} N{len(b):5d} win%{100*(R>0).mean():5.1f} avgR{R.mean():+.3f} "
          f"PF{pf:5.2f} tot$k{d.sum()/1e3:7.1f} worstR{R.min():+6.2f} "
          f"%<-3R{100*(R<-3).mean():4.1f}")


def main():
    df = pd.read_parquet(TRADES)
    stocks = df[df.channel == "detect_seasonal"].copy()
    emap = load_earnings_dates_map()

    # per (asof, ticker) — same for every variant
    keys = stocks[["asof", "ticker", "entry_date"]].drop_duplicates(["asof", "ticker"]).copy()
    win_end = stocks[stocks.variant == "nostop_sqrt"].set_index(["asof", "ticker"])["exit_date"]

    offs, inside = {}, {}
    no_data = 0
    for r in keys.itertuples():
        arr = emap.get(str(r.ticker).upper())
        k = (r.asof, r.ticker)
        offs[k] = signed_offset(r.asof, arr)
        if arr is None or len(arr) == 0:
            no_data += 1
            inside[k] = np.nan
            continue
        end = win_end.get(k)
        lo = np.datetime64(pd.Timestamp(r.entry_date).date())
        hi = np.datetime64(pd.Timestamp(end).date())
        inside[k] = bool(((arr >= lo) & (arr <= hi)).any())
    print(f"stock-long trade keys: {len(keys)} | no earnings data: {no_data}")

    mk = list(zip(stocks["asof"], stocks["ticker"]))
    stocks["e_off"] = [offs[k] for k in mk]
    stocks["e_inside"] = [inside[k] for k in mk]

    for v in ["nostop_sqrt", "ticket"]:
        b = stocks[stocks.variant == v]
        print(f"\n================ {v} — earnings inside hold window ================")
        line("print INSIDE hold", b[b.e_inside == True])   # noqa: E712
        line("no print in hold", b[b.e_inside == False])   # noqa: E712
        line("no earnings data", b[b.e_inside.isna()])

        print(f"---- {v} — signal-date offset to nearest print (TD) ----")
        buckets = [(-99, -22), (-21, -11), (-10, -6), (-5, -1), (0, 0),
                   (1, 5), (6, 10), (11, 21), (22, 99)]
        for lo_, hi_ in buckets:
            m = b[(b.e_off >= lo_) & (b.e_off <= hi_)]
            line(f"offset [{lo_:+d},{hi_:+d}]", m)

    # horizon split of the inside-hold cell (21d holds capture most prints)
    print("\n================ nostop_sqrt — inside-hold by horizon ================")
    ns = stocks[stocks.variant == "nostop_sqrt"]
    for h in [5, 10, 21]:
        line(f"h={h}d, print inside", ns[(ns.h == h) & (ns.e_inside == True)])   # noqa: E712
        line(f"h={h}d, clean",        ns[(ns.h == h) & (ns.e_inside == False)])  # noqa: E712

    # tail-loser attribution
    tail = ns[ns.R < -3].sort_values("R")
    print(f"\nno-stop tail losers (<-3R): {len(tail)} | with print inside hold: "
          f"{int((tail.e_inside == True).sum())}")  # noqa: E712
    print(tail[["asof", "ticker", "h", "R", "e_off", "e_inside"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
