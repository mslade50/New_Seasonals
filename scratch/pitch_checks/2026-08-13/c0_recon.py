"""c0 recon: confirm today's trigger states match the tape file before measuring.

Registry trap guarded here: pct_rank takes a PRICE series (it differences
internally). Print TODAY's value of every trigger statistic and eyeball it
against 00_surface_map.md / the tape sort.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

TK = ["IHI", "XLV", "XLU", "XLP", "FXI", "EEM", "DIA", "SPY", "QQQ", "IWM"]

px_map = load_prices(TK)
for t in TK:
    c = px_map[t]["Close"]
    r5 = pct_rank(c, 5).iloc[-1]
    r21 = pct_rank(c, 21).iloc[-1]
    r63 = pct_rank(c, 63).iloc[-1]
    hi52 = c.rolling(252).max().iloc[-1]
    dd = 100 * (c.iloc[-1] / hi52 - 1.0)
    sma200 = c.rolling(200).mean().iloc[-1]
    print(f"{t:5s} last={c.index[-1].date()} close={c.iloc[-1]:8.2f} "
          f"r5={r5:5.1f} r21={r21:5.1f} r63={r63:5.1f} "
          f"ret5={100*(c.iloc[-1]/c.iloc[-6]-1):+6.2f}% "
          f"ret21={100*(c.iloc[-1]/c.iloc[-22]-1):+6.2f}% "
          f"dd52wh={dd:+6.2f}% vs200d={100*(c.iloc[-1]/sma200-1):+6.2f}%")

print("\n--- joint states live today ---")
ihi, xlv, xlu, fxi, eem, dia, spy = (px_map[t]["Close"] for t in
                                     ["IHI", "XLV", "XLU", "FXI", "EEM", "DIA", "SPY"])
print("C6 IHI r21>=99 & dd<=-10:",
      pct_rank(ihi, 21).iloc[-1] >= 99,
      100 * (ihi.iloc[-1] / ihi.rolling(252).max().iloc[-1] - 1) <= -10)
print("C7 XLU r21<=10 & XLV r21>=85:",
      pct_rank(xlu, 21).iloc[-1] <= 10, pct_rank(xlv, 21).iloc[-1] >= 85)
print("C8 FXI r5<=20 & FXI r21>=80 & EEM ret5>0:",
      pct_rank(fxi, 5).iloc[-1] <= 20, pct_rank(fxi, 21).iloc[-1] >= 80,
      (eem.iloc[-1] / eem.iloc[-6] - 1) > 0)
print("C10 DIA r5<=20 & DIA r63>=80 & SPY ret5>0:",
      pct_rank(dia, 5).iloc[-1] <= 20, pct_rank(dia, 63).iloc[-1] >= 80,
      (spy.iloc[-1] / spy.iloc[-6] - 1) > 0)

print("\n--- SPY near 52w high (live gate that HURT the XLU cell) ---")
print(f"SPY dd52wh = {100*(spy.iloc[-1]/spy.rolling(252).max().iloc[-1]-1):+.3f}%")

print("\n--- cluster depth today (consecutive trigger sessions incl. today) ---")


def depth(mask):
    m = mask.values
    d = 0
    for v in m[::-1]:
        if v:
            d += 1
        else:
            break
    return d


c6m = (pct_rank(ihi, 21) >= 99) & (ihi / ihi.rolling(252).max() - 1 <= -0.10)
c7m = (pct_rank(xlu, 21) <= 10) & (pct_rank(xlv, 21) >= 85)
c8m = (pct_rank(fxi, 5) <= 20) & (pct_rank(fxi, 21) >= 80) & (eem.pct_change(5) > 0)
c10m = (pct_rank(dia, 5) <= 20) & (pct_rank(dia, 63) >= 80) & (spy.pct_change(5) > 0)
for lbl, m in [("C6", c6m), ("C7", c7m), ("C8", c8m), ("C10", c10m)]:
    m = m.fillna(False)
    print(f"{lbl}: live={bool(m.iloc[-1])} depth_today={depth(m)} n_days_hist={int(m.sum())}")
