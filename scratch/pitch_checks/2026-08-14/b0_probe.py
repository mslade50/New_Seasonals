"""Probe for C2/C2b/C9 (insurance breadth) and C5 (XLE vs USO).

Not a check. Establishes history depth, today's exact trigger values, and how
many days any candidate mask would ever fire, BEFORE a threshold is chosen.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pitch_lab import load_prices, pct_rank  # noqa: E402

ASOF = pd.Timestamp("2026-08-13")

INSURERS = ["HIG", "ALL", "TRV", "AIG", "MET", "PGR", "CB", "AFL", "PRU",
            "LNC", "WRB", "CINF", "L", "GL"]
OTHER = ["XLF", "SPY", "KRE", "XLE", "USO", "DBC", "XOP", "VLO", "COP", "XLV",
         "XLK", "XLY", "XLP", "XLU", "XLI", "XLB", "XLC", "XLRE", "IWM"]

px = load_prices(INSURERS + OTHER)

print("=" * 78)
print("1. history depth")
print("=" * 78)
for t in INSURERS + OTHER:
    if t not in px:
        print(f"  {t:8s} MISSING")
        continue
    c = px[t]["Close"].dropna()
    print(f"  {t:8s} {c.index[0].date()} .. {c.index[-1].date()}  n={len(c)}")

print("\n" + "=" * 78)
print("2. insurance complex, today's state (anchor 2026-08-13)")
print("=" * 78)
rows = []
for t in INSURERS:
    if t not in px:
        continue
    c = px[t]["Close"].dropna()
    c = c[c.index <= ASOF]
    rows.append({"tkr": t,
                 "ret5_pct": 100 * (c.iloc[-1] / c.iloc[-6] - 1),
                 "rank5": pct_rank(c, 5).iloc[-1],
                 "rank21": pct_rank(c, 21).iloc[-1],
                 "rank63": pct_rank(c, 63).iloc[-1]})
df = pd.DataFrame(rows).round(2).sort_values("rank5")
print(df.to_string(index=False))
print(f"  count rank5<=20: {(df['rank5'] <= 20).sum()} of {len(df)}")
print(f"  count rank5<=16.7: {(df['rank5'] <= 16.7).sum()} of {len(df)}")
print(f"  count rank5<=10: {(df['rank5'] <= 10).sum()} of {len(df)}")
print(f"  median rank63 = {df['rank63'].median():.1f};  "
      f"count rank63>=60: {(df['rank63'] >= 60).sum()}")
print(f"  XLF rank63 = {pct_rank(px['XLF']['Close'].dropna()[lambda s: s.index<=ASOF], 63).iloc[-1]:.1f}")

print("\n" + "=" * 78)
print("3. breadth-mask firing counts across history")
print("=" * 78)
# Panel of insurer closes; count of names available each day matters.
panel = pd.DataFrame({t: px[t]["Close"] for t in INSURERS if t in px}).sort_index()
panel = panel[panel.index <= ASOF]
r5 = pd.DataFrame({t: pct_rank(panel[t].dropna(), 5).reindex(panel.index)
                   for t in panel.columns})
r21 = pd.DataFrame({t: pct_rank(panel[t].dropna(), 21).reindex(panel.index)
                    for t in panel.columns})
r63 = pd.DataFrame({t: pct_rank(panel[t].dropna(), 63).reindex(panel.index)
                    for t in panel.columns})
navail = r5.notna().sum(axis=1)
print("  names with a valid rank5 by year:")
print(navail.groupby(navail.index.year).median().to_string())

xlf = px["XLF"]["Close"].dropna()
xlf_r63 = pct_rank(xlf, 63).reindex(panel.index)

for cut in (10, 20, 25):
    n_wash = (r5 <= cut).sum(axis=1)
    frac_wash = n_wash / navail
    for k in (0.5, 0.6, 0.7):
        for intact in (None, 60, 70):
            m = frac_wash >= k
            if intact is not None:
                med63 = r63.median(axis=1)
                m = m & (med63 >= intact)
            m = m & (navail >= 8)
            yrs = sorted(set(m[m].index.year))
            print(f"  rank5<={cut:2d}, frac>={k:.1f}, med_r63>={intact}: "
                  f"N={int(m.sum()):5d}  yrs={len(yrs)}  live_today="
                  f"{bool(m.reindex([ASOF]).fillna(False).iloc[0])}")

print("\n" + "=" * 78)
print("4. XLE vs USO 63d divergence: today and history")
print("=" * 78)
xle = px["XLE"]["Close"].dropna()
uso = px["USO"]["Close"].dropna()
idx = xle.index.intersection(uso.index)
idx = idx[idx <= ASOF]
xle, uso = xle.loc[idx], uso.loc[idx]
d63 = xle.pct_change(63) - uso.pct_change(63)
print(f"  today: XLE ret63 {100*xle.pct_change(63).iloc[-1]:+.2f}%, "
      f"USO ret63 {100*uso.pct_change(63).iloc[-1]:+.2f}%, "
      f"spread {100*d63.iloc[-1]:+.2f}pp")
print(f"  spread pctile in own history: "
      f"{100*(d63.iloc[-1] > d63.dropna()).mean():.1f}")
for thr in (0.10, 0.15, 0.19, 0.25):
    m = d63 >= thr
    print(f"  spread >= {100*thr:.0f}pp: N={int(m.sum()):5d} days, "
          f"yrs={sorted(set(m[m].index.year))}")
print(f"  USO unconditional drift: 5d {100*uso.pct_change(5).mean():+.3f}%  "
      f"10d {100*uso.pct_change(10).mean():+.3f}%  "
      f"63d {100*uso.pct_change(63).mean():+.3f}%")
print(f"  XLE unconditional drift: 5d {100*xle.pct_change(5).mean():+.3f}%  "
      f"10d {100*xle.pct_change(10).mean():+.3f}%")
