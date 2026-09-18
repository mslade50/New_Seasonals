"""Small caps 21d rank in the bottom decile while the S&P sits near its high.

Tonight: IWM -1.37%, 21d -3.11% for a trailing-252 rank of 11.5, and 4.74%
below its own 52w high, while SPY is 1.99% below its 52w high with a 21d rank
of 15.9. ETFs and cash indices only, so no contract-roll exposure.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (close_panel, cluster_note, declusters, era_split,  # noqa
                       fwd_ret, local_control, pct_rank, sign_test, summarize)

TICKERS = ["IWM", "SPY", "^GSPC", "^VIX", "^RUT"]
px = close_panel(TICKERS)

iwm = px["IWM"].dropna()
spy = px["SPY"].dropna()

iwm_rank21 = pct_rank(iwm, 21, 252)
spy_hi252 = spy.rolling(252).max()
spy_near_high = (spy / spy_hi252 - 1.0) >= -0.03

print("current readings 2026-09-09")
print(f"  IWM 21d rank            {iwm_rank21.iloc[-1]:.1f}")
print(f"  SPY dist to 252d high   {100 * (spy.iloc[-1] / spy_hi252.iloc[-1] - 1):.2f}%")
print(f"  IWM dist to 252d high   "
      f"{100 * (iwm.iloc[-1] / iwm.rolling(252).max().iloc[-1] - 1):.2f}%")

mask = (iwm_rank21 <= 15) & spy_near_high.reindex(iwm_rank21.index).fillna(False)
trig_all = iwm_rank21.index[mask.fillna(False)]
trig = declusters(pd.DatetimeIndex(trig_all), 10, pd.DatetimeIndex(iwm.index))
print(f"\ncell: IWM 21d rank <= 15 AND SPY within 3% of its 252d high")
print(f"  raw days {len(trig_all)}, declustered episodes {len(trig)}")
print(f"  years: {pd.Series([d.year for d in trig]).value_counts().sort_index().to_dict()}")

for name in ["IWM", "SPY", "^VIX"]:
    s = px[name].dropna()
    print(f"\n--- {name}")
    for h in (1, 5, 21):
        f = fwd_ret(s, h).reindex(trig).dropna()
        if len(f) == 0:
            continue
        st = summarize(f.values, f"h{h}")
        up = int((f.values > 0).sum())
        n = len(f)
        base = summarize(fwd_ret(s, h).dropna().values, "all days")
        ctrl_idx = local_control(pd.DatetimeIndex(s.index),
                                 pd.DatetimeIndex(f.index), 126)
        ctrl = summarize(fwd_ret(s, h).reindex(ctrl_idx).dropna().values, "local")
        print(f"  h={h:<2} n={st['n']:<4} mean {st['mean_pct']:+.3f}%  "
              f"med {st['median_pct']:+.3f}%  hit {st['hit']:.1f}%  "
              f"t {st['t']:+.2f}  {up}-{n - up} up  "
              f"sign p {sign_test(up, n):.4f}")
        print(f"        all-days {base['mean_pct']:+.3f}%  "
              f"local+/-126 {ctrl['mean_pct']:+.3f}%  "
              f"edge vs local {st['mean_pct'] - ctrl['mean_pct']:+.3f}%")
        eras = era_split(pd.DatetimeIndex(f.index), f.values)
        print("        era: " + " | ".join(
            f"{e['label']} n={e.get('n', 0)} "
            f"{e.get('mean_pct', float('nan')):+.3f}% hit {e.get('hit', float('nan')):.0f}%"
            for e in eras))
        print(f"        {cluster_note(pd.DatetimeIndex(f.index), f.values)}")

# does the gap itself close, i.e. IWM minus SPY forward
print("\n--- IWM minus SPY, same anchors (relative, lag=0)")
for h in (1, 5, 21):
    a = fwd_ret(iwm, h).reindex(trig)
    b = fwd_ret(spy, h).reindex(trig)
    d = (a - b).dropna()
    st = summarize(d.values, f"h{h}")
    up = int((d.values > 0).sum())
    n = len(d)
    allrel = (fwd_ret(iwm, h) - fwd_ret(spy, h)).dropna()
    print(f"  h={h:<2} n={st['n']:<4} mean {st['mean_pct']:+.3f}pts  "
          f"hit {st['hit']:.1f}%  t {st['t']:+.2f}  {up}-{n - up}  "
          f"sign p {sign_test(up, n):.4f}  "
          f"all-days {summarize(allrel.values)['mean_pct']:+.3f}pts")
