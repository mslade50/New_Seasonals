"""CPI print day: the index mean is zero, so measure the dispersion instead.

Also the QQQ-minus-SPY spread on the same anchors, which the engine reported
as two separate means (QQQ +0.115 vs SPY +0.029) without ever pairing them.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

TICKERS = ["SPY", "QQQ", "^GSPC"]
px = load_prices(TICKERS)
ev = load_events(["cpi"])
cpi = pd.DatetimeIndex(sorted(pd.to_datetime(ev["date"]).unique()))

spy = px["SPY"]["Close"]
qqq = px["QQQ"]["Close"]
dates = spy.index

# anchor = session before the print, so h1 is the print session
anchors = []
prints = []
for d in cpi:
    pos = dates.searchsorted(d)
    if pos <= 0 or pos >= len(dates):
        continue
    if dates[pos] != d:            # print not a session in the panel
        continue
    anchors.append(dates[pos - 1])
    prints.append(dates[pos])
anchors = pd.DatetimeIndex(anchors)
prints = pd.DatetimeIndex(prints)
print(f"CPI prints matched to sessions: {len(prints)}  {prints[0].date()} .. {prints[-1].date()}")

spy_r = spy.pct_change()
qqq_r = qqq.pct_change()

print_moves = spy_r.reindex(prints).dropna()
base = spy_r.drop(index=prints, errors="ignore").dropna()
base = base[base.index >= print_moves.index[0]]

print("\n--- SPY, print session close-to-close ---")
print(f"  n={len(print_moves)}  mean {print_moves.mean()*100:+.3f}%  "
      f"up {(print_moves > 0).sum()}-{(print_moves < 0).sum()} down")
print(f"  baseline non-print sessions same span: n={len(base)}  mean {base.mean()*100:+.3f}%")

for thr in (0.5, 1.0, 1.5, 2.0):
    p_hit = (print_moves.abs() > thr / 100).mean()
    b_hit = (base.abs() > thr / 100).mean()
    z = (p_hit - b_hit) / np.sqrt(b_hit * (1 - b_hit) / len(print_moves))
    print(f"  |move| > {thr:.1f}%:  print days {p_hit*100:5.1f}%   "
          f"other days {b_hit*100:5.1f}%   ratio {p_hit/b_hit:4.2f}x   z={z:+.2f}")

print(f"\n  mean |move| print {print_moves.abs().mean()*100:.3f}%  "
      f"vs other {base.abs().mean()*100:.3f}%  "
      f"ratio {print_moves.abs().mean()/base.abs().mean():.2f}x")
print(f"  stdev print {print_moves.std()*100:.3f}%  vs other {base.std()*100:.3f}%")

# era split on the absolute move
for cut_lo, cut_hi, lab in [(None, "2018-01-01", "pre-2018"), ("2018-01-01", None, "2018+")]:
    m = print_moves
    if cut_lo:
        m = m[m.index >= cut_lo]
    if cut_hi:
        m = m[m.index < cut_hi]
    b = base
    if cut_lo:
        b = b[b.index >= cut_lo]
    if cut_hi:
        b = b[b.index < cut_hi]
    print(f"  {lab}: mean|move| print {m.abs().mean()*100:.3f}% (n={len(m)})  "
          f"other {b.abs().mean()*100:.3f}%   ratio {m.abs().mean()/b.abs().mean():.2f}x   "
          f"P(>1%) {(m.abs()>0.01).mean()*100:.1f}% vs {(b.abs()>0.01).mean()*100:.1f}%")

# ---- QQQ minus SPY on the print session ----
print("\n--- QQQ minus SPY, print session ---")
both = pd.concat([qqq_r, spy_r], axis=1, keys=["QQQ", "SPY"]).dropna()
sprd = (both["QQQ"] - both["SPY"])
sp = sprd.reindex(prints).dropna()
sb = sprd.drop(index=prints, errors="ignore").dropna()
sb = sb[sb.index >= sp.index[0]]
s = summarize(sp.values, "QQQ-SPY on print")
print(f"  n={s['n']}  mean {s['mean_pct']:+.3f}%  hit {s['hit']:.1f}%  t={s['t']:+.2f}")
print(f"  record {(sp>0).sum()}-{(sp<0).sum()}  sign p {sign_test(int((sp>0).sum()), int(len(sp))):.4f}")
sc = summarize(sb.values, "control")
print(f"  all other sessions same span: n={sc['n']}  mean {sc['mean_pct']:+.3f}%  hit {sc['hit']:.1f}%")
print(f"  edge over control: {s['mean_pct']-sc['mean_pct']:+.3f}%")
for e in era_split(sp.index, sp.values):
    print(f"  era {e['label']:<10} n={e['n']:<4} mean {e['mean_pct']:+.3f}%  hit {e['hit']:.1f}%  t={e['t']:+.2f}")
print("  concentration:", cluster_note(sp.index, sp.values))

# does it survive dropping the biggest days, and is it a mega-cap-era artifact?
srt = sp.reindex(sp.abs().sort_values(ascending=False).index)
trimmed = sp.drop(index=srt.index[:5])
st = summarize(trimmed.values, "trim5")
print(f"  drop 5 largest |spread| days: n={st['n']} mean {st['mean_pct']:+.3f}% "
      f"record {(trimmed>0).sum()}-{(trimmed<0).sum()} "
      f"sign p {sign_test(int((trimmed>0).sum()), int(len(trimmed))):.4f}")

by_dec = sp.groupby((sp.index.year // 5) * 5)
print("  by half-decade:")
for k, v in by_dec:
    print(f"    {k}s: n={len(v):<3} mean {v.mean()*100:+.3f}%  up {(v>0).sum()}-{(v<0).sum()}")

# same conditioning on the other two top-tier prints, as a specificity control
for kind in ("nfp", "fomc_decision"):
    e2 = load_events([kind])
    dd = pd.DatetimeIndex(sorted(pd.to_datetime(e2["date"]).unique()))
    dd = dd[dd.isin(dates)]
    v = sprd.reindex(dd).dropna()
    v = v[v.index >= sp.index[0]]
    print(f"  control {kind}: n={len(v)} mean {v.mean()*100:+.3f}% "
          f"up {(v>0).sum()}-{(v<0).sum()} sign p {sign_test(int((v>0).sum()), int(len(v))):.4f}")
