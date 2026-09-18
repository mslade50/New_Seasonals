"""The pre-CPI vol decline, conditioned on how vol ENTERED the print.

Tonight ^VIX closed +8.38% at 17.84, its 5th consecutive up close, still 42.5% below its
own 252-day high. The engine's base cell (E:cpi|^VIX|k1, n=318, -0.86%, 120-195, BH pass)
pools every entry state. Split it.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np, pandas as pd
from pitch_lab import load_prices, load_events, sign_test, era_split, cluster_note

px = load_prices(['^VIX','^GSPC','SPY','^VIX3M'])
ev = load_events()
cpi = pd.DatetimeIndex(sorted(pd.to_datetime(ev.loc[ev['event'] == 'cpi', 'date'].unique())))

vix = px['^VIX']['Close'].dropna()
spx = px['^GSPC']['Close'].dropna()
d = vix.index

pairs = [(d[d < c][-1], c) for c in cpi if c in d and len(d[d < c])]
pairs = [(a, c) for a, c in pairs if a >= pd.Timestamp('1999-01-01')]

r1 = vix.pct_change()
# consecutive up closes ending at the anchor
up = (r1 > 0).astype(int)
streak = up * 0
run = 0
for i, u in enumerate(up.values):
    run = run + 1 if u else 0
    streak.iloc[i] = run
roll_max = vix.rolling(252).max()

def report(label, sel, s=vix):
    rows = [(a, (s.loc[c] / s.loc[a] - 1) * 100) for a, c in pairs if a in sel]
    if len(rows) < 3:
        print(f"  {label:56} n={len(rows)}"); return
    dd = pd.DatetimeIndex([r[0] for r in rows]); v = np.array([r[1] for r in rows])
    w = int((v > 0).sum()); n = len(v)
    t = v.mean() / (v.std(ddof=1) / np.sqrt(n))
    print(f"  {label:56} n={n:4} mean={v.mean():+7.3f}% up={w:3}/{n:3} ({100*w/n:4.1f}%) "
          f"t={t:+6.2f} signp(down)={sign_test(n-w,n):.4f}")
    return dd, v

print("VIX from the CPI eve close to the CPI print close\n")
allA = pd.DatetimeIndex([a for a, _ in pairs])
report("ALL CPI eves", set(allA))

print("\n-- by the eve session's own VIX move --")
report("eve VIX up 5% or more",  set(allA[r1.reindex(allA) >= 0.05]))
report("eve VIX up 0 to 5%",     set(allA[(r1.reindex(allA) > 0) & (r1.reindex(allA) < 0.05)]))
report("eve VIX down",           set(allA[r1.reindex(allA) <= 0]))

print("\n-- by consecutive up closes into the eve --")
for k in (1, 2, 3, 4, 5):
    report(f"eve is the {k}th+ consecutive VIX up close", set(allA[streak.reindex(allA) >= k]))

print("\n-- tonight's exact shape: 5+ up closes AND eve up 5%+ --")
sel5 = set(allA[(streak.reindex(allA) >= 5) & (r1.reindex(allA) >= 0.05)])
out = report("5+ up closes and eve +5% or more", sel5)
if out:
    dd, v = out
    print("      episodes:", ", ".join(f"{x.date()} {y:+.1f}%" for x, y in zip(dd, v)))

print("\n-- 5+ up closes, no move filter, detail --")
out = report("5+ consecutive up closes into the eve", set(allA[streak.reindex(allA) >= 5]))
if out:
    dd, v = out
    print("      episodes:", ", ".join(f"{x.date()} {y:+.1f}%" for x, y in zip(dd, v)))
    print("      era:", era_split(dd, v))
    print("      concentration:", cluster_note(dd, v))
    print("\n      same episodes, S&P 500 leg:")
    sv = np.array([(spx.loc[c] / spx.loc[a] - 1) * 100 for a, c in pairs if a in set(dd)])
    w = int((sv > 0).sum())
    print(f"      ^GSPC n={len(sv)} mean={sv.mean():+.3f}% up={w}/{len(sv)} signp={sign_test(w,len(sv)):.4f}")

print("\n-- control: how VIX behaves after 5+ up closes on ANY session (not just CPI eves) --")
anyd = d[(streak >= 5).reindex(d).fillna(False)]
anyd = anyd[anyd >= pd.Timestamp('1999-01-01')]
nxt = []
for a in anyd:
    i = d.get_loc(a)
    if i + 1 < len(d): nxt.append((vix.iloc[i+1] / vix.iloc[i] - 1) * 100)
nxt = np.array(nxt); w = int((nxt > 0).sum())
print(f"  any session with 5+ up closes  n={len(nxt)} mean={nxt.mean():+.3f}% up={w}/{len(nxt)} signp(down)={sign_test(len(nxt)-w,len(nxt)):.4f}")
