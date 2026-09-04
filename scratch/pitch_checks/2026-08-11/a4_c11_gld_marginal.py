"""C11 specific: GLD is the underlying of the position the book already owns.

C4 died on definition, excess sign and basket. C11 survived more of the
teardown, so it gets the question that actually decides it: what does adding
long GLD to a live long GDX buy McKinley that scaling the GDX leg would not?

Also: the registry's two crash episodes (2013-04 -13.07%, 2020-08 -5.70%),
the mean-vs-record question at the horizon whose hit rate was quoted (h=2),
and the CPI-in-window subset, since the live window contains both CPI and PPI.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (close_panel, fwd_lag, declusters, summarize, sign_test,  # noqa: E402
                       pct_rank, bootstrap_p_le0, event_in_window, show,
                       horizon_scan)

warnings.filterwarnings("ignore")

px = close_panel(["GDX", "GLD", "SLV", "SPY"])
idx = px.index
correct = pct_rank(px["GDX"], 5)
tf = (correct >= 95).fillna(False)

print("=" * 96)
print("C11-1  HORIZON SCAN on the CORRECT trigger -- is h=2's 71.6% hit real?")
print("=" * 96)
rows = []
for h in (1, 2, 3, 5, 10):
    leg = fwd_lag(px["GLD"], h, lag=1)
    v = leg.notna()
    e = declusters(idx[(tf & v).values], 5, idx)
    s = summarize(leg.loc[e].values, f"h={h} episodes")
    s["own_drift"] = round(100 * leg[v].mean(), 3)
    s["excess_pct"] = round(s["mean_pct"] - 100 * leg[v].mean(), 3)
    s["signp"] = round(sign_test(int((leg.loc[e].values > 0).sum()), len(e)), 4)
    s["boot_p"] = round(bootstrap_p_le0(leg.loc[e].values), 3)
    rows.append(s)
show(rows, "GLD on GDX 5d rank>=95 (correct definition), by horizon")
print("  recon claimed h=2 +0.50 / excess +0.41 / hit 71.6 / p 0.000 -- compare the h=2 row.")

print()
print("=" * 96)
print("C11-2  MEAN vs RECORD: is the payoff a high hit rate or a few fat wins?")
print("=" * 96)
for h in (2, 5):
    leg = fwd_lag(px["GLD"], h, lag=1)
    e = declusters(idx[(tf & leg.notna()).values], 5, idx)
    v = leg.loc[e].values
    w, l = v[v > 0], v[v <= 0]
    print(f"  h={h}: N={len(v)}  hit {100*(v>0).mean():.1f}%  mean {100*v.mean():+.3f}%  "
          f"median {100*np.median(v):+.3f}%")
    print(f"        avg win {100*w.mean():+.3f}% (N={len(w)})  avg loss {100*l.mean():+.3f}% "
          f"(N={len(l)})  win/loss ratio {abs(w.mean()/l.mean()):.2f}")
    print(f"        trimmed mean (drop best+worst 5%) {100*np.mean(np.sort(v)[max(1,len(v)//20):-max(1,len(v)//20)]):+.3f}%")

print()
print("=" * 96)
print("C11-3  THE REGISTRY'S CRASH EPISODES -- are they in THIS trigger's sample?")
print("=" * 96)
for h in (5, 10):
    leg = fwd_lag(px["GLD"], h, lag=1)
    e = declusters(idx[(tf & leg.notna()).values], 5, idx)
    v = pd.Series(leg.loc[e].values, index=e)
    worst = v.nsmallest(6)
    print(f"\n  h={h} six worst episodes:")
    for d, r in worst.items():
        print(f"     {d.date()}  {100*r:+7.2f}%")
    for tag, lo, hi in (("2013-04", "2013-04-01", "2013-05-01"),
                        ("2020-08", "2020-08-01", "2020-09-01")):
        hits = v[(v.index >= lo) & (v.index < hi)]
        print(f"     registry crash {tag}: {'IN sample -> ' + ', '.join(f'{d.date()} {100*r:+.2f}%' for d, r in hits.items()) if len(hits) else 'NOT a trigger episode'}")

print()
print("=" * 96)
print("C11-4  CPI IS IN THE LIVE WINDOW.  registry: 'GLD into CPI underperforms")
print("       GLD's own unconditional drift'.  Split this trigger on it.")
print("=" * 96)
for h in (2, 5):
    leg = fwd_lag(px["GLD"], h, lag=1)
    e = declusters(idx[(tf & leg.notna()).values], 5, idx)
    v = leg.loc[e].values
    for kinds in (("cpi",), ("cpi", "ppi")):
        fl = event_in_window(e, idx, h, 1, kinds)
        a, b = v[fl], v[~fl]
        print(f"  h={h} {'+'.join(kinds):<8} IN  N={len(a):<3} {100*a.mean():+7.3f}% "
              f"hit {100*(a>0).mean():5.1f}%  sign p {sign_test(int((a>0).sum()), len(a)):.3f}   |  "
              f"OUT N={len(b):<3} {100*b.mean():+7.3f}% hit {100*(b>0).mean():5.1f}%")

print()
print("=" * 96)
print("C11-5  MARGINAL CONTRIBUTION to a book that is ALREADY long GDX")
print("=" * 96)
H = 5
live = fwd_lag(px["GDX"], H, lag=0)          # the live 2026-08-10 leg
gld = fwd_lag(px["GLD"], H, lag=1)
v = live.notna() & gld.notna()
e = declusters(idx[(tf & v).values], 5, idx)
a, b = gld.loc[e].values, live.loc[e].values
print(f"  N={len(e)} episodes")
for w in (0.0, 0.25, 0.5, 1.0):
    comb = b + w * a                          # 1 unit live GDX + w units new GLD
    sh = comb.mean() / comb.std(ddof=1)
    print(f"  book = 1.0 x live GDX + {w:.2f} x new GLD:  mean {100*comb.mean():+7.3f}%  "
          f"sd {100*comb.std(ddof=1):6.3f}%  mean/sd {sh:+.4f}  "
          f"worst {100*comb.min():+7.2f}%  hit {100*(comb>0).mean():5.1f}%")
print("\n  ... and the honest comparison: just size the leg you ALREADY have.")
for w in (1.0, 1.25, 1.5):
    comb = w * b
    print(f"  book = {w:.2f} x live GDX alone:              mean {100*comb.mean():+7.3f}%  "
          f"sd {100*comb.std(ddof=1):6.3f}%  mean/sd {comb.mean()/comb.std(ddof=1):+.4f}  "
          f"worst {100*comb.min():+7.2f}%  hit {100*(comb>0).mean():5.1f}%")
print("\n  If mean/sd barely moves, the second leg is not diversification, it is size.")

print()
print("=" * 96)
print("C11-6  DOES THE GDX-THRUST TRIGGER ADD ANYTHING OVER 'GLD IS RALLYING'?")
print("=" * 96)
gld_rank = pct_rank(px["GLD"], 5)
for lbl, m in (("GDX 5d rank>=95 (the candidate)", tf),
               ("GLD 5d rank>=95 (the plain state)", (gld_rank >= 95).fillna(False)),
               ("BOTH", tf & (gld_rank >= 95).fillna(False)),
               ("GDX only, GLD NOT >=95", tf & ~(gld_rank >= 95).fillna(False))):
    leg = fwd_lag(px["GLD"], 5, lag=1)
    e = declusters(idx[(m & leg.notna()).values], 5, idx)
    if len(e) < 4:
        print(f"  {lbl:<36} N={len(e)} too few")
        continue
    v2 = leg.loc[e].values
    print(f"  {lbl:<36} N={len(e):<4} {100*v2.mean():+7.3f}%  hit {100*(v2>0).mean():5.1f}%  "
          f"excess {100*v2.mean() - 100*leg[leg.notna()].mean():+7.3f}%  "
          f"sign p {sign_test(int((v2>0).sum()), len(e)):.4f}")
print("  today: GLD 5d rank = "
      f"{gld_rank.loc[px['GLD'].dropna().index[-1]]:.1f}")
