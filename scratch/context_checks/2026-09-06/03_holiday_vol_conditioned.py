"""Is the post-Labor-Day VIX pop just weekend decay reversal, and does the
crushed entry (VIX 63d rank 9.1 tonight) change it?

Controls that matter:
  a) the eve session itself: if VIX falls into the holiday, the pop is the
     round trip and the honest sentence says so
  b) September Tuesdays that are NOT post-Labor-Day
  c) all Tuesdays
  d) the same low-vol entry state with no holiday in the way
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import close_panel, fwd_ret, summarize, sign_test, era_split, cluster_note, pct_rank  # noqa

px = close_panel(["^VIX", "^VIX3M", "^GSPC", "SPY"])
idx = px.index
vix = px["^VIX"].dropna()
spx = px["^GSPC"].dropna()

MID = {y for y in range(1996, 2030) if y % 4 == 2}

ld_anchor, ld_post = [], []
for yr in sorted(set(idx.year)):
    sept = pd.Timestamp(yr, 9, 1)
    lab = sept + pd.Timedelta(days=(7 - sept.weekday()) % 7)
    after, before = idx[idx > lab], idx[idx < lab]
    if not len(after) or not len(before) or (after[0] - lab).days > 6:
        continue
    ld_post.append(after[0]); ld_anchor.append(before[-1])
ld_anchor, ld_post = pd.DatetimeIndex(ld_anchor), pd.DatetimeIndex(ld_post)


def line(label, dates, s, h=1):
    d = pd.DatetimeIndex([x for x in dates if x in s.index])
    r = fwd_ret(s, h).reindex(d).dropna()
    if len(r) < 3:
        print(f"  {label:44} n={len(r)} thin"); return None
    v = r.values; up = int((v > 0).sum())
    st = summarize(v, label)
    print(f"  {label:44} n={len(v):5d} mean={st['mean_pct']:+7.3f}% "
          f"med={st['median_pct']:+7.3f}% {up}-{len(v)-up} hit={st['hit']:5.1f}% "
          f"t={st['t']:+5.2f} signp={sign_test(up, len(v)):.4f}")
    return r


print("=== a) the eve session itself: does VIX fall INTO Labor Day? ===")
# return of the eve session = anchor close vs the close before it
eve_prev = pd.DatetimeIndex([idx[idx.get_loc(d) - 1] for d in ld_anchor])
line("VIX on the Labor Day eve session", eve_prev, vix)
line("VIX on the post-Labor-Day session", ld_anchor, vix)
# round trip: eve close-2 -> post-holiday close (2 sessions)
line("VIX eve-1 close -> post-LD close (2 sessions)", eve_prev, vix, h=2)
print("  reading: if the 2-session round trip is near zero the pop is decay reversal")

print("\n=== b/c) Tuesday controls for VIX ===")
tues = idx[[d.weekday() == 1 for d in idx]]
sep_tues = pd.DatetimeIndex([d for d in tues if d.month == 9])
# anchors are the session BEFORE each Tuesday
def anchors_before(days):
    return pd.DatetimeIndex([idx[idx.get_loc(d) - 1] for d in days if idx.get_loc(d) > 0])
sep_t_anch = anchors_before(sep_tues)
sep_t_nonld = pd.DatetimeIndex([d for d in sep_t_anch if d not in set(ld_anchor)])
line("all September Tuesdays", sep_t_anch, vix)
line("September Tuesdays, post-Labor-Day removed", sep_t_nonld, vix)
line("all Tuesdays", anchors_before(tues), vix)
line("post-Labor-Day only", ld_anchor, vix)

print("\n=== d) conditioning on a crushed VIX entry ===")
rank63 = pct_rank(vix, 63, 252)
for lo, hi, nm in [(0, 15, "63d rank <= 15 (tonight 9.1)"), (15, 101, "63d rank > 15")]:
    m = (rank63 >= lo) & (rank63 < hi)
    sel = pd.DatetimeIndex([d for d in ld_anchor if d in m.index and bool(m.get(d, False))])
    line(f"post-LD, {nm}", sel, vix)
    print(f"      years: {sorted(set(d.year for d in sel))}")
# same crushed state with NO holiday next
allm = m.index[(rank63 >= 0) & (rank63 < 15)]
nohol = pd.DatetimeIndex([d for d in allm if d not in set(ld_anchor)])
line("CONTROL crushed VIX, any session, no holiday", nohol, vix)

print("\n=== the Labor Day cell in detail ===")
r = fwd_ret(vix, 1).reindex(pd.DatetimeIndex([d for d in ld_anchor if d in vix.index])).dropna()
print("  era:", [f"{e['label']} n={e['n']} mean={e['mean_pct']:+.2f}% hit={e['hit']:.1f}%"
                for e in era_split(r.index, r.values)])
print("  concentration:", cluster_note(r.index, r.values, 2))
srt = np.sort(r.values)
print(f"  drop the two best: n={len(srt)-2} mean={100*srt[:-2].mean():+.3f}% "
      f"med={100*np.median(srt[:-2]):+.3f}%")
print(f"  last 10 years: {[f'{d.year}:{100*v:+.1f}%' for d, v in zip(r.index, r.values)][-10:]}")
mid = pd.DatetimeIndex([d for d in r.index if d.year in MID])
print(f"  midterm years: {[f'{d.year}:{100*v:+.1f}%' for d, v in zip(r.index, r.values) if d.year in MID]}")

print("\n=== the equity side of the same session ===")
line("post-LD ^GSPC", ld_anchor, spx)
line("post-LD ^GSPC, midterm", pd.DatetimeIndex([d for d in ld_anchor if d.year in MID]), spx)
rv = fwd_ret(vix, 1).reindex(pd.DatetimeIndex([d for d in ld_anchor if d in vix.index]))
rs = fwd_ret(spx, 1).reindex(rv.index)
both = pd.DataFrame({"vix": rv, "spx": rs}).dropna()
up_vix = both[both.vix > 0]
print(f"  VIX up on {len(up_vix)} of {len(both)} post-LD sessions; on those the S&P "
      f"averaged {100*up_vix.spx.mean():+.3f}% and fell on {int((up_vix.spx<0).sum())}")
print(f"  both VIX up AND S&P down: {int(((both.vix>0)&(both.spx<0)).sum())} of {len(both)}")
print(f"  correlation of the two: {both.vix.corr(both.spx):+.2f}")
