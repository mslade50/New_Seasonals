"""C2 round 2 - is this a re-find of the 2026-08-07 midterm mid-August kill?

Registry (2026-08-07): "midterm mid-August seasonality - N=6, carried entirely
by 2002 (+8.68%); drop-two-best is negative. The midterm restriction ANTI-WORKS
at 21 td (SPY midterm +0.361% vs non-midterm +0.531%; IWM +0.269% vs +1.455%)."

That was measured at h=21. Today's candidate is the SHORT at h=10. Same days,
same conditioner, different horizon - so the question is whether h=10 is a
genuinely different cell or the same six years read at a different exit.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = close_panel(["SPY", "QQQ", "IWM"])
IDX = px.index
epi = declusters(pd.DatetimeIndex([d for d in IDX if d.month == 8
                                   and 15 <= d.day <= 19]), 5, IDX)
mid = pd.DatetimeIndex([d for d in epi if d.year % 4 == 2])

print("=" * 92)
print("1. reproduce the 08-07 h=21 finding, then walk the horizon (LONG basis)")
print("=" * 92)
rows = []
for tkr in ("SPY", "QQQ", "IWM"):
    s = px[tkr].dropna()
    for h in (5, 10, 15, 21):
        f = fwd_lag(s, h, 1)
        m, n = f.reindex(mid).dropna(), f.reindex(epi.difference(mid)).dropna()
        rows.append({"tkr": tkr, "h": h, "n_mid": len(m),
                     "midterm_pct": round(100 * m.mean(), 3),
                     "nonmid_pct": round(100 * n.mean(), 3),
                     "mid-non": round(100 * (m.mean() - n.mean()), 3),
                     "mid_rec": f"{int((m>0).sum())}-{int((m<=0).sum())}"})
show(rows, "long-basis midterm vs non-midterm across horizons")

print("\n" + "=" * 92)
print("2. the SAME SIX YEARS at h=10 and h=21 - one cell or two?")
print("=" * 92)
s = px["SPY"].dropna()
a = fwd_lag(s, 10, 1).reindex(mid).dropna()
b = fwd_lag(s, 21, 1).reindex(mid).dropna()
j = pd.DataFrame({"h10": 100 * a, "h21": 100 * b}).dropna()
print(j.round(2).to_string())
print(f"  corr(h10, h21) across the six midterm years = {j['h10'].corr(j['h21']):.3f}")
print("  -> same six draws, read at two exits. Not an independent finding.")

print("\n" + "=" * 92)
print("3. drop-one / drop-two on the SHORT at h=10, every ticker")
print("=" * 92)
for tkr in ("SPY", "QQQ", "IWM"):
    s = px[tkr].dropna()
    f = -fwd_lag(s, 10, 1)
    v = f.reindex(mid).dropna()
    srt = np.sort(v.values)
    w = int((v > 0).sum())
    print(f"  {tkr}: N={len(v)} mean {100*v.mean():+.3f}%  median "
          f"{100*np.median(v.values):+.3f}%  record {w}-{len(v)-w} sign p "
          f"{sign_test(w, len(v)):.4f}  drop-best {100*srt[:-1].mean():+.3f}%  "
          f"drop-two-best {100*srt[:-2].mean():+.3f}%")
    print(f"     years: {dict(zip([d.year for d in v.index], (100*v.values).round(2)))}")

print("\n" + "=" * 92)
print("4. the 08-07 anti-work check at h=21, SHORT basis (does midterm help?)")
print("=" * 92)
for tkr in ("SPY", "QQQ", "IWM"):
    s = px[tkr].dropna()
    f = -fwd_lag(s, 21, 1)
    m, n = f.reindex(mid).dropna(), f.reindex(epi.difference(mid)).dropna()
    print(f"  {tkr} h=21 SHORT: midterm {100*m.mean():+.3f}% ({int((m>0).sum())}-"
          f"{int((m<=0).sum())})  non-midterm {100*n.mean():+.3f}%  "
          f"diff {100*(m.mean()-n.mean()):+.3f}pp")
