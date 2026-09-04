"""C3 KILL CHECK — the COUNT of sectors simultaneously in the bottom 15% of
their 5-day rank, as a monotone conditioner.

Count-first is already done (00c): 239 of 1805 sessions carry a count >= 5, so
the live state sits at roughly the 13th percentile of rarity. The only thing
that could make a mid-range count tradeable is a DOSE RESPONSE: forward return
must move monotonically with the count. The 2026-08-31 inverted-U rule says an
interior bucket wearing an extremity label is a kill.

Two panels on purpose:
  - 11-sector panel (XLC from 2018-06) = today's literal definition, but only
    ~1805 sessions of history.
  - 9-sector panel (the 1999-vintage SPDRs, XLRE/XLC dropped) = 2000+ history,
    today's count is 4 of 9.
A dose response that exists on one panel and not the other is a definitional
artifact, not an effect.

Traded objects measured, both at lag=1 (the real MOC-tomorrow order):
  (a) SPY forward (the breadth read as an index call)
  (b) the WASHED BASKET: equal-weight of exactly the sectors that were <= 15
      that day, which is what "buy the washout" actually means.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import pandas as pd
import numpy as np

ASOF = pd.Timestamp("2026-08-31")
SEC11 = ["XLK", "XLF", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC", "XLE"]
SEC9 = ["XLK", "XLF", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLE"]
THRESH = 15.0

px = load_prices(SEC11 + ["SPY"])
C = pd.DataFrame({t: px[t]["Close"] for t in px}).dropna(how="all")
C = C[C.index <= ASOF]
spy = C["SPY"].dropna()

# rank frames computed on each ticker's OWN valid sessions (rolling_on_valid rule)
R5 = pd.DataFrame({t: pct_rank(C[t].dropna(), 5) for t in SEC11})


def count_series(cols):
    sub = R5[cols].dropna()
    return (sub <= THRESH).sum(axis=1)


def washed_fwd(cols, h, lag=1):
    """Equal-weight forward return of exactly the sectors below THRESH that day."""
    sub = R5[cols].dropna()
    fwd = pd.DataFrame({t: fwd_lag(C[t].dropna(), h, lag) for t in cols})
    fwd = fwd.reindex(sub.index)
    m = (sub <= THRESH)
    num = (fwd.where(m)).mean(axis=1, skipna=True)
    return num


def dose_table(cols, label, hs=(5, 10, 21)):
    cnt = count_series(cols)
    print(f"\n===== DOSE LADDER — {label} ({len(cols)} sectors, "
          f"{cnt.index[0].date()} .. {cnt.index[-1].date()}, {len(cnt)} sessions) =====")
    print("  today's count:", int(cnt.iloc[-1]),
          "->", [t for t in cols if R5[t].iloc[-1] <= THRESH])
    print("  count distribution:", cnt.value_counts().sort_index().to_dict())
    for h in hs:
        sf = fwd_lag(spy, h, 1)
        wb = washed_fwd(cols, h, 1)
        rows = []
        for c in sorted(cnt.unique()):
            d = cnt.index[cnt == c]
            rows.append({"count": int(c),
                         "n_days": len(d),
                         "SPY_mean_pct": round(100 * sf.reindex(d).mean(), 4),
                         "SPY_hit": round(100 * (sf.reindex(d) > 0).mean(), 1),
                         "WASHED_mean_pct": round(100 * wb.reindex(d).mean(), 4),
                         "WASHED_hit": round(100 * (wb.reindex(d) > 0).mean(), 1)})
        df = pd.DataFrame(rows)
        base_spy = 100 * sf.reindex(cnt.index).mean()
        print(f"\n  --- h={h} td, lag=1 ---   (all-days SPY base over this span "
              f"= {base_spy:+.4f}%)")
        print(df.to_string(index=False))
        # monotonicity: Spearman of count vs forward, day level
        m = cnt.reindex(sf.index).dropna()
        j = pd.DataFrame({"c": m, "spy": sf.reindex(m.index),
                          "wb": wb.reindex(m.index)}).dropna()
        print(f"  spearman(count, SPY fwd) = {j['c'].corr(j['spy'], method='spearman'):+.4f}"
              f"   spearman(count, WASHED fwd) = {j['c'].corr(j['wb'], method='spearman'):+.4f}"
              f"   (n={len(j)})")


for cols, lbl in ((SEC11, "11-sector (today's literal definition)"),
                  (SEC9, "9-sector (2000+ history)")):
    dose_table(cols, lbl)

# ---------------------------------------------------------------------------
# the live cell as a binary, with the standard battery, on both panels
# ---------------------------------------------------------------------------
for cols, lbl, rung in ((SEC11, "11-sector", 5), (SEC9, "9-sector", 4)):
    cnt = count_series(cols)
    mask = (cnt >= rung).reindex(spy.index, fill_value=False)
    px_spy = pd.DataFrame({"SPY": spy})
    variants = {}
    for k in range(max(1, rung - 2), rung + 3):
        variants[f"count>={k}"] = (cnt >= k).reindex(spy.index, fill_value=False)
    battery(px_spy, mask, [("SPY", 1.0)], 10,
            f"C3 {lbl}: SPY long, count>={rung} sectors at r5<={THRESH:.0f}",
            cost_bps=2.0, variants=variants, min_gap=10)

# ---------------------------------------------------------------------------
# lag profile on the live cell (registry trap)
# ---------------------------------------------------------------------------
print("\n===== LAG PROFILE (11-sector, count>=5, h=10) =====")
cnt = count_series(SEC11)
trig = spy.index.intersection(cnt.index[cnt >= 5])
for lag in (0, 1, 2, 3):
    r = fwd_lag(spy, 10, lag)
    epi = declusters(trig.intersection(r.dropna().index), 10, spy.index)
    print("  lag=%d  %s" % (lag, summarize(r.reindex(epi).values, f"lag{lag}")))
