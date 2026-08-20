"""C10: where does the post-opex drift accrue — overnight or intraday?

Decomposition, cheap by design. For each session t after the anchor:
    overnight(t) = Open[t] / Close[t-1] - 1
    intraday(t)  = Close[t] / Open[t]  - 1
and the two sum (compounded, but additively to first order) to the session's
close-to-close return. Summed over the h sessions after the entry close.

The placebo (registry 2026-08-10 rule): run the IDENTICAL decomposition on
NON-opex anchors matched on trading-day-of-month. If the split looks the same
there, the opex label carries nothing.

Cost: an overnight leg is an MOC-to-MOO round trip, ~8-10 bps all-in on SPY.
The bar is 5x, i.e. ~40-50 bps per window.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
warnings.filterwarnings("ignore")
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

px_raw = load_prices(["SPY", "IWM"])
spy = px_raw["SPY"]
iwm = px_raw["IWM"]
d = spy.index

ev = load_events(["opex"])
opex = pd.DatetimeIndex(sorted(set(ev["date"]) & set(d)))
pos = pd.Series(range(len(d)), index=d)


def legs(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    on = df["Open"] / df["Close"].shift(1) - 1.0
    intra = df["Close"] / df["Open"] - 1.0
    return on, intra


def window_sum(s: pd.Series, anchors: pd.DatetimeIndex, h: int,
               skip: int = 1) -> pd.Series:
    """Sum of `s` over sessions anchor+skip .. anchor+skip+h-1."""
    v = s.values
    out = {}
    for a in anchors:
        p = pos.get(a)
        if p is None or p + skip + h - 1 >= len(d):
            continue
        seg = v[p + skip: p + skip + h]
        if np.isnan(seg).any():
            continue
        out[a] = float(seg.sum())
    return pd.Series(out)


def tdom_of(x):
    m = d[(d.year == x.year) & (d.month == x.month)]
    return int(np.where(m == x)[0][0]) + 1


print("SANITY: Open column present and non-degenerate")
for t, df in (("SPY", spy), ("IWM", iwm)):
    on, intra = legs(df)
    print(f"  {t}: bars {len(df)}, mean overnight {100*on.mean():+.4f}%/day, "
          f"mean intraday {100*intra.mean():+.4f}%/day, "
          f"sum {100*(on.mean()+intra.mean()):+.4f}% vs close-to-close "
          f"{100*df['Close'].pct_change().mean():+.4f}%")

# non-opex tdom-matched control: same tdom, at least 4 td from any opex
opex_pos = set()
for a in opex:
    p = pos.get(a)
    if p is not None:
        for k in range(-4, 5):
            opex_pos.add(p + k)
opex_tdoms = sorted(set(tdom_of(x) for x in opex))
print(f"\nopex trading-day-of-month values observed: {opex_tdoms}")
ctrl = pd.DatetimeIndex([x for i, x in enumerate(d)
                         if i not in opex_pos and tdom_of(x) in opex_tdoms])
print(f"tdom-matched NON-opex control anchors: {len(ctrl)} "
      f"(>=4 td from any opex)")

for TK, df in (("SPY", spy), ("IWM", iwm)):
    on, intra = legs(df)
    cc = df["Close"].pct_change()
    print("\n\n" + "=" * 78)
    print(f"{TK}: post-opex window decomposition, entry at the OPEX close "
          f"(skip=1 -> first session after opex)")
    print("=" * 78)
    for lbl, anch in (("POOLED opex", opex),
                      ("AUGUST opex", pd.DatetimeIndex(
                          [x for x in opex if x.month == 8])),
                      ("tdom-matched NON-opex PLACEBO", ctrl)):
        print(f"\n--- {lbl} ---")
        for h in (1, 2, 3, 5, 10):
            o = window_sum(on, anch, h)
            i_ = window_sum(intra, anch, h)
            c = window_sum(cc, anch, h)
            j = o.index.intersection(i_.index).intersection(c.index)
            if len(j) < 4:
                continue
            print(f"  h={h:2d} N={len(j):4d}  close-to-close "
                  f"{100*c.loc[j].mean():+7.3f}%  =  overnight "
                  f"{100*o.loc[j].mean():+7.3f}% (hit "
                  f"{100*(o.loc[j]>0).mean():4.1f}%)  +  intraday "
                  f"{100*i_.loc[j].mean():+7.3f}% (hit "
                  f"{100*(i_.loc[j]>0).mean():4.1f}%)")

    # the decisive comparison: opex minus the tdom-matched placebo, per leg
    print(f"\n--- {TK}: OPEX MINUS TDOM-MATCHED PLACEBO, per leg ---")
    for h in (1, 2, 3, 5, 10):
        oo, oi = window_sum(on, opex, h), window_sum(intra, opex, h)
        co, ci = window_sum(on, ctrl, h), window_sum(intra, ctrl, h)
        print(f"  h={h:2d}  overnight excess "
              f"{100*(oo.mean()-co.mean()):+7.3f}pp   intraday excess "
              f"{100*(oi.mean()-ci.mean()):+7.3f}pp   "
              f"(N opex {len(oo)}, N placebo {len(co)})")
    print(f"\n--- {TK}: AUGUST opex minus tdom-matched placebo, per leg ---")
    aug = pd.DatetimeIndex([x for x in opex if x.month == 8])
    for h in (1, 2, 3, 5, 10):
        oo, oi = window_sum(on, aug, h), window_sum(intra, aug, h)
        co, ci = window_sum(on, ctrl, h), window_sum(intra, ctrl, h)
        print(f"  h={h:2d}  overnight excess "
              f"{100*(oo.mean()-co.mean()):+7.3f}pp   intraday excess "
              f"{100*(oi.mean()-ci.mean()):+7.3f}pp   "
              f"(N Aug opex {len(oo)})")

    # cost verdict on the best overnight leg
    print(f"\n--- {TK}: COST on the overnight leg (MOC->MOO, ~9 bps "
          f"round trip per night held) ---")
    for h in (1, 2, 3, 5):
        o = window_sum(on, opex, h)
        gross = 100 * 100 * o.mean()
        cost = 9.0 * h
        print(f"  h={h:2d}  gross {gross:+7.2f} bps over {h} night(s), "
              f"cost {cost:.0f} bps -> {gross/cost:+.2f}x  (need >= 5x)")
    # single best night
    print(f"\n--- {TK}: NIGHT-BY-NIGHT after opex (which single night?) ---")
    for k in range(1, 6):
        o = window_sum(on, opex, 1, skip=k)
        i_ = window_sum(intra, opex, 1, skip=k)
        co = window_sum(on, ctrl, 1, skip=k)
        ci = window_sum(intra, ctrl, 1, skip=k)
        print(f"  session +{k}: overnight {100*o.mean():+7.3f}% "
              f"(placebo {100*co.mean():+7.3f}%, excess "
              f"{100*(o.mean()-co.mean()):+6.3f}pp, {100*(o.mean()-co.mean())*100:+.1f} bps "
              f"vs a 9 bp cost)   intraday {100*i_.mean():+7.3f}% "
              f"(placebo {100*ci.mean():+7.3f}%)")

# ------------------------------------------------- mechanism measurability
print("\n\n" + "=" * 78)
print("MECHANISM MEASURABILITY (honesty rule for flow_mechanics)")
print("=" * 78)
for p in ("data/option_surface_history.parquet",
          "data/option_positioning_history.parquet"):
    f = Path(p)
    if not f.exists():
        print(f"  {p}: ABSENT")
        continue
    dfp = pd.read_parquet(f)
    dates = None
    for c in dfp.columns:
        if "date" in c.lower():
            dates = pd.to_datetime(dfp[c])
            break
    print(f"  {p}: {len(dfp)} rows, cols {list(dfp.columns)[:8]}"
          + (f", dates {dates.min().date()}..{dates.max().date()}"
             if dates is not None else ""))
print("  -> dealer gamma history does not exist in this repo. The stated "
      "mechanism (hedge unwind at the session boundary) cannot be measured "
      "here; only its price shadow can, and that is what the placebo tests.")
