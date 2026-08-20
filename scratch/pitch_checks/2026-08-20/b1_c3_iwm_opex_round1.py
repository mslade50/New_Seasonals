"""C3 round 1 + 2: long IWM out of August opex, entry MOC at the opex-1 close.

Anchor convention reproduced from scratch, not trusted from 03_recon_events.py:
the entry close is the session BEFORE opex (today's slot). To keep pitch_lab's
lag=1 convention honest the MASK sits on opex-2 and lag=1 carries the entry to
the opex-1 close.

Attack list from the brief: offset placebo ladder, month-of-year control,
midterm split, concentration / drop-best-years, the 2000-2004 registry problem,
the near-52w-high subset (the shape that INVERTED on 2026-08-17), cost.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
warnings.filterwarnings("ignore")
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

TK = ["IWM", "SPY"]
px = close_panel(TK)
d = px.index

ev = load_events(["opex"])
opex = pd.DatetimeIndex(sorted(set(ev["date"]) & set(d)))
print(f"opex events in the price index: {len(opex)}  "
      f"{opex.min().date()} .. {opex.max().date()}")


def anchor_at(offset: int) -> pd.DatetimeIndex:
    """Sessions `offset` trading days from each opex date."""
    pos = d.get_indexer(opex) + offset
    pos = pos[(pos >= 0) & (pos < len(d))]
    return d[pos]


ENTRY_OFF = -1                       # entry close = opex-1 = today
entry = anchor_at(ENTRY_OFF)
mask_dates = anchor_at(ENTRY_OFF - 1)  # signal bar, lag=1 -> entry at opex-1
aug_entry = entry[entry.month == 8]
aug_mask = mask_dates[mask_dates.month == 8]

print(f"\nSANITY: today 2026-08-20 is opex-1? next opex = "
      f"{ev[ev['date'] > '2026-08-19']['date'].iloc[0].date()}")
print(f"entry anchors all months N={len(entry)}, August N={len(aug_entry)}")
print("August entry anchors:", ", ".join(str(x.date()) for x in aug_entry))
tdom = []
for a in aug_entry:
    m = d[(d.year == a.year) & (d.month == 8)]
    tdom.append(int(np.where(m == a)[0][0]) + 1)
print(f"August entry anchor trading-day-of-month: min {min(tdom)} max {max(tdom)} "
      f"median {int(np.median(tdom))}   (today is tdom 14)")

m_all = pd.Series(False, index=d)
m_all.loc[mask_dates] = True
m_aug = pd.Series(False, index=d)
m_aug.loc[aug_mask] = True

# ---------------------------------------------------------------- round 1
for h in (3, 5, 10):
    battery(px, m_aug, [("IWM", 1.0)], h,
            f"C3 LONG IWM, AUGUST opex-1 entry", cost_bps=3.0,
            variants={"all months (pooled)": m_all},
            min_gap=15, event_kinds=("jackson_hole",))

# ---------------------------------------------------- offset placebo ladder
print("\n\n" + "=" * 78)
print("OFFSET PLACEBO LADDER — August anchors, k = -10..+5 sessions from opex")
print("=" * 78)
for h in (3, 5, 10):
    rows = []
    for k in range(-10, 6):
        a = anchor_at(k)
        a = a[a.month == 8]
        r = fwd_lag(px["IWM"], h, lag=0).reindex(a).dropna()
        if len(r) < 5:
            continue
        rows.append({"k": k, "n": len(r), "mean_pct": 100 * r.mean(),
                     "hit": 100 * (r > 0).mean(),
                     "true": "<== TRUE" if k == ENTRY_OFF else ""})
    df = pd.DataFrame(rows).sort_values("mean_pct", ascending=False)
    df["rank"] = range(1, len(df) + 1)
    print(f"\n--- h={h} ---")
    print(df.round(3).to_string(index=False))
    tr = df[df["k"] == ENTRY_OFF]
    print(f"  >>> TRUE ANCHOR (k={ENTRY_OFF}) RANKS {int(tr['rank'].iloc[0])} "
          f"of {len(df)} at h={h}")

# ------------------------------------------------------- month-of-year ctrl
print("\n\n" + "=" * 78)
print("MONTH-OF-YEAR CONTROL: the same opex-1 anchor, every month")
print("=" * 78)
for h in (5, 10):
    rows = []
    for mo in range(1, 13):
        a = entry[entry.month == mo]
        r = fwd_lag(px["IWM"], h, lag=0).reindex(a).dropna()
        rows.append({"month": mo, "n": len(r), "mean_pct": 100 * r.mean(),
                     "hit": 100 * (r > 0).mean()})
    df = pd.DataFrame(rows).sort_values("mean_pct", ascending=False)
    df["rank"] = range(1, 13)
    print(f"\n--- IWM h={h}, opex-1 anchor by month ---")
    print(df.round(3).to_string(index=False))
    print(f"  August ranks {int(df[df.month == 8]['rank'].iloc[0])} of 12")

# unconditional August window (the control that closed JH large caps)
print("\n--- UNCONDITIONAL AUGUST WINDOW: every August start at tdom "
      f"{min(tdom)}-{max(tdom)}, no event involved ---")
aug_days = d[d.month == 8]
tdom_map = {}
for y in sorted(set(aug_days.year)):
    m = d[(d.year == y) & (d.month == 8)]
    for i, x in enumerate(m):
        tdom_map[x] = i + 1
band = pd.DatetimeIndex([x for x in aug_days
                         if min(tdom) <= tdom_map[x] <= max(tdom)])
for h in (3, 5, 10):
    r_uc = fwd_lag(px["IWM"], h, lag=0).reindex(band).dropna()
    r_ev = fwd_lag(px["IWM"], h, lag=0).reindex(aug_entry).dropna()
    print(f"  h={h:2d}  unconditional Aug tdom band: {100*r_uc.mean():+.3f}% "
          f"over {len(r_uc)} starts (hit {100*(r_uc>0).mean():.0f}%)   "
          f"|  opex anchor: {100*r_ev.mean():+.3f}% over {len(r_ev)}   "
          f"|  anchor minus window = {100*(r_ev.mean()-r_uc.mean()):+.3f}pp")

# ------------------------------------------------------------ midterm split
print("\n\n" + "=" * 78)
print("MIDTERM SPLIT (year %% 4 == 2). 2026 IS MIDTERM.")
print("=" * 78)
for h in (3, 5, 10):
    r = fwd_lag(px["IWM"], h, lag=0).reindex(aug_entry).dropna()
    mid = r.index.year % 4 == 2
    show([summarize(r[mid].values, f"h={h} MIDTERM (N={int(mid.sum())})"),
          summarize(r[~mid].values, f"h={h} non-midterm (N={int((~mid).sum())})")])
    print(f"   midterm years: "
          f"{sorted(set(r.index[mid].year))}")

# ------------------------------------------------------ concentration/years
print("\n\n" + "=" * 78)
print("CONCENTRATION: by-year table and drop-best-k")
print("=" * 78)
for h in (5, 10):
    r = fwd_lag(px["IWM"], h, lag=0).reindex(aug_entry).dropna()
    byyr = pd.Series(100 * r.values, index=r.index.year).sort_values(
        ascending=False)
    print(f"\n--- h={h} by year (%) ---")
    print(byyr.round(2).to_string())
    v = np.sort(r.values)[::-1]
    print(f"  full mean {100*v.mean():+.3f}%  "
          f"drop-1 {100*v[1:].mean():+.3f}%  "
          f"drop-2 {100*v[2:].mean():+.3f}%  "
          f"drop-3 {100*v[3:].mean():+.3f}%")
    print(f"  {cluster_note(r.index, r.values, k=2)}")
    print(f"  worst window {100*v.min():+.2f}% in "
          f"{r.index[int(np.argmin(r.values))].year}")

# --------------------------------------------------------- era: registry's
print("\n\n" + "=" * 78)
print("ERA: the registry's 'run INTO August opex' died because 2000-2004 "
      "carried it. Does the run OUT have the same problem?")
print("=" * 78)
for h in (5, 10):
    r = fwd_lag(px["IWM"], h, lag=0).reindex(aug_entry).dropna()
    e1 = r[r.index.year <= 2004]
    e2 = r[(r.index.year >= 2005) & (r.index.year <= 2009)]
    e3 = r[r.index.year >= 2010]
    show([summarize(e1.values, f"h={h} 2000-2004"),
          summarize(e2.values, f"h={h} 2005-2009"),
          summarize(e3.values, f"h={h} 2010+")])

# ------------------------------------------- near-52w-high subset (LIVE)
print("\n\n" + "=" * 78)
print("LIVE SLICE: IWM is 1.10%% off its 52w high today. Registry 2026-08-17: "
      "'IWM at a 52w high into opex week' — the opex gate INVERTED it.")
print("=" * 78)
hi = rolling_on_valid(px["IWM"], lambda x: x.rolling(252).max())
off = (px["IWM"] / hi - 1.0) * 100
print(f"IWM off-high on the last bar: {off.iloc[-1]:.2f}%")
for h in (5, 10):
    r = fwd_lag(px["IWM"], h, lag=0).reindex(aug_entry).dropna()
    o = off.reindex(r.index)
    for thr in (-2.0, -3.0, -5.0):
        near = o >= thr
        far = ~near
        print(f"  h={h:2d} off-high >= {thr:+.0f}%: "
              f"{100*r[near.values].mean():+.3f}% N={int(near.sum())} "
              f"(hit {100*(r[near.values]>0).mean():.0f}%)   |  "
              f"farther: {100*r[far.values].mean():+.3f}% N={int(far.sum())}")
    # and the pooled all-month version of the same gate
    r2 = fwd_lag(px["IWM"], h, lag=0).reindex(entry).dropna()
    o2 = off.reindex(r2.index)
    n2 = o2 >= -2.0
    print(f"  h={h:2d} POOLED all months, off-high >= -2%: "
          f"{100*r2[n2.values].mean():+.3f}% N={int(n2.sum())}  |  "
          f"rest {100*r2[~n2.values].mean():+.3f}% N={int((~n2).sum())}")

# ------------------------------------------------------------- IWM vs SPY
print("\n\n" + "=" * 78)
print("IS THIS A SMALL-CAP STORY? IWM minus SPY on the same anchors")
print("=" * 78)
for h in (3, 5, 10):
    ri = fwd_lag(px["IWM"], h, lag=0).reindex(aug_entry).dropna()
    rs = fwd_lag(px["SPY"], h, lag=0).reindex(aug_entry).dropna()
    j = ri.index.intersection(rs.index)
    sp = (ri.loc[j] - rs.loc[j])
    print(f"  h={h:2d}  IWM {100*ri.loc[j].mean():+.3f}%  SPY "
          f"{100*rs.loc[j].mean():+.3f}%  spread {100*sp.mean():+.3f}% "
          f"(hit {100*(sp>0).mean():.0f}%, N={len(j)}, "
          f"sign p {sign_test(int((sp>0).sum()), len(j)):.4f})")
    # IWM's own unconditional beta-free base: all-days IWM minus SPY
    ai = fwd_lag(px["IWM"], h, lag=0).dropna()
    as_ = fwd_lag(px["SPY"], h, lag=0).dropna()
    jj = ai.index.intersection(as_.index)
    print(f"        all-days IWM-SPY base {100*(ai.loc[jj]-as_.loc[jj]).mean():+.3f}%")
