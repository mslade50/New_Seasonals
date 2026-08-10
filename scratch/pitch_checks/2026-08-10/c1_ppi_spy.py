"""C1 round 1 -- PPI day fingerprint on SPY + the back-to-back CPI->PPI pair.

Anchor convention (tradeable): anchor D = 2 td before the PPI session, so the
lag=1 MOC entry lands on the close BEFORE the print and h=1 exits on the PPI
session close. That is the only form today's calendar allows (entry MOC
2026-08-11, CPI 08-12 = h=1, PPI 08-13 = h=2).

Pre-specified hypothesis: h=1 (the PPI session's own move) with NO direction
prior. Everything else printed here is a SCAN and is labelled as such.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent

px = close_panel(["SPY"])
idx = px.index
pos = pd.Series(range(len(idx)), index=idx)

ev = load_events(["ppi", "cpi", "nfp", "fomc_decision"])


def ev_pos(kind: str) -> pd.DataFrame:
    """Trading-day position of each print (first session on/after the date)."""
    d = ev[ev.event == kind]["date"]
    rows = []
    for x in d:
        p = int(idx.searchsorted(x, side="left"))
        if p >= len(idx):
            continue
        rows.append({"date": x, "sess": idx[p], "pos": p,
                     "same_day": idx[p] == x})
    return pd.DataFrame(rows)


ppi = ev_pos("ppi")
cpi = ev_pos("cpi")
nfp = ev_pos("nfp")
fomc = ev_pos("fomc_decision")

print("PPI prints mapped:", len(ppi), " on-session:", int(ppi.same_day.sum()),
      " span", ppi.sess.min().date(), "..", ppi.sess.max().date())
print("CPI prints mapped:", len(cpi), " on-session:", int(cpi.same_day.sum()))

# ---------------------------------------------------------------- ordering
# For each month, is PPI before or after CPI, and how many sessions apart?
gaps = []
for _, r in ppi.iterrows():
    near = cpi.loc[(cpi["pos"] - r["pos"]).abs() <= 10]
    if near.empty:
        gaps.append({"ppi": r["sess"], "cpi": pd.NaT, "gap_td": np.nan})
        continue
    j = (near["pos"] - r["pos"]).abs().idxmin()
    gaps.append({"ppi": r["sess"], "cpi": cpi.loc[j, "sess"],
                 "gap_td": int(r["pos"] - cpi.loc[j, "pos"])})
gaps = pd.DataFrame(gaps)
print("\nPPI-minus-CPI session gap distribution (>0 = PPI AFTER CPI):")
print(gaps["gap_td"].value_counts().sort_index().to_string())
print("  PPI after CPI:", int((gaps.gap_td > 0).sum()),
      " PPI before CPI:", int((gaps.gap_td < 0).sum()),
      " same session:", int((gaps.gap_td == 0).sum()))
# this week: CPI Wed 08-12, PPI Thu 08-13 -> gap = +1
print("  gap == +1 (this week's shape):", int((gaps.gap_td == 1).sum()))
print("  gap == -1 (PPI one session BEFORE CPI):", int((gaps.gap_td == -1).sum()))

# ---------------------------------------------------------------- masks
def mask_from_anchor(anchor_pos: np.ndarray) -> pd.Series:
    m = pd.Series(False, index=idx)
    ap = anchor_pos[(anchor_pos >= 0) & (anchor_pos < len(idx))]
    m.iloc[ap] = True
    return m


# anchor = 2 td before the print session -> lag=1 entry is the prior close
ppi_anchor = (ppi["pos"] - 2).values
cpi_anchor = (cpi["pos"] - 2).values

m_ppi = mask_from_anchor(ppi_anchor)
m_cpi = mask_from_anchor(cpi_anchor)

# pair variants
pair_after = gaps.loc[gaps.gap_td == 1, "ppi"]        # this week's ordering
pair_before = gaps.loc[gaps.gap_td == -1, "ppi"]
isolated = gaps.loc[gaps.gap_td.abs() > 3, "ppi"]     # PPI far from any CPI

def mask_from_sess(sess) -> pd.Series:
    p = np.array([pos[s] - 2 for s in sess])
    return mask_from_anchor(p)


print("\n" + "=" * 78)
print("A. PRE-SPECIFIED CELL: long SPY, anchor 2td before PPI, h=1")
print("   (entry MOC the session before the print, exit the print session close)")
print("=" * 78)
battery(px, m_ppi, [("SPY", 1.0)], h=1, title="SPY long into PPI day",
        cost_bps=2.0, lag=1, min_gap=5, event_kinds=("cpi",),
        variants={"CPI anchor (contrast)": m_cpi})

print("\n" + "=" * 78)
print("B. SCAN (multiplicity applies): horizons 1..10 off the same PPI anchor")
print("=" * 78)
ppi_dates = idx[m_ppi.values]
show(horizon_scan(px, ppi_dates, [("SPY", 1.0)], hs=(1, 2, 3, 4, 5, 7, 10),
                  min_gap=5), "PPI anchor, SPY long, h scan")

print("\n" + "=" * 78)
print("C. PAIR ORDERING: does CPI-then-PPI differ from PPI-then-CPI?")
print("=" * 78)
for lbl, sess, h in [("CPI then PPI (gap +1, THIS WEEK)", pair_after, 1),
                     ("PPI then CPI (gap -1)", pair_before, 1),
                     ("PPI isolated (|gap| > 3td)", isolated, 1)]:
    mm = mask_from_sess(sess)
    r = vehicle_ret(px, [("SPY", 1.0)], h, 1)
    d = idx[mm.values & r.notna().values]
    print(f"\n--- {lbl}  N_sess={len(sess)} N_usable={len(d)} ---")
    if len(d) == 0:
        continue
    v = r.loc[d].values
    s = summarize(v, lbl)
    base = r.dropna()
    print(f"  mean {s['mean_pct']:+.3f}%  med {s['median_pct']:+.3f}%  "
          f"hit {s['hit']:.1f}%  t {s['t']:+.2f}  worst {s['worst_pct']:+.2f}%  "
          f"|  all-days ctrl {100*base.mean():+.3f}%  "
          f"edge {s['mean_pct']-100*base.mean():+.3f}pp")
    w = int((v > 0).sum())
    print(f"  record {w}-{len(v)-w}, sign p={sign_test(w, len(v)):.4f}, "
          f"bootstrap P(mean<=0)={bootstrap_p_le0(v):.3f}")
    yrs = pd.DatetimeIndex(d).year
    mid = (yrs % 4 == 2)
    show([summarize(v[mid], f"midterm (N={int(mid.sum())})"),
          summarize(v[~mid], f"non-midterm (N={int((~mid).sum())})")],
         "  midterm split")
    show(era_split(d, v), "  era split")

print("\n" + "=" * 78)
print("D. THE ACTUAL TRADE SHAPE: hold across BOTH prints")
print("   anchor = 2td before CPI when a PPI follows within 1-2 sessions,")
print("   entry MOC the session before CPI, h=2 or 3 covers CPI and PPI.")
print("=" * 78)
pair_cpi = gaps.loc[gaps.gap_td.isin([1, 2]), "cpi"].dropna()
m_pair = mask_from_sess(pair_cpi)
print(f"pair episodes (CPI followed by PPI within 2 td): N={len(pair_cpi)}")
for h in (2, 3, 5):
    battery(px, m_pair, [("SPY", 1.0)], h=h,
            title=f"SPY long across CPI->PPI pair, h={h}", cost_bps=2.0,
            lag=1, min_gap=5, event_kinds=("fomc_decision",))

print("\n" + "=" * 78)
print("E. MIDTERM SPLIT on the pre-specified h=1 PPI cell")
print("=" * 78)
r1 = vehicle_ret(px, [("SPY", 1.0)], 1, 1)
d1 = idx[m_ppi.values & r1.notna().values]
v1 = r1.loc[d1].values
yrs = pd.DatetimeIndex(d1).year
mid = yrs % 4 == 2
base = r1.dropna()
bmid = base.index.year % 4 == 2
show([summarize(v1[mid], f"PPI midterm (N={int(mid.sum())})"),
      summarize(v1[~mid], f"PPI non-midterm (N={int((~mid).sum())})"),
      summarize(base[bmid].values, "CTRL all days midterm"),
      summarize(base[~bmid].values, "CTRL all days non-midterm")],
     "midterm vs not, h=1")

print("\n" + "=" * 78)
print("F. AUGUST-ONLY / month-of-year fingerprint (scan, multiplicity applies)")
print("=" * 78)
mo = pd.DatetimeIndex(d1).month
rows = []
for m in range(1, 13):
    sel = mo == m
    if sel.sum():
        rows.append(summarize(v1[sel], f"month {m:02d} (N={int(sel.sum())})"))
show(rows, "PPI h=1 by calendar month")
