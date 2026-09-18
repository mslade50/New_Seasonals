"""C8: narrow leadership week. <=2 of 9 original SPDRs beat SPY over 5d while SPY is
within 2% of its 252 high; SPY forward h=5..10. Dose response + cap-weight confound."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
sys.path.insert(0, str(Path(__file__).resolve().parent))
from k2_common import cell, fmt, conc, near_high  # noqa

SPDR = ["XLK", "XLF", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLE"]
px = close_panel(SPDR + ["SPY"])
px = px[px.index >= "1999-01-01"]
r = px.pct_change(5)
nbeat = sum((r[t] > r["SPY"]).astype(int) for t in SPDR)
allv = px[SPDR].notna().all(axis=1)
nbeat = nbeat.where(allv)
nh = near_high(px["SPY"])
ew = r[SPDR].mean(axis=1)
gap = r["SPY"] - ew  # cap-weight over equal-weight sectors, 5d
gr = pct_rank((1 + gap).cumprod(), 1) if False else None
print("live nbeat", nbeat.iloc[-1], "SPY nh", bool(nh.iloc[-1]), "SPY-EW 5d gap pp",
      round(100 * gap.iloc[-1], 2), "XLK leads", bool(r["XLK"].iloc[-1] > r["SPY"].iloc[-1]))
M = (nbeat <= 2) & nh
L = [("SPY", 1.0)]
battery(px, M, L, 5, "C8 SPY long, <=2 of 9 beat SPY 5d & SPY nh", 3,
        variants={"<=1 & nh": (nbeat <= 1) & nh, "<=3 & nh": (nbeat <= 3) & nh,
                  "<=2 NO nh": nbeat <= 2, "<=2 & nh 3%": (nbeat <= 2) & near_high(px["SPY"], .03)},
        event_kinds=("fomc_decision",))

print("\n=== dose response across count (SPY nh subset, and all) ===")
for h in (5, 10):
    print(f"-- h={h}")
    for k in range(10):
        print(fmt(cell(px, (nbeat == k) & nh, L, h), f"count=={k} & nh"),
              "|", fmt(cell(px, nbeat == k, L, h), f"count=={k} all")[44:120])
    print(fmt(cell(px, nh, L, h), "SPY nh alone (control)"))
    print(fmt(cell(px, (nbeat <= 2) & ~nh, L, h), "<=2 & SPY NOT near high"))

print("\n=== cap-weight confound: SPY minus EW sectors 5d gap ===")
gq = gap.rolling(252).rank(pct=True) * 100
for h in (5, 10):
    print(f"-- h={h}")
    print(fmt(cell(px, (gq >= 90) & nh, L, h), "gap top decile & nh"))
    print(fmt(cell(px, M & (gq >= 90), L, h), "narrow & gap top decile"))
    print(fmt(cell(px, M & (gq < 90), L, h), "narrow & gap NOT top decile"))
    xlkl = r["XLK"] > r["SPY"]
    print(fmt(cell(px, M & xlkl, L, h), "narrow & XLK among leaders (live)"))
    print(fmt(cell(px, M & ~xlkl, L, h), "narrow & XLK not leading"))
c = cell(px, M, L, 5)
print("\nconcentration h5:", conc(c))
mid = pd.Series(px.index.year % 4 == 2, index=px.index)
for h in (5, 10):
    print(fmt(cell(px, M & (px.index < "2018-01-01"), L, h), f"pre-2018 h{h}"))
    print(fmt(cell(px, M & (px.index >= "2018-01-01"), L, h), f"2018+ h{h}"))
    print(fmt(cell(px, M & mid, L, h), f"midterm h{h}"))
