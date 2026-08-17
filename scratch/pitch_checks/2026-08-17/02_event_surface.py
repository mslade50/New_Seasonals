"""Stage B1 enumeration 1: every live calendar event x every asset class.

Today's anchor offsets: vix_expiry -2 td, opex -4 td, jackson_hole -9 td.
For each event we take the historical day sitting at the SAME offset and
measure the forward return to the event and a little past it, against the
instrument's own unconditional drift over the same span.

This is the sixty-cell grid the surface map has to give a verdict to. It is a
SEARCH, so anything that comes out of here is multiplicity-charged and needs a
mechanism before it earns a real check.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

BAR = pd.Timestamp("2026-08-14")

CLASSES = {
    "us_large": ["SPY", "QQQ"],
    "us_small": ["IWM"],
    "rates": ["TLT", "IEF"],
    "credit": ["HYG", "LQD"],
    "gold_miners": ["GLD", "GDX"],
    "other_metals": ["SLV"],
    "energy": ["USO", "XLE"],
    "dollar_fx": ["UUP", "DX-Y.NYB"],
    "international": ["EFA", "EEM", "FXI"],
    "volatility": ["SVXY", "^VIX"],
}
TICKERS = sorted({t for v in CLASSES.values() for t in v})

px = load_prices(TICKERS)
C = pd.DataFrame({t: px[t]["Close"] for t in TICKERS}).sort_index()
ALL = C.index

ev = load_events()
print("event kinds:", sorted(ev["event"].unique()))

# Today's live forward events and the td offset we sit at.
LIVE = [("vix_expiry", 2), ("opex", 4), ("jackson_hole", 9)]
# Horizons to look at from TODAY's analogue (entry lag=1, so h=1 is tomorrow's close).
HS = (1, 2, 3, 5, 8, 10)


def anchors(kind: str, off: int) -> pd.DatetimeIndex:
    """Sessions sitting exactly `off` trading days before an event of `kind`."""
    dates = pd.to_datetime(ev.loc[ev["event"] == kind, "date"].unique())
    out = []
    for d in dates:
        pos = ALL.searchsorted(d)          # first session >= event date
        if pos >= len(ALL):
            continue
        i = pos - off
        if i < 260 or i >= len(ALL) - 12:
            continue
        out.append(ALL[i])
    return pd.DatetimeIndex(sorted(set(out)))


def drift(s: pd.Series, h: int) -> float:
    return float(fwd_lag(s, h, 1).mean() * 100)


print("\n" + "=" * 118)
print("EVENT x ASSET CLASS GRID -- excess over own unconditional drift, percent")
print("entry lag=1 (MOC tomorrow), midterm-only column shown alongside all-years")
print("=" * 118)

for kind, off in LIVE:
    a = anchors(kind, off)
    a_mid = pd.DatetimeIndex([d for d in a if d.year % 4 == 2])
    print(f"\n### {kind}  (today sits {off} td before; N={len(a)} all-years, {len(a_mid)} midterm)")
    print(f"{'ticker':<10} {'class':<14} " + " ".join(f"h{h:<2}excess" for h in HS)
          + "   | " + " ".join(f"h{h:<2}mid" for h in HS))
    for cls, names in CLASSES.items():
        for t in names:
            s = C[t].dropna()
            aa = a[a.isin(s.index)]
            am = a_mid[a_mid.isin(s.index)]
            if len(aa) < 8:
                continue
            cells, mids = [], []
            for h in HS:
                f = fwd_lag(s, h, 1)
                v = f.reindex(aa).dropna()
                cells.append(f"{v.mean()*100 - drift(s, h):+8.3f}" if len(v) else "     n/a")
                vm = f.reindex(am).dropna()
                mids.append(f"{vm.mean()*100 - drift(s, h):+7.3f}" if len(vm) >= 4 else "    n/a")
            print(f"{t:<10} {cls:<14} " + " ".join(cells) + "   | " + " ".join(mids))
