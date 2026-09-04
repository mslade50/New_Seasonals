"""B1 recon: every live calendar event x every asset class, one anchor ladder.

Not a check. This tells the surface map which of the ~60 event x class cells
are worth spending a real check on. Anchor = the session k td BEFORE the event
(k=0 means the event lands inside the h=1 hold entered at the anchor close).
Entry is lag=1 MOC on the anchor+1 close, per the house convention, so a k=1
anchor is TODAY's run staging tonight's close.

Prints excess over each instrument's own same-span drift, which is the only
control that matters at this stage.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

CLASSES = {
    "us_large": ["SPY", "QQQ"],
    "us_small": ["IWM"],
    "rates": ["TLT", "IEF"],
    "credit": ["HYG", "LQD"],
    "gold_miners": ["GLD", "GDX"],
    "metals": ["SLV"],
    "energy": ["USO", "DBC", "XLE"],
    "dollar_fx": ["UUP"],
    "intl": ["EFA", "EEM", "FXI"],
    "vol": ["SVXY"],
}
TKRS = sorted({t for v in CLASSES.values() for t in v})
EVENTS = ["cpi", "ppi", "vix_expiry", "opex", "jackson_hole", "nfp"]

px = close_panel(TKRS)
idx = px.index
pos = pd.Series(range(len(idx)), index=idx)
ev = load_events(EVENTS)


def anchors(kind, k):
    """Sessions exactly k td before an event date (k=0 -> event day itself)."""
    out = []
    for d in ev.loc[ev["event"] == kind, "date"]:
        loc = idx.searchsorted(pd.Timestamp(d))
        if loc >= len(idx):          # registry 2026-08-11: never mint a fake anchor
            continue
        p = loc - k
        if 0 <= p < len(idx):
            out.append(idx[p])
    return pd.DatetimeIndex(sorted(set(out)))


print("recon: excess (cond mean - own same-span drift), percent, h=1 lag=1")
print("k = sessions between the ANCHOR close and the event; the tradeable")
print("entry is the anchor+1 close, so k=1 is an entry ON the event session.\n")

for kind in EVENTS:
    for k in (2, 1, 0):
        a = anchors(kind, k)
        if len(a) < 10:
            continue
        line = []
        for cls, tks in CLASSES.items():
            for t in tks:
                r = fwd_lag(px[t], 1, 1)
                v = r.loc[r.index.intersection(a)].dropna()
                if len(v) < 10:
                    continue
                span = (idx >= v.index[0]) & (idx <= v.index[-1])
                ctrl = r[span].dropna()
                exc = 100 * (v.mean() - ctrl.mean())
                hit = 100 * (v > 0).mean()
                line.append(f"{t}:{exc:+.3f}({hit:.0f}%,N{len(v)})")
        print(f"[{kind} k={k}] " + "  ".join(line))
    print()
