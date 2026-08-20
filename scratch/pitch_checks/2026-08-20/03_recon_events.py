"""Recon 3: the two live calendar events crossed with every asset class.
opex is +1 td, jackson_hole is +6 td. NFP/PPI/CPI are past the 10 td horizon cap.
Anchor convention: entry at the CLOSE of the session before the event (which is
today's close for opex), forward h sessions. Controls are each vehicle's own
unconditional drift over the same h."""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
warnings.filterwarnings("ignore")
from pitch_lab import *  # noqa

CLASSES = {
    "us_large": "SPY", "us_small": "IWM", "rates": "TLT", "credit": "HYG",
    "gold": "GLD", "metals": "SLV", "energy": "USO", "energy_eq": "XLE",
    "dollar_fx": "DX-Y.NYB", "intl": "EEM", "intl_dev": "EFA", "vol": "^VIX",
    "vol_veh": "SVXY", "tech": "XLK", "defensive": "XLV",
}
px = close_panel(sorted(set(CLASSES.values())))
d = px.index

def anchors(kind, offset=-1):
    ev = load_events([kind])
    e = pd.DatetimeIndex(sorted(set(ev["date"]) & set(d)))
    pos = d.get_indexer(e) + offset
    pos = pos[(pos >= 0) & (pos < len(d))]
    return d[pos]

def sweep(name, anch, hs, month=None):
    print(f"\n{'='*72}\n{name}  anchors={len(anch)}  "
          f"{anch.min().date()}->{anch.max().date()}")
    if month:
        anch = anch[anch.month == month]
        print(f"  restricted to month {month}: {len(anch)} anchors")
    hdr = f"{'class':<11}{'vehicle':<9}" + "".join(f"{'h='+str(h):>21}" for h in hs)
    print(hdr)
    for cls, tk in CLASSES.items():
        if tk not in px.columns:
            continue
        cells = []
        for h in hs:
            v = fwd_lag(px[tk], h, lag=0).reindex(anch).dropna()
            c = fwd_lag(px[tk], h, lag=0).dropna()
            if len(v) < 5:
                cells.append(f"{'n/a':>21}")
                continue
            exc = (v.mean() - c.mean()) * 100
            cells.append(f"{v.mean()*100:>+7.2f}%{(v>0).mean()*100:>4.0f}% ex{exc:>+6.2f}"[:21].rjust(21))
        print(f"{cls:<11}{tk:<9}" + "".join(cells))

sweep("OPEX, entry at the opex-1 close (today's slot), ALL months",
      anchors("opex", -1), (1, 2, 3, 5, 10))
sweep("OPEX, entry at the opex-1 close, AUGUST only",
      anchors("opex", -1), (1, 2, 3, 5, 10), month=8)

jh = anchors("jackson_hole", -6)
sweep("JACKSON HOLE, entry 6 sessions before the conference (today's slot)",
      jh, (3, 5, 6, 8, 10))
