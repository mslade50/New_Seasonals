"""C5 round 2 -- the interaction moved the number NEGATIVE, so the only
tradeable form of C5 today is the SHORT: fade XLE on a crude thrust when a CPI
print lands inside the hold. Full battery on that, because "the gate flipped
the sign" is a claim that has to survive concentration, era and horizon before
anything can be pitched off it.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (close_panel, fwd_lag, declusters, summarize, sign_test,  # noqa: E402
                       event_in_window, bootstrap_p_le0, cluster_note, battery,
                       horizon_scan, show)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)

px = close_panel(["USO", "XLE", "DBC", "SPY"])
s = px["XLE"].dropna()
uso_1d = px["USO"].pct_change()


def cpi_gated_mask(thr: float, h: int) -> pd.Series:
    """USO 1d >= thr AND a CPI print lands inside the lag=1, h-day hold."""
    m = (uso_1d >= thr).reindex(s.index).fillna(False)
    days = s.index[m.values]
    fl = event_in_window(days, s.index, h, 1, ("cpi",))
    out = pd.Series(False, index=s.index)
    out.loc[days[fl]] = True
    return out


H = 3
mask = cpi_gated_mask(0.03, H)
print(f"trigger days (USO>=3% & CPI in a {H}d hold): {int(mask.sum())}")

variants = {}
for thr in (0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06):
    variants[f"USO>=({100*thr:.1f}%) & CPI in hold"] = cpi_gated_mask(thr, H)

battery(px, mask, [("XLE", -1.0)], h=H,
        title="C5 SHORT: fade XLE, USO 1d>=+3% with CPI inside a 3d hold",
        cost_bps=4.0, variants=variants, min_gap=5)

# ---------------------------------------------------------------------------
# horizon scan -- the sign reversed by h=5 in the cross table; confirm
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("HORIZON SCAN on the short (mask recomputed per horizon, as it must be)")
print("=" * 100)
rows = []
for h in (1, 2, 3, 4, 5, 7, 10):
    mk = cpi_gated_mask(0.03, h)
    days = s.index[mk.values]
    f = fwd_lag(s, h, lag=1)
    epi = declusters(days, 5, s.index)
    v = -f.reindex(epi).dropna()
    if len(v) < 5:
        continue
    own = -100 * f.dropna().mean()
    st = summarize(v.values)
    rows.append({"h": h, "n": st["n"], "short_mean_pct": round(st["mean_pct"], 3),
                 "own_short_drift": round(own, 3),
                 "excess": round(st["mean_pct"] - own, 3),
                 "hit": round(st["hit"], 1), "t": round(st["t"], 2),
                 "signp": round(sign_test(int((v.values > 0).sum()), len(v)), 4),
                 "worst": round(st["worst_pct"], 2)})
print(pd.DataFrame(rows).to_string(index=False))

# ---------------------------------------------------------------------------
# does today's own bucket (USO >= 6%) show it at all?
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("TODAY'S BUCKET -- USO 1d >= 6% (today printed +6.73%), CPI in hold")
print("=" * 100)
for thr, lbl in ((0.05, ">=5%"), (0.06, ">=6% <-- TODAY")):
    for h in (1, 2, 3, 5):
        mk = cpi_gated_mask(thr, h)
        days = s.index[mk.values]
        epi = declusters(days, 5, s.index)
        f = fwd_lag(s, h, lag=1)
        v = -f.reindex(epi).dropna()
        if len(v) == 0:
            print(f"  {lbl} h={h}: no episodes")
            continue
        st = summarize(v.values)
        print(f"  {lbl:<16} h={h}  n={st['n']:<3} short mean {st['mean_pct']:+.3f}% "
              f"hit {st['hit']:.1f} signp {sign_test(int((v.values>0).sum()), len(v)):.4f} "
              f"dates {[str(d.date()) for d in v.index]}")

# ---------------------------------------------------------------------------
# era + midterm on the pooled >=3% short
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("ERA / MIDTERM / LOYO on the pooled >=3% short, h=3")
print("=" * 100)
mk = cpi_gated_mask(0.03, 3)
epi = declusters(s.index[mk.values], 5, s.index)
f3 = fwd_lag(s, 3, lag=1)
v = -f3.reindex(epi).dropna()
epi = v.index
own = -100 * f3.dropna().mean()
print(f"pooled: n={len(v)} short mean {100*v.mean():+.3f}% excess {100*v.mean()-own:+.3f}% "
      f"hit {100*(v.values>0).mean():.1f} signp {sign_test(int((v.values>0).sum()), len(v)):.4f}")
print(f"concentration: {cluster_note(epi, v.values)}")
for lbl, sel in (("pre-2018", epi.year < 2018), ("2018+", epi.year >= 2018),
                 ("midterm <-- TODAY", (epi.year % 4) == 2),
                 ("non-midterm", (epi.year % 4) != 2)):
    vv = v.values[sel]
    if len(vv) < 3:
        continue
    st = summarize(vv)
    print(f"  {lbl:<20} n={st['n']:<3} mean {st['mean_pct']:+.3f}% "
          f"excess {st['mean_pct']-own:+.3f}% hit {st['hit']:.1f} "
          f"signp {sign_test(int((vv>0).sum()), len(vv)):.4f}")
print("\nepisode dates and short returns:")
print(pd.Series((100 * v.values).round(2), index=[d.date() for d in epi]).to_string())

print("\nLOYO:")
for y in sorted(set(epi.year)):
    keep = v.values[epi.year != y]
    if len(keep) < 5:
        continue
    st = summarize(keep)
    print(f"  drop {y}: n={st['n']:<3} mean {st['mean_pct']:+.3f}% "
          f"excess {st['mean_pct']-own:+.3f}% t {st['t']:+.2f} "
          f"signp {sign_test(int((keep>0).sum()), len(keep)):.4f}")
