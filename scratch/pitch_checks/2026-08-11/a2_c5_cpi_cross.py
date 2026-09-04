"""C5 -- does the CPI print ADD anything to the crude thrust, or is it a
filter that does not filter?

Three passes:
  A. the PLAIN CPI-anchored energy cell on its own (XLE was never measured;
     measure it beside USO and DBC), vs a trading-day-of-month matched control.
  B. the C3 trigger SPLIT by whether CPI lands inside the hold -- at several
     thresholds so the split is not decided by 10 observations, and across
     horizons.
  C. the verdict: does the gate MOVE the number, and in which direction?

Anchor convention matches the surface map: anchor D = 2 sessions before the
print, so the lag=1 MOC entry lands on the session BEFORE the print and h=1
exits at the print session's close. That is exactly today's geometry.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (close_panel, load_events, fwd_lag, declusters, summarize,  # noqa: E402
                       sign_test, event_in_window, bootstrap_p_le0)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)

px = close_panel(["USO", "XLE", "DBC", "SPY", "XOP"])
idx = px.index
pos = pd.Series(range(len(idx)), index=idx)
uso_1d = px["USO"].pct_change()

# trading-day-of-month, for the calendar-matched control
tdom = pd.Series(idx, index=idx).groupby([idx.year, idx.month]).cumcount() + 1
tdom = pd.Series(tdom.values, index=idx)


def anchors_for(kind: str, offset: int = 2) -> pd.DatetimeIndex:
    """Trading day `offset` sessions BEFORE each print of `kind`."""
    ev = load_events([kind])["date"]
    out = []
    for d in ev:
        nxt = idx.searchsorted(d)
        if nxt >= len(idx) or idx[nxt] != d:
            nxt = idx.searchsorted(d, side="left")
            if nxt >= len(idx):
                continue
        p = nxt - offset
        if 0 <= p < len(idx):
            out.append(idx[p])
    return pd.DatetimeIndex(sorted(set(out)))


cpi_anchor = anchors_for("cpi", 2)
print(f"CPI anchors (2 sessions before the print): {len(cpi_anchor)}, "
      f"{cpi_anchor[0].date()} .. {cpi_anchor[-1].date()}")

# ---------------------------------------------------------------------------
# A. the plain CPI-anchored energy cell
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("A. PLAIN CPI-ANCHORED ENERGY CELL (no price state), tdom-matched control")
print("=" * 100)
rows = []
for tkr in ("XLE", "USO", "DBC", "SPY", "XOP"):
    ser = px[tkr].dropna()
    a = cpi_anchor.intersection(ser.index)
    for h in (1, 2, 3, 5):
        f = fwd_lag(ser, h, lag=1)
        v = f.reindex(a).dropna()
        if len(v) < 10:
            continue
        base_all = f.dropna()
        # tdom-matched control: all days sharing the anchors' tdom values
        want = set(tdom.reindex(v.index).dropna().astype(int))
        ctl_idx = tdom[tdom.isin(want)].index.intersection(base_all.index).difference(v.index)
        ctl = base_all.reindex(ctl_idx).dropna()
        st = summarize(v.values)
        rows.append({"tkr": tkr, "h": h, "n": st["n"],
                     "mean_pct": round(st["mean_pct"], 3),
                     "own_pct": round(100 * base_all.mean(), 3),
                     "tdom_ctl_pct": round(100 * ctl.mean(), 3),
                     "excess_own": round(st["mean_pct"] - 100 * base_all.mean(), 3),
                     "excess_tdom": round(st["mean_pct"] - 100 * ctl.mean(), 3),
                     "hit": round(st["hit"], 1), "t": round(st["t"], 2),
                     "signp": round(sign_test(int((v.values > 0).sum()), len(v)), 4)})
print(pd.DataFrame(rows).to_string(index=False))

# ---------------------------------------------------------------------------
# B. the C3 trigger split by CPI-in-hold, at several thresholds
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("B. C3 TRIGGER x CPI-IN-HOLD  (XLE, lag=1, declustered 5td)")
print("=" * 100)
s = px["XLE"].dropna()
rows = []
for thr in (0.03, 0.04, 0.045, 0.05, 0.06):
    m = (uso_1d >= thr).reindex(s.index).fillna(False)
    epi_all = declusters(s.index[m.values], 5, s.index)
    for h in (1, 2, 3, 5):
        f = fwd_lag(s, h, lag=1)
        v = f.reindex(epi_all).dropna()
        if len(v) < 8:
            continue
        fl = event_in_window(v.index, s.index, h, 1, ("cpi",))
        own = 100 * f.dropna().mean()
        for lbl, sel in (("CPI IN", fl), ("CPI OUT", ~fl)):
            vv = v.values[sel]
            if len(vv) < 3:
                continue
            st = summarize(vv)
            rows.append({"thr": f">={100*thr:.1f}%", "h": h, "cell": lbl, "n": st["n"],
                         "mean_pct": round(st["mean_pct"], 3),
                         "excess": round(st["mean_pct"] - own, 3),
                         "hit": round(st["hit"], 1),
                         "signp": round(sign_test(int((vv > 0).sum()), len(vv)), 4),
                         "worst": round(st["worst_pct"], 2)})
df = pd.DataFrame(rows)
print(df.pivot_table(index=["thr", "h"], columns="cell",
                     values=["n", "mean_pct", "excess", "hit"]).to_string())
print("\nfull table:")
print(df.to_string(index=False))

# ---------------------------------------------------------------------------
# B2. pooled across thresholds: is the CPI-IN degradation systematic?
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("B2. POOLED -- every USO>=3% day (max N), h=3, split by CPI in hold")
print("=" * 100)
m3 = (uso_1d >= 0.03).reindex(s.index).fillna(False)
epi3 = declusters(s.index[m3.values], 5, s.index)
f3 = fwd_lag(s, 3, lag=1)
v3 = f3.reindex(epi3).dropna()
fl3 = event_in_window(v3.index, s.index, 3, 1, ("cpi",))
own3 = 100 * f3.dropna().mean()
for lbl, sel in (("CPI IN hold", fl3), ("CPI OUT", ~fl3)):
    vv = v3.values[sel]
    st = summarize(vv)
    print(f"  {lbl:<14} n={st['n']:<4} mean {st['mean_pct']:+.3f}% "
          f"excess {st['mean_pct']-own3:+.3f}% hit {st['hit']:.1f} "
          f"t {st['t']:+.2f} signp {sign_test(int((vv>0).sum()), len(vv)):.4f} "
          f"worst {st['worst_pct']:+.2f}%")
d_in, d_out = v3.values[fl3], v3.values[~fl3]
se = np.sqrt(d_in.var(ddof=1) / len(d_in) + d_out.var(ddof=1) / len(d_out))
print(f"  IN minus OUT = {100*(d_in.mean()-d_out.mean()):+.3f}%  welch t {(d_in.mean()-d_out.mean())/se:+.2f}")

# CPI or PPI
fl_both = event_in_window(v3.index, s.index, 3, 1, ("cpi", "ppi"))
for lbl, sel in (("CPI or PPI IN", fl_both), ("neither", ~fl_both)):
    vv = v3.values[sel]
    st = summarize(vv)
    print(f"  {lbl:<14} n={st['n']:<4} mean {st['mean_pct']:+.3f}% "
          f"excess {st['mean_pct']-own3:+.3f}% hit {st['hit']:.1f} "
          f"signp {sign_test(int((vv>0).sum()), len(vv)):.4f}")

# ---------------------------------------------------------------------------
# B3. baseline: does CPI-in-hold hurt XLE on ALL days, not just trigger days?
#     (if it does, the "interaction" is just the CPI main effect)
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("B3. CONTROL -- CPI-in-hold effect on XLE across ALL days (the main effect)")
print("=" * 100)
allv = f3.dropna()
fl_all = event_in_window(allv.index, s.index, 3, 1, ("cpi",))
for lbl, sel in (("CPI IN hold", fl_all), ("CPI OUT", ~fl_all)):
    vv = allv.values[sel]
    st = summarize(vv)
    print(f"  {lbl:<14} n={st['n']:<5} mean {st['mean_pct']:+.3f}% hit {st['hit']:.1f} t {st['t']:+.2f}")
a_in, a_out = allv.values[fl_all], allv.values[~fl_all]
print(f"  MAIN EFFECT of CPI-in-hold on XLE (all days) = "
      f"{100*(a_in.mean()-a_out.mean()):+.3f}%")
print(f"  INTERACTION (trigger IN-OUT) minus (all-days IN-OUT) = "
      f"{100*((d_in.mean()-d_out.mean()) - (a_in.mean()-a_out.mean())):+.3f}%")

# ---------------------------------------------------------------------------
# C. verdict inputs
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("C. VERDICT INPUTS")
print("=" * 100)
print("Does the CPI gate MOVE the number, at the pitched 5% threshold, h=3?")
m5 = (uso_1d >= 0.05).reindex(s.index).fillna(False)
epi5 = declusters(s.index[m5.values], 5, s.index)
v5 = f3.reindex(epi5).dropna()
fl5 = event_in_window(v5.index, s.index, 3, 1, ("cpi",))
print(f"  ungated  n={len(v5)} mean {100*v5.mean():+.3f}%")
print(f"  CPI IN   n={int(fl5.sum())} mean {100*v5.values[fl5].mean():+.3f}%  "
      f"record {int((v5.values[fl5]>0).sum())}-{int((v5.values[fl5]<=0).sum())}  "
      f"sign p (for the LONG) {sign_test(int((v5.values[fl5]>0).sum()), int(fl5.sum())):.4f}  "
      f"sign p (for a SHORT) {sign_test(int((v5.values[fl5]<=0).sum()), int(fl5.sum())):.4f}")
print(f"  CPI OUT  n={int((~fl5).sum())} mean {100*v5.values[~fl5].mean():+.3f}%")
print(f"  the CPI-IN episodes: {[str(d.date()) for d in v5.index[fl5]]}")
print(f"  their returns: {[round(100*x,2) for x in v5.values[fl5]]}")
