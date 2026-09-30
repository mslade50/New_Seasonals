"""WMT round 2 follow-up: excess-over-drift concentration, residual vs XLP drop-two,
sign test against WMT's own 21d base hit rate."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitch_lab import *  # noqa
from sb2_common import *  # noqa

px = close_panel(["WMT", "XLP", "SPY"]).dropna()
w, x = px["WMT"], px["XLP"]
anc = anchors(px.index)
for lag in (1, 2):
    s = yearly(w, anc, lag, 21, 1)
    xs = yearly(x, anc, lag, 21, 1)
    base = fwd_lag(w, 21, lag).dropna()
    bhit = float((base > 0).mean())
    drift = float(base.mean())
    ex = s - drift
    o = ex.sort_values(ascending=False)
    print(f"\nlag={lag}: base hit {100*bhit:.0f}%, drift {100*drift:+.2f}%")
    print(f"  record {int((s>0).sum())}/{len(s)}, sign p vs base hit {sign_test(int((s>0).sum()), len(s), bhit):.3f}")
    print(f"  excess mean {100*ex.mean():+.2f}pp, drop1 {100*o.iloc[1:].mean():+.2f}, drop2 {100*o.iloc[2:].mean():+.2f}, "
          f"top2 {list(o.index[:2])} = {100*o.iloc[:2].sum():+.1f}pp of {100*ex.sum():+.1f}pp")
    ex_med = float(np.median(s) - np.median(base))
    print(f"  median excess {100*ex_med:+.2f}pp")
    r = (s - xs).dropna()
    xd = float(fwd_lag(w, 21, lag).sub(fwd_lag(x, 21, lag)).dropna().mean())
    ro = (r - xd).sort_values(ascending=False)
    print(f"  WMT-XLP resid mean {100*r.mean():+.2f}% (uncond spread {100*xd:+.2f}%), excess {100*(r.mean()-xd):+.2f}pp, "
          f"drop2 {100*ro.iloc[2:].mean():+.2f}pp, top2 {list(ro.index[:2])}; hit {int((r>0).sum())}/{len(r)}")
    print("  yrs:", ", ".join(f"{y}:{100*v:+.1f}" for y, v in s.items()))
