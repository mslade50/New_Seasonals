import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitch_lab import *  # noqa
from sb5_common import PX, IDX, H, round1_2, anchors, yearly, MIDTERMS

import numpy as np

res = round1_2("XLP")
print("\n### (f) defensive rotation: XLP-SPY excess vs SPY window return")
for lag, (s, sp, ex) in res.items():
    j = sp.reindex(ex.index)
    rho = np.corrcoef(ex.values, j.values)[0, 1]
    up, dn = ex[j > 0], ex[j <= 0]
    print(f" lag{lag}: corr(excess, SPY) {rho:+.2f} | SPY up yrs n={len(up)} excess {100*up.mean():+.2f}% "
          f"hit {int((up>0).sum())}/{len(up)} | SPY down yrs n={len(dn)} excess {100*dn.mean():+.2f}% "
          f"hit {int((dn>0).sum())}/{len(dn)}")
    xs = s.reindex(j.index)
    print(f"   XLP abs in SPY-up yrs {100*xs[j>0].mean():+.2f}% / SPY-down yrs {100*xs[j<=0].mean():+.2f}%")
    print("   midterm rows (yr: XLP / SPY / ex): " + ", ".join(
        f"{y}:{100*s[y]:+.2f}/{100*sp[y]:+.2f}/{100*ex[y]:+.2f}" for y in s.index if y in MIDTERMS))
    b = np.polyfit(j.values, xs.values, 1)
    print(f"   window beta XLP on SPY {b[0]:.2f}, alpha {100*b[1]:+.2f}%")
