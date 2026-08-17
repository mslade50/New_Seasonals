"""C10 - the h=2 pocket against its reference class + the live joint cell.

h=2 was the only GDX rank-100 pocket to survive concentration (excess +1.171pp,
13-5, top-2 episodes 3% of total, drop-top-3 +0.388pp, LOYO floor +0.999%,
both eras positive). Registry 2026-08-13: "a single-ticker result has to be
priced against its reference class, not just against its own bootstrap", and
that test is stronger than any within-instrument robustness check because it
measures how much dispersion the estimator manufactures at N~18 when nothing
is there.

Also: the live joint cell (rank 100 AND 21d ret >= 26% AND dd <= -20%), which
is the state actually on the tape this morning.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

NAMES = ["GDX", "GDXJ", "NEM", "AEM", "KGC", "GLD", "SLV",
         "XLE", "XLF", "SMH", "IWM", "EEM", "TLT", "USO", "SPY"]
px = close_panel(NAMES)
H = 2

print("===== reference class: 21d rank==100 -> h=2, identical rule =====")
rows, ex = [], []
for t in NAMES:
    s = px[t].dropna()
    if len(s) < 600:
        continue
    rk = pct_rank(s, 21)
    rr = fwd_lag(s, H, 1)
    d = s.index[(rk >= 100.0).values & rr.notna().values]
    e = declusters(d, 21, s.index)
    r = summarize(rr.loc[e].values, t)
    if not r["n"]:
        continue
    base = 100 * rr.dropna().mean()
    r["ctl_pct"] = round(base, 3)
    r["excess_pp"] = round(r["mean_pct"] - base, 3)
    r["se_pp"] = round(r["sd_pct"] / np.sqrt(r["n"]), 3)
    rows.append(r)
    ex.append(r["excess_pp"])
show(rows, "h=2, episodes, per name")
ex = np.array(ex)
se = np.array([r["se_pp"] for r in rows])
gdx = rows[0]["excess_pp"]
print(f"\n  cross-name excess: mean {ex.mean():+.3f}pp, observed sd "
      f"{ex.std(ddof=1):.3f}pp, mean sampling SE {se.mean():.3f}pp "
      f"-> dispersion ratio {ex.std(ddof=1)/se.mean():.2f}")
print(f"  GDX excess {gdx:+.3f}pp ranks {1+int((ex > gdx).sum())} of {len(ex)}")
# permutation: the distribution of "best name under no effect"
rng = np.random.default_rng(42)
best = []
for _ in range(2000):
    draw = rng.normal(0.0, se)
    best.append(draw.max())
best = np.array(best)
print(f"  permutation (each name ~ N(0, its own SE)): max-excess mean "
      f"{best.mean():+.3f}pp, 95th {np.percentile(best, 95):+.3f}pp; "
      f"P(max >= GDX's {gdx:+.3f}) = {(best >= gdx).mean():.3f}")

# ---------------------------------------------------------------------------
print("\n===== the LIVE joint cell on GDX =====")
s = px["GDX"].dropna()
rk = pct_rank(s, 21)
r21 = s.pct_change(21)
dd = s / s.rolling(252).max() - 1.0
for H in (2, 3, 5, 10):
    rr = fwd_lag(s, H, 1)
    rows = []
    base = 100 * rr.dropna().mean()
    for lbl, m in [("rank100 parent", rk >= 100.0),
                   ("+ ret>=26% (LIVE magnitude)", (rk >= 100.0) & (r21 >= 0.26)),
                   ("  ret in [20,26) (where the edge is)",
                    (rk >= 100.0) & (r21 >= 0.20) & (r21 < 0.26)),
                   ("+ ret>=26 + dd<=-20 (FULL LIVE)",
                    (rk >= 100.0) & (r21 >= 0.26) & (dd <= -0.20))]:
        e = declusters(s.index[m.values & rr.notna().values], 21, s.index)
        r = summarize(rr.loc[e].values, f"h={H} {lbl}")
        if r["n"]:
            r["excess_pp"] = round(r["mean_pct"] - base, 3)
            r["sign_p"] = round(sign_test(int((rr.loc[e].values > 0).sum()),
                                          r["n"]), 4)
        rows.append(r)
    show(rows, f"h={H}")
    if H == 2:
        e = declusters(s.index[((rk >= 100.0) & (r21 >= 0.26)).values
                               & rr.notna().values], 21, s.index)
        print("   ret>=26 episodes:", ", ".join(
            f"{d.date()} {100*r21.loc[d]:.0f}%->{100*rr.loc[d]:+.1f}%" for d in e))
