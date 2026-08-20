"""C5: guard my own kill. The reference-class permutation could be too easy
if the null's max is being set by two structurally odd vehicles - RSX (trading
frozen from 2022) and YINN (3x leveraged, so its excess scales 3x by
construction). Re-run on the clean 11-name class without them.

If P(max >= KWEB) stays large without the two outliers, the kill is real.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

FULL = ["EEM", "EFA", "EWJ", "EWT", "EWW", "EWY", "EWZ", "FXI", "INDA",
        "KWEB", "RSX", "VGK", "YINN"]
CLEAN = [t for t in FULL if t not in ("RSX", "YINN")]

px = close_panel(["DX-Y.NYB"] + FULL)
d = px.index
mask = (pct_rank(px["DX-Y.NYB"], 21) <= 2).reindex(d).fillna(False)
epi_full = declusters(d[mask.values], 21, d)
kstart = px["KWEB"].dropna().index[0]
span = d[d >= kstart]
epi = pd.DatetimeIndex([x for x in epi_full if x >= kstart])
posmap = pd.Series(range(len(span)), index=span)
base_pos = np.array([posmap[x] for x in epi])
B = 20000

for INTL in (CLEAN,):
    print(f"\n\n=========== reference class = {len(INTL)} names (RSX/YINN removed) "
          f"===========")
    print("  ", INTL)
    for h in (5, 10):
        M = np.full((len(span), len(INTL)), np.nan)
        for j, t in enumerate(INTL):
            M[:, j] = vehicle_ret(px, [(t, 1.0)], h).reindex(span).values
        ok = ~np.isnan(M)
        drift = np.nanmean(M, axis=0)
        Z = np.where(ok, M, 0.0)

        def excess(rows):
            cnt = ok[rows].sum(axis=0)
            mean = Z[rows].sum(axis=0) / np.maximum(cnt, 1)
            e = 100 * (mean - drift)
            e[cnt < 10] = np.nan
            return e

        kj = INTL.index("KWEB")
        obs = excess(base_pos)
        kx = obs[kj]
        order = np.argsort(-obs)
        print(f"\n  h={h}: observed excess by name")
        for j in order:
            print(f"     {INTL[j]:<6} {obs[j]:+.3f}pp"
                  + ("   <== KWEB" if j == kj else ""))
        rng = np.random.default_rng(7)
        maxes = np.empty(B)
        for b in range(B):
            picked = []
            while len(picked) < len(base_pos):
                c = int(rng.integers(0, len(span)))
                if all(abs(c - p) >= 21 for p in picked):
                    picked.append(c)
            maxes[b] = np.nanmax(excess(np.array(picked)))
        print(f"    P(max name excess >= KWEB {kx:+.3f}pp) = "
              f"{np.mean(maxes >= kx):.4f}   null max median "
              f"{np.median(maxes):+.3f}  p95 {np.percentile(maxes,95):+.3f}")

        maxesB = np.empty(B)
        for b in range(B):
            k = int(rng.integers(1, len(span)))
            maxesB[b] = np.nanmax(excess((base_pos + k) % len(span)))
        print(f"    circular-shift null: P = {np.mean(maxesB >= kx):.4f}   "
              f"median {np.median(maxesB):+.3f}")
