"""C11 round 2c: fix the reference-class null so it is not too generous.

NULL 2 in a4b pooled every peer's episode excesses into one bag (sd 6.46%,
inflated by RSX and KWEB) and then drew EWJ-sized samples from it. That
over-disperses the null for a 3.4%-sd name and makes the max-of-K enormous
(null median best-of-10 = +1.474%). Rebuild it properly:

NULL 3: impose the COMMON CLASS MEAN but keep each name's OWN episode
dispersion -- resample name c's own centered episode excesses, shifted to the
class mean, at its own N. Max over K. This asks exactly the family question:
if the washout rule has one class-wide effect, how often is the best of ten
as good as Japan?

Plus the direct test: Welch of EWJ's episode residuals against the pooled
OTHER-name residuals, which needs no simulation at all.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change  # noqa

import warnings
warnings.filterwarnings("ignore")

ASOF = pd.Timestamp("2026-08-20")
H = 5
PEERS = ["EWJ", "EWT", "EWW", "EWY", "EWZ", "EEM", "FXI", "INDA", "KWEB", "RSX"]

px = close_panel(PEERS + ["EFA"]).loc[:ASOF]
idx = px.index
r5 = {c: _valid_pct_change(px[c], 5) for c in px.columns}
rk5 = {c: pct_rank(px[c], 5) for c in px.columns}

ex, res, ns = {}, {}, {}
for c in PEERS:
    g = r5[c] - r5["EFA"]
    m = ((rk5[c] <= 5) & (g <= -0.025)).fillna(False)
    r = fwd_lag(px[c], H, 1)
    dd = px[[c, "EFA"]].pct_change().dropna()
    bb = float(np.polyfit(dd["EFA"], dd[c], 1)[0])
    rr = vehicle_ret(px, [(c, 1.0), ("EFA", -bb)], H, 1)
    e = declusters(idx[(m & r.notna() & rr.notna()).values], 5, idx)
    if len(e) < 5:
        continue
    ex[c] = 100 * r.loc[e].values - 100 * r.dropna().mean()
    res[c] = 100 * rr.loc[e].values - 100 * rr.dropna().mean()
    ns[c] = len(e)

names = list(ex)
print("name    N   excess    sd    residual   sd")
for c in names:
    print(f"{c:6s} {ns[c]:>3}  {ex[c].mean():+7.3f} {ex[c].std(ddof=1):6.2f}   "
          f"{res[c].mean():+7.3f} {res[c].std(ddof=1):6.2f}")

rng = np.random.default_rng(42)
NB = 20000


def null3(book):
    cm = float(np.mean([book[c].mean() for c in names]))          # equal-weight class mean
    cen = {c: book[c] - book[c].mean() + cm for c in names}
    out = np.empty(NB)
    for i in range(NB):
        out[i] = max(rng.choice(cen[c], size=ns[c], replace=True).mean() for c in names)
    return cm, out


for lbl, book in (("EXCESS", ex), ("BETA-NEUTRAL RESIDUAL", res)):
    cm, mx = null3(book)
    obs = book["EWJ"].mean()
    print(f"\n=== NULL 3 on {lbl} ===")
    print(f"  equal-weight class mean = {cm:+.3f}%   EWJ observed = {obs:+.3f}%")
    print(f"  null max-of-{len(names)}: median {np.median(mx):+.3f}  95th {np.percentile(mx,95):+.3f}")
    print(f"  P(max-of-{len(names)} >= EWJ) = {(mx >= obs).mean():.4f}")
    # direct Welch, no simulation
    other = np.concatenate([book[c] for c in names if c != "EWJ"])
    a = book["EWJ"]
    se = np.sqrt(a.var(ddof=1) / len(a) + other.var(ddof=1) / len(other))
    print(f"  Welch EWJ vs pooled other names: {a.mean()-other.mean():+.3f}pp  t {(a.mean()-other.mean())/se:+.2f}"
          f"   (others mean {other.mean():+.3f}%, N={len(other)})")
    # drop the two wildest names and redo
    keep = [c for c in names if c not in ("RSX", "KWEB")]
    cm2 = float(np.mean([book[c].mean() for c in keep]))
    cen2 = {c: book[c] - book[c].mean() + cm2 for c in keep}
    mx2 = np.array([max(rng.choice(cen2[c], size=ns[c], replace=True).mean() for c in keep)
                    for _ in range(NB)])
    print(f"  ex-RSX/KWEB (8 names): class mean {cm2:+.3f}%  P(max >= EWJ) = {(mx2 >= obs).mean():.4f}"
          f"  null median max {np.median(mx2):+.3f}")
