"""C5 attack 1: price KWEB against its reference class.

KWEB is 1 positive of 7 risk assets on the identical trigger in the recon.
Run the identical rule across every EM/intl vehicle the cache holds, then
permute to get P(max name excess >= KWEB's).  Two nulls:

  A. random anchors   - same episode count, random dates in the common span,
                        min gap 21 td
  B. circular shift   - the REAL trigger positions shifted by a random offset,
                        preserving the trigger set's own spacing

Both measure every name on the SAME drawn dates, so the enormous
cross-sectional correlation inside EM is preserved by construction.

Forward returns are precomputed into a (dates x names) matrix once; the
permutation loop is pure numpy indexing.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

INTL = ["EEM", "EFA", "EWJ", "EWT", "EWW", "EWY", "EWZ", "FXI", "INDA",
        "KWEB", "RSX", "VGK", "YINN"]
px = close_panel(["DX-Y.NYB"] + INTL)
d = px.index
rk21 = pct_rank(px["DX-Y.NYB"], 21)
mask = (rk21 <= 2).reindex(d).fillna(False)
epi_full = declusters(d[mask.values], 21, d)

kstart = px["KWEB"].dropna().index[0]
span = d[d >= kstart]
epi = pd.DatetimeIndex([x for x in epi_full if x >= kstart])
print(f"common span {span[0].date()} .. {span[-1].date()}  ({len(span)} sessions)")
print(f"episodes in span: {len(epi)}")
print("  ", ", ".join(str(x.date()) for x in epi))

B = 20000
rng = np.random.default_rng(20260820)
posmap = pd.Series(range(len(span)), index=span)
base_pos = np.array([posmap[x] for x in epi])
n_ep = len(base_pos)

for h in (5, 10):
    print(f"\n\n################ h={h} ################")
    # (len(span) x n_names) forward returns, NaN where undefined
    M = np.full((len(span), len(INTL)), np.nan)
    for j, t in enumerate(INTL):
        M[:, j] = vehicle_ret(px, [(t, 1.0)], h).reindex(span).values
    ok = ~np.isnan(M)
    drift = np.nanmean(np.where(ok, M, np.nan), axis=0)      # each name's own
    n_valid = ok.sum(axis=0)                                  # all-days N

    keep = n_valid >= 500
    names = [t for j, t in enumerate(INTL) if keep[j]]
    M, ok, drift = M[:, keep], ok[:, keep], drift[keep]
    Z = np.where(ok, M, 0.0)

    def excess(rows):
        cnt = ok[rows].sum(axis=0)
        s = Z[rows].sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
        e = 100 * (mean - drift)
        e[cnt < 10] = np.nan          # name must actually be alive on the draw
        return e, cnt

    obs, cnt = excess(base_pos)
    rows = []
    for j, t in enumerate(names):
        if np.isnan(obs[j]):
            continue
        v = M[base_pos, j]
        v = v[~np.isnan(v)]
        rows.append({"ticker": t, "N": int(cnt[j]),
                     "mean_pct": round(100 * v.mean(), 3),
                     "own_drift_pct": round(100 * drift[j], 3),
                     "excess_pp": round(obs[j], 3),
                     "hit": round(100 * (v > 0).mean(), 1)})
    rows.sort(key=lambda r: -r["excess_pp"])
    show(rows, f"observed reference class, h={h} (common span, same episodes)")
    kj = names.index("KWEB")
    kx = obs[kj]
    live = ~np.isnan(obs)
    print(f"  KWEB excess {kx:+.3f}pp | rank {1 + int((obs[live] > kx).sum())} "
          f"of {int(live.sum())} live names | "
          f"{int((obs[live] > 0).sum())}/{int(live.sum())} names positive")

    # ---------------- NULL A: random anchors, min gap 21
    maxes = np.empty(B)
    kn = np.empty(B)
    L = len(span)
    for b in range(B):
        picked = []
        while len(picked) < n_ep:
            c = int(rng.integers(0, L))
            if all(abs(c - p) >= 21 for p in picked):
                picked.append(c)
        e, _ = excess(np.array(picked))
        maxes[b] = np.nanmax(e)
        kn[b] = e[kj]
    print(f"\n  NULL A random anchors (B={B}, {n_ep} anchors, min gap 21):")
    print(f"    P(MAX name excess >= KWEB's {kx:+.3f}pp) = {np.mean(maxes >= kx):.4f}")
    print(f"    null max distribution: median {np.median(maxes):+.3f} "
          f"p75 {np.percentile(maxes,75):+.3f} p90 {np.percentile(maxes,90):+.3f} "
          f"p95 {np.percentile(maxes,95):+.3f}")
    print(f"    P(KWEB ALONE >= {kx:+.3f}) = {np.nanmean(kn >= kx):.4f} "
          f"(single name, no search charge)")

    # ---------------- NULL B: circular shift of the real trigger set
    maxesB = np.empty(B)
    knB = np.empty(B)
    for b in range(B):
        k = int(rng.integers(1, L))
        e, _ = excess((base_pos + k) % L)
        maxesB[b] = np.nanmax(e)
        knB[b] = e[kj]
    print(f"\n  NULL B circular shift (B={B}):")
    print(f"    P(MAX name excess >= KWEB's {kx:+.3f}pp) = {np.mean(maxesB >= kx):.4f}")
    print(f"    null max distribution: median {np.median(maxesB):+.3f} "
          f"p95 {np.percentile(maxesB,95):+.3f}")
    print(f"    P(KWEB ALONE >= {kx:+.3f}) = {np.nanmean(knB >= kx):.4f}")
