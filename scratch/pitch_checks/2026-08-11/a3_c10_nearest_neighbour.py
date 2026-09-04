"""C10 kill: nearest-neighbour tapes to today's joint state, and what followed.

Today's joint state (from the 2026-08-10 bar): SPY at a 52w high with z10 1.64
and 5d rank 85, TLT at a 52w low, a violent precious-metals thrust (GDX 5d rank
99.6), VIX 15.5 in the bottom half of its 63d range, CPI tomorrow.

The check has to answer THREE questions, and any one of them can kill it:
  Q1 HONEST DISTANCE. How far away is the nearest historical tape, in the
     joint standardised metric, compared with how far a TYPICAL day's nearest
     neighbour is? If today's neighbourhood is an outlier in the distance
     distribution, the verdict is "no comparable state exists".
  Q2 HONEST NEIGHBOURS. Do the k neighbours cluster in one year / one episode?
  Q3 HONEST EDGE. Forward SPY at h=1,3,5,10 (lag=1) vs unconditional, with a
     sign test, AND stability of the whole answer to k and to the feature set.
     A nearest-neighbour result with no mechanism is a data artifact whatever
     its N.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, fwd_lag, declusters, summarize, sign_test, bootstrap_p_le0,
    pct_rank, zscore,
)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)

TKRS = ["SPY", "TLT", "GLD", "GDX", "^VIX"]
px = close_panel(TKRS)
px = px.dropna(subset=["SPY"])
all_dates = px.index

spy, tlt, gld, gdx, vix = (px["SPY"], px["TLT"], px["GLD"], px["GDX"], px["^VIX"])

FEAT = pd.DataFrame({
    "spy_r5_rank": pct_rank(spy, 5),
    "spy_r21_rank": pct_rank(spy, 21),
    "spy_dist52wh": 100 * (spy / spy.rolling(252).max() - 1.0),
    "tlt_dist52wl": 100 * (tlt / tlt.rolling(252).min() - 1.0),
    "gld_r5_rank": pct_rank(gld, 5),
    "vix_pct63": vix.rolling(63).rank(pct=True) * 100.0,
})
FEAT_ALT = FEAT.copy()
FEAT_ALT["gld_r5_rank"] = pct_rank(gdx, 5)      # GDX variant (shorter history)
FEAT_ALT = FEAT_ALT.rename(columns={"gld_r5_rank": "gdx_r5_rank"})

TODAY = all_dates[-1]
print(f"today = {TODAY.date()} (freshest bar). raw feature vector:")
print(FEAT.loc[TODAY].round(2).to_string())
print(f"  (cross-check vs the brief: SPY 5d rank 85.3, dist52wh -0.03%, "
      f"z10 {zscore(spy).loc[TODAY]:.2f}, VIX {vix.loc[TODAY]:.2f})")


def neighbours(F, k=20, exclude_td=126, gap=21, label=""):
    F = F.dropna()
    if TODAY not in F.index:
        raise SystemExit("today has no complete feature vector")
    mu, sd = F.mean(), F.std(ddof=0)
    Z = (F - mu) / sd
    z_today = Z.loc[TODAY]
    d = np.sqrt(((Z - z_today) ** 2).sum(axis=1))
    usable = F.index[:-exclude_td] if exclude_td else F.index
    d_hist = d.reindex(usable).dropna().sort_values()
    # decluster: keep the closest of each cluster
    keep, taken = [], []
    posn = pd.Series(range(len(all_dates)), index=all_dates)
    for dt in d_hist.index:
        p = posn.get(dt)
        if p is None:
            continue
        if all(abs(p - q) >= gap for q in taken):
            keep.append(dt)
            taken.append(p)
        if len(keep) >= k:
            break
    return pd.DatetimeIndex(keep), d, d_hist, Z, z_today


print("\n" + "=" * 110)
print("Q1  HONEST DISTANCE: is there any comparable tape at all?")
print("=" * 110)
K = 20
nn, dist_all, d_hist, Z, z_today = neighbours(FEAT, k=K)
print(f"feature space = {list(FEAT.columns)}  ({FEAT.dropna().shape[0]} usable "
      f"sessions, {FEAT.dropna().index[0].date()} .. {FEAT.dropna().index[-1].date()})")
print(f"nearest historical session: {d_hist.index[0].date()} at distance "
      f"{d_hist.iloc[0]:.2f} sd-units (6 dims, so the null 'random pair' "
      f"distance is ~sqrt(2*6) = {np.sqrt(12):.2f})")
print(f"the {K} kept neighbours span distances "
      f"{dist_all.reindex(nn).min():.2f} .. {dist_all.reindex(nn).max():.2f}")

# BASELINE: for every day, how far is ITS own nearest neighbour?
Zc = Z.dropna()
arr = Zc.values
rng = np.random.default_rng(42)
samp = rng.choice(len(arr), size=min(600, len(arr)), replace=False)
nn_dists = []
for i in samp:
    dd = np.sqrt(((arr - arr[i]) ** 2).sum(axis=1))
    dd[i] = np.inf
    nn_dists.append(np.sort(dd)[:K])
nn_dists = np.array(nn_dists)
base_nn1 = nn_dists[:, 0]
base_nnk = nn_dists[:, K - 1]
print(f"\nBASELINE over 600 random sessions (same metric, no decluster):")
print(f"  a typical day's NEAREST neighbour sits at {np.median(base_nn1):.2f} "
      f"sd-units (10th pct {np.percentile(base_nn1, 10):.2f}, "
      f"90th {np.percentile(base_nn1, 90):.2f})")
print(f"  a typical day's {K}th neighbour sits at {np.median(base_nnk):.2f} "
      f"(90th pct {np.percentile(base_nnk, 90):.2f})")
raw_sorted = dist_all.reindex(Zc.index).sort_values()
print(f"  TODAY's nearest (no decluster) = {raw_sorted.iloc[0]:.2f}, "
      f"{K}th = {raw_sorted.iloc[K-1]:.2f}")
pct1 = 100 * (base_nn1 < raw_sorted.iloc[0]).mean()
pctk = 100 * (base_nnk < raw_sorted.iloc[K-1]).mean()
print(f"  -> today's nearest neighbour is FARTHER than {pct1:.0f}% of days' "
      f"nearest neighbours; its {K}th is farther than {pctk:.0f}% of days'.")
print(f"  per-feature gap of the single closest tape ({d_hist.index[0].date()}):")
print((Z.loc[d_hist.index[0]] - z_today).round(2).to_string())

print("\n" + "=" * 110)
print("Q2  HONEST NEIGHBOURS: where do they live?")
print("=" * 110)
tbl = pd.DataFrame({
    "date": [d.date() for d in nn],
    "dist_sd": dist_all.reindex(nn).round(2).values,
})
for c in FEAT.columns:
    tbl[c] = FEAT.reindex(nn)[c].round(1).values
print(tbl.to_string(index=False))
yrs = pd.Series([d.year for d in nn])
print(f"\n  year histogram of the {len(nn)} neighbours: "
      f"{yrs.value_counts().sort_index().to_dict()}")
top = yrs.value_counts()
print(f"  most-represented year {top.index[0]} holds {top.iloc[0]}/{len(nn)} "
      f"= {100*top.iloc[0]/len(nn):.0f}% of the neighbourhood")

print("\n" + "=" * 110)
print("Q3  HONEST EDGE: forward SPY from the neighbours vs unconditional")
print("=" * 110)
rows = []
for h in (1, 3, 5, 10):
    f = fwd_lag(spy, h, 1)
    v = f.reindex(nn).dropna()
    base = f.dropna()
    st = summarize(v.values)
    wins = int((v.values > 0).sum())
    rows.append(dict(h=h, N=st["n"], mean=round(st["mean_pct"], 3),
                     median=round(st["median_pct"], 3),
                     uncond=round(100 * base.mean(), 3),
                     excess=round(st["mean_pct"] - 100 * base.mean(), 3),
                     hit=round(st["hit"], 1), t=round(st["t"], 2),
                     signp=round(sign_test(wins, st["n"]), 4),
                     worst=round(st["worst_pct"], 2), best=round(st["best_pct"], 2),
                     bootP_le0=round(bootstrap_p_le0(v.values), 3)))
print(pd.DataFrame(rows).to_string(index=False))

print("\n  STABILITY to k (the one free parameter nobody pre-registers):")
srows = []
for k in (5, 10, 20, 30, 50):
    nk, dk, _, _, _ = neighbours(FEAT, k=k)
    for h in (1, 5):
        f = fwd_lag(spy, h, 1)
        v = f.reindex(nk).dropna()
        st = summarize(v.values)
        if not st["n"]:
            continue
        wins = int((v.values > 0).sum())
        srows.append(dict(k=k, h=h, N=st["n"], mean=round(st["mean_pct"], 3),
                          hit=round(st["hit"], 1),
                          signp=round(sign_test(wins, st["n"]), 4),
                          maxdist=round(dk.reindex(nk).max(), 2)))
print(pd.DataFrame(srows).to_string(index=False))

print("\n  STABILITY to the feature set (swap GLD 5d rank for GDX 5d rank):")
try:
    nn2, d2, _, _, _ = neighbours(FEAT_ALT, k=K)
    print(f"    overlap with the GLD neighbourhood: "
          f"{len(set(nn2) & set(nn))}/{K} dates")
    print(f"    GDX-variant neighbour years: "
          f"{pd.Series([d.year for d in nn2]).value_counts().sort_index().to_dict()}")
    for h in (1, 5):
        f = fwd_lag(spy, h, 1)
        v = f.reindex(nn2).dropna()
        st = summarize(v.values)
        wins = int((v.values > 0).sum())
        print(f"    h={h}: N={st['n']} mean={st['mean_pct']:+.3f}% "
              f"hit={st['hit']:.1f}% sign p={sign_test(wins, st['n']):.4f}")
except SystemExit as e:
    print("   ", e)

print("\n  DROP-THE-DOMINANT-YEAR: same cell with the most-represented year out")
drop_yr = int(top.index[0])
nn3 = pd.DatetimeIndex([d for d in nn if d.year != drop_yr])
for h in (1, 5):
    f = fwd_lag(spy, h, 1)
    v = f.reindex(nn3).dropna()
    st = summarize(v.values)
    if not st["n"]:
        continue
    wins = int((v.values > 0).sum())
    print(f"    ex-{drop_yr} h={h}: N={st['n']} mean={st['mean_pct']:+.3f}% "
          f"hit={st['hit']:.1f}% sign p={sign_test(wins, st['n']):.4f}")

print("\n" + "=" * 110)
print("MECHANISM: who is on the other side of a nearest-neighbour trade?")
print("=" * 110)
print("""  There is no answer to write. The feature vector was chosen this morning
  from the loudest things on the tape; the k that produced the headline was
  chosen after seeing the table; and the 'state' being matched is a
  six-dimensional coincidence rather than a forced seller, an informed
  disagreement or a neglected asset. Whatever the table below says, this is a
  descriptive statistic and not a trade.""")
