"""C2 round 2: the only horizon with a pulse in round 1 was h=2 (+0.420% PIT
pair vs +0.074% all days). Era split, concentration, definition neighbours
(residual 2.0/2.5/3.0pp and raw GDX-GLD gap) and 2018+ profile at h=1..5."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kC_c2_gdx_overshoot import build, pair_ret, eps  # noqa: E402

if __name__ == "__main__":
    px, rg, rl, beta, res = build()
    idx = px.index
    base = {h: pair_ret(px, beta, h) for h in (1, 2, 3, 5)}
    rows = []
    for h, pr in base.items():
        valid = pr.dropna()
        for lbl, m in [("resid>=2.0", res >= 0.020), ("resid>=2.5", res >= 0.025),
                       ("resid>=3.0", res >= 0.030), ("rawgap>=2.5", (rg - rl) >= 0.025),
                       ("rawgap>=3.0", (rg - rl) >= 0.030)]:
            e, v = eps(pr, m, h, idx)
            post = pd.DatetimeIndex(e) >= "2018-01-01"
            r = summarize(v, f"h={h} {lbl}")
            r["excess_pp"] = r["mean_pct"] - 100 * valid.mean()
            r["pre18_pct"] = 100 * np.nanmean(v[~post])
            r["post18_pct"] = 100 * np.nanmean(v[post]) if post.any() else np.nan
            r["n_post18"] = int(post.sum())
            rows.append(r)
    show(rows, "C2 neighbours x horizon (PIT-beta pair, episodes; excess vs all days)")
    pr = base[2]
    e, v = eps(pr, res >= 0.025, 2, idx)
    print("\nh=2 child:", cluster_note(e, v))
    yrs = pd.Series(v, index=pd.DatetimeIndex(e).year).groupby(level=0).agg(["sum", "count"])
    print((100 * yrs["sum"]).round(2).to_dict())
    ex = ~pd.DatetimeIndex(e).year.isin([2008, 2020])
    show([summarize(v[ex], "h=2 ex-2008/2020"),
          summarize(v[pd.DatetimeIndex(e) >= "2021-01-01"], "h=2 2021+")], "h=2 robustness")
    w = int((v > 0).sum())
    print(f"h=2 record {w}-{len(v)-w} sign p {sign_test(w, len(v)):.4f}")
    e10 = declusters(e, 10, idx)
    v10 = pr.loc[e10].values
    w = int((v10 > 0).sum())
    print(f"h=2 decluster 10td: N={len(e10)} mean {100*np.nanmean(v10):+.3f}% "
          f"record {w}-{len(v10)-w}")
    # the 3.0pp rung (today's 2.84 sits BELOW it): crisis-free and recent
    rows = []
    for h in (1, 2, 3):
        e3, v3 = eps(base[h], res >= 0.030, h, idx)
        yy = pd.DatetimeIndex(e3).year
        rows += [summarize(v3[~yy.isin([2008, 2020])], f"resid>=3.0 h={h} ex-2008/2020"),
                 summarize(v3[yy >= 2021], f"resid>=3.0 h={h} 2021+")]
    show(rows, "3.0pp rung robustness")
    b = float(beta.iloc[-1])
    print(f"\ncost: per-leg ~2.5 bps x (1+{b:.2f}) = {2.5*(1+b):.1f} bps round trip; "
          f"5x bar = {5*2.5*(1+b):.0f} bps")
