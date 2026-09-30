"""kB E1 round 1: short EEM against ex-ante beta-SPY after a dollar-breakout
session (DX-Y.NYB >= +0.4% on the day AND closing within 0.5% of its 252 high),
h=1..5. Pre-specified SHORT residual (continuation). Job: kill it.

LEADS with the reference-class test that killed the washout form: the identical
rule on every clean EM/intl vehicle in the cache, each against its own ex-ante
beta-SPY, plus a random-date permutation of the max-name excess.
Beta: rolling 252-session OLS of daily returns through the SIGNAL close (ex ante
for the t+1 MOC entry).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

INTL = ["EEM", "EFA", "FXI", "EWZ", "EWJ", "EWY", "EWT", "EWW", "KWEB", "INDA"]
DX = "DX-Y.NYB"
raw = load_prices(INTL + ["SPY", DX])
cl = {t: raw[t]["Close"].dropna() for t in raw}

dx = cl[DX]
dx1 = dx.pct_change()
dx_off = dx / dx.rolling(252).max() - 1.0
print(f"live DX {dx.iloc[-1]:.2f} 1d {100*dx1.iloc[-1]:+.2f}% off-high {100*dx_off.iloc[-1]:+.3f}% "
      "(literal 0.5% gate misses by 0.2 bp; threshold -0.00505 used)")

spy = cl["SPY"]


def resid_frame(t: str) -> pd.DataFrame:
    """Aligned on the vehicle's and SPY's common sessions; DX state reindexed."""
    px = pd.DataFrame({t: cl[t], "SPY": spy}).dropna()
    r = px.pct_change()
    cov = r[t].rolling(252).cov(r["SPY"])
    var = r["SPY"].rolling(252).var()
    px["beta"] = cov / var
    return px


def short_resid(px: pd.DataFrame, t: str, h: int, lag: int = 1) -> pd.Series:
    # short vehicle, long beta_t x SPY, beta fixed at the signal close
    return -fwd_lag(px[t], h, lag) + px["beta"] * fwd_lag(px["SPY"], h, lag)


def trig(px: pd.DataFrame, up: float = 0.004, near: float = 0.00505) -> pd.Series:
    m = (dx1 >= up) & (dx_off >= -near)
    return m.reindex(px.index).fillna(False).astype(bool)


def ep_stats(ret: pd.Series, mask: pd.Series, h: int):
    valid = ret.dropna().index
    s = mask.index[mask.values].intersection(valid)
    e = declusters(s, h, valid)
    return e, ret.loc[e].values, ret.loc[valid]


# ---------------- (A) reference class, h=1,3,5 ----------------
for h in (1, 3, 5):
    rows = []
    for t in INTL:
        px = resid_frame(t)
        ret = short_resid(px, t, h)
        e, v, base = ep_stats(ret, trig(px), h)
        d = summarize(v, t)
        if d["n"]:
            d["ctl_all"] = round(100 * base.mean(), 3)
            d["edge"] = round(d["mean_pct"] - d["ctl_all"], 3)
            w = int((v > 0).sum())
            d["rec"] = f"{w}-{len(v)-w}"
            d["sign_p"] = round(sign_test(w, len(v)), 4)
            d["beta_now"] = round(px["beta"].iloc[-1], 2)
            d["start"] = str(px.index[252].date())
        rows.append(d)
    show(sorted(rows, key=lambda r: -r.get("edge", -99)),
         f"REFERENCE CLASS h={h}: short vehicle vs ex-ante beta-SPY after DX breakout session")

# permutation max-of-N on a common sample (two sets: full-10 since 2013, long-history 8 since 2005)
for label, names in (("all 10 (2013+)", INTL), ("8 long-history (ex KWEB/INDA)", INTL[:8])):
    for h in (1, 5):
        frames = {t: resid_frame(t) for t in names}
        rets = pd.DataFrame({t: short_resid(frames[t], t, h) for t in names}).dropna()
        valid = rets.index
        m = trig(rets)
        s = valid[m.values]
        e = declusters(s, h, valid)
        obs = {t: rets[t].loc[e].mean() - rets[t].mean() for t in names}
        rng = np.random.default_rng(42)
        maxes = []
        for _ in range(2000):
            pick = valid[rng.choice(len(valid), size=len(e), replace=False)]
            maxes.append(max(rets[t].loc[pick].mean() - rets[t].mean() for t in names))
        maxes = np.asarray(maxes)
        rank_eem = 1 + sum(1 for t in names if obs[t] > obs["EEM"])
        print(f"\nPERMUTATION {label} h={h}: {len(e)} episodes {valid[0].date()}..{valid[-1].date()}")
        print("  excess (pp):", {t: round(100 * v, 3) for t, v in sorted(obs.items(), key=lambda x: -x[1])})
        print(f"  EEM ranks {rank_eem} of {len(names)}; P(max-of-{len(names)} >= EEM {100*obs['EEM']:+.3f}) = "
              f"{(maxes >= obs['EEM']).mean():.3f}; P(max >= family max {100*max(obs.values()):+.3f}) = "
              f"{(maxes >= max(obs.values())).mean():.3f}")

# ---------------- (B) EEM itself: full battery on fixed-beta legs + ex-ante residual detail ----------------
px = resid_frame("EEM")
print(f"\nEEM ex-ante beta now {px['beta'].iloc[-1]:.3f}; median beta {px['beta'].median():.3f}")
bmed = float(px["beta"].median())
cell = trig(px)
variants = {
    "DX >= +0.3% & near": trig(px, 0.003),
    "DX >= +0.6% & near": trig(px, 0.006),
    "DX >= +0.4% & within 1.0%": trig(px, 0.004, 0.010),
    "GATE-OFF: DX >= +0.4% any level": trig(px, 0.004, 9.0),
    "GATE-OFF: DX near high, any day": trig(px, -9.0, 0.00505),
    "COMPLEMENT: DX >= +0.4% NOT near": trig(px, 0.004, 9.0) & ~trig(px, -9.0, 0.00505),
}
for h in (1, 3, 5):
    battery(px[["EEM", "SPY"]], cell, [("EEM", -1.0), ("SPY", bmed)], h,
            f"E1 short EEM vs {bmed:.2f}x SPY (fixed median beta)", 2.5,
            variants=variants, event_kinds=("nfp",))

print("\n=== EX-ANTE BETA residual (the pitched construction), episodes ===")
for h in (1, 3, 5):
    ret = short_resid(px, "EEM", h)
    e, v, base = ep_stats(ret, cell, h)
    naked = -fwd_lag(px["EEM"], h, 1)
    w = int((v > 0).sum())
    rows = [summarize(v, f"h={h} short resid episodes"),
            summarize(base.values, f"h={h} short resid all days"),
            summarize(naked.loc[e].values, f"h={h} NAKED short EEM episodes")]
    show(rows)
    print(f"  record {w}-{len(v)-w} sign p {sign_test(w, len(v)):.4f}  bootstrap P(mean<=0) {bootstrap_p_le0(v):.3f}")
    show(era_split(e, v), f"  era split h={h}")
    mid = np.asarray(e.year % 4 == 2)
    show([summarize(v[mid], "midterm"), summarize(v[~mid], "non-midterm")], f"  midterm split h={h}")
    print("  " + cluster_note(e, v))
