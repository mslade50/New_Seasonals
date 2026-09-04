"""Monthly block bootstrap of the trade-weighted avgR contrasts (vectorized).
Resample calendar months with replacement; recompute trade-weighted avgR diff
from precomputed per-month (sum, count) arrays for each group."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:/Users/McKinley Slade/dev/New_Seasonals")
rng = np.random.default_rng(42)
NREP = 10000

trades = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
frag_ma = frag["63d"].dropna().rolling(10, min_periods=1).mean()
frag_ma.index = pd.to_datetime(frag_ma.index).normalize()
trades["Signal Date"] = pd.to_datetime(trades["Signal Date"]).dt.normalize()
start = frag_ma.index.min() + pd.Timedelta(days=20)
ovs = trades[trades.Strategy.str.contains("Overbot Vol", na=False)].copy()
ovs = ovs[ovs["Signal Date"] >= start].sort_values("Signal Date").reset_index(drop=True)
f = frag_ma.rename("frag").reset_index(); f.columns = ["Date", "frag"]
ovs["frag"] = pd.merge_asof(ovs[["Signal Date"]], f, left_on="Signal Date",
                            right_on="Date", tolerance=pd.Timedelta(days=5))["frag"].values
ovs = ovs.dropna(subset=["frag", "R_Multiple"]).copy()
ovs["ym"] = ovs["Signal Date"].dt.to_period("M")
ovs["gap_atr"] = (ovs["T+1 Open"] - ovs["Signal Close"]) / ovs["ATR"]
ovs["path"] = np.where(ovs.gap_atr > 0.25, "P1", "P2")
ovs["midterm"] = (ovs["Signal Date"].dt.year % 4 == 2)


def block_boot(df, mask_a, mask_b, la, lb, n=NREP):
    months = df.ym.unique()
    m_idx = {m: i for i, m in enumerate(months)}
    k = len(months)

    def per_month(mask):
        s = np.zeros(k); c = np.zeros(k)
        g = df[mask].groupby("ym")["R_Multiple"].agg(["sum", "size"])
        for m, row in g.iterrows():
            s[m_idx[m]] = row["sum"]; c[m_idx[m]] = row["size"]
        return s, c

    sa, ca = per_month(mask_a)
    sb, cb = per_month(mask_b)
    obs = sa.sum() / ca.sum() - sb.sum() / cb.sum()
    draws = rng.integers(0, k, size=(n, k))
    SA, CA = sa[draws].sum(axis=1), ca[draws].sum(axis=1)
    SB, CB = sb[draws].sum(axis=1), cb[draws].sum(axis=1)
    ok = (CA > 0) & (CB > 0)
    diffs = SA[ok] / CA[ok] - SB[ok] / CB[ok]
    se = diffs.std()
    z = obs / se
    p2 = 2 * min((diffs >= 0).mean(), (diffs <= 0).mean())
    print(f"{la} vs {lb}: obs diff {obs:+.3f}R, boot SE {se:.3f}, z={z:+.2f}, "
          f"boot two-sided p={p2:.3f}  (Na={int(ca.sum())}, Nb={int(cb.sum())})")


mid = (ovs.frag >= 21) & (ovs.frag < 44)
low = ovs.frag < 21
calm = ovs.frag < 3
hi = ovs.frag >= 44

print(f"monthly block bootstrap, trade-weighted avgR contrasts ({NREP} reps):")
block_boot(ovs, mid, low, "21-44", "frag<21")
block_boot(ovs, mid, calm, "21-44", "frag<3")
block_boot(ovs, mid, hi, "21-44", "frag>=44")
block_boot(ovs, calm, ~calm, "0-3", "rest")

nm = ovs[~ovs.midterm].copy()
print("\nnon-midterm years only:")
block_boot(nm, (nm.frag >= 21) & (nm.frag < 44), nm.frag < 21, "NM 21-44", "NM frag<21")
block_boot(nm, (nm.frag >= 21) & (nm.frag < 44), nm.frag < 3, "NM 21-44", "NM frag<3")

p1 = ovs[ovs.path == "P1"].copy()
print("\nP1 only:")
block_boot(p1, (p1.frag >= 21) & (p1.frag < 44), p1.frag < 3, "P1 21-44", "P1 frag<3")
p2df = ovs[ovs.path == "P2"].copy()
print("\nP2 only:")
block_boot(p2df, (p2df.frag >= 21) & (p2df.frag < 44), p2df.frag < 3, "P2 21-44", "P2 frag<3")
