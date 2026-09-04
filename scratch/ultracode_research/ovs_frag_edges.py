"""Band-edge sensitivity of the OVS mid-band dip (trade-weighted block bootstrap)."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:/Users/McKinley Slade/dev/New_Seasonals")
rng = np.random.default_rng(11)
NREP = 6000

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


def boot_z(df, mask_a, mask_b, n=NREP):
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
    return obs, obs / diffs.std(), int(ca.sum()), int(cb.sum())


print("edge sensitivity: [lo,hi) band vs frag<lo, trade-weighted block bootstrap")
for lo, hi in [(15, 40), (18, 42), (21, 44), (25, 44), (21, 50), (25, 50), (30, 50)]:
    m = (ovs.frag >= lo) & (ovs.frag < hi)
    b = ovs.frag < lo
    obs, z, na, nb = boot_z(ovs, m, b)
    a = ovs[m]
    print(f"  [{lo},{hi}): avgR {a.R_Multiple.mean():+.3f} (N={na})  diff {obs:+.3f}R z={z:+.2f}")

print("\ntrades per month by band (signal density):")
EDGES = [0, 3, 21, 44, 55, 100.001]
LABELS = ["0-3", "3-21", "21-44", "44-55", "55+"]
ovs["band"] = pd.cut(ovs.frag, bins=EDGES, labels=LABELS, include_lowest=True)
days = frag_ma[frag_ma.index >= start]
day_band = pd.cut(days, bins=EDGES, labels=LABELS, include_lowest=True)
tr = ovs.groupby("band", observed=False).size()
dd = day_band.value_counts()
out = pd.DataFrame({"trades": tr, "frag_days": dd, "tr_per_day": (tr / dd).round(3)})
print(out.to_string())
