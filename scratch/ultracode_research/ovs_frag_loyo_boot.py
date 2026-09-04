"""LOYO on the trade-weighted monthly block bootstrap of 21-44 vs frag<21
(and vs frag<3), plus drop-2026+2025 and P1-only LOYO."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:/Users/McKinley Slade/dev/New_Seasonals")
rng = np.random.default_rng(7)
NREP = 8000

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
ovs["yr"] = ovs["Signal Date"].dt.year
ovs["gap_atr"] = (ovs["T+1 Open"] - ovs["Signal Close"]) / ovs["ATR"]
ovs["path"] = np.where(ovs.gap_atr > 0.25, "P1", "P2")


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


print("LOYO, trade-weighted block bootstrap: 21-44 vs frag<21")
for drop in sorted(ovs.yr.unique()):
    d = ovs[ovs.yr != drop]
    obs, z, na, nb = boot_z(d, (d.frag >= 21) & (d.frag < 44), d.frag < 21)
    print(f"  drop {drop}: diff {obs:+.3f}R z={z:+.2f} (Na={na}, Nb={nb})")

print("\nLOYO: 21-44 vs frag<3")
for drop in sorted(ovs.yr.unique()):
    d = ovs[ovs.yr != drop]
    obs, z, na, nb = boot_z(d, (d.frag >= 21) & (d.frag < 44), d.frag < 3)
    print(f"  drop {drop}: diff {obs:+.3f}R z={z:+.2f} (Na={na}, Nb={nb})")

d = ovs[~ovs.yr.isin([2025, 2026])]
obs, z, na, nb = boot_z(d, (d.frag >= 21) & (d.frag < 44), d.frag < 21)
print(f"\ndrop BOTH 2025+2026: diff {obs:+.3f}R z={z:+.2f} (Na={na}, Nb={nb})")

print("\nLOYO, P1 only: 21-44 vs frag<3")
p1 = ovs[ovs.path == "P1"]
for drop in sorted(p1.yr.unique()):
    d = p1[p1.yr != drop]
    obs, z, na, nb = boot_z(d, (d.frag >= 21) & (d.frag < 44), d.frag < 3)
    print(f"  drop {drop}: diff {obs:+.3f}R z={z:+.2f} (Na={na}, Nb={nb})")

# sanity: same bootstrap applied to 55+ vs 21-44 (the U-shape right side)
obs, z, na, nb = boot_z(ovs, ovs.frag >= 55, (ovs.frag >= 21) & (ovs.frag < 44))
print(f"\n55+ vs 21-44: diff {obs:+.3f}R z={z:+.2f} (Na={na}, Nb={nb})")
d = ovs[ovs.yr != 2022]
obs, z, na, nb = boot_z(d, d.frag >= 55, (d.frag >= 21) & (d.frag < 44))
print(f"55+ vs 21-44 ex-2022: diff {obs:+.3f}R z={z:+.2f} (Na={na}, Nb={nb})")

# throttle replay stats trade-weighted for the report
mid = (ovs.frag >= 21) & (ovs.frag < 44)
print(f"\nmid-band totR {ovs[mid].R_Multiple.sum():+.1f} over {mid.sum()} trades "
      f"({mid.mean()*100:.0f}% of OVS)")
