"""What would the pending throttle rec (1.0x through 50, taper to 0.5x by 60,
no boost) have done in 2026? Plus LOYO robustness of the pre/post-break split."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

trades = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
frag_ma = frag["63d"].dropna().rolling(10, min_periods=1).mean()
frag_ma.index = pd.to_datetime(frag_ma.index).normalize()

mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet")
spy = mp[mp.ticker == "SPY"].set_index("date")["Close"].sort_index()
spy.index = pd.to_datetime(spy.index).normalize()
dd = (spy / spy.rolling(252).max() - 1) * 100

trades["Signal Date"] = pd.to_datetime(trades["Signal Date"]).dt.normalize()
t = trades[trades["Signal Date"] >= frag_ma.index.min() + pd.Timedelta(days=20)].copy().sort_values("Signal Date")
t["frag"] = pd.merge_asof(t[["Signal Date"]], frag_ma.rename("frag").reset_index(),
                          left_on="Signal Date", right_on="Date",
                          tolerance=pd.Timedelta(days=5))["frag"].values
t["spy_dd"] = dd.reindex(t["Signal Date"], method="ffill").values
t = t.dropna(subset=["frag", "R_Multiple"])
is_ovs = t["Strategy"].str.contains("Overbot Vol|OVS", case=False, na=False)
nb = t[~is_ovs].copy()
nb["yr"] = nb["Signal Date"].dt.year


def pending_mult(f):
    if f <= 50:
        return 1.0
    if f >= 60:
        return 0.5
    return 1.0 - (f - 50) / 10 * 0.5


nb["mult"] = nb.frag.map(pending_mult)
nb["R_adj"] = nb.R_Multiple * nb.mult

for yr in sorted(nb.yr.unique()):
    g = nb[nb.yr == yr]
    cut = g[g.mult < 1.0]
    print(f"{yr}: totR {g.R_Multiple.sum():+7.1f} -> {g.R_adj.sum():+7.1f} "
          f"(delta {g.R_adj.sum()-g.R_Multiple.sum():+6.1f}R, {len(cut)} trades throttled)")
tot = nb
print(f"ALL : totR {tot.R_Multiple.sum():+7.1f} -> {tot.R_adj.sum():+7.1f} "
      f"(delta {tot.R_adj.sum()-tot.R_Multiple.sum():+6.1f}R)")

# phase-gated variant: taper applies ONLY while SPY within 2% of 52w high
nb["mult_gated"] = np.where(nb.spy_dd > -2, nb.mult, 1.0)
nb["R_gated"] = nb.R_Multiple * nb.mult_gated
print("\nphase-gated throttle (taper only when SPY dd > -2%):")
for yr in sorted(nb.yr.unique()):
    g = nb[nb.yr == yr]
    print(f"{yr}: plain {g.R_adj.sum()-g.R_Multiple.sum():+6.1f}R  gated {g.R_gated.sum()-g.R_Multiple.sum():+6.1f}R")
print(f"ALL : plain {nb.R_adj.sum()-nb.R_Multiple.sum():+6.1f}R  gated {nb.R_gated.sum()-nb.R_Multiple.sum():+6.1f}R")

# LOYO: pre-break vs post-break >=50 split, drop each year
hi = nb[nb.frag >= 50].copy()
hi["phase"] = np.where(hi.spy_dd > -2, "pre", np.where(hi.spy_dd < -3, "post", "mid"))
print("\nLOYO of hi-frag phase split (avgR pre / post, all years incl 2026):")
years = sorted(hi.yr.unique())
full_pre = hi[hi.phase == "pre"].R_Multiple.mean()
full_post = hi[hi.phase == "post"].R_Multiple.mean()
print(f"  full: pre {full_pre:+.3f} (N={len(hi[hi.phase=='pre'])}) post {full_post:+.3f} (N={len(hi[hi.phase=='post'])})")
for y in years:
    sub = hi[hi.yr != y]
    p1 = sub[sub.phase == "pre"]; p2 = sub[sub.phase == "post"]
    if len(p1) >= 5 and len(p2) >= 5:
        print(f"  drop {y}: pre {p1.R_Multiple.mean():+.3f} (N={len(p1)}) post {p2.R_Multiple.mean():+.3f} (N={len(p2)}) "
              f"gap {p2.R_Multiple.mean()-p1.R_Multiple.mean():+.3f}")

# what the pending rec would cut per phase, historically
hist = nb[nb.yr < 2026]
th = hist[hist.mult < 1.0]
print(f"\nhistorical trades the pending taper touches: N={len(th)}, avgR {th.R_Multiple.mean():+.3f}")
for ph, cond in [("pre-break", th.spy_dd > -2), ("post-break", th.spy_dd < -3),
                 ("mid", (th.spy_dd <= -2) & (th.spy_dd >= -3))]:
    s = th[cond]
    print(f"  {ph}: N={len(s)}, avgR {s.R_Multiple.mean():+.3f}, totR {s.R_Multiple.sum():+.1f}")
