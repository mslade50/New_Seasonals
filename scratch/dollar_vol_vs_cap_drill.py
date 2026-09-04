"""Drill-down on dollar_vol_vs_cap_events: sanity list, cap buckets,
year-clustered t-stats, tier overlap."""
import os

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ev = pd.read_parquet(os.path.join(_ROOT, "scratch", "dollar_vol_vs_cap_events.parquet"))
pd.set_option("display.width", 200)

print("== all 41 daily-tier episodes (sanity check) ==")
daily = ev[ev["tier"] == "daily"].sort_values("date")
cols = ["ticker", "date", "mktcap", "ratio_1d", "day_ret", "fwd_21d", "fwd_63d"]
d = daily[cols].copy()
d["mktcap"] = (d["mktcap"] / 1e6).round(0)
for c in ["ratio_1d", "day_ret", "fwd_21d", "fwd_63d"]:
    d[c] = d[c].round(2)
print(d.to_string(index=False))

print("\n== cap buckets (monthly tier, 63d excess) ==")
mon = ev[ev["tier"] == "monthly"].copy()
mon["bucket"] = pd.cut(mon["mktcap"], [0, 300e6, 2e9, 10e9, np.inf],
                       labels=["micro<300M", "small<2B", "mid<10B", "large>10B"])
for tier_df, label in [(mon, "monthly"), (ev[ev["tier"] == "weekly"].copy(), "weekly")]:
    if "bucket" not in tier_df:
        tier_df["bucket"] = pd.cut(tier_df["mktcap"], [0, 300e6, 2e9, 10e9, np.inf],
                                   labels=["micro<300M", "small<2B", "mid<10B", "large>10B"])
    g = tier_df.groupby("bucket", observed=True)["xs_63d"].agg(
        N="count", mean=lambda s: s.mean() * 100, med=lambda s: s.median() * 100,
        win=lambda s: (s > 0).mean() * 100)
    print(f"-- {label}")
    print(g.round(2))

print("\n== year-clustered t (mean of yearly mean xs, both tiers/horizons) ==")
for tier in ["weekly", "monthly"]:
    sub = ev[ev["tier"] == tier].copy()
    sub["year"] = sub["date"].dt.year
    for h in [21, 63]:
        yearly = sub.groupby("year")[f"xs_{h}d"].mean().dropna()
        t = yearly.mean() / (yearly.std() / np.sqrt(len(yearly)))
        pos = (yearly > 0).mean() * 100
        print(f"{tier} xs_{h}d: {len(yearly)} yrs, mean-of-yearly {yearly.mean()*100:.2f}%, "
              f"clustered t={t:.2f}, {pos:.0f}% of years positive")

print("\n== tier overlap: monthly episodes that were ALSO daily/weekly that day ==")
mon2 = ev[ev["tier"] == "monthly"]
print("monthly with ratio_1d>1 same day:", int((mon2["ratio_1d"] > 1).sum()),
      "| with ratio_5d>1:", int((mon2["ratio_5d"] > 1).sum()), "of", len(mon2))

print("\n== 'monthly-only' (5d ratio < 1 at trigger): the slow-burn cohort ==")
slow = mon2[mon2["ratio_5d"] < 1.0]
for h in [5, 10, 21, 63]:
    x = slow[f"xs_{h}d"].dropna()
    print(f"xs_{h}d: N={len(x)} mean {x.mean()*100:+.2f}% med {x.median()*100:+.2f}% "
          f"win {(x>0).mean()*100:.0f}% t={x.mean()/(x.std()/np.sqrt(len(x))):.2f}")

print("\n== biggest 63d winners/losers (weekly tier) ==")
wk = ev[ev["tier"] == "weekly"].dropna(subset=["fwd_63d"])
show = ["ticker", "date", "mktcap", "ratio_5d", "day_ret", "fwd_63d"]
print(wk.nlargest(8, "fwd_63d")[show].to_string(index=False))
print(wk.nsmallest(8, "fwd_63d")[show].to_string(index=False))
