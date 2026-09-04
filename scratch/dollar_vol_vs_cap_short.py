"""Short-side cut of the dollar-vol-vs-cap events: condition on trigger-day
character (up-spike vs down day), ratio magnitude, and cap bucket, looking
for any cohort with reliably NEGATIVE forward excess returns."""
import os

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ev = pd.read_parquet(os.path.join(_ROOT, "scratch", "dollar_vol_vs_cap_events.parquet"))
pd.set_option("display.width", 200)


def stats(sub: pd.DataFrame, label: str) -> None:
    if len(sub) < 5:
        print(f"{label}: N={len(sub)} (too few)")
        return
    parts = [f"{label}: N={len(sub)}"]
    for h in (5, 10, 21, 63):
        x = sub[f"xs_{h}d"].dropna()
        if len(x) < 5:
            continue
        t = x.mean() / (x.std() / np.sqrt(len(x)))
        parts.append(f"{h}d {x.mean()*100:+.1f}/{x.median()*100:+.1f}% "
                     f"(win {(x>0).mean()*100:.0f}%, t {t:+.1f})")
    print("  ".join(parts))


for tier in ("daily", "weekly", "monthly"):
    sub = ev[ev["tier"] == tier].copy()
    print(f"\n===== {tier} (N={len(sub)}) — xs mean/median per horizon =====")
    stats(sub[sub["day_ret"] > 0.10], "trigger day +10%+ (blowoff)")
    stats(sub[sub["day_ret"] > 0.25], "trigger day +25%+")
    stats(sub[(sub["day_ret"] > 0) & (sub["day_ret"] <= 0.10)], "mild up day")
    stats(sub[sub["day_ret"] < 0], "down trigger day")
    stats(sub[sub["day_ret"] < -0.10], "trigger day -10%- (capitulation)")

print("\n===== worst-case short outcomes: fwd_63d tail in the +25% blowoff cohort =====")
blow = ev[(ev["day_ret"] > 0.25)].dropna(subset=["fwd_63d"])
print(f"N={len(blow)}, fwd_63d distribution (a SHORT loses these):")
print((blow["fwd_63d"].describe(percentiles=[.1, .25, .5, .75, .9, .95]) * 100).round(1))
big = blow.nlargest(6, "fwd_63d")[["ticker", "date", "tier", "day_ret", "fwd_63d"]]
print(big.to_string(index=False))

print("\n===== 21d-horizon short PnL if shorting every +10% blowoff trigger =====")
for tier in ("weekly", "monthly"):
    b = ev[(ev["tier"] == tier) & (ev["day_ret"] > 0.10)].dropna(subset=["xs_21d"])
    if len(b) < 5:
        continue
    pnl = -b["xs_21d"]
    yearly = b.assign(y=b["date"].dt.year).groupby("y")["xs_21d"].mean().mul(-1)
    t = yearly.mean() / (yearly.std() / np.sqrt(len(yearly))) if len(yearly) > 2 else np.nan
    print(f"{tier}: N={len(b)} shorts, mean {pnl.mean()*100:+.1f}%, med {pnl.median()*100:+.1f}%, "
          f"win {(pnl>0).mean()*100:.0f}%, worst single {pnl.min()*100:+.0f}%, "
          f"clustered t {t:+.2f} over {len(yearly)} yrs")
