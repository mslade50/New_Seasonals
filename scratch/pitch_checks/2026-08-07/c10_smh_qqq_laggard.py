"""C10 "Laggard semis turning": long SMH / short QQQ equal dollar, 10 td.
Trigger: SMH rank63 <= 10 AND SMH rank5 >= 80.

Controls: (1) the pair's own unconditional 10td drift over the same window,
(2) the deep-laggard-only cell (rank63<=10, no snapback gate) to isolate what
the rank5 gate actually adds, (3) all-days baseline over full history.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import *  # noqa

import numpy as np
import pandas as pd

px = load_prices(["SMH", "QQQ"])
smh = px["SMH"]["Close"].dropna()
qqq = px["QQQ"]["Close"].dropna()
cal = smh.index.intersection(qqq.index)
smh, qqq = smh.reindex(cal), qqq.reindex(cal)
pos = pd.Series(range(len(cal)), index=cal)
ev = load_events()
cpi_d = set(ev.loc[ev.event == "cpi", "date"])

r63, r5, r21 = pct_rank(smh, 63), pct_rank(smh, 5), pct_rank(smh, 21)


def pair(h):
    """Equal-dollar long SMH / short QQQ, held h td, no rebalance."""
    return fwd_ret(smh, h) - fwd_ret(qqq, h)


print(f"common calendar {cal[0].date()} .. {cal[-1].date()}  n={len(cal)}")
print(f"pair unconditional drift, full history: "
      f"10td mean={100*pair(10).mean():+.3f}%  5td={100*pair(5).mean():+.3f}%  "
      f"21td={100*pair(21).mean():+.3f}%")

trig = cal[(r63 <= 10) & (r5 >= 80)]
trig = pd.DatetimeIndex([d for d in trig if not np.isnan(pair(10).get(d, np.nan))])
lag_only = cal[(r63 <= 10)]
print(f"\ntrigger days (rank63<=10 & rank5>=80): {len(trig)}  "
      f"first={trig[0].date() if len(trig) else 'na'} "
      f"last={trig[-1].date() if len(trig) else 'na'}")
print(f"laggard-only days (rank63<=10): {len(lag_only)}")

# --- 1) does it exist? -----------------------------------------------------
rows = []
for h in [5, 10, 21]:
    p = pair(h)
    v = p.reindex(trig).to_numpy()
    s = summarize(v, f"TRIGGER h={h}")
    s["ctl_uncond_pct"] = 100 * p.mean()
    s["ctl_laggard_only_pct"] = 100 * np.nanmean(p.reindex(lag_only).to_numpy())
    s["ctl_window_pct"] = 100 * p.loc[trig[0]:trig[-1]].mean() if len(trig) else np.nan
    rows.append(s)
show(rows, "1) conditional pair return vs controls")

# legs, so we can see which side does the work
for h in [10]:
    show([summarize(fwd_ret(smh, h).reindex(trig).to_numpy(), f"SMH leg h={h}"),
          summarize(fwd_ret(qqq, h).reindex(trig).to_numpy(), f"QQQ leg h={h}"),
          summarize(fwd_ret(smh, h).to_numpy(), "SMH all-days"),
          summarize(fwd_ret(qqq, h).to_numpy(), "QQQ all-days")], "leg decomposition")

# --- 2) era + episode concentration ---------------------------------------
p10 = pair(10)
v10 = p10.reindex(trig).to_numpy()
show(era_split(trig, v10), "2) era split (h=10)")
print("\n  trigger dates by year:")
yr = pd.Series(1, index=trig).groupby(trig.year).sum()
print("   ", dict(yr))
top = pd.Series(v10, index=trig).sort_values()
print(f"  worst 3: {[(str(d.date()), round(100*x,2)) for d, x in top.head(3).items()]}")
print(f"  best  3: {[(str(d.date()), round(100*x,2)) for d, x in top.tail(3).items()]}")

# --- 3) decluster ----------------------------------------------------------
print("\n=== 3) declustering (min gap = 10 td) ===")
drows = []
for h in [5, 10, 21]:
    dc = declusters(trig, h, cal)
    v = pair(h).reindex(dc).to_numpy()
    s = summarize(v, f"episodes h={h}")
    s["p_le0_boot"] = bootstrap_p_le0(v)
    drows.append(s)
show(drows, "episode level")
dc10 = declusters(trig, 10, cal)
print(f"  episodes h=10: {[str(d.date()) for d in dc10]}")
print(f"  episode years: {sorted(set(dc10.year))}")

# --- 4) sensitivity --------------------------------------------------------
sens = []
for a in [5, 10, 15, 20]:
    for b in [70, 80, 90]:
        t = cal[(r63 <= a) & (r5 >= b)]
        v = pair(10).reindex(t).to_numpy()
        s = summarize(v, f"r63<={a} r5>={b}")
        s["n_epi"] = len(declusters(t, 10, cal))
        sens.append(s)
show(sens, "4) threshold sensitivity (h=10)")

# --- 6) CPI inside the window ---------------------------------------------
print("\n=== 6) CPI print inside the 10 td hold? ===")
ins = np.array([any(x in cpi_d for x in cal[pos[d] + 1: pos[d] + 11]) for d in trig])
show([summarize(v10[ins], "CPI inside"), summarize(v10[~ins], "no CPI inside")],
     "CPI split (h=10)")
allc = np.array([any(x in cpi_d for x in cal[pos[d] + 1: pos[d] + 11])
                 for d in cal[:-11]])
print(f"  base rate of a CPI landing in any 10td window: {100*allc.mean():.1f}%")

print("\n=== 5) cost: SMH ~2bp + QQQ ~1bp per side = 6bp round trip = 0.060% ===")
print("     5x hurdle = 0.300%.")
