"""C11 "Mega-cap laggard snapback": long AAPL / short QQQ equal dollar, 5 and
10 td. Trigger: AAPL rank5 <= 10 AND QQQ rank5 >= 85.

Controls: (1) the pair's own unconditional drift over the same window (AAPL has
a massive unconditional outperformance of QQQ - that is the control that
matters), (2) AAPL-weak-only (no QQQ gate), (3) all-days baseline.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import *  # noqa

import numpy as np
import pandas as pd

px = load_prices(["AAPL", "QQQ"])
aapl = px["AAPL"]["Close"].dropna()
qqq = px["QQQ"]["Close"].dropna()
cal = aapl.index.intersection(qqq.index)
aapl, qqq = aapl.reindex(cal), qqq.reindex(cal)
pos = pd.Series(range(len(cal)), index=cal)
cpi_d = set(load_events(["cpi"])["date"])

ra, rq = pct_rank(aapl, 5), pct_rank(qqq, 5)


def pair(h):
    return fwd_ret(aapl, h) - fwd_ret(qqq, h)


print(f"calendar {cal[0].date()} .. {cal[-1].date()}  n={len(cal)}")
print("AAPL/QQQ pair unconditional drift (this is the control that kills or "
      "saves the cell):")
for h in [5, 10, 21]:
    print(f"   h={h:2d}: mean={100*pair(h).mean():+.3f}%  "
          f"median={100*pair(h).median():+.3f}%  hit={100*(pair(h)>0).mean():.1f}%")

trig = pd.DatetimeIndex([d for d in cal[(ra <= 10) & (rq >= 85)]
                         if not np.isnan(pair(10).get(d, np.nan))])
aapl_only = pd.DatetimeIndex([d for d in cal[ra <= 10]
                              if not np.isnan(pair(10).get(d, np.nan))])
print(f"\ntrigger days: {len(trig)}   AAPL-weak-only days: {len(aapl_only)}")
if len(trig):
    print(f"  first={trig[0].date()} last={trig[-1].date()}")

# --- 1) exists? ------------------------------------------------------------
rows = []
for h in [5, 10, 21]:
    p = pair(h)
    v = p.reindex(trig).to_numpy()
    s = summarize(v, f"TRIGGER h={h}")
    s["ctl_uncond_pct"] = 100 * p.mean()
    s["ctl_aaplweak_pct"] = 100 * np.nanmean(p.reindex(aapl_only).to_numpy())
    s["edge_vs_uncond"] = s["mean_pct"] - 100 * p.mean()
    rows.append(s)
show(rows, "1) conditional pair return vs controls")

show([summarize(fwd_ret(aapl, 10).reindex(trig).to_numpy(), "AAPL leg h=10"),
      summarize(fwd_ret(qqq, 10).reindex(trig).to_numpy(), "QQQ leg h=10"),
      summarize(fwd_ret(aapl, 10).to_numpy(), "AAPL all-days"),
      summarize(fwd_ret(qqq, 10).to_numpy(), "QQQ all-days")], "leg decomposition")

# --- 2) era + concentration ------------------------------------------------
for h in [5, 10]:
    v = pair(h).reindex(trig).to_numpy()
    show(era_split(trig, v), f"2) era split h={h}")
v10 = pair(10).reindex(trig).to_numpy()
print("\n  trigger dates by year:", dict(pd.Series(1, index=trig).groupby(trig.year).sum()))
srt = pd.Series(v10, index=trig).sort_values()
print(f"  worst 3 (h=10): {[(str(d.date()), round(100*x,2)) for d, x in srt.head(3).items()]}")
print(f"  best  3 (h=10): {[(str(d.date()), round(100*x,2)) for d, x in srt.tail(3).items()]}")

# --- 3) decluster ----------------------------------------------------------
drows = []
for h in [5, 10]:
    dc = declusters(trig, h, cal)
    v = pair(h).reindex(dc).to_numpy()
    s = summarize(v, f"episodes h={h}")
    s["p_le0_boot"] = bootstrap_p_le0(v)
    drows.append(s)
show(drows, "3) episode level (decluster min gap = horizon)")
dc10 = declusters(trig, 10, cal)
print(f"  episodes h=10 ({len(dc10)}): {[str(d.date()) for d in dc10]}")
# how much of the episode mean is one episode?
ve = pair(10).reindex(dc10).to_numpy()
if len(ve) > 1:
    i = int(np.nanargmax(np.abs(ve - np.nanmean(ve))))
    m2 = np.nanmean(np.delete(ve, i))
    print(f"  drop the single most-influential episode "
          f"({dc10[i].date()}, {100*ve[i]:+.2f}%): mean {100*np.nanmean(ve):+.3f}% "
          f"-> {100*m2:+.3f}%")

# --- 4) sensitivity --------------------------------------------------------
sens = []
for a in [5, 10, 15, 20]:
    for b in [80, 85, 90]:
        t = pd.DatetimeIndex([d for d in cal[(ra <= a) & (rq >= b)]
                              if not np.isnan(pair(10).get(d, np.nan))])
        for h in [5, 10]:
            s = summarize(pair(h).reindex(t).to_numpy(), f"ra<={a} rq>={b} h={h}")
            s["n_epi"] = len(declusters(t, h, cal))
            sens.append(s)
show(sens, "4) threshold sensitivity")

# --- 6) CPI in window ------------------------------------------------------
print("\n=== 6) CPI print inside the hold? ===")
for h in [5, 10]:
    v = pair(h).reindex(trig).to_numpy()
    ins = np.array([any(x in cpi_d for x in cal[pos[d] + 1: pos[d] + h + 1]) for d in trig])
    show([summarize(v[ins], f"h={h} CPI inside"),
          summarize(v[~ins], f"h={h} no CPI inside")], f"CPI split h={h}")

print("\n=== 5) cost: AAPL ~1bp + QQQ ~1bp per side = 4bp round trip = 0.040% ===")
print("     5x hurdle = 0.200%.")
