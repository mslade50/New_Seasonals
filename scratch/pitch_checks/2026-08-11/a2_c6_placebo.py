"""C6 round 2 -- the anchor placebo.

C6's whole claim is that the CPI PRINT adds something to a UNG short. CPI
always lands in the 10th-15th of the month, so the "CPI anchor" is also a
trading-day-of-month position. The honest test is a PLACEBO: score the exact
same short on anchors shifted +/-1..10 sessions away from the real print. If
the shifted anchors pay the same, the CPI print is decoration and the cell is
a mid-month calendar position -- a filter that does not filter.

Second test: a proper difference-of-means bootstrap on the EXCESS, not on the
raw short return (which is just the structural bleed and will always look
significant).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (close_panel, load_events, fwd_lag, declusters, summarize,  # noqa: E402
                       sign_test)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)

px = close_panel(["UNG"])
u = px["UNG"].dropna()
uidx = u.index
H = 5
f = fwd_lag(u, H, lag=1)
short_all = -f.dropna()
BASE = 100 * short_all.mean()
ev = load_events(["cpi"])["date"]


def anchors(offset: int) -> pd.DatetimeIndex:
    """`offset` sessions before each CPI print (offset=2 is the real anchor)."""
    out = []
    for d in ev:
        p = uidx.searchsorted(d, side="left") - offset
        if 0 <= p < len(uidx):
            out.append(uidx[p])
    return pd.DatetimeIndex(sorted(set(out)))


print("=" * 100)
print("PLACEBO: the same UNG short, anchored k sessions before the CPI print")
print(f"(k=2 is C6's real anchor. baseline always-short h={H} = {BASE:+.3f}%)")
print("=" * 100)
rows = []
for k in range(-8, 13):
    a = anchors(k)
    v = -f.reindex(a).dropna()
    if len(v) < 30:
        continue
    st = summarize(v.values)
    se = np.sqrt(v.values.var(ddof=1) / len(v) + short_all.values.var(ddof=1) / len(short_all))
    rows.append({"k_sessions_before_print": k, "n": st["n"],
                 "short_pct": round(st["mean_pct"], 3),
                 "excess": round(st["mean_pct"] - BASE, 3),
                 "welch_t": round((v.values.mean() - short_all.values.mean()) / se, 2),
                 "hit": round(st["hit"], 1),
                 "signp": round(sign_test(int((v.values > 0).sum()), len(v)), 4),
                 "REAL": "  <-- C6" if k == 2 else ""})
df = pd.DataFrame(rows)
print(df.to_string(index=False))
real = df[df["k_sessions_before_print"] == 2]["excess"].iloc[0]
others = df[df["k_sessions_before_print"] != 2]["excess"]
print(f"\nREAL anchor excess {real:+.3f}%")
print(f"placebo anchors: mean {others.mean():+.3f}%, median {others.median():+.3f}%, "
      f"min {others.min():+.3f}%, max {others.max():+.3f}%")
print(f"placebos beating the real anchor: {int((others >= real).sum())} of {len(others)}")
print(f"--> percentile of the real anchor among placebos: "
      f"{100*float((others < real).mean()):.0f}")

# ---------------------------------------------------------------------------
# a genuine mid-month control: every day with the same tdom, no CPI required
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("MID-MONTH POSITION control -- is this just 'short UNG in the 2nd week'?")
print("=" * 100)
tdom = pd.Series(pd.Series(uidx, index=uidx).groupby([uidx.year, uidx.month]).cumcount().values + 1,
                 index=uidx)
real_a = anchors(2)
real_tdoms = sorted(set(tdom.reindex(real_a).dropna().astype(int)))
print(f"the real anchor's trading-day-of-month values: {real_tdoms}")
rows = []
for t in range(1, 22):
    sel = tdom[tdom == t].index
    v = short_all.reindex(sel).dropna()
    if len(v) < 30:
        continue
    st = summarize(v.values)
    rows.append({"tdom": t, "n": st["n"], "short_pct": round(st["mean_pct"], 3),
                 "excess_vs_alldays": round(st["mean_pct"] - BASE, 3),
                 "hit": round(st["hit"], 1),
                 "is_cpi_tdom": t in real_tdoms})
print(pd.DataFrame(rows).to_string(index=False))
cpi_t = [r for r in rows if r["is_cpi_tdom"]]
non_t = [r for r in rows if not r["is_cpi_tdom"]]
print(f"\nmean excess on CPI-anchor tdoms      : {np.mean([r['excess_vs_alldays'] for r in cpi_t]):+.3f}%")
print(f"mean excess on every other tdom      : {np.mean([r['excess_vs_alldays'] for r in non_t]):+.3f}%")

# ---------------------------------------------------------------------------
# bootstrap the EXCESS (difference of means), not the raw short
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("BOOTSTRAP ON THE EXCESS (the raw short bootstrap only measures the bleed)")
print("=" * 100)
rng = np.random.default_rng(42)
v = -f.reindex(anchors(2)).dropna()
a_ = v.values
b_ = short_all.values
diffs = (rng.choice(a_, size=(10000, len(a_))).mean(axis=1)
         - rng.choice(b_, size=(10000, len(b_))).mean(axis=1))
print(f"  excess point estimate {100*(a_.mean()-b_.mean()):+.3f}%")
print(f"  bootstrap 95% CI [{100*np.percentile(diffs,2.5):+.3f}%, {100*np.percentile(diffs,97.5):+.3f}%]")
print(f"  bootstrap P(excess <= 0) = {float((diffs <= 0).mean()):.3f}")

# and the raw short, for contrast, so the difference is unmistakable
raw = rng.choice(a_, size=(10000, len(a_))).mean(axis=1)
print(f"\n  for contrast -- RAW short bootstrap P(mean <= 0) = {float((raw <= 0).mean()):.3f}")
print("  that number is the UNG BLEED passing a test, not the CPI print passing one.")

# ---------------------------------------------------------------------------
# what would turn it on
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("TURN-ON ARITHMETIC")
print("=" * 100)
se_now = np.sqrt(a_.var(ddof=1) / len(a_) + b_.var(ddof=1) / len(b_))
print(f"  current excess {100*(a_.mean()-b_.mean()):+.3f}% on n={len(a_)}, welch t "
      f"{(a_.mean()-b_.mean())/se_now:+.2f}")
need_mean = 2.0 * se_now
print(f"  to reach welch t=2.0 at this n, the excess would have to be "
      f"{100*need_mean:+.3f}% (it is {100*(a_.mean()-b_.mean()):+.3f}%)")
n_need = (2.0 / ((a_.mean() - b_.mean()) / se_now)) ** 2 * len(a_)
print(f"  or, holding the effect size, n would have to reach ~{n_need:.0f} CPI prints "
      f"(~{n_need/12:.0f} years); there are {len(a_)}.")
