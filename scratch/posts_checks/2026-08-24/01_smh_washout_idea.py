"""Idea check: long semis (SMH) after a five-session washout under an index
that is still near its high.

Tonight's state (2026-08-24): SMH -7.96% over 5 sessions (5d rank 4.0,
63d rank 0.4, 18% off its 52w high) while SPY sits 1.85% below its 52w high
and 8% above its 200d. QQQ is in its own bottom decile (rank_5d 7.1).
Question: does buying the semis basket after a bottom-5% week, with the
broad tape still within 3% of a high, pay over the next 5-10 sessions
once you enter at the NEXT close (lag-1) and decluster?

Kill attempts: full-history control, local control, era split, cluster
concentration, midterm years, and the reference-class check that the
edge is not all below-200d (bear rebound) episodes.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    cluster_note, declusters, era_split, fwd_lag, load_prices, local_control,
    pct_rank, sign_test, summarize, wilder_atr,
)

px = load_prices(["SMH", "SPY", "QQQ"])
smh_raw, spy_raw, qqq_raw = px["SMH"], px["SPY"], px["QQQ"]
smh, spy, qqq = smh_raw["Close"], spy_raw["Close"], qqq_raw["Close"]
idx = smh.index.intersection(spy.index)
smh, spy = smh.reindex(idx), spy.reindex(idx)
qqq = qqq.reindex(idx)

atr = pd.Series(wilder_atr(smh_raw["High"], smh_raw["Low"], smh_raw["Close"]),
                index=smh_raw.index).reindex(idx)
rank5 = pct_rank(smh, 5)
spy_hi = spy.rolling(252).max()
spy_dist = spy / spy_hi - 1.0
sma200 = smh.rolling(200).mean()

print("SMH panel", idx[0].date(), "->", idx[-1].date(), "n", len(idx))
print("tonight: SMH close %.2f | 5d %+.2f%% | 5d rank %.1f | Wilder-14 ATR %.4f (%.2f%%) | "
      "SPY dist 52w high %+.2f%%"
      % (smh.iloc[-1], (smh.iloc[-1] / smh.iloc[-6] - 1) * 100, rank5.iloc[-1],
         atr.iloc[-1], atr.iloc[-1] / smh.iloc[-1] * 100, spy_dist.iloc[-1] * 100))

raw_mask = (rank5 < 5) & (spy_dist > -0.03)
raw = idx[raw_mask.fillna(False)]
raw = raw[raw < idx[-1]]
trig = declusters(raw, 10, idx)
print(f"\ntrigger: SMH 5d rank < 5 AND SPY within 3% of 252d high -> raw n={len(raw)}, "
      f"declustered@10td n={len(trig)}")
print("dates:", [d.strftime("%Y-%m-%d") for d in trig])


def report(series: pd.Series, dates: pd.DatetimeIndex, label: str) -> None:
    for h in (1, 2, 3, 5, 10):
        f = fwd_lag(series, h).reindex(dates).dropna()
        if not len(f):
            continue
        s = summarize(f.values)
        nup = int((f > 0).sum())
        allc = summarize(fwd_lag(series, h).dropna().values)
        loc = summarize(fwd_lag(series, h).reindex(local_control(series.index, dates, 126)).dropna().values)
        print(f"  {label} h{h:<2} n={s['n']:<3} mean={s['mean_pct']:+.3f}%  med={s['median_pct']:+.2f}%  "
              f"{nup}-{len(f)-nup} up  t={s['t']:+.2f}  sign_p={sign_test(nup, len(f)):.4f}  "
              f"| all {allc['mean_pct']:+.3f}%  local {loc['mean_pct']:+.3f}%  "
              f"| worst {s['worst_pct']:+.1f}% best {s['best_pct']:+.1f}%")


print("\n-- SMH, MOC tomorrow (lag-1), exit h sessions later")
report(smh, trig, "SMH")
print("\n-- QQQ as the vehicle on the same trigger")
report(qqq, trig, "QQQ")

for h in (5, 10):
    f = fwd_lag(smh, h).reindex(trig).dropna()
    print(f"\nera h{h} SMH:", [(e["label"], e["n"], round(e.get("mean_pct", np.nan), 3),
                                round(e.get("hit", np.nan), 1)) for e in era_split(f.index, f.values)])
    print(f"concentration h{h} SMH:", cluster_note(f.index, f.values))

# reference class: SMH above / below its own 200d at trigger
above = (smh / sma200 - 1).reindex(trig) > 0
for lab, sub in (("SMH above 200d", trig[above.fillna(False)]),
                 ("SMH below 200d", trig[~above.fillna(True)])):
    fs = fwd_lag(smh, 5).reindex(sub).dropna()
    if not len(fs):
        print(f"  {lab}: n=0")
        continue
    s = summarize(fs.values)
    nup = int((fs > 0).sum())
    print(f"  {lab} h5: n={s['n']:<3} mean={s['mean_pct']:+.3f}%  {nup}-{len(fs)-nup} up  "
          f"t={s['t']:+.2f}  sign_p={sign_test(nup, len(fs)):.4f}  worst={s['worst_pct']:+.2f}%")

# midterm years
mid = pd.DatetimeIndex([d for d in trig if d.year % 4 == 2])
fm = fwd_lag(smh, 5).reindex(mid).dropna()
if len(fm):
    s = summarize(fm.values)
    nup = int((fm > 0).sum())
    print(f"  midterm yrs h5: n={s['n']:<3} mean={s['mean_pct']:+.3f}%  {nup}-{len(fm)-nup} up  "
          f"sign_p={sign_test(nup, len(fm)):.4f}")

# ungated parent: SMH 5d rank < 5 regardless of SPY (does the gate do anything?)
par = idx[(rank5 < 5).fillna(False)]
par = declusters(par[par < idx[-1]], 10, idx)
print(f"\n-- ungated parent (SMH 5d rank < 5, any tape) n={len(par)}")
report(smh, par, "SMH-parent")
comp = par.difference(trig)
print(f"-- complement (parent minus gated) n={len(comp)}")
report(smh, comp, "SMH-compl")

# the 8-percent-plus five-day drop version: magnitude rather than rank
r5 = smh / smh.shift(5) - 1
mag = idx[((r5 <= -0.075) & (spy_dist > -0.03)).fillna(False)]
mag = declusters(mag[mag < idx[-1]], 10, idx)
print(f"\n-- magnitude form: SMH 5d <= -7.5% with SPY within 3% of high, n={len(mag)}")
print("dates:", [d.strftime("%Y-%m-%d") for d in mag])
report(smh, mag, "SMH-mag")
