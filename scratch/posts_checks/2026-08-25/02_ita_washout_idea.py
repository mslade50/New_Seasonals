"""Idea check: long the defense ETF (ITA) after a five-session washout while
the index sits near its high.

Tonight's state (2026-08-25): ITA 5d rank 0.8 (lowest on the 218-name tape),
z10 -1.40, with LMT/RTX/NOC/GD all bottom-5% on 5d and XLI z10 -1.12. SPY is
1.54% below its 52w high. Same shape as last night's semis candidate, which
was killed on the era split and two-episode concentration. Question: does
buying ITA at the NEXT close after a bottom-5% week (SPY within 3% of a
high) pay over 5-10 sessions, declustered at 10 td?

Kill attempts: full-history and local controls, era split, cluster
concentration, midterm years, 200d reference class, the ungated parent and
its complement (does the SPY gate do anything), and XLI as the vehicle.
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

px = load_prices(["ITA", "SPY", "XLI"])
ita_raw, spy_raw, xli_raw = px["ITA"], px["SPY"], px["XLI"]
ita, spy, xli = ita_raw["Close"], spy_raw["Close"], xli_raw["Close"]
idx = ita.index.intersection(spy.index)
ita, spy, xli = ita.reindex(idx), spy.reindex(idx), xli.reindex(idx)

atr = pd.Series(wilder_atr(ita_raw["High"], ita_raw["Low"], ita_raw["Close"]),
                index=ita_raw.index).reindex(idx)
rank5 = pct_rank(ita, 5)
spy_dist = spy / spy.rolling(252).max() - 1.0
sma200 = ita.rolling(200).mean()

print("ITA panel", idx[0].date(), "->", idx[-1].date(), "n", len(idx))
print("tonight: ITA close %.2f | 5d %+.2f%% | 5d rank %.1f | Wilder-14 ATR %.4f (%.2f%%) | "
      "SPY dist 52w high %+.2f%% | ITA vs 200d %+.2f%%"
      % (ita.iloc[-1], (ita.iloc[-1] / ita.iloc[-6] - 1) * 100, rank5.iloc[-1],
         atr.iloc[-1], atr.iloc[-1] / ita.iloc[-1] * 100, spy_dist.iloc[-1] * 100,
         (ita.iloc[-1] / sma200.iloc[-1] - 1) * 100))

raw = idx[((rank5 < 5) & (spy_dist > -0.03)).fillna(False)]
raw = raw[raw < idx[-1]]
trig = declusters(raw, 10, idx)
print(f"\ntrigger: ITA 5d rank < 5 AND SPY within 3% of 252d high -> raw n={len(raw)}, "
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


print("\n-- ITA, MOC tomorrow (lag-1), exit h sessions later")
report(ita, trig, "ITA")
print("\n-- XLI as the vehicle on the same trigger")
report(xli, trig, "XLI")

for h in (5, 10):
    f = fwd_lag(ita, h).reindex(trig).dropna()
    print(f"\nera h{h} ITA:", [(e["label"], e["n"], round(e.get("mean_pct", np.nan), 3),
                                round(e.get("hit", np.nan), 1)) for e in era_split(f.index, f.values)])
    print(f"concentration h{h} ITA:", cluster_note(f.index, f.values))

above = (ita / sma200 - 1).reindex(trig) > 0
for lab, sub in (("ITA above 200d", trig[above.fillna(False)]),
                 ("ITA below 200d", trig[~above.fillna(True)])):
    fs = fwd_lag(ita, 5).reindex(sub).dropna()
    if not len(fs):
        print(f"  {lab}: n=0")
        continue
    s = summarize(fs.values)
    nup = int((fs > 0).sum())
    print(f"  {lab} h5: n={s['n']:<3} mean={s['mean_pct']:+.3f}%  {nup}-{len(fs)-nup} up  "
          f"t={s['t']:+.2f}  sign_p={sign_test(nup, len(fs)):.4f}  worst={s['worst_pct']:+.2f}%")

mid = pd.DatetimeIndex([d for d in trig if d.year % 4 == 2])
fm = fwd_lag(ita, 5).reindex(mid).dropna()
if len(fm):
    s = summarize(fm.values)
    nup = int((fm > 0).sum())
    print(f"  midterm yrs h5: n={s['n']:<3} mean={s['mean_pct']:+.3f}%  {nup}-{len(fm)-nup} up  "
          f"sign_p={sign_test(nup, len(fm)):.4f}")

par = idx[(rank5 < 5).fillna(False)]
par = declusters(par[par < idx[-1]], 10, idx)
print(f"\n-- ungated parent (ITA 5d rank < 5, any tape) n={len(par)}")
report(ita, par, "ITA-parent")
comp = par.difference(trig)
print(f"-- complement (parent minus gated) n={len(comp)}")
report(ita, comp, "ITA-compl")
