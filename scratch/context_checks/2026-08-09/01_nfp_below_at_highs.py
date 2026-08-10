"""Friday's payroll print, conditioned the way Friday actually looked.

NFP -23k vs +80k consensus (an outright job loss) but unemployment 4.1% vs
4.2% expected, and the tape rallied into a 52w high on ^GSPC. Three questions
the engine's base cell cannot answer:

1. Does the below-consensus payroll cell survive the era split and a
   concentration check, or is the +0.32% IWM mean two 2020 episodes?
2. What happens when the SAME report sends both signals (payrolls below
   consensus, unemployment ALSO below)? That configuration is what printed.
3. Does the cell hold when the index is already AT a 52w high, i.e. when the
   dovish read has nothing left to reprice?

Also checks the ^VIX +3.26% number, which is internally odd next to equities
up and is the classic shape of a mean carried by a handful of spikes.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from macro_releases import load_macro_releases  # noqa: E402
from pitch_lab import cluster_note, fwd_ret, load_prices, sign_test, summarize, show  # noqa: E402

ASOF = pd.Timestamp("2026-08-07")
SUBJECTS = ["SPY", "QQQ", "IWM", "^GSPC", "TLT", "^TNX", "^VIX", "GC=F"]

px = load_prices(SUBJECTS)
rel = load_macro_releases(events=["nfp", "unemployment_rate"], end=ASOF)

nfp = rel[rel["event_id"] == "nfp"]
ur = rel[rel["event_id"] == "unemployment_rate"]

nfp_below = pd.DatetimeIndex(sorted(nfp.loc[nfp["surprise_label"] == "below",
                                            "release_date"].unique()))
ur_below = pd.DatetimeIndex(sorted(ur.loc[ur["surprise_label"] == "below",
                                          "release_date"].unique()))
both = nfp_below.intersection(ur_below)

print(f"below-consensus NFP prints            : {len(nfp_below)}")
print(f"below-consensus unemployment prints   : {len(ur_below)}")
print(f"BOTH on the same report (Friday's cfg): {len(both)}")
print(f"  most recent: {[str(d.date()) for d in both[-6:]]}")


def cell(ticker: str, anchors: pd.DatetimeIndex, label: str, h: int = 1) -> dict:
    close = px[ticker]["Close"].astype(float)
    f = fwd_ret(close, h)
    idx = anchors.intersection(f.dropna().index)
    vals = f.loc[idx].values
    row = summarize(vals, label)
    if row["n"]:
        up = int((vals > 0).sum())
        row["record"] = f"{up}-{row['n'] - up}"
        row["sign_p"] = round(sign_test(max(up, row["n"] - up), row["n"]), 4)
    return row


print("\n" + "=" * 78)
print("1. below-consensus payrolls, next session, era split and concentration")
print("=" * 78)
for ticker in ("IWM", "QQQ", "SPY", "TLT", "^TNX", "^VIX"):
    close = px[ticker]["Close"].astype(float)
    f = fwd_ret(close, 1)
    idx = nfp_below.intersection(f.dropna().index)
    vals = f.loc[idx].values
    pre = np.asarray(idx) < np.datetime64(pd.Timestamp("2018-01-01"))
    rows = [cell(ticker, nfp_below, "all"),
            summarize(vals[pre], "pre-2018"),
            summarize(vals[~pre], "2018+")]
    show(rows, f"{ticker} after a below-consensus payroll print")
    print(f"  median {100 * np.median(vals):+.3f}%   "
          f"{cluster_note(pd.DatetimeIndex(idx), vals)}")

print("\n" + "=" * 78)
print("2. the configuration that actually printed: payrolls below AND")
print("   unemployment below on the same report")
print("=" * 78)
rows = []
for ticker in SUBJECTS:
    rows.append(cell(ticker, both, ticker))
show(rows, "next session after a 'both below' report")
rows5 = [cell(t, both, t, h=5) for t in SUBJECTS]
show(rows5, "next WEEK after a 'both below' report")

print("\n" + "=" * 78)
print("3. does it hold when the index is already at a 52w high?")
print("=" * 78)
gspc = px["^GSPC"]["Close"].astype(float)
hi52 = gspc.rolling(252, min_periods=252).max()
near_high = (gspc / hi52 - 1.0) >= -0.01          # within 1% of the 52w high
at_high_dates = pd.DatetimeIndex(near_high.index[near_high.fillna(False).values])

for name, anchors in (("payrolls below", nfp_below), ("both below", both)):
    hot = anchors.intersection(at_high_dates)
    cold = anchors.difference(at_high_dates)
    print(f"\n  {name}: {len(hot)} with the index within 1% of its 52w high, "
          f"{len(cold)} without")
    rows = []
    for ticker in ("SPY", "QQQ", "IWM", "TLT"):
        rows.append(cell(ticker, hot, f"{ticker} at highs"))
        rows.append(cell(ticker, cold, f"{ticker} not at highs"))
    show(rows, f"  {name}, split on the index being at a 52w high")

print("\n" + "=" * 78)
print("4. the VIX number: mean vs median, and what carries it")
print("=" * 78)
vix = px["^VIX"]["Close"].astype(float)
f = fwd_ret(vix, 1)
idx = nfp_below.intersection(f.dropna().index)
vals = f.loc[idx].values
order = np.argsort(-np.abs(vals))[:5]
print(f"  n={len(vals)}  mean {100 * vals.mean():+.2f}%  "
      f"median {100 * np.median(vals):+.2f}%  "
      f"record {(vals > 0).sum()}-{(vals < 0).sum()}")
print("  five largest absolute moves:")
for i in order:
    print(f"    {pd.Timestamp(idx[i]).date()}  {100 * vals[i]:+7.2f}%")
trimmed = np.sort(vals)[2:-2]
print(f"  mean excluding the two largest each way: {100 * trimmed.mean():+.2f}%")
