"""Roll-seam guard on tonight's futures outliers.

The 2026-09-06 brief excluded grains, softs and metals because their 09-04 bars
sat on a continuous-contract roll seam: "the tape's largest move, coffee at
-9.70%, is a seam and not a price." Tonight coffee prints -10.58% and would be
the 3rd worst session in 26 years, and CL=F, HG=F, CT=F, ZW=F and ZC=F all show
large moves or 52-week highs. Every one of them has to clear this check first.

A genuine session move sits INSIDE its own bar: the close is bounded by the
day's low and high, and the open is near the prior close. A roll seam shows up
as a close-to-close jump the bar's own range cannot contain, usually with a
volume discontinuity as liquidity moves to the next contract.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import load_prices

TK = ["KC=F", "CC=F", "CT=F", "SB=F", "ZW=F", "ZC=F", "ZS=F",
      "HG=F", "GC=F", "SI=F", "CL=F", "NG=F"]
px = load_prices(TK)

print("Bars 2026-09-01 .. 2026-09-08, with the gap test\n")
for t in TK:
    df = px[t].loc["2026-08-28":"2026-09-08"]
    if df.empty:
        print(f"{t}: no bars"); continue
    print(f"--- {t} ---")
    prev_close = None
    for d, row in df.iterrows():
        o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
        v = row.get("Volume", np.nan)
        gap = (o / prev_close - 1.0) * 100 if prev_close else np.nan
        ret = (c / prev_close - 1.0) * 100 if prev_close else np.nan
        # a seam signature: the OPEN itself jumps, and the close sits inside
        # a range that never contains the prior close
        contains_prev = (prev_close is not None and l <= prev_close <= h)
        flag = ""
        if prev_close is not None:
            if abs(gap) > 3.0 and not contains_prev:
                flag = "  <== OPEN GAP + prior close OUTSIDE the bar's range"
            elif not contains_prev and abs(ret) > 3.0:
                flag = "  <== prior close outside today's range"
        print(f"  {d.date()} O={o:9.3f} H={h:9.3f} L={l:9.3f} C={c:9.3f} "
              f"V={v:>12,.0f} gap={gap:7.2f}% ret={ret:7.2f}% "
              f"prev_in_range={contains_prev}{flag}"
              if prev_close is not None else
              f"  {d.date()} O={o:9.3f} H={h:9.3f} L={l:9.3f} C={c:9.3f} V={v:>12,.0f}")
        prev_close = c
    print()

print("=" * 74)
print("Volume discontinuity check: today's volume vs the trailing 20d median")
print("=" * 74)
for t in TK:
    df = px[t]
    if "Volume" not in df.columns or df.empty:
        continue
    v = df["Volume"].dropna()
    if len(v) < 30 or pd.Timestamp("2026-09-08") not in v.index:
        continue
    today = v.loc["2026-09-08"]
    med = v.loc[:"2026-09-05"].tail(20).median()
    r = df["Close"].pct_change(fill_method=None).loc["2026-09-08"] * 100
    print(f"  {t:6s} ret={r:+7.2f}%  vol={today:>12,.0f}  20d median={med:>12,.0f}  "
          f"ratio={today/med if med else float('nan'):6.2f}x")
