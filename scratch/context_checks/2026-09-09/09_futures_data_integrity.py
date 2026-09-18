"""How much of tonight's commodity tape is real?

08_roll_seam_check produced unusable volume multiples because the cache holds
ZERO volume for recent sessions on most futures. Establish (a) how wide the
zero-volume hole is, (b) whether the 09-08 bars were revised since last
night's brief was written, and (c) what the last 12 sessions actually look
like on the loud names.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import load_prices  # noqa: E402

LOUD = ["KC=F", "SB=F", "CT=F", "ZC=F", "ZS=F", "ZW=F", "HG=F", "SI=F",
        "GC=F", "PL=F", "CL=F", "NG=F", "CC=F"]

px = load_prices(LOUD)

print("=" * 78)
print("A. Zero-volume hole: last session with nonzero volume before 2026-09-09")
print("=" * 78)
for t in LOUD:
    df = px[t].dropna(subset=["Close"]).sort_index()
    hist = df[df.index < "2026-09-09"]
    nz = hist[hist["Volume"].fillna(0) > 0]
    last_nz = nz.index[-1].date() if len(nz) else None
    tail30 = hist.tail(30)
    n_zero = int((tail30["Volume"].fillna(0) == 0).sum())
    print(f"  {t:<6} last nonzero volume {last_nz}   "
          f"zero-volume sessions in the last 30: {n_zero}/30")

print()
print("=" * 78)
print("B. Last 12 sessions, loud names: close, pct, gap from prior close,")
print("   and whether the bar traded back through the prior close")
print("=" * 78)
for t in ["KC=F", "SB=F", "CT=F", "ZC=F", "HG=F", "SI=F", "CL=F"]:
    df = px[t].dropna(subset=["Close"]).sort_index().tail(13)
    print(f"\n--- {t}")
    prev = None
    for d, r in df.iterrows():
        if prev is None:
            prev = float(r["Close"])
            continue
        ret = 100 * (r["Close"] / prev - 1)
        gap = 100 * (r["Open"] / prev - 1)
        spans = float(r["Low"]) <= prev <= float(r["High"])
        print(f"   {d.date()}  close {float(r['Close']):>10.3f}  "
              f"{ret:+7.2f}%  gap {gap:+7.2f}%  "
              f"spans_prev={'Y' if spans else 'N'}  "
              f"vol {int(r['Volume']) if r['Volume'] == r['Volume'] else 'NA':>9}")
        prev = float(r["Close"])

print()
print("=" * 78)
print("C. How ordinary is a -8.5% KC=F session, and how ordinary is a bar")
print("   that gaps 9%+ and never trades back through the prior close?")
print("=" * 78)
kc = px["KC=F"].dropna(subset=["Close"]).sort_index()
prev_c = kc["Close"].shift(1)
ret = kc["Close"] / prev_c - 1
gap = kc["Open"] / prev_c - 1
spans = (kc["Low"] <= prev_c) & (kc["High"] >= prev_c)
big_down = ret <= -0.08
print(f"  KC=F history: {kc.index[0].date()} to {kc.index[-1].date()}, "
      f"{len(kc)} sessions")
print(f"  sessions <= -8%: {int(big_down.sum())}")
print(f"  of those, bars that NEVER traded through the prior close: "
      f"{int((big_down & ~spans).sum())}")
print(f"  of those, bars that DID trade through: {int((big_down & spans).sum())}")
print()
print("  the ten most recent <= -8% sessions:")
for d in kc.index[big_down][-10:]:
    print(f"    {d.date()}  {100 * ret[d]:+7.2f}%  gap {100 * gap[d]:+7.2f}%  "
          f"spans_prev={'Y' if spans[d] else 'N'}")

print()
print("=" * 78)
print("D. 252d-high claims: is each 'at a 252d high' close a spanning bar?")
print("=" * 78)
for t in ["HG=F", "ZC=F", "ZS=F", "ZW=F", "SB=F"]:
    df = px[t].dropna(subset=["Close"]).sort_index()
    c = df["Close"]
    hi252 = c.rolling(252).max()
    last = c.index[-1]
    dist = 100 * (c.iloc[-1] / hi252.iloc[-1] - 1)
    pc = float(c.iloc[-2])
    bar = df.iloc[-1]
    spans = float(bar["Low"]) <= pc <= float(bar["High"])
    print(f"  {t:<6} {last.date()} close {float(c.iloc[-1]):>10.3f}  "
          f"{dist:+6.2f}% vs its 252d high   spans_prev={'Y' if spans else 'N'}")
