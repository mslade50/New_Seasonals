"""Stage B1 recon: compute TODAY's value of every trigger the surface map will
cite, straight from master_prices, so no dismissal rests on a summary field or
a rank trap. Prints the watchlist verdict inputs first, then candidate states.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

NAMES = ["SPY", "QQQ", "IWM", "TLT", "IEF", "LQD", "HYG", "GLD", "GDX", "SLV", "NEM",
         "USO", "XLE", "XOP", "UNG", "DBC", "UUP", "EFA", "EEM", "FXI", "EWZ", "EWJ",
         "^VIX", "^VIX3M", "^MOVE", "^SKEW", "SVXY", "XLU", "XRT", "IHI", "SMH", "^TNX",
         "MU", "NKE", "CSCO", "^GSPC", "^NDX"]

px = load_prices(NAMES)
print("loaded:", sorted(k for k in px))
missing = [n for n in NAMES if n not in px]
print("MISSING FROM CACHE:", missing)

for t in sorted(px):
    s = px[t]["Close"]
    print(f"{t:<10} last {s.index[-1].date()}  close {s.iloc[-1]:.2f}  n={len(s)}  from {s.index[0].date()}")

print("\n=== distance-to-extreme on each instrument's OWN series (never a panel) ===")
print(f"{'tkr':<10}{'52wH%':>9}{'52wL%':>9}{'200d%':>9}{'r5':>7}{'r21':>7}{'r63':>7}{'z10':>7}")
for t in sorted(px):
    s = px[t]["Close"]
    hi = s.rolling(252).max().iloc[-1]
    lo = s.rolling(252).min().iloc[-1]
    sma = s.rolling(200).mean().iloc[-1]
    last = s.iloc[-1]
    r5 = pct_rank(s, 5).iloc[-1]
    r21 = pct_rank(s, 21).iloc[-1]
    r63 = pct_rank(s, 63).iloc[-1]
    z = zscore(s, 10).iloc[-1]
    print(f"{t:<10}{100*(last/hi-1):>9.2f}{100*(last/lo-1):>9.2f}{100*(last/sma-1):>9.2f}"
          f"{r5:>7.1f}{r21:>7.1f}{r63:>7.1f}{z:>7.2f}")

print("\n=== W6 / W13: IG-complex 52w-low rung and its cluster depth ===")
def dist_low(t, lb=252):
    s = px[t]["Close"]
    return 100 * (s / s.rolling(lb).min() - 1)

dl = {t: dist_low(t) for t in ("TLT", "IEF", "LQD")}
idx = px["TLT"].index
tight = (dl["TLT"] <= 0.5) & (dl["IEF"] <= 1.0) & (dl["LQD"] <= 1.0)
tight = tight.dropna()
live = tight.iloc[-1]
print("tight rung live today:", bool(live),
      f"(TLT {dl['TLT'].iloc[-1]:.2f}%, IEF {dl['IEF'].iloc[-1]:.2f}%, LQD {dl['LQD'].iloc[-1]:.2f}%)")
trig = tight[tight].index
print("total trigger days:", len(trig), "| last 12:", [str(d.date()) for d in trig[-12:]])
if len(trig) >= 2:
    all_d = tight.index
    pos = {d: i for i, d in enumerate(all_d)}
    gap = pos[trig[-1]] - pos[trig[-2]]
    print(f"sessions since PRIOR trigger day: {gap} (W6 needs >= 10)")
    # episode depth
    depth = 1
    for i in range(len(trig) - 1, 0, -1):
        if pos[trig[i]] - pos[trig[i - 1]] <= 10:
            depth += 1
        else:
            break
    print("cluster depth (<=10td chaining):", depth)

print("\n=== W13: SPY within 0.5% of 52w high AND TLT within 1% of 52w low ===")
s = px["SPY"]["Close"]
spy_h = 100 * (s / s.rolling(252).max() - 1)
print(f"SPY dist 52w high {spy_h.iloc[-1]:.3f}% (needs >= -0.5)  TLT dist 52w low {dl['TLT'].iloc[-1]:.3f}% (needs <= 1.0)")
print("joint live:", bool(spy_h.iloc[-1] >= -0.5 and dl["TLT"].iloc[-1] <= 1.0))

print("\n=== W7: skew spike legs ===")
sk = px["^SKEW"]["Close"]
print(f"^SKEW level {sk.iloc[-1]:.2f} | rank5 {pct_rank(sk,5).iloc[-1]:.1f} (needs >=95)")
print(f"SPY off 52w high {spy_h.iloc[-1]:.2f}% (needs < -1.0) | midterm year: True (needs False)")

print("\n=== W4 / W8 / W9 / W10: ranks the triggers name ===")
for t, legs in [("GDX", "5d>=95"), ("GLD", "5d<95"), ("USO", "5d>=90 & 63d<=20"),
                ("IHI", "21d==100"), ("FXI", "5d<=20 & 21d>=80"), ("EEM", "5d ret > 0")]:
    s2 = px[t]["Close"]
    print(f"{t:<5} r5 {pct_rank(s2,5).iloc[-1]:>6.1f}  r21 {pct_rank(s2,21).iloc[-1]:>6.1f}  "
          f"r63 {pct_rank(s2,63).iloc[-1]:>6.1f}  ret5 {100*(s2.iloc[-1]/s2.iloc[-6]-1):>7.2f}%  [{legs}]")

print("\n=== W5: USO one-day pop in ATR terms ===")
u = px["USO"]
atr = wilder_atr(u["High"], u["Low"], u["Close"])
d1 = u["Close"].iloc[-1] - u["Close"].iloc[-2]
print(f"USO 1d ${d1:.3f} = {d1/atr[-1]:.2f} ATR ({100*d1/u['Close'].iloc[-2]:.2f}%), needs [5,6)% and >=1.50 ATR")

print("\n=== CANDIDATE STATES ===")
print("\n-- C: ^MOVE one-day spike --")
mv = px["^MOVE"]["Close"]
r1 = mv.pct_change()
print(f"^MOVE level {mv.iloc[-1]:.2f}  1d {100*r1.iloc[-1]:.2f}%  "
      f"pctile of 1d moves (full hist) {100*(r1 < r1.iloc[-1]).mean():.1f}  "
      f"level pctile full {100*(mv < mv.iloc[-1]).mean():.1f} / 252d {100*(mv.tail(252) < mv.iloc[-1]).mean():.1f}")
print("^MOVE history from", mv.index[0].date(), "n =", len(mv))

print("\n-- B: VIX pop without spot damage --")
v = px["^VIX"]["Close"]
spyr = s.pct_change()
vr = v.pct_change()
print(f"^VIX {v.iloc[-1]:.2f}  1d {100*vr.iloc[-1]:.2f}%   SPY 1d {100*spyr.iloc[-1]:.2f}%")
joint = (vr >= 0.05) & (spyr > -0.0075)
joint = joint.dropna()
print(f"days with VIX 1d >= +5% and SPY 1d > -0.75%: {int(joint.sum())} of {len(joint)} "
      f"({100*joint.mean():.2f}%), from {joint.index[0].date()}")
print("last 10 such days:", [str(d.date()) for d in joint[joint].index[-10:]])

print("\n-- A: month-end stock/bond divergence --")
sp21 = s.pct_change(21)
tl21 = px["TLT"]["Close"].pct_change(21)
div = (sp21 - tl21).dropna()
print(f"SPY 21d {100*sp21.iloc[-1]:.2f}%  TLT 21d {100*tl21.iloc[-1]:.2f}%  divergence {100*div.iloc[-1]:.2f}pp")
print(f"divergence pctile full hist {100*(div < div.iloc[-1]).mean():.1f}")
import pandas as pd
tdom = pd.Series(1, index=s.index).groupby([s.index.year, s.index.month]).cumsum()
print("today's trading-day-of-month:", int(tdom.iloc[-1]),
      "| sessions left in month:", int((s.index.month == s.index[-1].month).sum() - tdom.iloc[-1]),
      "(cache ends today, so month-end count is partial)")

print("\n-- H: miner/metal ratio extreme --")
ratio = (px["GDX"]["Close"] / px["GLD"]["Close"]).dropna()
r21r = ratio.pct_change(21)
print(f"GDX/GLD 21d change {100*r21r.iloc[-1]:.2f}pp  pctile {100*(r21r < r21r.iloc[-1]).mean():.1f}  "
      f"rank252 {pct_rank(ratio,21).iloc[-1]:.1f}")

print("\n-- D: international leadership --")
for t in ("EFA", "EWJ", "EEM"):
    s2 = px[t]["Close"]
    hi = 100 * (s2.iloc[-1] / s2.rolling(252).max().iloc[-1] - 1)
    print(f"{t} off 52w high {hi:.2f}% | 63d {100*(s2.iloc[-1]/s2.iloc[-64]-1):.2f}% vs SPY {100*(s.iloc[-1]/s.iloc[-64]-1):.2f}%")

print("\n-- G: MU / semis extension --")
for t in ("MU", "SMH"):
    s2 = px[t]["Close"]
    sma = s2.rolling(200).mean().iloc[-1]
    print(f"{t} {100*(s2.iloc[-1]/sma-1):.1f}% above 200d, pctile of that stat "
          f"{100*((s2/s2.rolling(200).mean()).dropna() < (s2.iloc[-1]/sma)).mean():.1f}")

print("\n-- E: NKE deep capitulation --")
n = px["NKE"]["Close"]
print(f"NKE off 52w high {100*(n.iloc[-1]/n.rolling(252).max().iloc[-1]-1):.2f}%  "
      f"vs 200d {100*(n.iloc[-1]/n.rolling(200).mean().iloc[-1]-1):.2f}%  "
      f"at 52w low: {bool(n.iloc[-1] <= n.rolling(252).min().iloc[-1]*1.0001)}")

print("\n=== live events in the window ===")
ev = load_events()
recent = ev[(ev["date"] >= "2026-08-01") & (ev["date"] <= "2026-09-20")]
print(recent.to_string())
