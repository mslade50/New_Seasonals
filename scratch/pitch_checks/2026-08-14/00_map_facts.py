"""Surface-map recon for 2026-08-14. Facts the map's verdicts must cite.

Not a check. Enumerates today's live states so every B1 cell gets a number
attached to its verdict instead of an impression.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pitch_lab import load_events, load_prices, pct_rank  # noqa: E402

ASOF = pd.Timestamp("2026-08-13")

CLASSES = {
    "us_large": ["SPY", "QQQ", "^GSPC", "^NDX"],
    "us_small": ["IWM"],
    "rates": ["TLT", "IEF", "^TNX"],
    "credit": ["HYG", "LQD"],
    "gold_miners": ["GLD", "GDX"],
    "other_metals": ["SLV"],
    "energy": ["USO", "UNG", "DBC", "XLE"],
    "dollar_fx": ["UUP", "DX-Y.NYB"],
    "international": ["EFA", "EEM", "FXI"],
    "volatility": ["^VIX", "^VIX3M", "^MOVE", "SVXY", "^SKEW"],
}

INSURERS = ["HIG", "ALL", "TRV", "AIG", "MET", "PGR", "CB", "AFL", "PRU", "LNC"]
RETAIL = ["TJX", "ROST", "TGT", "HD", "LOW", "WMT", "M", "KSS", "BBY", "DG"]

print("=" * 72)
print("1. EVENT WINDOW (entry MOC 2026-08-14 = anchor+1; legal exits <= +10 td)")
print("=" * 72)
ev = load_events()
w = ev[(ev["date"] >= "2026-08-01") & (ev["date"] <= "2026-09-10")]
print(w.to_string(index=False))

print("\n" + "=" * 72)
print("2. LEVEL vs RANK for the vol complex (the 2026-08-10 MOVE trap)")
print("=" * 72)
allt = sorted({t for v in CLASSES.values() for t in v} | set(INSURERS) | set(RETAIL)
              | {"XLF", "KRE", "SMH", "XBI", "XLV", "XLK", "XLY", "XLP", "XLU",
                 "XLI", "XLB", "XLC", "XLRE", "XOP", "NVDA", "GOOG", "AAPL",
                 "META", "MSFT", "MU", "INTC", "ADI"})
px = load_prices(allt)

for t in ["^VIX", "^VIX3M", "^MOVE", "^SKEW"]:
    if t not in px:
        continue
    c = px[t]["Close"].dropna()
    c = c[c.index <= ASOF]
    lvl_pct = 100 * (c.iloc[-1] > c.iloc[-252:]).mean()
    r21 = pct_rank(c, 21).iloc[-1]
    r5 = pct_rank(c, 5).iloc[-1]
    print(f"  {t:8s} last {c.iloc[-1]:9.2f}  LEVEL pctile(252d) {lvl_pct:5.1f}   "
          f"ret21 rank {r21:5.1f}   ret5 rank {r5:5.1f}   "
          f"ret21 {100*(c.iloc[-1]/c.iloc[-22]-1):+6.2f}%")

print("\n" + "=" * 72)
print("3. INSURANCE COMPLEX — how synchronized is the washout?")
print("=" * 72)
rows = []
for t in INSURERS:
    if t not in px:
        continue
    c = px[t]["Close"].dropna()
    c = c[c.index <= ASOF]
    if len(c) < 300:
        continue
    rows.append({
        "tkr": t,
        "ret5_pct": 100 * (c.iloc[-1] / c.iloc[-6] - 1),
        "rank5": pct_rank(c, 5).iloc[-1],
        "rank21": pct_rank(c, 21).iloc[-1],
        "rank63": pct_rank(c, 63).iloc[-1],
        "dist200_pct": 100 * (c.iloc[-1] / c.iloc[-200:].mean() - 1),
    })
print(pd.DataFrame(rows).round(2).to_string(index=False))

print("\n" + "=" * 72)
print("4. RETAIL into next week's prints")
print("=" * 72)
earn = pd.read_parquet("data/earnings_calendar.parquet")
earn["date"] = pd.to_datetime(earn["date"] if "date" in earn else earn.iloc[:, 1])
nxt = earn[(earn["date"] >= "2026-08-14") & (earn["date"] <= "2026-08-31")]
rows = []
for t in RETAIL:
    if t not in px:
        continue
    c = px[t]["Close"].dropna()
    c = c[c.index <= ASOF]
    if len(c) < 300:
        continue
    d = nxt[nxt["ticker"] == t]["date"]
    rows.append({
        "tkr": t,
        "print": str(d.iloc[0].date()) if len(d) else "-",
        "ret5_pct": 100 * (c.iloc[-1] / c.iloc[-6] - 1),
        "rank5": pct_rank(c, 5).iloc[-1],
        "rank21": pct_rank(c, 21).iloc[-1],
    })
print(pd.DataFrame(rows).round(2).to_string(index=False))

print("\n" + "=" * 72)
print("5. 63d WASHOUT + SHORT-TERM THRUST cross-asset cluster")
print("=" * 72)
rows = []
for cls, tks in CLASSES.items():
    for t in tks:
        if t not in px:
            continue
        c = px[t]["Close"].dropna()
        c = c[c.index <= ASOF]
        if len(c) < 300:
            continue
        rows.append({
            "class": cls, "tkr": t,
            "rank5": pct_rank(c, 5).iloc[-1],
            "rank21": pct_rank(c, 21).iloc[-1],
            "rank63": pct_rank(c, 63).iloc[-1],
            "ret63_pct": 100 * (c.iloc[-1] / c.iloc[-64] - 1),
        })
df = pd.DataFrame(rows).round(1)
print(df.sort_values("rank63").to_string(index=False))

print("\n" + "=" * 72)
print("6. SEMIS / NVDA into the 2026-08-26 print")
print("=" * 72)
for t in ["SMH", "NVDA", "QQQ", "MU", "INTC", "ADI"]:
    if t not in px:
        continue
    c = px[t]["Close"].dropna()
    c = c[c.index <= ASOF]
    print(f"  {t:6s} rank5 {pct_rank(c,5).iloc[-1]:5.1f}  rank21 "
          f"{pct_rank(c,21).iloc[-1]:5.1f}  rank63 {pct_rank(c,63).iloc[-1]:5.1f}  "
          f"ret63 {100*(c.iloc[-1]/c.iloc[-64]-1):+6.2f}%  "
          f"dist52wh {100*(c.iloc[-1]/c.iloc[-252:].max()-1):+6.2f}%")

print("\n" + "=" * 72)
print("7. WATCHLIST triggers — today's values")
print("=" * 72)
tlt = px["TLT"]["Close"].dropna(); tlt = tlt[tlt.index <= ASOF]
ief = px["IEF"]["Close"].dropna(); ief = ief[ief.index <= ASOF]
lqd = px["LQD"]["Close"].dropna(); lqd = lqd[lqd.index <= ASOF]
hyg = px["HYG"]["Close"].dropna(); hyg = hyg[hyg.index <= ASOF]


def dist_low(s):
    return 100 * (s.iloc[-1] / s.iloc[-252:].min() - 1)


def dist_high(s):
    return 100 * (s.iloc[-1] / s.iloc[-252:].max() - 1)


print(f"  W6 tight rung: TLT {dist_low(tlt):.2f}% off 52wl (need <=0.5), "
      f"IEF {dist_low(ief):.2f}% (<=1.0), LQD {dist_low(lqd):.2f}% (<=1.0)")
tight = ((tlt / tlt.rolling(252).min() - 1 <= 0.005)
         & (ief / ief.rolling(252).min() - 1 <= 0.010)
         & (lqd / lqd.rolling(252).min() - 1 <= 0.010))
fired = tight[tight].index
recent = [d for d in fired if d >= ASOF - pd.Timedelta(days=60)]
print(f"     episode depth: trigger days in last 60cd = {len(recent)}, "
      f"first = {recent[0].date() if recent else '-'}")
print(f"  W2 credit divergence: HYG {dist_high(hyg):+.2f}% off 52wh (need >=-0.5), "
      f"LQD {dist_low(lqd):.2f}% off 52wl (need <=2.0)")
gld = px["GLD"]["Close"].dropna(); gld = gld[gld.index <= ASOF]
gdx = px["GDX"]["Close"].dropna(); gdx = gdx[gdx.index <= ASOF]
print(f"  W4 GLD/GDX divergence: GDX rank5 {pct_rank(gdx,5).iloc[-1]:.1f} (need >=95), "
      f"GLD rank5 {pct_rank(gld,5).iloc[-1]:.1f} (need <95)")
uso = px["USO"]["Close"].dropna(); uso = uso[uso.index <= ASOF]
print(f"  W5 XLE crude pop: USO 1d {100*(uso.iloc[-1]/uso.iloc[-2]-1):+.2f}% "
      f"(need +5..6% and >=1.5 ATR)")
skew = px["^SKEW"]["Close"].dropna(); skew = skew[skew.index <= ASOF]
spy = px["SPY"]["Close"].dropna(); spy = spy[spy.index <= ASOF]
print(f"  W7 skew spike: ^SKEW rank5 {pct_rank(skew,5).iloc[-1]:.1f} (need >=95), "
      f"SPY {dist_high(spy):+.2f}% off 52wh (need <-1), midterm=YES (need NO)")
fxi = px["FXI"]["Close"].dropna(); fxi = fxi[fxi.index <= ASOF]
eem = px["EEM"]["Close"].dropna(); eem = eem[eem.index <= ASOF]
print(f"  W10 FXI break: FXI rank5 {pct_rank(fxi,5).iloc[-1]:.1f} (need <=20), "
      f"FXI rank21 {pct_rank(fxi,21).iloc[-1]:.1f} (need >=80), "
      f"EEM ret5 {100*(eem.iloc[-1]/eem.iloc[-6]-1):+.2f}% (need >0)")
ihi_ok = "IHI" not in px
print(f"  W9 IHI: reference-class gate, requires Cochran Q p<0.05 across 27 ETFs "
      f"(measured 0.544 on 2026-08-13) -- structural, cannot flip in one session")
