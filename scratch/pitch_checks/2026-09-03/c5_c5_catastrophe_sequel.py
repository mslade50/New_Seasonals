"""C5 round 1: the two-name California utility catastrophe.

PCG -26.84% and EIX -25.93% over five sessions, both ~-30% off their 52-week
highs, while XLU is -1.93% and no other utility sits below a 5-day rank of 22.

Nearest-neighbour question: across the whole ~1120-name cache, what happens in
the 1 to 10 sessions after a >= 20% five-day collapse in a name whose SECTOR is
untouched? Split utilities against everything else. Report the LEFT TAIL
honestly -- a name that falls 26% in a week can fall another 26%.

Honest framing this owes and cannot escape:
  - the driver is wildfire / legal liability news that exists in no series here
  - master_prices holds only names alive in TODAY's universe, so every analogue
    that went to zero is missing. The measured sequel is an UPPER BOUND.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import ROOT, load_prices, show, summarize, sign_test

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)

DROP = -0.20
HS = (1, 2, 3, 5, 10)
MIN_GAP = 10          # sessions between kept episodes, per ticker

print("=" * 78)
print("C5  sequel to a >= 20% five-day collapse, universe-wide")
print("=" * 78)

mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet",
                     columns=["date", "ticker", "Close"])
mp["date"] = pd.to_datetime(mp["date"])
W = mp.pivot_table(index="date", columns="ticker", values="Close", aggfunc="last")
W = W.sort_index()
print(f"panel {W.shape[0]} sessions x {W.shape[1]} tickers  "
      f"{W.index[0].date()} .. {W.index[-1].date()}")

sec = pd.read_parquet(ROOT / "data" / "sector_map.parquet").set_index("ticker")["sector"]
UTIL = set(sec[sec == "Utilities"].index)
print(f"sector map: {len(sec)} names, {len(UTIL)} utilities; "
      f"PCG={sec.get('PCG')}  EIX={sec.get('EIX')}")

r5 = W / W.shift(5) - 1.0
fwd = {h: W.shift(-(1 + h)) / W.shift(-1) - 1.0 for h in HS}   # lag=1 entry
TRIG = (r5 <= DROP)
print(f"\nraw trigger cells (>= {abs(DROP):.0%} 5-day drop): {int(TRIG.values.sum())}")

# sector-untouched conditioner: the name's SECTOR MEDIAN 5d return that day
sector_of = {t: sec.get(t, "UNKNOWN") for t in W.columns}
sec_series = pd.Series(sector_of)
sec_med = {}
for s in sorted(set(sec_series.values)):
    cols = sec_series[sec_series == s].index
    cols = [c for c in cols if c in r5.columns]
    if len(cols) >= 5:
        sec_med[s] = r5[cols].median(axis=1)
SECMED = pd.DataFrame({t: sec_med.get(sector_of[t], pd.Series(np.nan, index=r5.index))
                       for t in W.columns})
UNTOUCHED = SECMED > -0.05      # peers down less than 5% over the same five days

# ---- collect episodes ------------------------------------------------------
recs = []
pos = pd.Series(range(len(W.index)), index=W.index)
for t in W.columns:
    col = TRIG[t]
    d = W.index[col.fillna(False).values]
    if len(d) == 0:
        continue
    last = -10 ** 9
    for x in d:
        p = int(pos[x])
        if p - last < MIN_GAP:
            continue
        last = p
        rec = {"ticker": t, "date": x, "sector": sector_of[t],
               "is_util": t in UTIL,
               "r5": float(r5.at[x, t]),
               "sec_med5": float(SECMED.at[x, t]) if not pd.isna(SECMED.at[x, t]) else np.nan}
        for h in HS:
            rec[f"h{h}"] = float(fwd[h].at[x, t])
        recs.append(rec)
E = pd.DataFrame(recs)
E["untouched"] = E["sec_med5"] > -0.05
print(f"declustered episodes (min gap {MIN_GAP} td): {len(E)}  "
      f"tickers {E['ticker'].nunique()}  {E['date'].min().date()} .. {E['date'].max().date()}")
print(f"  utilities: {int(E['is_util'].sum())}   non-utilities: {int((~E['is_util']).sum())}")
print(f"  with an UNTOUCHED sector (peer median 5d > -5%): {int(E['untouched'].sum())}")

# ---- the sequel, honestly --------------------------------------------------
def block(sub: pd.DataFrame, label: str) -> list[dict]:
    rows = []
    for h in HS:
        v = sub[f"h{h}"].dropna().values
        r = summarize(v, f"{label} h={h}")
        if r["n"]:
            r["p05_pct"] = round(100 * float(np.percentile(v, 5)), 2)
            r["p25_pct"] = round(100 * float(np.percentile(v, 25)), 2)
            r["lose>10%"] = round(100 * float((v <= -0.10).mean()), 1)
            r["lose>20%"] = round(100 * float((v <= -0.20).mean()), 1)
            w = int((v > 0).sum())
            r["sign_p"] = round(sign_test(w, len(v)), 4)
        rows.append(r)
    return rows


show(block(E, "ALL"), "sequel: every >=20% 5-day collapse in the cache")
show(block(E[E["is_util"]], "UTIL"), "sequel: utilities only")
show(block(E[~E["is_util"]], "NON-UTIL"), "sequel: everything else")
show(block(E[E["untouched"]], "UNTOUCHED-SECTOR"),
     "sequel: collapse with the sector untouched (the LIVE conditioner)")
show(block(E[E["untouched"] & E["is_util"]], "UTIL+UNTOUCHED"),
     "sequel: utilities whose sector is untouched (the exact live cell)")
show(block(E[~E["untouched"]], "SECTOR-ALSO-HIT"),
     "sequel: collapse with the sector also down >5% (the complement)")

# ---- the unconditional control on the same names ---------------------------
print("\n" + "=" * 78)
print("CONTROL: unconditional forward returns on the SAME tickers, all days")
print("=" * 78)
names = sorted(E["ticker"].unique())
rows = []
for h in HS:
    v = fwd[h][names].values.ravel()
    v = v[~np.isnan(v)]
    r = summarize(v, f"all days, {len(names)} names h={h}")
    r["p05_pct"] = round(100 * float(np.percentile(v, 5)), 2)
    r["lose>10%"] = round(100 * float((v <= -0.10).mean()), 1)
    r["lose>20%"] = round(100 * float((v <= -0.20).mean()), 1)
    rows.append(r)
show(rows, "unconditional baseline")

# ---- era stability ---------------------------------------------------------
print("\n" + "=" * 78)
print("ERA SPLIT on the live cell (untouched sector)")
print("=" * 78)
U = E[E["untouched"]]
for cut, lbl in ((2018, "2018"),):
    a, b = U[U["date"].dt.year < cut], U[U["date"].dt.year >= cut]
    rows = []
    for h in (3, 5, 10):
        rows.append(summarize(a[f"h{h}"].dropna().values, f"pre-{lbl} h={h}"))
        rows.append(summarize(b[f"h{h}"].dropna().values, f"{lbl}+  h={h}"))
    show(rows, "era split, untouched-sector cell")

# ---- what did the two live names themselves do, historically? -------------
print("\n" + "=" * 78)
print("PCG / EIX own history of >=20% five-day collapses")
print("=" * 78)
own = E[E["ticker"].isin(["PCG", "EIX"])]
if len(own):
    print(own[["ticker", "date", "r5", "sec_med5"] + [f"h{h}" for h in HS]]
          .assign(**{c: lambda d, c=c: (100 * d[c]).round(2)
                     for c in ["r5", "sec_med5"] + [f"h{h}" for h in HS]})
          .to_string(index=False))
else:
    print("  (none)")
for t in ("PCG", "EIX"):
    s = W[t].dropna()
    print(f"{t}: last {s.iloc[-1]:.2f}  5d {100*(s.iloc[-1]/s.iloc[-6]-1):+.2f}%  "
          f"252d max {s.rolling(252).max().iloc[-1]:.2f}  "
          f"dist {100*(s.iloc[-1]/s.rolling(252).max().iloc[-1]-1):+.2f}%")

# ---- SURVIVORSHIP: the bias that makes all of the above an upper bound -----
print("\n" + "=" * 78)
print("SURVIVORSHIP CHECK -- how many analogue names are STILL quoted today?")
print("=" * 78)
last_bar = W.index[-1]
alive = {t: bool(W[t].dropna().index[-1] >= last_bar - pd.Timedelta(days=7))
         for t in names}
n_alive = sum(alive.values())
print(f"{n_alive} of {len(names)} analogue tickers still print a bar in the last "
      f"week ({100*n_alive/len(names):.1f}%).")
print("The cache universe IS today's universe (CLAUDE.md survivorship caveat), so "
      "a name that collapsed and then delisted at zero contributes NO episode. "
      "Every sequel number above is therefore an upper bound on the real one.")

# ---- BOOK OVERLAP ----------------------------------------------------------
print("\n" + "=" * 78)
print("BOOK OVERLAP -- does the scanner already harvest this?")
print("=" * 78)
led = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
led["Signal Date"] = pd.to_datetime(led["Signal Date"])
key = set(zip(E["ticker"], E["date"]))
# scanner signals within [0, +5] td of a collapse in the SAME ticker
posn = pd.Series(range(len(W.index)), index=W.index)
hits = []
for _, row in led.iterrows():
    t, d = row["Ticker"], row["Signal Date"]
    if t not in TRIG.columns or d not in posn.index:
        continue
    p = int(posn[d])
    lo = max(0, p - 5)
    seg = TRIG[t].iloc[lo:p + 1]
    if bool(seg.fillna(False).any()):
        hits.append(row["Strategy"])
print(f"ledger rows whose signal date sits within 5 td AFTER a >=20% 5-day "
      f"collapse in the same ticker: {len(hits)} of {len(led)}")
print(pd.Series(hits).value_counts().to_string() if hits else "  (none)")
print("\nledger rows on PCG / EIX ever:",
      int(led["Ticker"].isin(["PCG", "EIX"]).sum()))
print(led[led["Ticker"].isin(["PCG", "EIX"])][["Strategy", "Ticker", "Signal Date",
                                               "R_Multiple"]].to_string(index=False)
      if led["Ticker"].isin(["PCG", "EIX"]).any() else "  (none)")

print("\nDONE C5")
