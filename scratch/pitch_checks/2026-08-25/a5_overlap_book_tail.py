"""Mutual overlap between C1/C6/C9/C10, book overlap, and tomorrow's tail risk.

Three questions the four candidates share and that no single-candidate script
can answer:

  1. Are C1, C6, C9 and C10 one trade?  Leg-return correlations over the hold,
     not prose.
  2. Book overlap: the scan staged OLV LONGS in D, ETR (utilities), AMKR, ON,
     POWI, WWD (semis/industrial) and OVS SHORTS in AMGN, CAG, CMCSA, DIS,
     GIS, KO (the defensive winners).  C1/C6 duplicate the semis longs and
     C10 duplicates the defensive shorts.  Quantify with beta, not adjectives.
  3. Tail risk inside the hold: NVDA reports 2026-08-26 (+1 td) and Jackson
     Hole is 2026-08-28 (+3 td).  Both land inside anything with h>=3.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

pd.set_option("display.width", 240)

SECT = ["XLK", "XLV", "XLP", "XLU", "XLI", "XLF", "XLY", "XLE", "XLB"]
OLV_LONG = ["D", "ETR", "AMKR", "ON", "POWI", "WWD"]
OVS_SHORT = ["AMGN", "CAG", "CMCSA", "DIS", "GIS", "KO"]
TK = SECT + ["SPY", "SMH", "NVDA", "QQQ"] + OLV_LONG + OVS_SHORT
px = close_panel(TK)
r = px.pct_change()

# ---------------------------------------------------------------- 1. mutual
print("########## 1. MUTUAL OVERLAP - are the four one trade? ##########")
H = 5
fw = {t: fwd_lag(px[t], H) for t in TK}
veh = pd.DataFrame({
    "C1 long XLK": fw["XLK"],
    "C6 long XLK+XLI / short XLV+XLP": 0.5 * fw["XLK"] + 0.5 * fw["XLI"]
                                       - 0.5 * fw["XLV"] - 0.5 * fw["XLP"],
    "C9 long XLI": fw["XLI"],
    "C10 short XLV": -fw["XLV"],
}).dropna()
print(f"\n  h={H} overlapping-window return correlations (full history, N={len(veh)}):")
print((veh.corr()).round(3).to_string())
recent = veh.loc["2024-01-01":]
print(f"\n  same, 2024+ only (N={len(recent)}):")
print(recent.corr().round(3).to_string())

print("\n  pairwise OLS beta of each candidate on C1 (long XLK), h=5 full history:")
x = veh["C1 long XLK"].values
for c in veh.columns[1:]:
    y = veh[c].values
    b = np.polyfit(x, y, 1)
    resid = y - (b[0] * x + b[1])
    print(f"    {c:<34} beta {b[0]:+.3f}   R2 "
          f"{1 - resid.var()/y.var():.3f}   resid mean {100*resid.mean():+.3f}%")

print("\n  TODAY the four candidates trade these names:")
print("    C1  : long XLK")
print("    C6  : long XLK + XLI, short XLV + XLP")
print("    C9  : long XLI")
print("    C10 : short XLV")
print("  Union = {XLK, XLI, XLV, XLP}. C6 is literally C1 + C9 long and C10 short")
print("  plus XLP. There is ONE object here, expressed four ways.")

# ---------------------------------------------------------------- 2. book
print("\n\n########## 2. BOOK OVERLAP (staged this morning) ##########")
win = r.loc["2025-08-25":].dropna(how="any")
print(f"  daily-return window {win.index[0].date()} -> {win.index[-1].date()} (N={len(win)})")


def rel(basket: list[str], proxy: str, label: str) -> None:
    b = win[basket].mean(axis=1)
    p = win[proxy]
    beta = np.polyfit(p.values, b.values, 1)[0]
    c = np.corrcoef(p.values, b.values)[0, 1]
    print(f"  {label:<52} corr {c:+.3f}  beta {beta:+.3f}  R2 {c**2:.3f}")


rel(["AMKR", "ON", "POWI"], "XLK", "OLV semis longs (AMKR/ON/POWI) vs XLK")
rel(["AMKR", "ON", "POWI"], "SMH", "OLV semis longs vs SMH")
rel(["WWD"], "XLI", "OLV industrial long (WWD) vs XLI")
rel(["D", "ETR"], "XLU", "OLV utility longs (D/ETR) vs XLU")
rel(OVS_SHORT, "XLV", "OVS defensive shorts (6 names) vs XLV")
rel(OVS_SHORT, "XLP", "OVS defensive shorts vs XLP")
rel(["AMGN"], "XLV", "  AMGN alone vs XLV")
rel(["KO", "GIS", "CAG"], "XLP", "  KO/GIS/CAG vs XLP")
print("\n  Reading: C1/C6's long tech leg is already on the book through three OLV")
print("  semis longs, and C10's short XLV is the SAME DIRECTIONAL BET as six")
print("  staged OVS shorts in defensive names. Both would concentrate, not diversify.")

# ---------------------------------------------------------------- 3. tail
print("\n\n########## 3. TAIL RISK IN THE HOLD - NVDA tomorrow, Jackson Hole Friday ##########")
earn = pd.read_parquet("data/earnings_calendar.parquet")
col = "date" if "date" in earn.columns else earn.columns[0]
nv = earn[earn["ticker"] == "NVDA"] if "ticker" in earn.columns else pd.DataFrame()
if len(nv):
    d = pd.to_datetime(nv[col]).dropna()
    d = d[(d >= px.index[0]) & (d <= px.index[-1])]
    d = pd.DatetimeIndex(sorted(set(d.dt.normalize())))
    print(f"  NVDA prints in the price span: {len(d)}")
    # the session AFTER the print (report is after the close -> next session reacts)
    pos = pd.Series(range(len(px.index)), index=px.index)
    react = []
    for dt in d:
        p = pos.get(dt)
        if p is None or p + 1 >= len(px.index):
            continue
        react.append((px.index[p + 1], r["XLK"].iloc[p + 1], r["SMH"].iloc[p + 1],
                      r["XLV"].iloc[p + 1], r["XLI"].iloc[p + 1]))
    R = pd.DataFrame(react, columns=["date", "XLK", "SMH", "XLV", "XLI"]).dropna()
    print(f"  reaction-session moves (N={len(R)}):")
    for c in ["XLK", "SMH", "XLV", "XLI"]:
        v = R[c].values
        print(f"    {c}: mean {100*v.mean():+.3f}%  sd {100*v.std(ddof=1):.3f}%  "
              f"worst {100*v.min():+.2f}%  best {100*v.max():+.2f}%  "
              f"|move|>2% on {100*(np.abs(v)>0.02).mean():.0f}% of prints")
        u = r[c].dropna()
        print(f"       vs unconditional sd {100*u.std(ddof=1):.3f}%  -> "
              f"{v.std(ddof=1)/u.std(ddof=1):.2f}x normal")
    R2 = R[R["date"] >= "2020-01-01"]
    print(f"  2020+ only (N={len(R2)}): XLK sd {100*R2['XLK'].std(ddof=1):.3f}%, "
          f"worst {100*R2['XLK'].min():+.2f}%; SMH sd {100*R2['SMH'].std(ddof=1):.3f}%, "
          f"worst {100*R2['SMH'].min():+.2f}%")
else:
    print("  NVDA rows not found in the earnings parquet under the expected schema")

print("\n  Jackson Hole: the registry has closed this anchor on seven asset classes")
print("  (pre-speech class mean +0.010pp, 2026-08-24 re-sweep). It is not an edge,")
print("  but a h>=3 hold carries the speech as pure variance.")
jh = load_events(["jackson_hole"])["date"]
jh = pd.DatetimeIndex([d for d in jh if px.index[0] <= d <= px.index[-1]])
pos = pd.Series(range(len(px.index)), index=px.index)
moves = []
for dt in jh:
    p = pos.get(dt)
    if p is None:
        near = px.index[px.index.searchsorted(dt)]
        p = pos.get(near)
    if p is None or p >= len(px.index):
        continue
    moves.append((px.index[p], r["XLK"].iloc[p], r["XLV"].iloc[p]))
M = pd.DataFrame(moves, columns=["date", "XLK", "XLV"]).dropna()
print(f"  speech-session moves (N={len(M)}): XLK mean {100*M['XLK'].mean():+.3f}% "
      f"sd {100*M['XLK'].std(ddof=1):.3f}% worst {100*M['XLK'].min():+.2f}%  |  "
      f"XLV mean {100*M['XLV'].mean():+.3f}% sd {100*M['XLV'].std(ddof=1):.3f}%")
