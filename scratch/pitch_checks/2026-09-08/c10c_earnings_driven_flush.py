"""C10c -- the live instance is a POST-EARNINGS flush, which is the opposite
mechanism to the one the long side needs.

CPB reported 2026-09-03 and is -8.59% over five sessions; SJM reported
08-26; HRL 08-27. GIS (-7.85%) reports 09-23 and MKC 10-01, so those two are
clean. Question: does the historical cell pay the same when the flushed
members just printed as when they did not? Post-earnings-announcement drift
says a fresh earnings loser keeps drifting; short-term reversal says it
bounces. They cannot both be true of the same basket.

Coverage caveat printed below: data/earnings_calendar.parquet does not run
the full 2001-2026 trigger span, so this split is measured over the years it
does cover and is quoted as such.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

warnings.filterwarnings("ignore")

FOOD = ["CPB", "GIS", "TSN", "HRL", "SYY", "CAG", "SJM", "MKC", "KR", "HSY"]
RANK_MAX, N_MIN, XLP_FLOOR, H = 10.0, 3, 25.0, 5

PXD = load_prices(FOOD + ["XLP", "SPY"])
IDX = PXD["SPY"].index
C = pd.DataFrame({t: PXD[t]["Close"] for t in PXD}).reindex(IDX)
R5 = pd.DataFrame({t: pct_rank(C[t], 5) for t in C.columns})
F5 = pd.DataFrame({t: fwd_lag(C[t], H, 1) for t in C.columns})

M = (R5[FOOD] <= RANK_MAX).fillna(False) & C[FOOD].notna()
v = F5[FOOD].where(M).mean(axis=1)
gate = (R5["XLP"] >= XLP_FLOOR).fillna(False)
trig = IDX[((M.sum(axis=1) >= N_MIN) & gate).values]

ec = pd.read_parquet(ROOT / "data" / "earnings_calendar.parquet")
ec["date"] = pd.to_datetime(ec["date"])
ec = ec[ec["ticker"].isin(FOOD)]
cov = ec.groupby("ticker")["date"].agg(["min", "max", "count"])
print("earnings coverage for the group:")
print(cov.to_string())
FIRST = cov["min"].max()
print(f"\n  the split below is only meaningful from {FIRST.date()} onward "
      f"(the latest per-name coverage start)")

# per name, was there a print in the 5 sessions ENDING on the signal day?
pos = pd.Series(range(len(IDX)), index=IDX)
edates = {t: pd.DatetimeIndex(sorted(ec[ec.ticker == t]["date"].unique()))
          for t in FOOD}
rows = []
valid = v.dropna().index
t_all = pd.DatetimeIndex(trig).intersection(valid)
epi = declusters(t_all, H, valid)
for d in epi:
    if d < FIRST:
        continue
    p = pos[d]
    lo = IDX[max(0, p - 5)]
    flushed = [t for t in FOOD if bool(M.loc[d, t])]
    n_earn = sum(1 for t in flushed
                 if ((edates[t] > lo) & (edates[t] <= d)).any())
    rows.append({"date": d, "ret": float(v.loc[d]), "n_flushed": len(flushed),
                 "n_earn": n_earn, "frac": n_earn / max(1, len(flushed))})
df = pd.DataFrame(rows)
print(f"\n  {len(df)} episodes inside earnings coverage "
      f"({df.date.min().date()} .. {df.date.max().date()})")
for lbl, sub in (("NO flushed member printed in the prior 5 sessions",
                  df[df.n_earn == 0]),
                 ("AT LEAST ONE flushed member printed", df[df.n_earn >= 1]),
                 ("HALF OR MORE of the flushed members printed",
                  df[df.frac >= 0.5])):
    if len(sub) == 0:
        print(f"  {lbl}: 0 episodes")
        continue
    x = sub["ret"].values
    w = int((x > 0).sum())
    print(f"  {lbl}: n {len(x):3d}  mean {100*x.mean():+.3f}%  "
          f"rec {w}-{len(x)-w}  sign p {sign_test(w, len(x)):.4f}  "
          f"worst {100*x.min():+.2f}%")
print("\n  LIVE 2026-09-08: flushed = CPB GIS TSN SJM MKC; CPB printed 09-03 "
      "(inside 5 sessions), SJM 08-26 and HRL 08-27 are outside, GIS 09-23 and "
      "MKC 10-01 are ahead -> 1 of 5 is a fresh earnings loser, and it is the "
      "deepest one.")
print("  No group member prints inside a 5-session hold from 2026-09-08 "
      "(KR 09-11 is in the group but NOT in the flush at r5 68.3).")
