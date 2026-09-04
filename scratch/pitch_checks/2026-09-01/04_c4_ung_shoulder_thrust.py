"""C4 KILL CHECK — natural gas THRUSTING into the September shoulder season.

Live: UNG z10 +1.34, r5 76.2, +3.84% over 5 sessions, but -37.63% below its
52-week high and -10.75% below its 200-day.

The registry's UNG kill is about a DIFFERENT state (long at a 52-week low), so
this candidate is legitimate — but its numbers are mandatory here and are
charged in ABSOLUTE terms:
  "UNG's structural bleed is -0.90%/10 td and -28.65%/yr (buy-and-hold -99.85%
   over 19.3 years). A cell that is positive in EXCESS while negative in
   ABSOLUTE is dead for an outright long."
So every table below reports the ABSOLUTE mean first and the excess second.

Three separable legs, each tested alone (gate attribution both ways):
  A. the THRUST leg (z10 / r5), all months
  B. the SEPTEMBER / shoulder leg, no thrust
  C. the conjunction, which is the candidate
If the seasonal leg does not filter, nothing may be attributed to it.

Plus the futures question: does NG=F escape the ETF's roll decay? The repo's
own memory warns continuous futures roll gaps fire price-state triggers as
fake moves, so the series is checked for roll gaps BEFORE it is believed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import pandas as pd
import numpy as np

ASOF = pd.Timestamp("2026-08-31")
PX = load_prices(["UNG", "NG=F", "SPY", "USO", "DBC"])
PX = {t: d[d.index <= ASOF] for t, d in PX.items()}
ung = PX["UNG"]["Close"].dropna()
ngf = PX["NG=F"]["Close"].dropna()

# ---------------------------------------------------------------------------
# 0. CHARGE THE BLEED, in absolute terms, before anything else
# ---------------------------------------------------------------------------
print("===== 0. THE BLEED, CHARGED FIRST (UNG all days, lag=1) =====")
for H in (5, 10, 21):
    r = fwd_lag(ung, H, 1)
    s = summarize(r.dropna().values, f"UNG all days h={H}")
    print("  h=%2d  N=%d  ABSOLUTE mean %+.3f%%  median %+.3f%%  hit %.1f%%"
          % (H, s["n"], s["mean_pct"], s["median_pct"], s["hit"]))
print("  UNG buy-and-hold since %s: %+.2f%%  (%.2f -> %.2f)"
      % (ung.index[0].date(), 100 * (ung.iloc[-1] / ung.iloc[0] - 1),
         ung.iloc[0], ung.iloc[-1]))
print("  NG=F buy-and-hold over the same span: %+.2f%%"
      % (100 * (ngf.reindex(ung.index).ffill().iloc[-1]
                / ngf.reindex(ung.index).ffill().dropna().iloc[0] - 1)))

# ---------------------------------------------------------------------------
# 0b. ROLL-GAP AUDIT on the continuous futures series before believing it
# ---------------------------------------------------------------------------
print("\n===== 0b. NG=F ROLL-GAP AUDIT (repo memory: roll gaps fire price-state "
      "triggers as fake moves) =====")
ngr = ngf.pct_change()
big = ngr[ngr.abs() > 0.08]
print("  NG=F daily |move| > 8%%: %d of %d sessions (%.2f%%)"
      % (len(big), len(ngr.dropna()), 100 * len(big) / len(ngr.dropna())))
o = PX["NG=F"]["Open"].reindex(ngf.index)
overnight = (o / ngf.shift(1) - 1.0).dropna()
print("  NG=F overnight |gap| > 5%%: %d sessions; largest %+.1f%% on %s"
      % (int((overnight.abs() > 0.05).sum()), 100 * overnight.abs().max(),
         overnight.abs().idxmax().date()))
mm = overnight.groupby(overnight.index.day).mean()
print("  mean overnight gap by day-of-month (roll clustering shows up here):")
print("   ", {int(k): round(100 * v, 3) for k, v in mm.items()})
print("  VERDICT ON THE FUTURES SERIES: the continuous series carries "
      "unadjusted roll steps; any 'thrust' measured on it is partly the roll. "
      "It is used below ONLY as a directional cross-check, never as the vehicle.")

# ---------------------------------------------------------------------------
# 1. COUNT FIRST + the three legs
# ---------------------------------------------------------------------------
z10 = zscore(ung, 10)
r5 = pct_rank(ung, 5)
dd = ung / ung.rolling(252, min_periods=252).max() - 1.0
print("\n===== 1. LIVE STATE + COUNT FIRST =====")
print("  UNG z10 %+.2f  r5 %.1f  dd %.2f%%  month=9" % (
    z10.iloc[-1], r5.iloc[-1], 100 * dd.iloc[-1]))

THRUST = ((z10 >= 1.0) & (r5 >= 75)).fillna(False)
SEP = pd.Series(ung.index.month == 9, index=ung.index)
SHOULDER = pd.Series(ung.index.month.isin([9, 10]), index=ung.index)
CELL = (THRUST & SEP).fillna(False)
CELL_SH = (THRUST & SHOULDER).fillna(False)
DEEP = (CELL & (dd <= -0.25)).fillna(False)

for lbl, m in (("A. THRUST alone (z10>=1 & r5>=75), all months", THRUST),
               ("B. SEPTEMBER alone (no thrust)", SEP),
               ("B2. SEP+OCT shoulder alone", SHOULDER),
               ("C. THRUST & September (THE CELL)", CELL),
               ("C2. THRUST & Sep/Oct shoulder", CELL_SH),
               ("D. CELL & dd<=-25% (today's depth)", DEEP)):
    t = ung.index[m.reindex(ung.index, fill_value=False).values]
    print("  %-46s %4d days | %2d ep(gap10) | %2d ep(gap21) | yrs %s"
          % (lbl, len(t), len(declusters(t, 10, ung.index)),
             len(declusters(t, 21, ung.index)),
             sorted(set(t.year))))

# ---------------------------------------------------------------------------
# 2. ABSOLUTE-FIRST table for every leg
# ---------------------------------------------------------------------------
print("\n===== 2. ABSOLUTE FIRST, EXCESS SECOND (UNG long, lag=1, episodes gap 10) =====")
for H in (3, 5, 10, 21):
    r = fwd_lag(ung, H, 1)
    drift = r.dropna().mean()
    rows = []
    for lbl, m in (("A THRUST all months", THRUST),
                   ("B SEPTEMBER alone", SEP),
                   ("B2 Sep/Oct shoulder alone", SHOULDER),
                   ("C THRUST & Sep (CELL)", CELL),
                   ("C2 THRUST & shoulder", CELL_SH),
                   ("D CELL & dd<=-25%", DEEP)):
        t = ung.index[m.reindex(ung.index, fill_value=False).values]
        t = t.intersection(r.dropna().index)
        e = declusters(t, 10, ung.index)
        s = summarize(r.reindex(e).values, lbl)
        if s["n"]:
            s["excess_pp"] = round(s["mean_pct"] - 100 * drift, 3)
            wins = int((r.reindex(e).values > 0).sum())
            s["sign_p"] = round(sign_test(wins, s["n"]), 4)
            s["n_days"] = len(t)
        rows.append(s)
    rows.append(summarize(r.dropna().values, "ALL DAYS (the bleed)"))
    show(rows, f"h={H}  (ABSOLUTE mean_pct is the number that decides an "
              f"outright long)")

# ---------------------------------------------------------------------------
# 3. the battery on the cell
# ---------------------------------------------------------------------------
px_ung = pd.DataFrame({"UNG": ung})
print("\n===== 3. HORIZON SCAN =====")
t_cell = ung.index[CELL.reindex(ung.index, fill_value=False).values]
show(horizon_scan(px_ung, t_cell, [("UNG", 1.0)], hs=(1, 2, 3, 5, 10, 21), min_gap=10),
     "horizon scan, THE CELL (episodes gap 10)")

variants = {
    "z10>=0.75 & r5>=70 & Sep": ((z10 >= 0.75) & (r5 >= 70) & SEP).fillna(False),
    "z10>=1.0 & r5>=75 & Sep (CELL)": CELL,
    "z10>=1.5 & r5>=85 & Sep": ((z10 >= 1.5) & (r5 >= 85) & SEP).fillna(False),
    "THRUST, NON-September": (THRUST & ~SEP).fillna(False),
    "SEPTEMBER, NO thrust": (SEP & ~THRUST).fillna(False),
    "THRUST & shoulder(Sep+Oct)": CELL_SH,
}
for H in (5, 10):
    battery(px_ung, CELL, [("UNG", 1.0)], H,
            "C4 UNG long, thrust into September", cost_bps=15.0,
            variants=variants, min_gap=10)

# ---------------------------------------------------------------------------
# 4. LAG PROFILE
# ---------------------------------------------------------------------------
print("\n===== 4. LAG PROFILE (the cell) =====")
for H in (5, 10):
    for lag in (0, 1, 2, 3):
        r = fwd_lag(ung, H, lag)
        e = declusters(t_cell.intersection(r.dropna().index), 10, ung.index)
        s = summarize(r.reindex(e).values, "")
        print("  h=%2d lag=%d N=%2d ABS mean %+.3f%% hit %.1f%% t %s"
              % (H, lag, s["n"], s["mean_pct"], s["hit"],
                 f"{s['t']:+.2f}" if s["n"] > 1 else "na"))

# ---------------------------------------------------------------------------
# 5. the futures cross-check (directional only, per 0b)
# ---------------------------------------------------------------------------
print("\n===== 5. NG=F CROSS-CHECK (same cell, futures series; roll-contaminated) =====")
z10f = zscore(ngf, 10)
r5f = pct_rank(ngf, 5)
SEPf = pd.Series(ngf.index.month == 9, index=ngf.index)
CELLf = ((z10f >= 1.0) & (r5f >= 75) & SEPf).fillna(False)
tf = ngf.index[CELLf.reindex(ngf.index, fill_value=False).values]
for H in (5, 10, 21):
    r = fwd_lag(ngf, H, 1)
    e = declusters(tf.intersection(r.dropna().index), 10, ngf.index)
    s = summarize(r.reindex(e).values, f"NG=F cell h={H}")
    b = summarize(r.dropna().values, f"NG=F all days h={H}")
    print("  h=%2d  cell N=%d ABS %+.3f%% hit %.1f%%  |  all days %+.3f%%  "
          "-> excess %+.3f pp"
          % (H, s["n"], s["mean_pct"], s["hit"], b["mean_pct"],
             s["mean_pct"] - b["mean_pct"]))
print("  NG=F September month-of-year, all days, h=10: %+.3f%% vs all-months %+.3f%%"
      % (100 * fwd_lag(ngf, 10, 1)[ngf.index.month == 9].mean(),
         100 * fwd_lag(ngf, 10, 1).mean()))

# ---------------------------------------------------------------------------
# 6. cost, stated honestly for a wide commodity ETF
# ---------------------------------------------------------------------------
print("\n===== 6. COST =====")
for H in (5, 10):
    r = fwd_lag(ung, H, 1)
    e = declusters(t_cell.intersection(r.dropna().index), 10, ung.index)
    m = 100 * r.reindex(e).mean()
    for c in (10, 15, 25):
        print("  h=%2d episode ABS mean %+.3f%% = %+.1f bps -> at %d bps round "
              "trip that is %.1fx cost (need >=5x)" % (H, m, m * 100, c, m * 100 / c))
