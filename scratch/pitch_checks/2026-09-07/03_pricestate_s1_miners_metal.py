"""S1 -- gold miners strong on a 21d rank while the metal sits below its 200d SMA.

Live 2026-09-04: GDX 21d rank ~80, +25.9% over 63d, +10.5% over its 200d;
NEM 21d rank ~90; GLD 2.1% BELOW its 200d and 18% off its 52w high; SLV 8.4%
below its 200d. The equity is leading and the metal is not confirming.

Cells measured (all lag=1, episodes declustered at h td):
  A. GDX long
  B. GLD long
  C. NEM long
  D. pair long GDX / short GLD at the trailing-252d beta (PIT beta, no lookahead)
  E. the reverse pair (long GLD / short GDX) == -D, reported for completeness
  F. GDXJ long (the higher-beta expression)
Trigger: pct_rank(GDX, 21d) >= 75  AND  GLD close < GLD 200d SMA.
Sensitivity: rank thresholds 70 / 75 / 80 / 90, and the NEM-led variant.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _survey_lib import (  # noqa: E402
    align, cell, close_panel, cluster_note, declusters, era_split, fwd_lag,
    hscan, load_prices, local_control, np, pct_rank, pd, roll_max, sign_test,
    sma, state_line, summarize, show,
)

TICKERS = ["GDX", "GLD", "SLV", "NEM", "GDXJ", "SPY"]
PX = load_prices(TICKERS)
IDX = PX["GDX"].index
C = {t: PX[t]["Close"] for t in PX}

print("=" * 78)
print("S1  MINERS AT A HIGH, METAL BELOW ITS 200d   (asof 2026-09-04, entry lag=1)")
print("=" * 78)

# ---------------------------------------------------------------- live state
print("\nLIVE STATE VERIFICATION (recomputed from data/master_prices.parquet):")
for t in ["GDX", "NEM", "GDXJ", "GLD", "SLV"]:
    s = C[t]
    r21 = 100 * (s.iloc[-1] / s.iloc[-22] - 1)
    r63 = 100 * (s.iloc[-1] / s.iloc[-64] - 1)
    rk = pct_rank(s, 21).iloc[-1]
    hi = roll_max(s, 252).iloc[-1]
    m200 = sma(s, 200).iloc[-1]
    print(f"  {t:5s} last {s.iloc[-1]:9.2f}  21d {r21:+7.2f}%  63d {r63:+7.2f}%  "
          f"rank21 {rk:5.1f}  off52wh {100 * (s.iloc[-1] / hi - 1):+6.2f}%  "
          f"vs200d {100 * (s.iloc[-1] / m200 - 1):+6.2f}%")
print(f"  freshest bar: {IDX[-1].date()}")

# ------------------------------------------------------------------ triggers
rank_gdx = pct_rank(C["GDX"], 21)
rank_nem = align(pct_rank(C["NEM"], 21), IDX)
gld_below200 = align(C["GLD"] < sma(C["GLD"], 200), IDX).fillna(False).astype(bool)
slv_below200 = align(C["SLV"] < sma(C["SLV"], 200), IDX).fillna(False).astype(bool)

TRIG = IDX[(rank_gdx >= 75).values & gld_below200.values]
print(f"\nTrigger definition: rank21(GDX) >= 75 AND GLD < 200d SMA")
print(f"  trigger days: {len(TRIG)}   span {TRIG[0].date()} .. {TRIG[-1].date()}"
      f"   (GDX history starts {IDX[0].date()})")
print(f"  live day qualifies: {IDX[-1] in TRIG}   "
      f"(rank21 GDX {rank_gdx.iloc[-1]:.1f}, GLD below 200d {bool(gld_below200.iloc[-1])})")

# ------------------------------------------------------------------ PIT beta
rg = C["GDX"].pct_change()
rl = C["GLD"].pct_change()
both = pd.concat([rg, rl], axis=1, keys=["g", "l"]).dropna()
beta = (both["g"].rolling(252).cov(both["l"]) / both["l"].rolling(252).var())
beta = align(beta, IDX)
print(f"\nPIT 252d beta GDX~GLD: live {beta.iloc[-1]:.2f}  "
      f"median over triggers {beta.loc[TRIG].median():.2f}  "
      f"(full-sample median {beta.median():.2f})")


def gdx(h):
    return fwd_lag(C["GDX"], h, 1)


def gld(h):
    return fwd_lag(C["GLD"], h, 1)


def pair_beta(h):
    return gdx(h) - beta * gld(h)


def pair_11(h):
    return gdx(h) - gld(h)


def nem(h):
    return align(fwd_lag(C["NEM"], h, 1), IDX)


def gdxj(h):
    return align(fwd_lag(C["GDXJ"], h, 1), IDX)


# ------------------------------------------------------------- horizon sweeps
hscan(gdx, TRIG, "A. GDX long")
hscan(gld, TRIG, "B. GLD long")
hscan(nem, TRIG, "C. NEM long")
hscan(pair_beta, TRIG, "D. long GDX / short beta*GLD (PIT beta)")
hscan(pair_11, TRIG, "D2. long GDX / short 1x GLD (equal notional)")
hscan(gdxj, TRIG, "F. GDXJ long")

# ------------------------------------------------------- full cells at h=5,10
for h in (5, 10):
    cell(gdx(h), TRIG, h, "A. GDX long")
    cell(pair_beta(h), TRIG, h, "D. long GDX / short beta*GLD")
    cell(gld(h), TRIG, h, "B. GLD long")

# ------------------------------------------------------------- E. reverse pair
h = 5
rev = -pair_beta(h)
c = cell(rev, TRIG, h, "E. REVERSE pair: long beta*GLD / short GDX")

# --------------------------------------------------------------- sensitivity
print("\n=== S1 THRESHOLD SENSITIVITY (h=5 episodes, pair_beta and GDX) ===")
rows = []
variants = {
    "rank21 GDX>=70 & GLD<200d": (rank_gdx >= 70).values & gld_below200.values,
    "rank21 GDX>=75 & GLD<200d": (rank_gdx >= 75).values & gld_below200.values,
    "rank21 GDX>=80 & GLD<200d": (rank_gdx >= 80).values & gld_below200.values,
    "rank21 GDX>=90 & GLD<200d": (rank_gdx >= 90).values & gld_below200.values,
    "rank21 GDX>=75, GLD ABOVE 200d": (rank_gdx >= 75).values & ~gld_below200.values,
    "rank21 GDX>=75 only (no metal gate)": (rank_gdx >= 75).values,
    "rank21 NEM>=85 & GLD<200d": (rank_nem >= 85).values & gld_below200.values,
    "GDX>=75 & GLD<200d & SLV<200d": ((rank_gdx >= 75).values & gld_below200.values
                                      & slv_below200.values),
}
for lbl, m in variants.items():
    t = IDX[m]
    for nm, f in (("GDX", gdx), ("pairB", pair_beta)):
        r = f(h)
        tt = pd.DatetimeIndex(t).intersection(r.dropna().index)
        if len(tt) == 0:
            rows.append({"variant": lbl, "leg": nm, "n": 0})
            continue
        epi = declusters(tt, h, r.dropna().index)
        ep = r.loc[epi].values
        w = int((ep > 0).sum())
        rows.append({"variant": lbl, "leg": nm, "n_days": len(tt), "n": len(epi),
                     "mean_pct": round(100 * ep.mean(), 3),
                     "hit": round(100 * (ep > 0).mean(), 1),
                     "worst_pct": round(100 * ep.min(), 2),
                     "rec": f"{w}-{len(epi) - w}",
                     "sign_p": round(sign_test(w, len(epi)), 4)})
print(pd.DataFrame(rows).to_string(index=False))

# ------------------------------------------------------------------ era split
print("\n=== S1 ERA SPLIT (h=5 episodes) ===")
for nm, f in (("GDX", gdx), ("pairB", pair_beta), ("GLD", gld)):
    r = f(5)
    tt = TRIG.intersection(r.dropna().index)
    epi = declusters(tt, 5, r.dropna().index)
    show(era_split(epi, r.loc[epi].values), nm)

print("\n=== S1 COST NOTE ===")
print("  single ETF round trip ~4-6 bps; the GDX/GLD pair ~8-12 bps.")
print("  'worth a slot' bar quoted by the survey brief = 3x cost, so a single")
print("  leg needs ~+0.18% and the pair ~+0.36% per episode at the pitched h.")
