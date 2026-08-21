"""C8 round 1: long SLV against short GLD on the drawdown divergence inside a
joint metals thrust.

Live: SLV -41.61% from its 52w high, GLD -16.26%, gap -25.35pp. SLV 5d +6.02%
(rank 73.4), 21d +14.35%; SLV 63d -10.29% against GLD -0.51%.

Registry adjacency this must clear:
  2026-08-10 "Silver thrust from deep inside a drawdown" -- the drawdown
             conditioner points the WRONG way (deep-dd +1.378% at h=10 vs
             near-high +1.780%); 8%->10% thrust nudge flips h=5 to -4.229%.
  2026-08-11 "Adding a second metals leg beside a live one".
  2026-08-18 "Miner-versus-metal ratio reversion after a maximal thrust".

Attack order: (0) live state, (1) beta(SLV|GLD) -- an equal-dollar pair is a
levered silver bet unless proven otherwise, (2) the pair vs controls,
(3) the drawdown conditioner's DOSE RESPONSE (the registry's kill), (4) leg
attribution: does the GLD short add anything to the beta-neutral residual,
(5) threshold neighbours.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change  # noqa

import warnings
warnings.filterwarnings("ignore")

ASOF = pd.Timestamp("2026-08-20")
H = 5

px = close_panel(["SLV", "GLD", "GDX", "SPY"]).loc[:ASOF]
px = px.dropna(subset=["SLV", "GLD"])
idx = px.index

slv5 = pct_rank(px["SLV"], 5)
gld5 = pct_rank(px["GLD"], 5)
slv_hi = rolling_on_valid(px["SLV"], lambda x: x.rolling(252).max())
gld_hi = rolling_on_valid(px["GLD"], lambda x: x.rolling(252).max())
slv_dd = px["SLV"] / slv_hi - 1.0
gld_dd = px["GLD"] / gld_hi - 1.0
gap = slv_dd - gld_dd

print("=" * 100)
print("C8-0  LIVE STATE", ASOF.date())
print("=" * 100)
print(f"  SLV dd 52wh {100*slv_dd.loc[ASOF]:7.2f}%   GLD dd 52wh {100*gld_dd.loc[ASOF]:7.2f}%   "
      f"gap {100*gap.loc[ASOF]:7.2f}pp")
print(f"  SLV 5d rank {slv5.loc[ASOF]:5.1f}  GLD 5d rank {gld5.loc[ASOF]:5.1f}   "
      f"SLV 5d ret {100*_valid_pct_change(px['SLV'],5).loc[ASOF]:.2f}%  "
      f"SLV 21d ret {100*_valid_pct_change(px['SLV'],21).loc[ASOF]:.2f}%")
print(f"  SLV 63d ret {100*_valid_pct_change(px['SLV'],63).loc[ASOF]:.2f}%  "
      f"GLD 63d ret {100*_valid_pct_change(px['GLD'],63).loc[ASOF]:.2f}%")

# ------------------------------------------------------------------ 1. beta
print("\n" + "=" * 100)
print("C8-1  BETA(SLV on GLD).  an equal-dollar pair is levered silver unless this is ~1")
print("=" * 100)
rs = px["SLV"].pct_change()
rg = px["GLD"].pct_change()
d = pd.concat([rs, rg], axis=1).dropna()
d.columns = ["s", "g"]


def beta(sub):
    return float(np.polyfit(sub["g"], sub["s"], 1)[0])


print(f"  full sample beta = {beta(d):.3f}   corr = {d['s'].corr(d['g']):.3f}  N={len(d)}")
for lo, hi in (("2006", "2013"), ("2013", "2018"), ("2018", "2022"), ("2022", "2027")):
    sub = d[(d.index >= lo) & (d.index < hi)]
    if len(sub) > 50:
        print(f"  {lo}-{hi}: beta {beta(sub):.3f}  corr {sub['s'].corr(sub['g']):.3f}  N={len(sub)}")
B = beta(d)
sub252 = d.iloc[-252:]
B252 = beta(sub252)
print(f"  trailing 252d beta = {B252:.3f}")

# ------------------------------------------------------ 2. the cell + battery
mask = ((slv5 >= 70) & (gld5 >= 70) & (gap <= -0.20)).fillna(False)
print(f"\n  cell = SLV 5d rank>=70 AND GLD 5d rank>=70 AND (SLV dd - GLD dd) <= -20pp")
print(f"  trigger days = {int(mask.sum())}   fires today = {bool(mask.loc[ASOF])}")

variants = {}
for g in (-0.10, -0.15, -0.20, -0.25, -0.30):
    variants[f"gap<={int(100*g)}pp"] = ((slv5 >= 70) & (gld5 >= 70) & (gap <= g)).fillna(False)
for r in (60, 70, 80, 90):
    variants[f"both 5d rank>={r}"] = ((slv5 >= r) & (gld5 >= r) & (gap <= -0.20)).fillna(False)

battery(px, mask, [("SLV", 1.0), ("GLD", -1.0)], H,
        "C8  equal-dollar SLV - GLD | joint thrust + 20pp drawdown gap",
        cost_bps=4.0, variants=variants, min_gap=5, event_kinds=("cpi", "ppi"))

# ---------------------------------------------- 3. beta-neutral + attribution
print("\n" + "=" * 100)
print("C8-3  LEG ATTRIBUTION and BETA-NEUTRALISATION at beta =", round(B, 3))
print("=" * 100)
epi = declusters(idx[(mask & fwd_lag(px["SLV"], H).notna()).values], 5, idx)
rows = []
for lbl, legs in (("SLV outright", [("SLV", 1.0)]),
                  ("GLD outright (the SHORT leg, long-side sign)", [("GLD", 1.0)]),
                  ("equal-dollar SLV - GLD", [("SLV", 1.0), ("GLD", -1.0)]),
                  (f"beta-neutral SLV - {B:.2f}*GLD", [("SLV", 1.0), ("GLD", -B)]),
                  (f"beta-neutral (252d beta {B252:.2f})", [("SLV", 1.0), ("GLD", -B252)])):
    r = vehicle_ret(px, legs, H, 1)
    base = r.dropna()
    x = r.loc[epi].values
    s = summarize(x, lbl)
    s["own_drift"] = round(100 * base.mean(), 3)
    s["excess_pct"] = round(s["mean_pct"] - 100 * base.mean(), 3)
    s["signp"] = round(sign_test(int((x > 0).sum()), len(x)), 4)
    rows.append(s)
show(rows, f"h={H}, {len(epi)} episodes")

# ------------------------------------------- 4. dose response on the drawdown
print("\n" + "=" * 100)
print("C8-4  DOSE RESPONSE on the drawdown gap -- the registry's own kill re-run")
print("      thesis needs: the DEEPER the gap, the BIGGER the catch-up")
print("=" * 100)
r_bn = vehicle_ret(px, [("SLV", 1.0), ("GLD", -B)], H, 1)
r_ed = vehicle_ret(px, [("SLV", 1.0), ("GLD", -1.0)], H, 1)
r_out = fwd_lag(px["SLV"], H, 1)
thrust = ((slv5 >= 70) & (gld5 >= 70)).fillna(False)
print(f"  joint-thrust days total = {int(thrust.sum())}")
for lo, hi in ((-1.0, -0.30), (-0.30, -0.20), (-0.20, -0.10), (-0.10, 0.0), (0.0, 1.0)):
    m = (thrust & (gap > lo) & (gap <= hi)).fillna(False)
    e = declusters(idx[(m & r_bn.notna()).values], 5, idx)
    if len(e) < 3:
        print(f"  gap ({100*lo:+.0f},{100*hi:+.0f}]pp : N={len(e)} too few")
        continue
    a, b, c = r_bn.loc[e].values, r_ed.loc[e].values, r_out.loc[e].values
    print(f"  gap ({100*lo:+.0f},{100*hi:+.0f}]pp : N={len(e):<4} "
          f"beta-neutral {100*a.mean():+7.3f}% (hit {100*(a>0).mean():5.1f}%)  "
          f"eq-dollar {100*b.mean():+7.3f}%  SLV outright {100*c.mean():+7.3f}%")

# ------------------------------------------- 5. gate attribution: thrust alone
print("\n" + "=" * 100)
print("C8-5  GATE ATTRIBUTION -- does either gate filter?")
print("=" * 100)
for lbl, m in (("all days (no gate)", pd.Series(True, index=idx)),
               ("joint thrust ONLY", thrust),
               ("gap<=-20pp ONLY (no thrust)", (gap <= -0.20).fillna(False)),
               ("BOTH (the cell)", mask)):
    e = declusters(idx[(m & r_bn.notna()).values], 5, idx)
    if len(e) < 3:
        print(f"  {lbl:<32} N={len(e)}")
        continue
    a = r_bn.loc[e].values
    b = r_ed.loc[e].values
    print(f"  {lbl:<32} N={len(e):<5} beta-neutral {100*a.mean():+7.3f}% hit {100*(a>0).mean():5.1f}% "
          f"signp {sign_test(int((a>0).sum()), len(a)):.4f}   |  eq-dollar {100*b.mean():+7.3f}%")

# ------------------------------------------- 6. horizon
print("\n" + "=" * 100)
print("C8-6  HORIZON SCAN, beta-neutral and equal-dollar")
print("=" * 100)
show(horizon_scan(px, epi, [("SLV", 1.0), ("GLD", -B)], hs=(1, 2, 3, 5, 10), min_gap=5),
     f"beta-neutral SLV - {B:.2f}*GLD")
show(horizon_scan(px, epi, [("SLV", 1.0), ("GLD", -1.0)], hs=(1, 2, 3, 5, 10), min_gap=5),
     "equal-dollar SLV - GLD")
