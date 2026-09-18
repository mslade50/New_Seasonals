"""S2 -- investment grade / duration pinned at the 252d floor while high yield
sits at the 252d ceiling. Pure rates repricing with no credit stress.

Live 2026-09-04: ^TNX 4.784 (0.25% off its 52w high, rank21 ~69, +9.7% over its
200d); TLT 1.44% above its 52w LOW, IEF 0.44% above, LQD 0.25% above, while HYG
is 0.41% BELOW its 52w HIGH.

Cells (lag=1, episodes declustered at h td):
  A. TLT long   B. IEF long   C. LQD long   D. HYG long
  E. pair long LQD / short HYG (the quality-spread convergence trade)
  F. the reverse pair, long HYG / short LQD
Trigger: LQD within 1% of its trailing-252 LOW  AND  HYG within 1% of its
trailing-252 HIGH.  Widened and narrowed in the sensitivity block, plus the
TLT-floor variant and the yield-side (^TNX near 252d high) variant.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _survey_lib import (  # noqa: E402
    align, cell, declusters, era_split, fwd_lag, hscan, load_prices, np,
    pct_rank, pd, roll_max, roll_min, show, sign_test, sma, summarize,
)

TICKERS = ["TLT", "IEF", "LQD", "HYG", "^TNX", "SPY", "JNK", "AGG"]
PX = load_prices(TICKERS)
IDX = PX["HYG"].index          # HYG is the binding history (2007-04 inception)
C = {t: PX[t]["Close"] for t in PX}

print("=" * 78)
print("S2  IG/DURATION AT THE 252d FLOOR, HY AT THE 252d CEILING  (asof 2026-09-04)")
print("=" * 78)

print("\nLIVE STATE VERIFICATION:")
for t in ["TLT", "IEF", "LQD", "HYG", "AGG", "JNK", "^TNX"]:
    if t not in C:
        print(f"  {t:6s} NOT IN CACHE")
        continue
    s = C[t]
    hi, lo = roll_max(s, 252).iloc[-1], roll_min(s, 252).iloc[-1]
    print(f"  {t:6s} last {s.iloc[-1]:9.3f}  vs52wLOW {100 * (s.iloc[-1] / lo - 1):+6.2f}%  "
          f"vs52wHIGH {100 * (s.iloc[-1] / hi - 1):+6.2f}%  rank21 {pct_rank(s, 21).iloc[-1]:5.1f}  "
          f"vs200d {100 * (s.iloc[-1] / sma(s, 200).iloc[-1] - 1):+6.2f}%")
print(f"  freshest bar: {IDX[-1].date()}")

# ------------------------------------------------------------------ triggers
lqd_floor = C["LQD"] <= 1.01 * roll_min(C["LQD"], 252)
hyg_ceil = C["HYG"] >= 0.99 * roll_max(C["HYG"], 252)
tlt_floor = C["TLT"] <= 1.02 * roll_min(C["TLT"], 252)
tnx_ceil = align(C["^TNX"] >= 0.99 * roll_max(C["^TNX"], 252), IDX).fillna(False)

lqd_floor = align(lqd_floor, IDX).fillna(False).astype(bool)
hyg_ceil = align(hyg_ceil, IDX).fillna(False).astype(bool)
tlt_floor = align(tlt_floor, IDX).fillna(False).astype(bool)
tnx_ceil = tnx_ceil.astype(bool)

TRIG = IDX[(lqd_floor & hyg_ceil).values]
print("\nTrigger: LQD within 1% of its 252d LOW  AND  HYG within 1% of its 252d HIGH")
print(f"  trigger days: {len(TRIG)}"
      + (f"   span {TRIG[0].date()} .. {TRIG[-1].date()}" if len(TRIG) else ""))
print(f"  live day qualifies: {IDX[-1] in TRIG}"
      f"   (LQD floor {bool(lqd_floor.iloc[-1])}, HYG ceiling {bool(hyg_ceil.iloc[-1])})")
if len(TRIG):
    print(f"  distinct years: {sorted(set(TRIG.year))}")


def leg(t):
    return lambda h: align(fwd_lag(C[t], h, 1), IDX)


tlt, ief, lqd, hyg = leg("TLT"), leg("IEF"), leg("LQD"), leg("HYG")


def pair_lqd_hyg(h):
    return lqd(h) - hyg(h)


def pair_hyg_lqd(h):
    return hyg(h) - lqd(h)


if len(TRIG) == 0:
    print("\n##### S2 CONJUNCTION HAS NEVER FIRED. Dead as stated. #####")
else:
    for nm, f in (("A. TLT long", tlt), ("B. IEF long", ief), ("C. LQD long", lqd),
                  ("D. HYG long", hyg), ("E. long LQD / short HYG", pair_lqd_hyg),
                  ("F. long HYG / short LQD", pair_hyg_lqd)):
        hscan(f, TRIG, nm)
    for h in (5, 10):
        for nm, f in (("A. TLT long", tlt), ("C. LQD long", lqd),
                      ("E. long LQD / short HYG", pair_lqd_hyg)):
            cell(f(h), TRIG, h, nm)

# --------------------------------------------------------------- sensitivity
print("\n=== S2 SENSITIVITY / PARENT ATTRIBUTION (h=5 episodes) ===")
variants = {
    "CONJ: LQD<=1%>252low & HYG>=1%<252high": (lqd_floor & hyg_ceil).values,
    "CONJ loose 2%/2%": ((C["LQD"].reindex(IDX) <= 1.02 * roll_min(C["LQD"], 252).reindex(IDX))
                         & (C["HYG"].reindex(IDX) >= 0.98 * roll_max(C["HYG"], 252).reindex(IDX))).fillna(False).values,
    "CONJ tight 0.5%/0.5%": ((C["LQD"].reindex(IDX) <= 1.005 * roll_min(C["LQD"], 252).reindex(IDX))
                             & (C["HYG"].reindex(IDX) >= 0.995 * roll_max(C["HYG"], 252).reindex(IDX))).fillna(False).values,
    "PARENT: LQD at 252d floor only": lqd_floor.values,
    "PARENT: HYG at 252d ceiling only": hyg_ceil.values,
    "TLT floor(2%) & HYG ceiling(1%)": (tlt_floor & hyg_ceil).values,
    "TNX at 252d high & HYG ceiling": (tnx_ceil & hyg_ceil).values,
    "TNX at 252d high only": tnx_ceil.values,
}
rows = []
h = 5
for lbl, m in variants.items():
    t0 = IDX[np.asarray(m, dtype=bool)]
    for nm, f in (("TLT", tlt), ("LQD", lqd), ("HYG", hyg), ("LQD-HYG", pair_lqd_hyg)):
        r = f(h)
        tt = pd.DatetimeIndex(t0).intersection(r.dropna().index)
        if len(tt) == 0:
            rows.append({"variant": lbl, "leg": nm, "n_days": 0, "n": 0})
            continue
        epi = declusters(tt, h, r.dropna().index)
        ep = r.loc[epi].values
        w = int((ep > 0).sum())
        base = float(r.dropna().mean())
        rows.append({"variant": lbl, "leg": nm, "n_days": len(tt), "n": len(epi),
                     "mean_pct": round(100 * ep.mean(), 3),
                     "edge_all_pct": round(100 * (ep.mean() - base), 3),
                     "hit": round(100 * (ep > 0).mean(), 1),
                     "worst_pct": round(100 * ep.min(), 2),
                     "rec": f"{w}-{len(epi) - w}",
                     "sign_p": round(sign_test(w, len(epi)), 4)})
print(pd.DataFrame(rows).to_string(index=False))

if len(TRIG):
    print("\n=== S2 ERA SPLIT (h=5 episodes) ===")
    for nm, f in (("TLT", tlt), ("LQD", lqd), ("LQD-HYG", pair_lqd_hyg)):
        r = f(5)
        tt = TRIG.intersection(r.dropna().index)
        epi = declusters(tt, 5, r.dropna().index)
        show(era_split(epi, r.loc[epi].values), nm)

print("\n=== S2 COST NOTE ===")
print("  TLT/LQD/HYG round trip ~4-6 bps (HYG/LQD spreads are wider than SPY's);")
print("  the LQD-HYG pair ~8-12 bps -> needs ~+0.36% per episode to clear 3x.")
