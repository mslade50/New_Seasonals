"""S4 -- a sector washed out on a 21d rank while the index sits at its high.

Live 2026-09-04: SPY 0.99% off its 52w high, but XLI rank21 7.1 (-5.14% 21d),
ITA 2.8 (-9.75%), XLRE 15.9, IYR 16.3, VNQ 15.9, XLY 27.4. Breadth: 66.5% of a
218-name tape above its 200d SMA but only 37.2% with a 21d rank above 50.

POOLED design, as briefed: the nine SPDR sectors (XLB XLE XLF XLI XLK XLP XLU
XLV XLY), trigger = own rank21 <= 10 AND SPY within 2% of its trailing-252 high.
Sector fixed effects are applied two ways, both reported, because they answer
different questions:
  FE-1 (demeaned)  : sector forward return minus that sector's OWN all-days mean
                     over the same span. Removes the level of each sector's drift.
  FE-2 (vs SPY)    : sector forward return minus SPY's forward return over the
                     identical window. The tradeable relative expression.
plus the raw pooled long. Every observation is declustered WITHIN sector at h td
before pooling, so one sector's month-long washout is one episode, not twenty.
Then XLI's own cell is compared against the pooled family.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _survey_lib import (  # noqa: E402
    align, cell, declusters, fwd_lag, load_prices, np, pct_rank, pd, roll_max,
    show, sign_test, sma, summarize, bootstrap_p_le0, local_control, era_split,
)

SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
EXTRA = ["SPY", "ITA", "XLRE", "IYR", "VNQ"]
PX = load_prices(SECTORS + EXTRA)
IDX = PX["SPY"].index
C = {t: PX[t]["Close"] for t in PX}

print("=" * 78)
print("S4  SECTOR 21d-RANK WASHOUT UNDER AN INDEX AT ITS HIGH  (asof 2026-09-04)")
print("=" * 78)

spy_hi = roll_max(C["SPY"], 252)
spy_near_hi = (C["SPY"] >= 0.98 * spy_hi)

print("\nLIVE STATE VERIFICATION:")
print(f"  SPY {C['SPY'].iloc[-1]:.2f}  off 52w high {100 * (C['SPY'].iloc[-1] / spy_hi.iloc[-1] - 1):+.2f}%"
      f"   within 2% of 252d high: {bool(spy_near_hi.iloc[-1])}")
for t in SECTORS + ["ITA", "XLRE", "IYR", "VNQ"]:
    s = C[t]
    print(f"  {t:5s} rank21 {pct_rank(s, 21).iloc[-1]:5.1f}   21d ret "
          f"{100 * (s.iloc[-1] / s.iloc[-22] - 1):+6.2f}%   off52wh "
          f"{100 * (s.iloc[-1] / roll_max(s, 252).iloc[-1] - 1):+6.2f}%")
print(f"  freshest bar: {IDX[-1].date()}")

RANK = {t: pct_rank(C[t], 21) for t in SECTORS + ["ITA", "XLRE", "IYR", "VNQ"]}
live_hits = [t for t in SECTORS if RANK[t].iloc[-1] <= 10]
print(f"\n  SPDR sectors live in the cell (rank21<=10 & SPY within 2% of high): "
      f"{live_hits or 'NONE'}")
print(f"  XLI rank21 = {RANK['XLI'].iloc[-1]:.1f}  -> XLI in cell: {RANK['XLI'].iloc[-1] <= 10}")


def pooled(h: int, rank_max: float = 10, spy_gate: bool = True,
           tickers=SECTORS, verbose: bool = False):
    """Return (raw, fe1_demeaned, fe2_vs_spy, per-sector rows) for the pooled cell."""
    spyf = fwd_lag(C["SPY"], h, 1)
    raw, fe1, fe2, rows, dates = [], [], [], [], []
    for t in tickers:
        f = align(fwd_lag(C[t], h, 1), IDX)
        valid = f.dropna().index
        m = (align(RANK[t] <= rank_max, IDX).fillna(0).astype(bool)).values
        if spy_gate:
            m = m & align(spy_near_hi, IDX).fillna(0).astype(bool).values
        trig = pd.DatetimeIndex(IDX[m]).intersection(valid)
        if len(trig) == 0:
            rows.append({"sector": t, "n": 0})
            continue
        epi = declusters(trig, h, valid)
        ep = f.loc[epi].values
        span = valid[(valid >= trig[0]) & (valid <= trig[-1])]
        own_mean = float(f.loc[span].mean())          # FE-1 baseline (same span)
        rel = (f - spyf).loc[epi].dropna().values      # FE-2 (drop the odd
        # session where the sector has a bar and SPY's forward window does not)
        raw.append(ep)
        fe1.append(ep - own_mean)
        fe2.append(rel)
        dates.append(epi)
        w = int((ep > 0).sum())
        wr = int((rel > 0).sum())
        rows.append({"sector": t, "n_days": len(trig), "n": len(epi),
                     "raw_pct": round(100 * ep.mean(), 3),
                     "own_drift_pct": round(100 * own_mean, 3),
                     "fe1_pct": round(100 * (ep.mean() - own_mean), 3),
                     "fe2_vsSPY_pct": round(100 * rel.mean(), 3),
                     "hit_raw": round(100 * (ep > 0).mean(), 1),
                     "worst_pct": round(100 * ep.min(), 2),
                     "rec_raw": f"{w}-{len(epi) - w}",
                     "rec_vsSPY": f"{wr}-{len(rel) - wr}",
                     "signp_vsSPY": round(sign_test(wr, len(rel)), 4)})
    return (np.concatenate(raw) if raw else np.array([]),
            np.concatenate(fe1) if fe1 else np.array([]),
            np.concatenate(fe2) if fe2 else np.array([]),
            pd.DataFrame(rows),
            dates)


print("\n" + "=" * 78)
print("POOLED 9-SECTOR CELL: rank21<=10 AND SPY within 2% of its 252d high")
print("(declustered within sector at h td, then pooled)")
print("=" * 78)
summary = []
for h in (1, 2, 3, 5, 10):
    raw, fe1, fe2, rows, _ = pooled(h)
    r_raw = summarize(raw, f"h={h} pooled RAW long")
    r_fe1 = summarize(fe1, f"h={h} pooled FE-1 (minus own drift)")
    r_fe2 = summarize(fe2, f"h={h} pooled FE-2 (minus SPY)")
    for r, tag in ((r_raw, "RAW"), (r_fe1, "FE1"), (r_fe2, "FE2vsSPY")):
        w = int((np.asarray({"RAW": raw, "FE1": fe1, "FE2vsSPY": fe2}[tag]) > 0).sum())
        n = r["n"]
        summary.append({"h": h, "basis": tag, "n_pooled_epi": n,
                        "mean_pct": round(r["mean_pct"], 3),
                        "median_pct": round(r["median_pct"], 3),
                        "hit": round(r["hit"], 1), "t": round(r["t"], 2),
                        "worst_pct": round(r["worst_pct"], 2),
                        "rec": f"{w}-{n - w}",
                        "sign_p": round(sign_test(w, n), 5),
                        "bootP<=0": round(bootstrap_p_le0(
                            {"RAW": raw, "FE1": fe1, "FE2vsSPY": fe2}[tag]), 4)})
print(pd.DataFrame(summary).to_string(index=False))

print("\n=== PER-SECTOR BREAKDOWN at h=5 ===")
_, _, _, rows5, _ = pooled(5)
print(rows5.to_string(index=False))
print("\n=== PER-SECTOR BREAKDOWN at h=10 ===")
_, _, _, rows10, _ = pooled(10)
print(rows10.to_string(index=False))

# ------------------------------------------------------ control: no SPY gate
print("\n" + "=" * 78)
print("CONTROL / GATE ATTRIBUTION: same washout WITHOUT the SPY-near-high gate")
print("=" * 78)
ctrl = []
for h in (1, 2, 3, 5, 10):
    raw, fe1, fe2, _, _ = pooled(h, spy_gate=False)
    for tag, v in (("RAW", raw), ("FE1", fe1), ("FE2vsSPY", fe2)):
        r = summarize(v, tag)
        w = int((v > 0).sum())
        ctrl.append({"h": h, "basis": tag, "n": r["n"],
                     "mean_pct": round(r["mean_pct"], 3), "hit": round(r["hit"], 1),
                     "t": round(r["t"], 2), "rec": f"{w}-{r['n'] - w}",
                     "sign_p": round(sign_test(w, r["n"]), 5)})
print(pd.DataFrame(ctrl).to_string(index=False))

print("\n=== RANK-THRESHOLD SENSITIVITY (pooled, h=5) ===")
sens = []
for rk in (5, 10, 20, 30):
    raw, fe1, fe2, _, _ = pooled(5, rank_max=rk)
    for tag, v in (("RAW", raw), ("FE1", fe1), ("FE2vsSPY", fe2)):
        r = summarize(v, tag)
        w = int((v > 0).sum())
        sens.append({"rank<=": rk, "basis": tag, "n": r["n"],
                     "mean_pct": round(r["mean_pct"], 3), "hit": round(r["hit"], 1),
                     "t": round(r["t"], 2), "rec": f"{w}-{r['n'] - w}",
                     "sign_p": round(sign_test(w, r["n"]), 5)})
print(pd.DataFrame(sens).to_string(index=False))

# ------------------------------------------------------------------ XLI cell
print("\n" + "=" * 78)
print("XLI's OWN CELL vs the pooled family")
print("=" * 78)
spy_gate_al = align(spy_near_hi, IDX).fillna(0).astype(bool)
for t in ["XLI", "ITA"]:
    m = align(RANK[t] <= 10, IDX).fillna(0).astype(bool) & spy_gate_al
    trig = IDX[m.values]
    print(f"\n{t}: {len(trig)} trigger days"
          + (f", {trig[0].date()} .. {trig[-1].date()}, yrs {len(set(trig.year))}"
             if len(trig) else ""))
    if len(trig) == 0:
        continue
    for h in (5, 10):
        f = align(fwd_lag(C[t], h, 1), IDX)
        cell(f, trig, h, f"{t} long | rank21<=10 & SPY within 2% of high")
        rel = f - fwd_lag(C["SPY"], h, 1)
        cell(rel, trig, h, f"{t} MINUS SPY | rank21<=10 & SPY within 2% of high")

print("\n=== S4 REAL-ESTATE COMPLEX (not in the 9-sector pool; XLRE/IYR/VNQ) ===")
for t in ["XLRE", "IYR", "VNQ"]:
    m = align(RANK[t] <= 20, IDX).fillna(0).astype(bool) & spy_gate_al
    trig = IDX[m.values]
    f = align(fwd_lag(C[t], 5, 1), IDX)
    tt = pd.DatetimeIndex(trig).intersection(f.dropna().index)
    if len(tt) == 0:
        print(f"  {t}: no triggers")
        continue
    epi = declusters(tt, 5, f.dropna().index)
    ep = f.loc[epi].values
    base = float(f.dropna().mean())
    w = int((ep > 0).sum())
    print(f"  {t:5s} rank21<=20 & SPY high: n_days {len(tt):4d} n_epi {len(epi):3d} "
          f"h=5 mean {100 * ep.mean():+.3f}%  all-days {100 * base:+.3f}%  "
          f"edge {100 * (ep.mean() - base):+.3f}pp  rec {w}-{len(epi) - w}  "
          f"sign p {sign_test(w, len(epi)):.4f}")

print("\n=== S4 CELL COUNT AND COST ===")
print("  grid searched here: 9 sectors x 5 horizons x 3 bases x 4 rank thresholds")
print("  = 540 sector-level cells, plus 30 pooled summaries. Any pulse is UNCHARGED.")
print("  sector ETF round trip ~4-6 bps; the sector-minus-SPY pair ~8-10 bps.")
