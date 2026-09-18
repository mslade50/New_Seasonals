"""S6b -- the dial extreme reads as SIZE DISPERSION, not as an index-direction bet.

03_pricestate_s6_fragility_dial_extreme.py found that the outright SPY short at a
dial extreme is a coin flip (h=10 conjunction 8-8, sign p 0.60) but that IWM's
degradation is far larger and more consistent than SPY's at every horizon:
h=10 top-decile IWM -1.630% vs a dial-era baseline of +0.507%, record 14-25, i.e.
the SHORT side is 25-14. This script measures the relative expression directly --
long SPY / short IWM, and long QQQ / short IWM -- so the market-direction bet is
netted out of the number instead of sitting inside it.

Same vintage caveat as S6, restated because it is load-bearing: the fragility
parquet is true point-in-time only from 2026-07-02. Everything earlier is a
recompute vintage that drifted up to ~7 points on the 63d dial, so the historical
episodes here are an UPPER BOUND on what was knowable.

This is also where the repo's own institutional memory is directly relevant: the
book-wide dial throttle was killed at aggregate PIT t = -0.23, and the sizing
memory records the dial as "a correlation instrument". A dispersion/relative read
is the form that survives that verdict; an outright directional one is not.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _survey_lib import (  # noqa: E402
    align, cell, cluster_note, declusters, era_split, fwd_lag, hscan,
    load_prices, np, pd, show, sign_test, summarize, bootstrap_p_le0,
)

ROOT = Path(__file__).resolve().parents[3]
FRAG = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
PX = load_prices(["SPY", "QQQ", "IWM"])   # RSP is not in the price cache
IDX = PX["SPY"].index
C = {t: PX[t]["Close"] for t in PX}

ma = FRAG["63d"].rolling(10).mean()
q90 = ma.expanding(252).quantile(0.90)
rise21 = ma - ma.shift(21)

m_top = align((ma >= q90).fillna(False), IDX).fillna(0).astype(bool)
m_conj = m_top & align((rise21 > 25).fillna(False), IDX).fillna(0).astype(bool)
m_dial = align(ma.notna(), IDX).fillna(0).astype(bool)
DIAL_DAYS = IDX[m_dial.values]

print("=" * 78)
print("S6b  DIAL EXTREME AS A SIZE-DISPERSION STATE (long SPY / short IWM)")
print("=" * 78)
print(f"  live 10d-MA-63d {ma.iloc[-1]:.2f}  rise21 {rise21.iloc[-1]:+.2f}  "
      f"top-decile cut {q90.iloc[-1]:.2f}")
print(f"  live in top decile: {bool(m_top.iloc[-1])}   live in conjunction: {bool(m_conj.iloc[-1])}")
print("  VINTAGE: true PIT only from 2026-07-02; earlier rows are a recompute")
print("  vintage that drifted up to ~7 dial points. Upper bound, not a record.")


def rel(a, b):
    def f(h):
        return align(fwd_lag(C[a], h, 1), IDX) - align(fwd_lag(C[b], h, 1), IDX)
    return f


def solo(t):
    def f(h):
        return align(fwd_lag(C[t], h, 1), IDX)
    return f


VEH = {
    "SPY - IWM": rel("SPY", "IWM"),
    "QQQ - IWM": rel("QQQ", "IWM"),
    "IWM outright SHORT": lambda h: -align(fwd_lag(C["IWM"], h, 1), IDX),
}
MASKS = {"(a) dial top decile (expanding cut)": m_top,
         "(d) top decile AND rose >25 pts": m_conj}

for vname, vf in VEH.items():
    for mname, m in MASKS.items():
        hscan(vf, IDX[m.values], f"{vname}  |  {mname}")

print("\n" + "=" * 78)
print("vs the DIAL-ERA baseline (the only honest control: the dial's own span)")
print("=" * 78)
rows = []
for h in (1, 2, 3, 5, 10):
    for vname, vf in VEH.items():
        r = vf(h)
        valid = r.dropna().index
        dv = pd.DatetimeIndex(DIAL_DAYS).intersection(valid)
        base = float(r.loc[dv].mean())
        for mname, m in MASKS.items():
            t = pd.DatetimeIndex(IDX[m.values]).intersection(valid)
            epi = declusters(t, h, valid)
            ep = r.loc[epi].values
            w = int((ep > 0).sum())
            rows.append({"h": h, "veh": vname, "mask": mname[:3], "n": len(epi),
                         "mean_pct": round(100 * ep.mean(), 3),
                         "dial_era_base_pct": round(100 * base, 3),
                         "edge_pct": round(100 * (ep.mean() - base), 3),
                         "hit": round(100 * (ep > 0).mean(), 1),
                         "worst_pct": round(100 * ep.min(), 2),
                         "rec": f"{w}-{len(epi) - w}",
                         "sign_p": round(sign_test(w, len(epi)), 4),
                         "bootP<=0": round(bootstrap_p_le0(ep), 3)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n" + "=" * 78)
print("FULL CELLS: long SPY / short IWM at the dial top decile")
print("=" * 78)
for h in (5, 10):
    cell(VEH["SPY - IWM"](h), IDX[m_top.values], h, "(a) dial top decile -> long SPY / short IWM")
    cell(VEH["SPY - IWM"](h), IDX[m_conj.values], h, "(d) top decile AND rose>25 -> long SPY / short IWM")

print("\n=== VINTAGE SPLIT and LEAVE-ONE-YEAR-OUT (SPY-IWM, mask (a)) ===")
for h in (5, 10):
    r = VEH["SPY - IWM"](h)
    valid = r.dropna().index
    epi = declusters(pd.DatetimeIndex(IDX[m_top.values]).intersection(valid), h, valid)
    ep = r.loc[epi].values
    pre = epi < pd.Timestamp("2026-07-02")
    show([summarize(ep[pre], f"h={h} recompute vintage (pre 2026-07-02)"),
          summarize(ep[~pre], f"h={h} true PIT (2026-07-02+)")], f"vintage split h={h}")
    rows = []
    for y in sorted(set(epi.year)):
        keep = epi.year != y
        v = ep[keep]
        w = int((v > 0).sum())
        rows.append({"h": h, "drop_year": y, "n_dropped": int((~keep).sum()),
                     "n_left": len(v), "mean_pct": round(100 * v.mean(), 3),
                     "rec": f"{w}-{len(v) - w}",
                     "sign_p": round(sign_test(w, len(v)), 4)})
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"  concentration: {cluster_note(epi, ep, k=3)}")

print("\n=== S6b DIAL-THRESHOLD SENSITIVITY (SPY-IWM, h=5 and h=10) ===")
rows = []
for q in (0.80, 0.85, 0.90, 0.95):
    cut = ma.expanding(252).quantile(q)
    m = align((ma >= cut).fillna(False), IDX).fillna(0).astype(bool)
    for h in (5, 10):
        r = VEH["SPY - IWM"](h)
        valid = r.dropna().index
        t = pd.DatetimeIndex(IDX[m.values]).intersection(valid)
        epi = declusters(t, h, valid)
        ep = r.loc[epi].values
        base = float(r.loc[pd.DatetimeIndex(DIAL_DAYS).intersection(valid)].mean())
        w = int((ep > 0).sum())
        rows.append({"quantile": q, "h": h, "n_days": len(t), "n": len(epi),
                     "mean_pct": round(100 * ep.mean(), 3),
                     "edge_pct": round(100 * (ep.mean() - base), 3),
                     "hit": round(100 * (ep > 0).mean(), 1),
                     "worst_pct": round(100 * ep.min(), 2),
                     "rec": f"{w}-{len(epi) - w}",
                     "sign_p": round(sign_test(w, len(epi)), 4)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== S6b COST NOTE ===")
print("  SPY/IWM pair round trip ~5-8 bps -> the 3x bar is ~+0.15-0.24% per episode.")
