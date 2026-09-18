"""C10 / C11 round 1 -- a food-and-packaged-staples subgroup flush under a
sector that has not moved.

Live 2026-09-04: CPB r5 2.8 (-8.59% 5d), GIS 4.0 (-7.85%), TSN 2.8 (-6.32%),
SJM 9.5, MKC 9.9 -- five of ten group members at a 5-day rank <= 10 -- while
XLP itself is r5 27.4, -1.02% over five sessions and only -4.98% off its 52w
high.

Trigger (pre-declared, the live reading is inside every leg):
    >= N of the group at 5-day rank <= 10 on the same session
    AND XLP's own 5-day rank >= FLOOR   (the "sector has not moved" gate)

Vehicles:
    V1  equal-weight basket of the FLUSHED members that day  (the traded thing)
    V2  equal-weight basket of ALL group members             (fixed membership)
    V3  V1 minus XLP                                          (the pair, C10)
    V4  V1 minus SPY
C10 is the long side of V1/V3, C11 the short side of the same object.

MANDATORY control, per the registry (2026-09-07, sector washout under an
index at its high): the SAME flush WITHOUT the intact-sector gate. That is
where the positive edge lived last time, and if it lives there again the gate
is decoration.

Survivorship note stated up front: the group is today's survivors. K, KHC and
LW are absent from master_prices, so three real packaged-food names (two of
them serial flush candidates) cannot be in it. That biases the long side
UPWARD, not down.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

warnings.filterwarnings("ignore")

GROUP = ["CPB", "GIS", "TSN", "HRL", "SYY", "CAG", "SJM", "MKC", "KR", "HSY"]
WIDE = GROUP + ["MDLZ", "KDP"]          # membership sensitivity (later starts)
RANK_MAX = 10.0
N_MIN = 3
XLP_FLOOR = 25.0
H = 5

PX = load_prices(sorted(set(WIDE + ["XLP", "SPY"])))
IDX = PX["SPY"].index
C = pd.DataFrame({t: PX[t]["Close"] for t in PX}).reindex(IDX)
R5 = pd.DataFrame({t: pct_rank(C[t], 5) for t in C.columns})

print("=" * 78)
print(f"C10/C11  FOOD-AND-PACKAGED-STAPLES FLUSH UNDER AN INTACT SECTOR "
      f"(asof {IDX[-1].date()})")
print("=" * 78)
live = R5.iloc[-1]
hits = [t for t in GROUP if live[t] <= RANK_MAX]
print(f"  group r5: " + "  ".join(f"{t} {live[t]:.1f}" for t in GROUP))
print(f"  members at r5<=10 TODAY: {hits}  (count {len(hits)})")
print(f"  XLP r5 {live['XLP']:.1f}   SPY r5 {live['SPY']:.1f}")
print(f"  trigger legs live: count>={N_MIN} {len(hits) >= N_MIN}, "
      f"XLP r5>={XLP_FLOOR} {live['XLP'] >= XLP_FLOOR}")


def build(group, rank_max=RANK_MAX, h=H):
    """Return (member-mask df, flushed-basket fwd ret, all-member fwd ret)."""
    M = (R5[group] <= rank_max).fillna(False) & C[group].notna()
    F = pd.DataFrame({t: fwd_lag(C[t], h, 1) for t in group})
    v1 = F.where(M).mean(axis=1)          # flushed members only
    v2 = F.mean(axis=1)                   # all members
    return M, v1, v2


def cell(ret, trig, h, label, min_gap=None, show_dates=0):
    valid = ret.dropna().index
    t = pd.DatetimeIndex(trig).intersection(valid)
    if len(t) == 0:
        print(f"  {label}: NO TRIGGERS")
        return None
    epi = declusters(t, min_gap or h, valid)
    ep = ret.loc[epi].values
    loc = local_control(valid, t)
    span = valid[(valid >= t[0]) & (valid <= t[-1])]
    base_all = float(ret.loc[valid].mean())
    base_span = float(ret.loc[span].mean())
    base_loc = float(ret.loc[loc].mean()) if len(loc) else np.nan
    w = int((ep > 0).sum())
    d = {"label": label, "n_days": len(t), "n": len(epi),
         "mean_pct": 100 * ep.mean(), "day_mean_pct": 100 * float(ret.loc[t].mean()),
         "hit": 100 * (ep > 0).mean(),
         "ctl_all": 100 * base_all, "ctl_span": 100 * base_span,
         "ctl_loc": 100 * base_loc,
         "edge_all": 100 * (ep.mean() - base_all),
         "edge_loc": 100 * (ep.mean() - base_loc),
         "worst": 100 * ep.min(), "rec": f"{w}-{len(epi)-w}",
         "sign_p": sign_test(w, len(epi)), "bootP": bootstrap_p_le0(ep),
         "t": summarize(ep)["t"], "yrs": len(set(epi.year)),
         "epi": epi, "ep": ep}
    print(f"  {label:<46s} Nd {d['n_days']:4d} Ne {d['n']:3d} "
          f"mean {d['mean_pct']:+7.3f}% (day {d['day_mean_pct']:+7.3f}%) "
          f"all {d['ctl_all']:+6.3f}% loc {d['ctl_loc']:+6.3f}% "
          f"edge {d['edge_all']:+6.3f}pp t {d['t']:+5.2f} rec {d['rec']:>7s} "
          f"p {d['sign_p']:.4f} bootP {d['bootP']:.3f} worst {d['worst']:+6.2f}%")
    if show_dates:
        print("      episodes: " + ", ".join(str(x.date()) for x in epi[:show_dates]))
    return d


M, v1, v2 = build(GROUP)
cnt = M.sum(axis=1)
xlp_r5 = R5["XLP"]
gate = (xlp_r5 >= XLP_FLOOR).fillna(False)
base_mask = (cnt >= N_MIN)
trig_gated = IDX[(base_mask & gate).values]
trig_ungated = IDX[base_mask.values]

print(f"\n  trigger days: gated {len(trig_gated)}, ungated {len(trig_ungated)} "
      f"({100*len(trig_gated)/max(1,len(trig_ungated)):.0f}% kept by the gate)")

pair = v1 - fwd_lag(C["XLP"], H, 1)
vsspy = v1 - fwd_lag(C["SPY"], H, 1)
allpair = v2 - fwd_lag(C["XLP"], H, 1)

print("\n" + "=" * 78)
print(f"1. THE CELL vs CONTROLS  (h={H}, lag=1, episodes declustered at h)")
print("=" * 78)
print(" -- GATED (XLP r5 >= 25): the pitched cell")
g = {}
g["V1 flushed basket"] = cell(v1, trig_gated, H, "V1 flushed basket LONG")
g["V2 all members"] = cell(v2, trig_gated, H, "V2 all-member basket LONG")
g["V3 pair"] = cell(pair, trig_gated, H, "V3 flushed basket MINUS XLP")
g["V4 vs SPY"] = cell(vsspy, trig_gated, H, "V4 flushed basket MINUS SPY")
g["V2-XLP"] = cell(allpair, trig_gated, H, "V2 all-member MINUS XLP")

print("\n -- UNGATED (no XLP floor): the registry's attribution control")
u = {}
u["V1"] = cell(v1, trig_ungated, H, "V1 flushed basket LONG (ungated)")
u["V3"] = cell(pair, trig_ungated, H, "V3 pair (ungated)")
u["V4"] = cell(vsspy, trig_ungated, H, "V4 vs SPY (ungated)")

print("\n -- THE COMPLEMENT: flush WITH the sector ALSO down (XLP r5 < 25)")
trig_comp = IDX[(base_mask & ~gate).values]
cell(v1, trig_comp, H, "V1 flushed basket LONG (sector down too)")
cell(pair, trig_comp, H, "V3 pair (sector down too)")

print("\n" + "=" * 78)
print("2. GATE ATTRIBUTION -- day-level AND episode-level, separately")
print("=" * 78)
for lbl, ret in (("V1", v1), ("V3 pair", pair)):
    va = ret.dropna().index
    tg = pd.DatetimeIndex(trig_gated).intersection(va)
    tu = pd.DatetimeIndex(trig_ungated).intersection(va)
    tc = pd.DatetimeIndex(trig_comp).intersection(va)
    eg = declusters(tg, H, va); eu = declusters(tu, H, va); ec = declusters(tc, H, va)
    print(f"  {lbl}: DAY-LEVEL  gated {100*ret.loc[tg].mean():+.3f}% (N {len(tg)}) | "
          f"ungated {100*ret.loc[tu].mean():+.3f}% (N {len(tu)}) | "
          f"complement {100*ret.loc[tc].mean():+.3f}% (N {len(tc)}) | "
          f"gate value {100*(ret.loc[tg].mean()-ret.loc[tu].mean()):+.3f}pp")
    print(f"  {lbl}: EPISODE    gated {100*ret.loc[eg].mean():+.3f}% (N {len(eg)}) | "
          f"ungated {100*ret.loc[eu].mean():+.3f}% (N {len(eu)}) | "
          f"complement {100*ret.loc[ec].mean():+.3f}% (N {len(ec)}) | "
          f"gate value {100*(ret.loc[eg].mean()-ret.loc[eu].mean()):+.3f}pp")
    print(f"  {lbl}: the gate discards {100*(1-len(tg)/len(tu)):.0f}% of days")

print("\n" + "=" * 78)
print("3. ERA SPLIT + CONCENTRATION (gated episodes)")
print("=" * 78)
for lbl in ("V1 flushed basket", "V3 pair"):
    d = g[lbl]
    if d is None:
        continue
    print(f"\n  {lbl}:")
    show(era_split(d["epi"], d["ep"]), f"{lbl} era split")
    print("   " + cluster_note(d["epi"], d["ep"], k=3))
    yr = pd.Series(d["ep"]).groupby(d["epi"].year.values).agg(["mean", "count"])
    yr["mean"] = (100 * yr["mean"]).round(2)
    print("   per-year (mean %, n): " +
          "  ".join(f"{y}:{r['mean']:+.2f}/{int(r['count'])}" for y, r in yr.iterrows()))
    print("   episode dates: " + ", ".join(str(x.date()) for x in d["epi"]))

print("\n" + "=" * 78)
print("4. THRESHOLD SENSITIVITY (gated, episodes, h=5) -- V1 then V3")
print("=" * 78)
for nmin in (2, 3, 4, 5):
    for rk in (5, 10, 20):
        MM, vv1, _ = build(GROUP, rank_max=rk)
        cc = MM.sum(axis=1)
        tt = IDX[((cc >= nmin) & gate).values]
        pp = vv1 - fwd_lag(C["XLP"], H, 1)
        va = vv1.dropna().index
        t2 = pd.DatetimeIndex(tt).intersection(va)
        if len(t2) < 5:
            print(f"  n>={nmin} rank<={rk}: {len(t2)} days -- too few")
            continue
        e2 = declusters(t2, H, va)
        a = vv1.loc[e2].values
        b = pp.loc[e2].dropna().values
        wa = int((a > 0).sum()); wb = int((b > 0).sum())
        print(f"  n>={nmin} rank<={rk}: days {len(t2):4d} epi {len(e2):3d}  "
              f"V1 {100*a.mean():+.3f}% rec {wa}-{len(a)-wa} p {sign_test(wa,len(a)):.3f}  |  "
              f"V3 {100*b.mean():+.3f}% rec {wb}-{len(b)-wb} p {sign_test(wb,len(b)):.3f}")

print("\n  XLP floor sensitivity (n>=3, rank<=10):")
for fl in (0, 15, 25, 35, 50):
    tt = IDX[(base_mask & (xlp_r5 >= fl).fillna(False)).values]
    va = v1.dropna().index
    t2 = pd.DatetimeIndex(tt).intersection(va)
    e2 = declusters(t2, H, va)
    a = v1.loc[e2].values
    b = pair.loc[e2].dropna().values
    wa = int((a > 0).sum()); wb = int((b > 0).sum())
    print(f"    XLP r5 >= {fl:2d}: days {len(t2):4d} epi {len(e2):3d}  "
          f"V1 {100*a.mean():+.3f}% rec {wa}-{len(a)-wa}  |  V3 {100*b.mean():+.3f}% "
          f"rec {wb}-{len(b)-wb}")

print("\n  membership sensitivity (add MDLZ+KDP, n>=3 rank<=10, gated):")
Mw, vw1, _ = build(WIDE)
cw = Mw.sum(axis=1)
tw = IDX[((cw >= N_MIN) & gate).values]
cell(vw1, tw, H, "V1 wide (12 names) LONG")
cell(vw1 - fwd_lag(C["XLP"], H, 1), tw, H, "V3 wide pair")

print("\n" + "=" * 78)
print("5. COST")
print("=" * 78)
print("  single-stock round trip ~5 bps each. An EQUAL-WEIGHT basket costs ~5 bps")
print("  of NOTIONAL (each leg is 1/n of capital); the house's sum-of-legs")
print("  convention would read 5 x n. Both are quoted below; the pair adds XLP at")
print("  ~4 bps.")
for lbl in ("V1 flushed basket", "V3 pair"):
    d = g[lbl]
    if d is None:
        continue
    bps = d["mean_pct"] * 100
    nlegs = int(M.sum(axis=1).loc[trig_gated].mean().round())
    rt_w = 5.0 if lbl.startswith("V1") else 9.0
    rt_sum = 5.0 * nlegs + (0 if lbl.startswith("V1") else 4.0)
    print(f"  {lbl}: episode mean {d['mean_pct']:+.3f}% = {bps:+.1f} bps; avg legs "
          f"{nlegs}; weighted RT {rt_w} bps -> {bps/rt_w:.1f}x; "
          f"sum-of-legs RT {rt_sum:.0f} bps -> {bps/rt_sum:.1f}x")

print("\n" + "=" * 78)
print("6. EVENT-IN-WINDOW + horizon scan (gated V1 and V3)")
print("=" * 78)
for lbl, ret in (("V1", v1), ("V3 pair", pair)):
    va = ret.dropna().index
    tg = pd.DatetimeIndex(trig_gated).intersection(va)
    e = declusters(tg, H, va)
    fl = event_in_window(e, IDX, H, 1, ("cpi", "fomc_decision"))
    ep = ret.loc[e].values
    print(f"  {lbl}: CPI/FOMC IN window N={int(fl.sum())} mean "
          f"{100*ep[fl].mean():+.3f}% | OUT N={int((~fl).sum())} mean "
          f"{100*ep[~fl].mean():+.3f}%")
    rows = []
    for h in (1, 2, 3, 5, 7, 10):
        MM, vv1, _ = build(GROUP, h=h)
        r = vv1 - fwd_lag(C["XLP"], h, 1) if lbl == "V3 pair" else vv1
        va2 = r.dropna().index
        t2 = pd.DatetimeIndex(IDX[(base_mask & gate).values]).intersection(va2)
        e2 = declusters(t2, h, va2)
        v = r.loc[e2].values
        w = int((v > 0).sum())
        base = float(r.loc[va2].mean())
        rows.append({"h": h, "n": len(e2), "mean_pct": round(100*v.mean(), 3),
                     "ctl_all": round(100*base, 3),
                     "edge_pp": round(100*(v.mean()-base), 3),
                     "hit": round(100*(v > 0).mean(), 1),
                     "rec": f"{w}-{len(v)-w}",
                     "sign_p": round(sign_test(w, len(v)), 4)})
    show(rows, f"{lbl} horizon scan (gated episodes)")
