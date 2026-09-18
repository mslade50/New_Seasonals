"""A1 ROUND 2 -- the mandatory five.

Round 1 (k1_tlt_ig_floor.py) found the PARKED ARITHMETIC DOES NOT REPRODUCE on
the arm's own words: the watchlist says "the FIRST trigger day in >= 10 trading
sessions" and that population has 11 members (10 resolvable), not 17/18. The
parked 17 came from pitch_lab.declusters, which restarts the clock at the last
KEPT day and therefore re-anchors MID-cluster. Forensics here.

Then:
  1. decluster + concentration
  2. definition neighbours -- the 2026-09-07 note claims BOTH IEF/LQD rung
     neighbours are wrong-signed (0.5% -0.173pp, 1.0% +0.385pp, 1.5% -0.073pp).
     Reproduce that under the FRESHNESS population, which is what actually arms.
  3. era / midterm / rate-regime split
  4. GATE ATTRIBUTION on all three legs, day level AND episode level (the
     2026-09-07 anchor-swap trap: the IEF leg moved the episode mean +0.321pp
     purely by re-anchoring Sept-2022)
  5. reference class: the identical rule on the duration/credit family
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

ASOF = pd.Timestamp("2026-09-10")
TK = ["TLT", "IEF", "LQD", "AGG", "TIP", "SPY", "^TNX"]
raw = load_prices(TK)
IDX = raw["TLT"].index
PX = pd.DataFrame({t: raw[t]["Close"].reindex(IDX) for t in TK})
PX = PX.rename(columns={"^TNX": "TNX"})
PX["TNX"] = PX["TNX"].ffill()
POS = pd.Series(range(len(IDX)), index=IDX)


def above_low(t, n=252):
    s = raw[t]["Close"]
    return ((s / s.rolling(n).min() - 1.0) * 100).reindex(IDX)


def first_in(mask, gap=10):
    days = IDX[mask.reindex(IDX, fill_value=False).values]
    keep, last = [], -10 ** 9
    for d in days:
        p = int(POS[d])
        if p - last >= gap:
            keep.append(d)
        last = p
    return pd.DatetimeIndex(keep)


def rung(tlt=0.5, ief=1.0, lqd=1.0, n=252):
    return ((above_low("TLT", n) <= tlt) & (above_low("IEF", n) <= ief)
            & (above_low("LQD", n) <= lqd)).fillna(False)


R1 = fwd_lag(PX["TLT"], 1, 1)
BASE = R1.dropna().mean()

print("=" * 78)
print("0. REPRODUCTION FORENSICS -- why 17 and not 11")
print("=" * 78)
T = rung()
days = IDX[T.values]
epi_fresh = first_in(T, 10)
epi_dec = declusters(days, 10, IDX)
print(f"  trigger DAYS all history                 : {len(days)}")
print(f"  pitch_lab.declusters(gap=10)             : {len(epi_dec)}  "
      f"<- the parked convention")
print(f"  FIRST trigger day in >= 10 td (the ARM)  : {len(epi_fresh)}")
extra = pd.DatetimeIndex(epi_dec).difference(epi_fresh)
print(f"  in declusters but NOT fresh-first ({len(extra)}): "
      f"{[str(d.date()) for d in extra]}")
print("  those are MID-CLUSTER re-anchors: a trigger fired the session before, "
      "so they are not what the arm describes.")
for d in extra:
    p = int(POS[d])
    prev = [x for x in days if x < d]
    g = p - int(POS[prev[-1]]) if prev else None
    print(f"    {d.date()}: gap since previous TRIGGER day = {g} td, "
          f"r={100*R1.get(d, np.nan):+.3f}%")
show([summarize(R1.loc[epi_dec].dropna().values,
                f"declusters population (N={len(R1.loc[epi_dec].dropna())})"),
      summarize(R1.loc[epi_fresh].dropna().values,
                f"fresh-first population (N={len(R1.loc[epi_fresh].dropna())})"),
      summarize(R1.loc[extra].dropna().values,
                f"the mid-cluster re-anchors only (N={len(R1.loc[extra].dropna())})")],
     "the two populations")
# vintage check: what did the parked script see on 2026-08-12?
d812 = days[days <= pd.Timestamp("2026-08-12")]
print(f"\n  trigger days through 2026-08-12 (parked vintage): {len(d812)}; "
      f"declusters -> {len(declusters(d812, 10, IDX))} "
      f"(parked reported 17 episodes + 52 later = 69 days)")

print("\n" + "=" * 78)
print("1. DECLUSTER + CONCENTRATION on the ARM's own population")
print("=" * 78)
v = R1.loc[epi_fresh].dropna()
print(f"  {cluster_note(v.index, v.values)}")
print("  leave-one-year-out:")
for y in sorted(set(v.index.year)):
    s = v[v.index.year != y]
    w = int((s > 0).sum())
    print(f"    drop {y}: N={len(s):2d} {100*s.mean():+.3f}% "
          f"excess {100*(s.mean()-BASE):+.3f}pp  {w}-{len(s)-w} "
          f"sign p {sign_test(w, len(s)):.4f}")
print("  drop the single best episode:")
s = v.drop(v.idxmax())
w = int((s > 0).sum())
print(f"    N={len(s)} {100*s.mean():+.3f}% excess {100*(s.mean()-BASE):+.3f}pp "
      f"{w}-{len(s)-w} sign p {sign_test(w, len(s)):.4f}")

print("\n" + "=" * 78)
print("2. DEFINITION NEIGHBOURS -- the knife edge")
print("=" * 78)
print("  (a) hold TLT at 0.5%, walk the IEF/LQD rung together:")
for k in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0):
    e = first_in(rung(0.5, k, k), 10)
    vv = R1.loc[e].dropna()
    if len(vv) == 0:
        print(f"    IEF/LQD <= {k:4.2f}%: no episodes")
        continue
    w = int((vv > 0).sum())
    print(f"    IEF/LQD <= {k:4.2f}%: N={len(vv):3d} {100*vv.mean():+.3f}% "
          f"excess {100*(vv.mean()-BASE):+.3f}pp hit {100*(vv>0).mean():.1f}% "
          f"sign p {sign_test(w, len(vv)):.4f}")
print("  (b) hold IEF/LQD at 1.0%, walk the TLT rung:")
for k in (0.10, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
    e = first_in(rung(k, 1.0, 1.0), 10)
    vv = R1.loc[e].dropna()
    if len(vv) == 0:
        continue
    w = int((vv > 0).sum())
    print(f"    TLT <= {k:4.2f}%: N={len(vv):3d} {100*vv.mean():+.3f}% "
          f"excess {100*(vv.mean()-BASE):+.3f}pp hit {100*(vv>0).mean():.1f}% "
          f"sign p {sign_test(w, len(vv)):.4f}")
print("  (c) the LOW-window lookback (252 is a choice):")
for n in (126, 189, 252, 315, 378, 504):
    e = first_in(rung(0.5, 1.0, 1.0, n), 10)
    vv = R1.loc[e].dropna()
    if len(vv) == 0:
        continue
    w = int((vv > 0).sum())
    print(f"    lookback {n:3d}: N={len(vv):3d} {100*vv.mean():+.3f}% "
          f"excess {100*(vv.mean()-BASE):+.3f}pp hit {100*(vv>0).mean():.1f}% "
          f"sign p {sign_test(w, len(vv)):.4f}")
print("  (d) the same walk under the PARKED (declusters) convention, to see "
      "whether the knife edge is a property of the cell or of the convention:")
for k in (0.5, 1.0, 1.5, 2.0):
    dd = IDX[rung(0.5, k, k).values]
    e = declusters(dd, 10, IDX)
    vv = R1.loc[e].dropna()
    w = int((vv > 0).sum())
    print(f"    IEF/LQD <= {k:4.2f}%: N={len(vv):3d} {100*vv.mean():+.3f}% "
          f"excess {100*(vv.mean()-BASE):+.3f}pp hit {100*(vv>0).mean():.1f}% "
          f"sign p {sign_test(w, len(vv)):.4f}")

print("\n" + "=" * 78)
print("3. ERA / MIDTERM / RATE REGIME")
print("=" * 78)
show(era_split(pd.DatetimeIndex(v.index), v.values), "pre-2018 / 2018+")
mt = pd.DatetimeIndex(v.index).year % 4 == 2
show([summarize(v[mt].values, f"MIDTERM (N={int(mt.sum())})"),
      summarize(v[~mt].values, f"non-midterm (N={int((~mt).sum())})")], "cycle")
tnx = PX["TNX"]
rising = (tnx - tnx.shift(252)) > 0
show([summarize(v[rising.loc[v.index].values].values, "yields UP over 252 sessions"),
      summarize(v[~rising.loc[v.index].values].values, "yields DOWN")],
     "rate regime (by construction the cell is nearly all rising)")
hi_lvl = tnx.loc[v.index] >= 3.0
show([summarize(v[hi_lvl.values].values, f"^TNX >= 3.0% (N={int(hi_lvl.sum())})"),
      summarize(v[~hi_lvl.values].values, f"^TNX < 3.0% (N={int((~hi_lvl).sum())})")],
     "yield LEVEL regime -- today ^TNX is 4.94")

print("\n" + "=" * 78)
print("4. GATE ATTRIBUTION, DAY LEVEL AND EPISODE LEVEL, ALL THREE LEGS")
print("=" * 78)
TL, IE, LQ = above_low("TLT"), above_low("IEF"), above_low("LQD")
legs = {
    "TLT<=0.5 ALONE":                (TL <= 0.5),
    "TLT<=0.5 & IEF<=1.0":           (TL <= 0.5) & (IE <= 1.0),
    "TLT<=0.5 & LQD<=1.0":           (TL <= 0.5) & (LQ <= 1.0),
    "TLT<=0.5 & IEF & LQD (CELL)":   rung(),
    "IEF<=1.0 & LQD<=1.0, no TLT":   (IE <= 1.0) & (LQ <= 1.0),
}
for lbl, m in legs.items():
    m = m.fillna(False)
    dd = IDX[m.values]
    dv = R1.loc[dd].dropna()
    ev = R1.loc[first_in(m, 10)].dropna()
    print(f"  {lbl:<30s} DAY n={len(dv):4d} {100*dv.mean():+.3f}% "
          f"hit {100*(dv>0).mean():.1f}%  |  FRESH n={len(ev):3d} "
          f"{100*ev.mean():+.3f}% hit {100*(ev>0).mean():.1f}%")
print("\n  DISCARDED COMPLEMENTS (does the gate keep the good half?):")
cell = rung().fillna(False)
for lbl, m in (("IEF gate: TLT&LQD pass but IEF FAILS",
                ((TL <= 0.5) & (LQ <= 1.0) & (IE > 1.0)).fillna(False)),
               ("LQD gate: TLT&IEF pass but LQD FAILS",
                ((TL <= 0.5) & (IE <= 1.0) & (LQ > 1.0)).fillna(False)),
               ("TLT gate: IEF&LQD pass but TLT FAILS",
                ((IE <= 1.0) & (LQ <= 1.0) & (TL > 0.5)).fillna(False))):
    dd = IDX[m.values]
    dv = R1.loc[dd].dropna()
    ev = R1.loc[first_in(m, 10)].dropna()
    print(f"  {lbl:<38s} DAY n={len(dv):4d} {100*dv.mean():+.3f}%  |  "
          f"FRESH n={len(ev):3d} {100*ev.mean():+.3f}% "
          f"hit {100*(ev>0).mean() if len(ev) else float('nan'):.1f}%")
print("\n  ANCHOR-SWAP CHECK: which fresh anchors move when IEF/LQD are dropped?")
a_full = list(first_in(rung(), 10))
a_tlt = list(first_in((TL <= 0.5).fillna(False), 10))
print(f"    fresh anchors with all three legs ({len(a_full)}): "
      f"{[str(d.date()) for d in a_full]}")
print(f"    fresh anchors on TLT alone ({len(a_tlt)}): "
      f"{[str(d.date()) for d in a_tlt]}")

print("\n" + "=" * 78)
print("5. REFERENCE CLASS -- the identical rule across the duration/credit family")
print("=" * 78)
fam = ["TLT", "IEF", "LQD", "AGG", "TIP"]
res = []
for t in fam:
    r = fwd_lag(PX[t], 1, 1)
    b = r.dropna().mean()
    vv = r.loc[epi_fresh].dropna()
    w = int((vv > 0).sum())
    res.append((t, len(vv), 100 * vv.mean(), 100 * (vv.mean() - b),
                100 * (vv > 0).mean(), sign_test(w, len(vv)),
                100 * r.dropna().std(ddof=1)))
res.sort(key=lambda x: -x[3])
print(f"  {'vehicle':<8}{'N':>4}{'mean%':>9}{'excess pp':>11}{'hit%':>7}"
      f"{'signp':>8}{'daily sd%':>11}")
for t, n, m_, e_, h_, p_, sd in res:
    print(f"  {t:<8}{n:>4}{m_:>9.3f}{e_:>11.3f}{h_:>7.1f}{p_:>8.4f}{sd:>11.3f}")
rank = [t for t, *_ in res].index("TLT") + 1
print(f"  TLT ranks {rank} of {len(res)} on excess. Excess per unit of daily sd:")
for t, n, m_, e_, h_, p_, sd in res:
    print(f"    {t:<6} {e_/sd:+.3f}")
# curve trade -- is this duration or is it TLT specifically?
rc = fwd_lag(PX["TLT"], 1, 1) - fwd_lag(PX["IEF"], 1, 1)
vc = rc.loc[epi_fresh].dropna()
print(f"  TLT minus IEF (duration-neutralised-ish): N={len(vc)} "
      f"{100*vc.mean():+.3f}% vs all-days {100*rc.dropna().mean():+.3f}% "
      f"hit {100*(vc>0).mean():.1f}%")
