"""C11 -- fragility dial ma10(63d) >= 85 while the VIX is calm (r21 <= 25),
tested on VOLATILITY rather than on direction.

Live 2026-08-28: ma10(63d) 87.6 (raw 63d 89.0), ^VIX r21 18.3, VIX 14.43,
VIX3M 17.48 (its exact 252d minimum), ratio 0.8255 (15.9th pctile).

Trade: the vol-EXPANSION expression -- SHORT SVXY (and, priced separately,
LONG UVXY). NOT the directional index claim killed 2026-08-27 (a8_c8_dial85)
and recorded dead in CLAUDE.md at PIT t = -0.23.

Kill tests, in order (vintage FIRST -- it has already flipped one cell):
 1. VINTAGE: run everything on data/rd2_fragility.parquet (SIZING, append-only
    PIT only since 2026-07-02) AND data/rd2_fragility_ts.parquet (raw-basis
    research recompute). Report agreement. Disagreement alone is a kill.
 2. Sample size / coverage: days, declustered episodes, distinct YEARS.
 3. Gate attribution: bare dial>=85 / bare calm-VIX / joint.
 4. MECHANISM, separate from P&L: does realized vol over the next 5/10/21
    sessions rise vs trailing realized vol, MORE than on matched calm-VIX days
    without the dial clause?
 5. Cost AND CARRY: the unconditional short-SVXY drift over the hold IS the
    carry; charge it. Plus the SVXY/UVXY pass-through ratio to spot ^VIX.
 6. Era / midterm / concentration / sign test.
 7. Threshold ladder: dial >= 70/75/80/85/90 x VIX r21 <= 15/25/35.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

TK = ["SPY", "SVXY", "UVXY"]
pm = load_prices(TK)
px = pd.DataFrame({t: pm[t]["Close"] for t in TK}).dropna(how="all")

raw = pd.read_parquet(ROOT / "data" / "master_prices.parquet")
raw["date"] = pd.to_datetime(raw["date"])


def ser(t, col="Close"):
    g = raw[raw.ticker == t].sort_values("date").set_index("date")
    return g[~g.index.duplicated(keep="last")][col]


vix, vix3m, spy = ser("^VIX"), ser("^VIX3M"), ser("SPY")

# ---------------------------------------------------------------- 1. VINTAGE
print("=" * 78)
print("1. VINTAGE -- the sizing parquet vs the research recompute")
print("=" * 78)
VIN = {}
for name, f in (("SIZING (rd2_fragility)", "rd2_fragility.parquet"),
                ("RESEARCH (rd2_fragility_ts)", "rd2_fragility_ts.parquet")):
    p = ROOT / "data" / f
    if not p.exists():
        print(f"  {name}: ABSENT")
        continue
    d = pd.read_parquet(p)
    d.index = pd.to_datetime(d.index)
    VIN[name] = d["63d"].rolling(10).mean().dropna()
    print(f"  {name}: {len(d)} rows {d.index.min().date()} .. {d.index.max().date()}")

a, b = VIN["SIZING (rd2_fragility)"], VIN["RESEARCH (rd2_fragility_ts)"]
common = a.index.intersection(b.index)
diff = (a.loc[common] - b.loc[common])
print(f"\n  common span {common.min().date()} .. {common.max().date()} "
      f"({len(common)} days)")
print(f"  ma10(63d) difference: mean {diff.mean():+.2f}  sd {diff.std():.2f}  "
      f"max |diff| {diff.abs().max():.2f} on {diff.abs().idxmax().date()}")
for thr in (85,):
    ga, gb = (a.loc[common] >= thr), (b.loc[common] >= thr)
    agree = int((ga == gb).sum())
    print(f"  >= {thr} gate: SIZING ON {int(ga.sum())} days, RESEARCH ON "
          f"{int(gb.sum())} days, agree on {agree}/{len(common)} "
          f"({100*agree/len(common):.1f}%), BOTH ON {int((ga & gb).sum())}")
print("  NOTE: the research recompute ENDS 2026-05-07, so it cannot see the")
print("  Aug-2026 episode that today's reading belongs to. Any 2026 evidence")
print("  in this cell exists on ONE vintage only.")

# ---------------------------------------------------------- masks per vintage
vix_r21 = pct_rank(vix, 21, 252)


def build(dial: pd.Series, idx: pd.DatetimeIndex, dthr=85, vthr=25):
    d = dial.reindex(idx)
    v = vix_r21.reindex(idx)
    return ((d >= dthr).fillna(False), (v <= vthr).fillna(False),
            ((d >= dthr) & (v <= vthr)).fillna(False), d, v)


print("\n" + "=" * 78)
print("2. SAMPLE SIZE AND COVERAGE, per vintage (SVXY calendar)")
print("=" * 78)
IS = px["SVXY"].dropna().index
for name, dial in VIN.items():
    MD, MV, MJ, dvals, vvals = build(dial, IS)
    trig = IS[MJ.values]
    epi = declusters(trig, 10, IS)
    yrs = sorted(set(trig.year))
    print(f"\n  {name}:")
    print(f"    dial>=85 days {int(MD.sum())} | VIX r21<=25 days {int(MV.sum())} "
          f"| JOINT days {len(trig)} | episodes(min_gap 10) {len(epi)}")
    print(f"    distinct years: {yrs}")
    if len(trig):
        print("    per-year day counts:",
              dict(pd.Series(1, index=trig).groupby(trig.year).sum()))

DIAL = VIN["SIZING (rd2_fragility)"]
MD, MV, MJ, dv, vv = build(DIAL, IS)
print(f"\n  live 2026-08-28 on SIZING vintage: ma10 {dv.iloc[-1]:.1f}, "
      f"VIX r21 {vv.iloc[-1]:.1f}, fires = {bool(MJ.iloc[-1])}")

# --------------------------------------------------- 3. GATE ATTRIBUTION
print("\n" + "=" * 78)
print("3. GATE ATTRIBUTION -- SHORT SVXY (long vol), episodes min_gap 10")
print("   [SVXY -0.5x era only, 2018-02-06+, so the vehicle is the live one]")
print("=" * 78)
E0 = pd.Timestamp("2018-02-06")
pxs = pd.DataFrame({"SVXY": px["SVXY"], "UVXY": px["UVXY"]}).dropna()
IS2 = pxs.index
MD2, MV2, MJ2, dv2, vv2 = build(DIAL, IS2)
era = IS2 >= E0
for h in (3, 5, 10):
    ret = vehicle_ret(pxs, [("SVXY", -1.0)], h, 1)
    valid = ret.notna().values & era
    rows = []
    for lbl, m in (("(a) dial>=85 only", MD2), ("(b) VIX r21<=25 only", MV2),
                   ("(c) JOINT", MJ2),
                   ("--- all days (2018+)", pd.Series(True, index=IS2))):
        t = IS2[m.reindex(IS2, fill_value=False).values & valid]
        e = declusters(t, 10, IS2)
        r = summarize(ret.loc[e].values, lbl)
        r["n_days"] = len(t)
        rows.append(r)
    show(rows, f"h={h} SHORT SVXY")
    A, B, C = (rows[0].get("mean_pct", np.nan), rows[1].get("mean_pct", np.nan),
               rows[2].get("mean_pct", np.nan))
    print(f"  JOIN VALUE h={h}: joint {C:+.3f}% vs better parent {max(A,B):+.3f}%"
          f"  -> {C-max(A,B):+.3f}pp")

# ---------------------------------------------------------- 4. MECHANISM TEST
print("\n" + "=" * 78)
print("4. MECHANISM (separate from P&L): does REALIZED VOL actually expand?")
print("   measure = ann. realized vol of SPY over the next h sessions divided")
print("   by its trailing 21d realized vol (ratio > 1 = expansion)")
print("=" * 78)
r = spy.pct_change()
trail = r.rolling(21).std()
IDX = spy.index
MDs, MVs, MJs, _, _ = build(DIAL, IDX)
rows = []
for h in (5, 10, 21):
    fwd = r.shift(-1).rolling(h).std().shift(-(h - 1))  # vol of D+1..D+h
    ratio = (fwd / trail)
    ok = ratio.notna()
    for lbl, m in (("JOINT dial>=85 & calmVIX", MJs),
                   ("calm VIX only", MVs),
                   ("dial>=85 only", MDs),
                   ("all days (dial era 2016+)",
                    pd.Series(IDX >= DIAL.index.min(), index=IDX))):
        t = IDX[m.reindex(IDX, fill_value=False).values & ok.values
                & (IDX >= DIAL.index.min())]
        e = declusters(t, 10, IDX)
        if not len(e):
            continue
        vals = ratio.loc[e].values
        rows.append({"h": h, "cell": lbl, "n_epi": len(e),
                     "mean_ratio": round(float(np.mean(vals)), 3),
                     "med_ratio": round(float(np.median(vals)), 3),
                     "pct_expand": round(100 * float((vals > 1).mean()), 1)})
show(rows, "realized-vol expansion ratio")

# ------------------------------------------------------- 5. COST AND CARRY
print("\n" + "=" * 78)
print("5. COST AND CARRY")
print("=" * 78)
rs = pxs["SVXY"].pct_change()
ru = pxs["UVXY"].pct_change()
rv = vix.reindex(pxs.index).pct_change()
m = (pxs.index >= E0)
ok = rs.notna() & rv.notna() & ru.notna() & pd.Series(m, index=pxs.index)
bs = np.polyfit(rv[ok].values, rs[ok].values, 1)[0]
bu = np.polyfit(rv[ok].values, ru[ok].values, 1)[0]
print(f"  PASS-THROUGH to spot ^VIX (2018-02-06+, N={int(ok.sum())} days):")
print(f"    SVXY beta to VIX = {bs:+.3f}  -> SHORT SVXY captures {-bs:+.3f} of a "
      f"VIX move")
print(f"    UVXY beta to VIX = {bu:+.3f}")
print("  i.e. a +10% VIX day delivers roughly "
      f"{-bs*10:+.1f}% to short SVXY and {bu*10:+.1f}% to long UVXY")

print("\n  CARRY = the unconditional drift of the position over the hold "
      "(2018-02-06+):")
for h in (3, 5, 10):
    for legs, lbl in (([("SVXY", -1.0)], "SHORT SVXY"), ([("UVXY", 1.0)], "LONG UVXY")):
        rr = vehicle_ret(pxs, legs, h, 1)
        base = rr[(pxs.index >= E0)].dropna()
        print(f"    h={h:2d} {lbl}: unconditional mean {100*base.mean():+.3f}%  "
              f"median {100*base.median():+.3f}%  hit {100*(base>0).mean():.1f}%")

ratio_ts = (vix / vix3m).dropna()
rr_pct = rolling_on_valid(ratio_ts, lambda x: x.rolling(252).rank(pct=True) * 100)
today_ratio = float(ratio_ts.iloc[-1])
print(f"\n  term structure today: VIX/VIX3M {today_ratio:.4f} "
      f"(trailing-252 pctile {rr_pct.iloc[-1]:.1f})")
print("  carry in TODAY'S term-structure bucket (ratio <= 0.85, 2018+):")
steep = (ratio_ts.reindex(pxs.index) <= 0.85)
for h in (3, 5, 10):
    for legs, lbl in (([("SVXY", -1.0)], "SHORT SVXY"), ([("UVXY", 1.0)], "LONG UVXY")):
        rr = vehicle_ret(pxs, legs, h, 1)
        sel = rr[(pxs.index >= E0) & steep.fillna(False).values].dropna()
        print(f"    h={h:2d} {lbl}: mean {100*sel.mean():+.3f}%  "
              f"hit {100*(sel>0).mean():.1f}%  (N={len(sel)})")

# ------------------------------------------- 6/7. era, midterm, ladder
print("\n" + "=" * 78)
print("6/7. THRESHOLD LADDER (SHORT SVXY h=5, 2018+, episodes min_gap 10)")
print("=" * 78)
ret5 = vehicle_ret(pxs, [("SVXY", -1.0)], 5, 1)
base5 = ret5[(pxs.index >= E0)].dropna()
rows = []
for dthr in (70, 75, 80, 85, 90):
    for vthr in (15, 25, 35):
        _, _, MJx, _, _ = build(DIAL, IS2, dthr, vthr)
        t = IS2[MJx.values & ret5.notna().values & era]
        e = declusters(t, 10, IS2)
        if not len(e):
            rows.append({"label": f"dial>={dthr} & VIXr21<={vthr}", "n": 0,
                         "n_days": len(t)})
            continue
        v = ret5.loc[e].values
        rr = summarize(v, f"dial>={dthr} & VIXr21<={vthr}")
        rr["n_days"] = len(t)
        rr["edge_vs_all"] = round(rr["mean_pct"] - 100 * base5.mean(), 3)
        order = np.argsort(-v)
        rr["drop_best3"] = round(100 * float(np.mean(np.delete(v, order[:3]))), 3) \
            if len(v) > 3 else np.nan
        rows.append(rr)
show(rows, f"ladder (all-days 2018+ control = {100*base5.mean():+.3f}%)")

print("\n" + "=" * 78)
print("6b. ERA / MIDTERM / CONCENTRATION / SIGN TEST at the pitched rung")
print("=" * 78)
t = IS2[MJ2.values & ret5.notna().values & era]
e = declusters(t, 10, IS2)
v = ret5.loc[e].values
print(f"  episodes N={len(e)}: {[str(d.date()) for d in e]}")
if len(v):
    w = int((v > 0).sum())
    print(f"  mean {100*np.mean(v):+.3f}%  median {100*np.median(v):+.3f}%  "
          f"record {w}-{len(v)-w}  sign p = {sign_test(w, len(v)):.4f}")
    print(f"  vs all-days-2018+ control {100*base5.mean():+.3f}%  -> edge "
          f"{100*(np.mean(v)-base5.mean()):+.3f}pp")
    print(f"  {cluster_note(e, v, k=2)}")
    mid = np.array([(d.year % 4) == 2 for d in e])
    show([summarize(v[mid], f"midterm (N={int(mid.sum())})"),
          summarize(v[~mid], f"non-midterm (N={int((~mid).sum())})")], "midterm")

# also: FULL dial history including the pre-2018 SVXY era, stated not pooled
print("\n  [pre-2018 SVXY era is NOT pooled: the dial series starts 2016-07 and")
print("   SVXY was -1.0x until 2018-02-05, so only ~1.5 years overlap at a")
print("   different leverage. Day count in that overlap with the joint state:",
      int((MJ2.values & (IS2 < E0)).sum()), "]")
