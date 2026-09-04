"""C11 round 1 — long QQQ while tech's 63d rank is bottom-quintile and the
index's is not.

Provenance: BY-PRODUCT of the 2026-08-24 kill of long SPY / short QQQ
(data/pitch_negative_registry.md, "Cells swept and empty (2026-08-24)"):
"on days tech's 63-day rank is bottom-quintile while the index's is not,
QQQ LONG pays +0.508% at h=5".  Script c3_c8_spy_qqq_pair.py trigger B.

The PRE-SPECIFIED object is exactly that sentence: long QQQ, h=5, on
  QQQ r63 rank <= 20  AND  SPY r63 rank > 20
and it carries NO multiplicity charge.  Everything else in this file is a
GRID and is reported separately with its charge (script e1b).

Obligations discharged here: (0) premise on VALID SESSIONS with a pad-basis
contrast, (1) the pre-specified cell alone + battery, (2) mask overlap with
the corpse it came from, both directions, (3) THE DIAL, (4) 200d tape
over-selection, (5) concentration + LOYO, (6) era incl. 2000-02/2008 removal,
(7) midterm split, (8) definition neighbours, (9) gate attribution.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

pd.set_option("display.width", 250)
ROOT = Path(__file__).resolve().parents[3]

NAMES = ["QQQ", "SPY", "XLK", "SMH", "IWM", "^NDX", "DIA"]
pxa = load_prices(NAMES)
CAL = pxa["SPY"]["Close"].dropna().index
px = pd.DataFrame({t: pxa[t]["Close"] for t in NAMES}).reindex(CAL)

print("=" * 104)
print("0. PREMISE on VALID SESSIONS (pitch_lab.pct_rank -> rolling_on_valid), pad-basis contrast")
print("=" * 104)


def rank_valid(t, n, lb=252):
    return pct_rank(pxa[t]["Close"].dropna(), n, lb).reindex(CAL)


def rank_pad(t, n, lb=252):
    s = px[t]
    r = s.pct_change(n)
    return r.rolling(lb).rank(pct=True) * 100.0


R = {t: rank_valid(t, 63) for t in NAMES}
for t in NAMES:
    print(f"  {t:6s} r63 valid-session = {R[t].iloc[-1]:5.1f}   "
          f"pad-basis = {rank_pad(t, 63).iloc[-1]:5.1f}   "
          f"delta = {R[t].iloc[-1] - rank_pad(t, 63).iloc[-1]:+.2f}")
print(f"  last bar = {CAL[-1].date()}   (all US-listed, same calendar -> pad and valid agree)")

qr, sr = R["QQQ"], R["SPY"]
CELL = ((qr <= 20) & (sr > 20)).fillna(False)          # PRE-SPECIFIED, registry prose
CELL_SCRIPT = ((qr <= 20) & (sr >= 25)).fillna(False)  # what c3_c8 trigger B literally coded
print(f"\n  PRE-SPEC cell (QQQ r63<=20 & SPY r63>20) fires today: {bool(CELL.iloc[-1])}   "
      f"days ever = {int(CELL.sum())}")
print(f"  08-24 script form (SPY r63>=25)          fires today: {bool(CELL_SCRIPT.iloc[-1])}   "
      f"days ever = {int(CELL_SCRIPT.sum())}   "
      f"(NOTE: today SPY r63={sr.iloc[-1]:.1f} -> the SCRIPT form does NOT fire)")

print("\n" + "=" * 104)
print("1. THE PRE-SPECIFIED CELL, ALONE, UNCHARGED: long QQQ, h=5, lag=1")
print("=" * 104)
battery(px, CELL, [("QQQ", 1.0)], 5, "PRE-SPEC long QQQ | QQQ r63<=20 & SPY r63>20",
        cost_bps=2.5, min_gap=5)

ret5 = vehicle_ret(px, [("QQQ", 1.0)], 5)
valid5 = ret5.dropna().index
trig5 = CAL[CELL.values].intersection(valid5)
epi5 = declusters(trig5, 5, valid5)
ep5 = ret5.loc[epi5].values
base_up = float((ret5.loc[valid5] > 0).mean())
w = int((ep5 > 0).sum())
print(f"\n  SIGN TEST vs QQQ's OWN unconditional h=5 up-rate {100*base_up:.1f}%: "
      f"{w}-{len(ep5)-w}, p = {sign_test(w, len(ep5), base_up):.4f}   "
      f"(vs coin p = {sign_test(w, len(ep5)):.4f})")
print(f"  bootstrap P(mean<=0) on episodes = {bootstrap_p_le0(ep5):.4f}")

print("\n" + "=" * 104)
print("2. IS IT THE CORPSE? mask overlap vs the 2026-08-24 killed pair's triggers")
print("=" * 104)


def dist_52wh(c, look=252):
    return (c / c.rolling(look).max() - 1.0)


spy_d = rolling_on_valid(px["SPY"], lambda x: x / x.rolling(252).max() - 1.0)
qqq_d = rolling_on_valid(px["QQQ"], lambda x: x / x.rolling(252).max() - 1.0)
gap = spy_d - qqq_d
PAIR_A = ((gap >= 0.0272 - 1e-9) & (spy_d > -0.03)).fillna(False)   # pitched trigger A
PAIR_B = CELL_SCRIPT                                                # trigger B, the by-product's home

A = set(CAL[CELL.values])
for lbl, m in (("pair trigger A (gap>=2.72pp & SPY>-3%)", PAIR_A),
               ("pair trigger B (QQQ<=20 & SPY>=25)", PAIR_B)):
    B = set(CAL[m.values])
    i = A & B
    print(f"  {lbl:44s} n={len(B):>4d}  shared {len(i):>4d}  "
          f"-> {100*len(i)/max(len(A),1):5.1f}% of C11, {100*len(i)/max(len(B),1):5.1f}% of it")
print(f"  C11 n={len(A)}")
print("  READ: C11 is the SAME OBJECT as pair-trigger-B traded outright (the short leg flipped "
      "to an outright long). Trigger A, the leg the 08-24 pitch actually proposed, is a "
      "different mask.")

print("\n" + "=" * 104)
print("3. THE DIAL — data/rd2_fragility.parquet, ma10 of the 63d column")
print("=" * 104)
frg = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
frg.index = pd.to_datetime(frg.index)
ma10 = frg["63d"].rolling(10).mean()
TODAY_DIAL = float(ma10.iloc[-1])
print(f"  series {frg.index[0].date()} .. {frg.index[-1].date()}   today ma10(63d) = {TODAY_DIAL:.1f}")
print("  VINTAGE: rows before 2026-07-02 are the RECOMPUTE vintage; 2026-07-02+ is append-only PIT. "
      "Everything below uses the file as-is (recompute vintage for all history before 2026-07-02).")

dv = ma10.reindex(epi5)
have = dv.notna()
print(f"\n  episodes with ANY dial reading: {int(have.sum())} of {len(epi5)} "
      f"(dial starts {ma10.dropna().index[0].date()})")
if have.sum():
    print(f"  MAX dial ever observed on this cell = {dv[have].max():.1f}   vs TODAY {TODAY_DIAL:.1f}")
    print(f"  dial distribution on the cell: min {dv[have].min():.1f}  p50 {dv[have].median():.1f}  "
          f"p90 {dv[have].quantile(0.9):.1f}")
    rows = []
    for lbl, sel in (("dial < 50", dv < 50), ("dial >= 50", dv >= 50),
                     ("dial < 70", dv < 70), ("dial >= 70", dv >= 70),
                     ("dial >= 85 (today's zone)", dv >= 85)):
        s = sel.fillna(False).values
        rows.append(summarize(ep5[s], f"{lbl} (N={int(s.sum())})"))
    show(rows, "3a. cell split by dial (episodes with a reading)")
    rows2 = [summarize(ep5[~have.values], f"NO dial reading, pre-2016 (N={int((~have).sum())})"),
             summarize(ep5[have.values], f"dial era 2016+ (N={int(have.sum())})")]
    show(rows2, "3b. dial-era vs pre-dial")
    for d, v in zip(epi5[have.values], ep5[have.values]):
        print(f"    {d.date()}  dial {ma10.loc[d]:5.1f}   ret {100*v:+.2f}%")

print("\n" + "=" * 104)
print("4. TAPE OVER-SELECTION — fraction of trigger days above SPY's 200d (repo base rate 71.6%)")
print("=" * 104)
sma200 = rolling_on_valid(px["SPY"], lambda x: x.rolling(200).mean())
above = (px["SPY"] > sma200)
print(f"  all days with a 200d: {100*above[sma200.notna()].mean():.1f}%")
print(f"  C11 trigger DAYS above 200d: {100*above.reindex(trig5).mean():.1f}%  (N={len(trig5)})")
print(f"  C11 EPISODES above 200d:     {100*above.reindex(epi5).mean():.1f}%  (N={len(epi5)})")
sel = above.reindex(epi5).fillna(False).values
show([summarize(ep5[sel], f"episodes ABOVE 200d (N={int(sel.sum())})"),
      summarize(ep5[~sel], f"episodes BELOW 200d (N={int((~sel).sum())})")],
     "4a. cell split by SPY vs 200d")
sma_pct = (px["SPY"] / sma200 - 1.0)
print(f"  today SPY is {100*sma_pct.iloc[-1]:+.2f}% over its 200d")
hi = (sma_pct.reindex(epi5) > 0.05).fillna(False).values
show([summarize(ep5[hi], f"episodes >5% OVER 200d (N={int(hi.sum())})"),
      summarize(ep5[~hi], f"episodes <=5% over (N={int((~hi).sum())})")],
     "4b. split at +5% over the 200d (today +8.1%)")

print("\n" + "=" * 104)
print("5. CONCENTRATION — top-2, best year, drop-best, drop-best-2, LOYO")
print("=" * 104)
print("  " + cluster_note(epi5, ep5))
order = np.argsort(ep5)[::-1]
print(f"  best episodes: " + ", ".join(
    f"{epi5[i].date()} {100*ep5[i]:+.2f}%" for i in order[:4]))
print(f"  worst episodes: " + ", ".join(
    f"{epi5[i].date()} {100*ep5[i]:+.2f}%" for i in order[-4:]))
d1, d2 = np.delete(ep5, order[0]), np.delete(ep5, order[:2])
show([summarize(ep5, "all episodes"), summarize(d1, "drop best"),
      summarize(d2, "drop best 2")], "5a. drop-best")
yrs = pd.DatetimeIndex(epi5).year
loyo = []
for y in sorted(set(yrs)):
    keep = ep5[yrs != y]
    loyo.append({"drop_year": y, "n_dropped": int((yrs == y).sum()),
                 "mean_pct": round(100 * keep.mean(), 3),
                 "t": round(float(keep.mean() / (keep.std(ddof=1) / np.sqrt(len(keep)))), 2)})
lo = pd.DataFrame(loyo).sort_values("mean_pct")
print("\n5b. LOYO (worst 5):")
print(lo.head(5).to_string(index=False))
print(f"  LOYO mean floor = {lo['mean_pct'].min():+.3f}%   t floor = {lo['t'].min():+.2f}")
byyr = pd.Series(ep5, index=yrs).groupby(level=0).agg(["count", "mean", "sum"])
byyr["mean"] *= 100
byyr["sum"] *= 100
print("\n5c. by year (episodes):")
print(byyr.round(3).to_string())

print("\n" + "=" * 104)
print("6. ERA — pre/post 2018, and with the tech-bear artifacts removed")
print("=" * 104)
show(era_split(epi5, ep5), "6a. pre-2018 / 2018+")
dts = pd.DatetimeIndex(epi5)
for lbl, keep in (("ex 2000-2002 (dotcom)", ~dts.year.isin([2000, 2001, 2002])),
                  ("ex 2008-2009 (GFC)", ~dts.year.isin([2008, 2009])),
                  ("ex 2022 (tech bear)", dts.year != 2022),
                  ("ex 2000-02 & 2008-09 & 2022", ~dts.year.isin([2000, 2001, 2002, 2008, 2009, 2022])),
                  ("2010+ only", dts.year >= 2010),
                  ("2015+ only", dts.year >= 2015)):
    show([summarize(ep5[keep.values if hasattr(keep, "values") else keep], lbl)], "")

print("\n" + "=" * 104)
print("7. MIDTERM SPLIT (year %% 4 == 2)")
print("=" * 104)
mid = (dts.year % 4 == 2)
show([summarize(ep5[mid], f"midterm years (N={int(mid.sum())})"),
      summarize(ep5[~mid], f"non-midterm (N={int((~mid).sum())})")], "")
print("  midterm episode years:", sorted(set(dts.year[mid])))

print("\n" + "=" * 104)
print("8. DEFINITION NEIGHBOURS (this is a GRID — charged in e1b, reported raw here)")
print("=" * 104)
grid = []
for lb in (42, 63, 126):
    rq = rank_valid("QQQ", lb)
    rs = rank_valid("SPY", lb)
    for q in (15, 20, 25, 30):
        for sthr in (20, 25, 30):
            m = ((rq <= q) & (rs > sthr)).fillna(False)
            t = CAL[m.values].intersection(valid5)
            if len(t) == 0:
                continue
            e = declusters(t, 5, valid5)
            v = ret5.loc[e].values
            grid.append({"lb": lb, "q<=": q, "spy>": sthr, "n_days": len(t), "n_epi": len(e),
                         "mean_pct": round(100 * v.mean(), 3),
                         "hit": round(100 * (v > 0).mean(), 1),
                         "t": round(float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))), 2)
                         if len(v) > 1 else np.nan,
                         "live": bool(m.iloc[-1])})
g = pd.DataFrame(grid)
print(g.to_string(index=False))
print(f"\n  grid cells = {len(g)};  positive mean in {int((g['mean_pct']>0).sum())} of {len(g)};  "
      f"|t|>=2 in {int((g['t'].abs()>=2).sum())};  max t = {g['t'].max():.2f} "
      f"({g.loc[g['t'].idxmax(), ['lb','q<=','spy>']].to_dict()})")
print(f"  cells LIVE today = {int(g['live'].sum())} of {len(g)}; "
      f"of those, positive mean = {int((g['live'] & (g['mean_pct']>0)).sum())}")

print("\n" + "=" * 104)
print("9. GATE ATTRIBUTION — does 'and the index is not' do anything?")
print("=" * 104)
rows = []
for lbl, m in (("QQQ r63<=20 ALONE (no SPY leg)", (qr <= 20)),
               ("PRE-SPEC: QQQ<=20 & SPY>20", CELL),
               ("QQQ<=20 & SPY<=20 (the EXCLUDED half)", (qr <= 20) & (sr <= 20)),
               ("SPY r63>20 ALONE", (sr > 20)),
               ("ALL DAYS", pd.Series(True, index=CAL))):
    mm = m.fillna(False) if hasattr(m, "fillna") else m
    t = CAL[mm.values].intersection(valid5)
    e = declusters(t, 5, valid5)
    v = ret5.loc[e].values
    r = summarize(v, f"{lbl}")
    r["n_days"] = len(t)
    rows.append(r)
show(rows, "9a. gate on / gate off / excluded half (h=5 episodes)")
mA = ((qr <= 20)).fillna(False)
tA = CAL[mA.values].intersection(valid5)
eA = declusters(tA, 5, valid5)
print(f"  GATE DELTA: pre-spec {100*ep5.mean():+.3f}%  minus  parent-alone "
      f"{100*ret5.loc[eA].mean():+.3f}%  =  {100*(ep5.mean()-ret5.loc[eA].mean()):+.3f}pp")
print(f"  the SPY gate removes {len(eA)-len(epi5)} of {len(eA)} parent episodes")
