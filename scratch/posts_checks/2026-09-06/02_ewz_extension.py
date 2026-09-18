"""EWZ extension: is the Friday 2026-09-04 state (5d rank 95.2, z10 1.6,
5d +6.5%) worth anything forward?

Three cell definitions, each tested on its own:
  R.  5d return trailing-252 percentile rank >= 95   (pitch_lab.pct_rank,
      the same definition build_pitch_state uses for `rank_5d`)
  Z.  z10 >= 1.5, where z10 = 10d return / (21d close-to-close sd * sqrt(10)).
      That is `build_pitch_state._metrics_for`'s definition, NOT
      pitch_lab.zscore (whose docstring claims the same thing and computes a
      252d standardisation of the 10d return instead). CLAUDE.md pins the
      _metrics_for form for anything that has to agree with the tape block,
      and the 1.6 reading quoted tonight came from the tape block.
  P.  plain 5d return >= +6%

Episodes are declustered at 5 td. Forward returns are LAG-1: the signal is
measured on the anchor close, entry is the NEXT close (MOC Tuesday from
tonight's Friday anchor), exit h sessions after that. Horizons 1/2/3/5/10.

Controls: all-days drift in the same lag-1 form, and the MIRROR cell (rank
<= 5, z10 <= -1.5, 5d <= -6%), which is the honest question for an extension
idea -- does the tail carry information at all, or only the down tail?

Splits: era 2018, midterm years (year %% 4 == 2), top-two concentration,
worst/best with dates, exact sign test alongside t.

Also run on SPY as the traded vehicle off EWZ's trigger (does a Brazil
extension say anything about the index? prior: no).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    cluster_note, declusters, era_split, fwd_lag, load_prices, pct_rank,
    sign_test, summarize, wilder_atr,
)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
ASOF = pd.Timestamp("2026-09-04")

raw = load_prices(["EWZ", "SPY"])
ewz, spy = raw["EWZ"], raw["SPY"]
ec = ewz["Close"].dropna()
sc = spy["Close"].dropna()
print(f"EWZ bars {ec.index[0].date()} .. {ec.index[-1].date()}  n={len(ec)}")

# ------------------------------------------------------------------ freeze
a = pd.Series(wilder_atr(ewz["High"], ewz["Low"], ewz["Close"]),
              index=ewz.index).reindex(ec.index)
print("\n=== FREEZE (Friday 2026-09-04) ===")
print(f"  EWZ close {ec.iloc[-1]:.4f}  Wilder-14 ATR {a.iloc[-1]:.4f} "
      f"({100*a.iloc[-1]/ec.iloc[-1]:.2f}% of close)")
sa = pd.Series(wilder_atr(spy["High"], spy["Low"], spy["Close"]),
               index=spy.index).reindex(sc.index)
print(f"  SPY close {sc.iloc[-1]:.4f}  Wilder-14 ATR {sa.iloc[-1]:.4f} "
      f"({100*sa.iloc[-1]/sc.iloc[-1]:.2f}% of close)")

# ------------------------------------------------------------ state series
rank5 = pct_rank(ec, 5, 252)
ret5 = ec.pct_change(5)
ret10 = ec.pct_change(10)
vol21 = ec.pct_change().rolling(21).std()
z10 = ret10 / (vol21 * np.sqrt(10))          # _metrics_for definition

print("\n=== does the state actually fire on Friday's bar? ===")
print(f"  EWZ rank_5d = {rank5.iloc[-1]:.1f}   (gate >= 95 -> "
      f"{'FIRES' if rank5.iloc[-1] >= 95 else 'no'})")
print(f"  EWZ z10     = {z10.iloc[-1]:.2f}    (gate >= 1.5 -> "
      f"{'FIRES' if z10.iloc[-1] >= 1.5 else 'no'})")
print(f"  EWZ ret_5d  = {100*ret5.iloc[-1]:.2f}%  (gate >= +6% -> "
      f"{'FIRES' if ret5.iloc[-1] >= 0.06 else 'no'})")
print(f"  (state file quoted rank 95.2 / z10 1.6 / 5d +6.5%)")

CELLS = {
    "R rank5>=95":  rank5 >= 95,
    "Z z10>=1.5":   z10 >= 1.5,
    "P ret5>=+6%":  ret5 >= 0.06,
}
MIRROR = {
    "R rank5<=5":   rank5 <= 5,
    "Z z10<=-1.5":  z10 <= -1.5,
    "P ret5<=-6%":  ret5 <= -0.06,
}
HS = (1, 2, 3, 5, 10)


def run(label, mask, series, idx, tag):
    """One cell x one vehicle, all horizons, lag-1."""
    trig = idx[mask.reindex(idx, fill_value=False).values]
    trig = trig[trig <= ASOF]
    epi = declusters(trig, 5, idx)
    print(f"\n--- {label}  vehicle {tag}  (day-level N={len(trig)}, "
          f"declustered-5td episodes N={len(epi)}, "
          f"span {epi[0].date() if len(epi) else 'n/a'} .. "
          f"{epi[-1].date() if len(epi) else 'n/a'}) ---")
    for h in HS:
        f = fwd_lag(series, h, 1)
        v = f.reindex(epi).dropna()
        if len(v) == 0:
            print(f"   h={h:<3} n=0")
            continue
        st = summarize(v.values)
        nup = int((v > 0).sum())
        allf = f.dropna()
        allf = allf[allf.index <= ASOF]
        print(f"   h={h:<3} n={st['n']:<4} mean={st['mean_pct']:+.3f}%  "
              f"med={st['median_pct']:+.3f}%  {nup}-{len(v)-nup} ({st['hit']:.1f}%)  "
              f"t={st['t']:+.2f}  sp={sign_test(nup, len(v)):.4f}  | ALL-DAYS "
              f"{100*allf.mean():+.3f}% hit {100*(allf>0).mean():.1f}%  | EDGE "
              f"{st['mean_pct']-100*allf.mean():+.3f}pp  | worst {st['worst_pct']:+.2f}% "
              f"({v.idxmin().date()})  best {st['best_pct']:+.2f}% ({v.idxmax().date()})")
    return epi


def detail(label, epi, series, h, tag):
    f = fwd_lag(series, h, 1)
    v = f.reindex(epi).dropna()
    if len(v) == 0:
        return
    print(f"\n   [detail] {label} vehicle {tag} h={h}")
    print("     era:", [(e["label"], e["n"], round(e.get("mean_pct", np.nan), 3),
                         round(e.get("hit", np.nan), 1))
                        for e in era_split(v.index, v.values)])
    print("     concentration:", cluster_note(v.index, v.values))
    mid = v[[d.year % 4 == 2 for d in v.index]]
    non = v[[d.year % 4 != 2 for d in v.index]]
    print(f"     midterm n={len(mid)} {int((mid>0).sum())}-{int((mid<=0).sum())} "
          f"mean={100*mid.mean() if len(mid) else float('nan'):+.3f}%  |  "
          f"non-midterm n={len(non)} {int((non>0).sum())}-{int((non<=0).sum())} "
          f"mean={100*non.mean() if len(non) else float('nan'):+.3f}%")
    yr = pd.Series(v.values, index=v.index).groupby(v.index.year).mean()
    print("     by year (mean %):",
          {int(y): round(100 * r, 2) for y, r in yr.items()})
    print("     episode dates:", ", ".join(str(d.date()) for d in v.index))


idx = ec.index
print("\n\n########## EWZ AS THE VEHICLE ##########")
epis = {}
for label, m in CELLS.items():
    epis[label] = run(label, m, ec, idx, "EWZ")
for label, m in MIRROR.items():
    run("MIRROR " + label, m, ec, idx, "EWZ")

print("\n\n########## detail on the extension cells (h=5) ##########")
for label in CELLS:
    detail(label, epis[label], ec, 5, "EWZ")
print("\n########## detail on the extension cells (h=10) ##########")
for label in CELLS:
    detail(label, epis[label], ec, 10, "EWZ")

print("\n\n########## SPY AS THE VEHICLE, EWZ TRIGGER ##########")
sidx = sc.index
for label, m in CELLS.items():
    run(label, m.reindex(sidx, fill_value=False), sc, sidx, "SPY")

# ----------------------------------------------- overlap between the cells
print("\n\n########## how much do the three cells overlap? ##########")
for a1 in CELLS:
    for a2 in CELLS:
        if a1 >= a2:
            continue
        s1 = set(idx[CELLS[a1].reindex(idx, fill_value=False).values])
        s2 = set(idx[CELLS[a2].reindex(idx, fill_value=False).values])
        print(f"  {a1} n={len(s1)}  &  {a2} n={len(s2)}  ->  both "
              f"n={len(s1 & s2)}  (jaccard {len(s1&s2)/max(1,len(s1|s2)):.2f})")

print("\nDONE.")

# =====================================================================
# ROUND 2 falsification, prompted by round 1: the Z cell (z10 >= 1.5) is the
# only one with an edge, but it shows the SAME edge on SPY as the vehicle
# (h=3 t +2.91, h=10 t +2.29), which is what a market-beta effect looks like
# rather than a Brazil effect. Three tests:
#   (a) the EXACT conjunction that fires tonight (all three gates at once)
#   (b) a z10 ladder -- is the effect monotonic in z10, or a threshold artefact
#   (c) EWZ minus SPY (dollar-neutral, beta 1), which strips the market leg
# =====================================================================
from pitch_lab import vehicle_ret  # noqa: E402

panel = pd.DataFrame({"EWZ": ec, "SPY": sc}).dropna()
pidx = panel.index

conj = (rank5 >= 95) & (z10 >= 1.5) & (ret5 >= 0.06)
print("\n\n########## (a) THE EXACT CONJUNCTION FIRING TONIGHT ##########")
print(f"  conjunction fires on Friday's bar: {bool(conj.iloc[-1])}")
epi_conj = run("CONJ rank>=95 & z10>=1.5 & ret5>=+6%", conj, ec, idx, "EWZ")
detail("CONJ", epi_conj, ec, 5, "EWZ")
detail("CONJ", epi_conj, ec, 10, "EWZ")
run("CONJ", conj, sc, sidx, "SPY")

print("\n\n########## (b) z10 LADDER (episodes, EWZ vehicle, h=10) ##########")
bands = [("z10 in [0.5,1.0)", (z10 >= 0.5) & (z10 < 1.0)),
         ("z10 in [1.0,1.5)", (z10 >= 1.0) & (z10 < 1.5)),
         ("z10 in [1.5,2.0)", (z10 >= 1.5) & (z10 < 2.0)),
         ("z10 >= 2.0", z10 >= 2.0)]
f10 = fwd_lag(ec, 10, 1)
for lbl, m in bands:
    trig = idx[m.reindex(idx, fill_value=False).values]
    trig = trig[trig <= ASOF]
    e = declusters(trig, 5, idx)
    v = f10.reindex(e).dropna()
    if len(v) == 0:
        print(f"  {lbl:<20} n=0")
        continue
    st = summarize(v.values)
    nup = int((v > 0).sum())
    print(f"  {lbl:<20} n={st['n']:<4} mean={st['mean_pct']:+.3f}%  "
          f"med={st['median_pct']:+.3f}%  {nup}-{len(v)-nup} ({st['hit']:.1f}%)  "
          f"t={st['t']:+.2f}  sp={sign_test(nup, len(v)):.4f}  "
          f"worst {st['worst_pct']:+.2f}%")
print(f"  (tonight's z10 = {z10.iloc[-1]:.2f}, so tonight sits in [1.5,2.0))")

print("\n\n########## (c) EWZ minus SPY, dollar-neutral (strips market beta) "
      "##########")
for lbl, m in (("Z z10>=1.5", z10 >= 1.5), ("CONJ", conj),
               ("R rank5>=95", rank5 >= 95)):
    trig = pidx[m.reindex(pidx, fill_value=False).values]
    trig = trig[trig <= ASOF]
    e = declusters(trig, 5, pidx)
    print(f"\n  --- {lbl}: LONG EWZ / SHORT SPY, episodes N={len(e)} ---")
    for h in HS:
        r = vehicle_ret(panel, [("EWZ", 1.0), ("SPY", -1.0)], h, 1)
        v = r.reindex(e).dropna()
        allr = r.dropna()
        allr = allr[allr.index <= ASOF]
        if len(v) == 0:
            print(f"    h={h:<3} n=0")
            continue
        st = summarize(v.values)
        nup = int((v > 0).sum())
        print(f"    h={h:<3} n={st['n']:<4} mean={st['mean_pct']:+.3f}%  "
              f"med={st['median_pct']:+.3f}%  {nup}-{len(v)-nup} ({st['hit']:.1f}%)  "
              f"t={st['t']:+.2f}  sp={sign_test(nup, len(v)):.4f}  | ALL-DAYS "
              f"{100*allr.mean():+.3f}%  | EDGE {st['mean_pct']-100*allr.mean():+.3f}pp"
              f"  | worst {st['worst_pct']:+.2f}% ({v.idxmin().date()})")

print("\nROUND 2 DONE.")
