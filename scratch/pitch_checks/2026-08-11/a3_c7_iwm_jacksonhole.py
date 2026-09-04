"""C7 kill: short IWM, anchored 14 sessions before Jackson Hole (entry lag=1
= 13 sessions before JH), h=1.

Stated suspicion in the brief: a 13-sessions-early anchor is mid-August
calendar position wearing an event label (the d4b_vix_week_vs_monthpos trap).

My own suspicion is sharper and testable: 13 sessions before a late-August
Jackson Hole IS, in most years, the session before or of the AUGUST CPI PRINT.
If so the "Jackson Hole" label is decoration on a CPI cell -- and the surface
map already DISMISSED IWM into CPI (h=1 excess -0.026 across all months).

Tests:
  (a) what dates are these, really: tdom of the entry, and the distance from
      the entry session to that year's August CPI / PPI / NFP prints
  (b) the same calendar position with NO Jackson Hole anchor (plain August
      trading-day-of-month), and the same tdom in every OTHER month
  (c) within-month paired excess (entry day vs the rest of that August)
  (d) the midterm subset, reported separately (2026 IS midterm)
  (e) year histogram, drop-best, and whether SPY/QQQ do the same thing (is
      this an IWM story or a whole-tape story?)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, load_events, fwd_lag, declusters, summarize, sign_test,
    bootstrap_p_le0,
)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)

TKRS = ["IWM", "SPY", "QQQ"]
px = close_panel(TKRS).dropna()
all_dates = px.index
pos = pd.Series(np.arange(len(all_dates)), index=all_dates)
TDOM = pd.Series(all_dates, index=all_dates).groupby(
    [all_dates.year, all_dates.month]).cumcount() + 1

OFFSET = 14   # matches 01_event_class_recon.py TODAY_OFFSETS['jackson_hole']
H = 1

ev_all = load_events()
jh = pd.DatetimeIndex(sorted(ev_all.loc[ev_all.event == "jackson_hole", "date"].unique()))

anch, meta = [], []
for d in jh:
    loc = all_dates.searchsorted(d)
    # a JH date past the end of the price panel has NO anchor. searchsorted
    # returns len(all_dates) there, and len - OFFSET silently produces a fake
    # anchor 14 sessions before the last bar (this bit my first run: 2026's
    # "anchor" came out 2026-07-22). 01_event_class_recon.py guards it with
    # `if loc >= len(all_dates): continue` -- match that.
    if loc >= len(all_dates):
        print(f"  [skip] {d.date()} is past the last price bar "
              f"({all_dates[-1].date()}) -- this is the LIVE one, no fwd return")
        continue
    j = loc - OFFSET
    if 0 <= j < len(all_dates) and j + 2 < len(all_dates):
        a = all_dates[j]
        entry, exit_ = all_dates[j + 1], all_dates[j + 2]
        anch.append(a)
        yr = a.year
        yev = ev_all[(ev_all.date.dt.year == yr) & (ev_all.date.dt.month == 8)]
        def nearest(kind):
            s = yev.loc[yev.event == kind, "date"]
            return s.iloc[0] if len(s) else pd.NaT
        cpi_d, ppi_d, nfp_d = nearest("cpi"), nearest("ppi"), nearest("nfp")
        meta.append(dict(
            year=yr, jh=d.date(), anchor=a.date(), entry=entry.date(),
            exit=exit_.date(), tdom_entry=int(TDOM.loc[entry]),
            aug_cpi=cpi_d.date() if pd.notna(cpi_d) else None,
            cpi_gap_td=(all_dates.searchsorted(cpi_d) - (j + 1)) if pd.notna(cpi_d) else None,
            ppi_gap_td=(all_dates.searchsorted(ppi_d) - (j + 1)) if pd.notna(ppi_d) else None,
            nfp_gap_td=(all_dates.searchsorted(nfp_d) - (j + 1)) if pd.notna(nfp_d) else None,
        ))
anch = pd.DatetimeIndex(sorted(set(anch)))
anch = declusters(anch, 5, all_dates)
M = pd.DataFrame(meta)

r_iwm = fwd_lag(px["IWM"], H, 1)
r_spy = fwd_lag(px["SPY"], H, 1)
r_qqq = fwd_lag(px["QQQ"], H, 1)
M["iwm_pct"] = [100 * r_iwm.get(pd.Timestamp(a), np.nan)
                for a in pd.to_datetime(M["anchor"])]
M["spy_pct"] = [100 * r_spy.get(pd.Timestamp(a), np.nan)
                for a in pd.to_datetime(M["anchor"])]
M["midterm"] = M["year"] % 4 == 2

print("=" * 130)
print("(a) WHAT DATES ARE THESE, REALLY. gap_td = sessions from the ENTRY close")
print("    to that print (0 = the entry session IS the print day, +1 = the")
print("    print lands on the exit session = inside the h=1 hold).")
print("=" * 130)
print(M.to_string(index=False))

inside = M["cpi_gap_td"].isin([1])
onday = M["cpi_gap_td"].isin([0])
print(f"\n  August CPI lands INSIDE the h=1 hold (gap +1) in {int(inside.sum())} "
      f"of {len(M)} years; ON the entry session (gap 0) in {int(onday.sum())}.")
print(f"  cpi_gap_td distribution: {M['cpi_gap_td'].value_counts().sort_index().to_dict()}")
print(f"  entry tdom distribution: {M['tdom_entry'].value_counts().sort_index().to_dict()}")

print("\n" + "=" * 130)
print("(b) THE SAME CALENDAR POSITION WITH NO JACKSON HOLE ANCHOR")
print("=" * 130)
mode_tdom = sorted(M["tdom_entry"].value_counts().index[:3].tolist())
print(f"    the entry lands on August trading-day-of-month {mode_tdom} most often.")

def cell(name, ser, dates):
    v = ser.reindex(pd.DatetimeIndex(dates)).dropna()
    st = summarize(v.values)
    if not st["n"]:
        return None
    wins = int((v.values > 0).sum())          # LONG wins
    losses = st["n"] - wins
    return dict(cell=name, N=st["n"], long_mean=round(st["mean_pct"], 3),
                long_hit=round(st["hit"], 1),
                short_signp=round(sign_test(losses, st["n"]), 4),
                t=round(st["t"], 2), worst_for_short=round(st["best_pct"], 2))

# anchor dates whose ENTRY sits on one of those tdoms, in August, no JH needed
ent_of = {a: all_dates[pos[a] + 1] for a in anch if pos[a] + 1 < len(all_dates)}
tdom_set = set(M["tdom_entry"].tolist())

rows = [cell("JH-anchored (the candidate)", r_iwm, anch)]
# plain August tdom-matched: anchors = the session before an August day whose
# tdom is in the JH entry tdom set
aug_mask = (all_dates.month == 8) & TDOM.reindex(all_dates).isin(tdom_set).values
aug_entries = all_dates[aug_mask]
aug_anch = pd.DatetimeIndex([all_dates[pos[d] - 1] for d in aug_entries if pos[d] >= 1])
rows.append(cell(f"plain AUGUST tdom {sorted(tdom_set)} (no JH label)", r_iwm, aug_anch))
# the same tdom in every other month
for mth, nm in ((None, "ALL months, same tdom set"),):
    m2 = TDOM.reindex(all_dates).isin(tdom_set).values
    ents = all_dates[m2]
    an2 = pd.DatetimeIndex([all_dates[pos[d] - 1] for d in ents if pos[d] >= 1])
    rows.append(cell(nm, r_iwm, an2))
rows.append(cell("ALL AUGUST days", r_iwm, all_dates[all_dates.month == 8]))
rows.append(cell("ALL DAYS (unconditional)", r_iwm, all_dates))
print(pd.DataFrame([r for r in rows if r]).to_string(index=False))

# the CPI decomposition: is it the August CPI session?
aug_cpi_dates = pd.DatetimeIndex(sorted(
    ev_all[(ev_all.event == "cpi") & (ev_all.date.dt.month == 8)]["date"].unique()))
cpi_eve = []
for d in aug_cpi_dates:
    j = all_dates.searchsorted(d) - 2
    if 0 <= j < len(all_dates):
        cpi_eve.append(all_dates[j])
cpi_eve = pd.DatetimeIndex(sorted(set(cpi_eve)))
rows2 = [cell("AUGUST CPI eve, IWM (anchor -2, h=1 = the print)", r_iwm, cpi_eve),
         cell("ALL-MONTH CPI eve, IWM (the map already dismissed this)", r_iwm,
              declusters(pd.DatetimeIndex(sorted(set(
                  all_dates[max(0, all_dates.searchsorted(d) - 2)]
                  for d in sorted(ev_all[ev_all.event == "cpi"]["date"].unique())
                  if all_dates.searchsorted(d) >= 2))), 5, all_dates))]
print("\n  the CPI decomposition:")
print(pd.DataFrame([r for r in rows2 if r]).to_string(index=False))

print("\n" + "=" * 130)
print("(c) WITHIN-MONTH PAIRED EXCESS: the entry day's h=1 vs the rest of that")
print("    same August. Kills the 'August is a bad month' explanation.")
print("=" * 130)
paired = []
for a in anch:
    e = all_dates[pos[a] + 1]
    yr, mo = e.year, e.month
    same = all_dates[(all_dates.year == yr) & (all_dates.month == mo)]
    others = pd.DatetimeIndex([d for d in same if d != e])
    oa = pd.DatetimeIndex([all_dates[pos[d] - 1] for d in others if pos[d] >= 1])
    mine = r_iwm.get(a, np.nan)
    rest = r_iwm.reindex(oa).dropna().mean()
    if np.isfinite(mine) and np.isfinite(rest):
        paired.append(dict(year=yr, mine=100 * mine, rest_of_aug=100 * rest,
                           paired_excess=100 * (mine - rest)))
P = pd.DataFrame(paired)
print(P.round(3).to_string(index=False))
pe = P["paired_excess"].values / 100.0
st = summarize(pe)
neg = int((pe < 0).sum())
print(f"\n  paired excess: N={st['n']} mean={st['mean_pct']:+.3f}pp "
      f"median={st['median_pct']:+.3f}pp  down {neg}/{st['n']} "
      f"sign p (short) = {sign_test(neg, st['n']):.4f}  t={st['t']:+.2f}")
print(f"  bootstrap P(mean >= 0) for the SHORT = "
      f"{1 - bootstrap_p_le0(pe):.3f}")

print("\n" + "=" * 130)
print("(d) MIDTERM SUBSET (2026 IS a midterm year) + (e) year histogram")
print("=" * 130)
for lbl, m in (("midterm years", M["midterm"]), ("non-midterm", ~M["midterm"])):
    sub = M[m]
    v = sub["iwm_pct"].dropna().values / 100.0
    st = summarize(v)
    losses = int((v < 0).sum())
    print(f"  {lbl:>14}: N={st['n']:<3} mean={st['mean_pct']:+.3f}% "
          f"down {losses}/{st['n']} ({100*losses/st['n']:.0f}%) "
          f"short sign p={sign_test(losses, st['n']):.4f} "
          f"years={sorted(sub['year'].tolist())}")
    print(f"                  values: "
          f"{[f'{y}:{r:+.2f}' for y, r in zip(sub['year'], sub['iwm_pct'].round(2))]}")

v = M["iwm_pct"].dropna().values / 100.0
st = summarize(v)
losses = int((v < 0).sum())
print(f"\n  FULL CELL: N={st['n']} mean={st['mean_pct']:+.3f}% "
      f"down {losses}/{st['n']} short sign p={sign_test(losses, st['n']):.4f} "
      f"t={st['t']:+.2f} worst-for-short {st['best_pct']:+.2f}%")
order = np.argsort(v)                    # most negative first = best for short
yrs = M.dropna(subset=["iwm_pct"])["year"].values
print(f"  best 3 years for the short: "
      f"{[(int(yrs[i]), round(100*v[i], 2)) for i in order[:3]]}")
print(f"  drop the 1 best short year: mean = {100*np.delete(v, order[0]).mean():+.3f}%")
print(f"  drop the 2 best:            mean = {100*np.delete(v, order[:2]).mean():+.3f}%")
print(f"  drop the 3 best:            mean = {100*np.delete(v, order[:3]).mean():+.3f}%")
print(f"  median (robust)             = {100*np.median(v):+.3f}%")

print("\n  is this IWM or the whole tape?")
rows3 = [cell("IWM", r_iwm, anch), cell("SPY", r_spy, anch), cell("QQQ", r_qqq, anch)]
sp = (r_iwm - r_spy)
rows3.append(cell("IWM - SPY spread", sp, anch))
print(pd.DataFrame([r for r in rows3 if r]).to_string(index=False))

print("\n  era split (pre-2013 / 2013+):")
for lbl, m in (("2000-2012", M["year"] <= 2012), ("2013-2025", M["year"] >= 2013)):
    sub = M[m]["iwm_pct"].dropna().values / 100.0
    if len(sub) == 0:
        continue
    st = summarize(sub)
    losses = int((sub < 0).sum())
    print(f"    {lbl}: N={st['n']} mean={st['mean_pct']:+.3f}% down "
          f"{losses}/{st['n']} sign p={sign_test(losses, st['n']):.4f}")

print("\n  paired excess with the two crash years removed (2010 flash-crash")
print("  summer, 2011 US downgrade) -- the concentration test on the control-")
print("  adjusted number, not the raw one:")
pe2 = P[~P["year"].isin([2010, 2011])]["paired_excess"].values / 100.0
st2 = summarize(pe2)
n2 = int((pe2 < 0).sum())
print(f"    N={st2['n']} mean={st2['mean_pct']:+.3f}pp down {n2}/{st2['n']} "
      f"sign p (short)={sign_test(n2, st2['n']):.4f} t={st2['t']:+.2f}")
pe3 = P[P["year"] % 4 == 2]["paired_excess"].values / 100.0
if len(pe3):
    st3 = summarize(pe3)
    n3 = int((pe3 < 0).sum())
    print(f"    MIDTERM paired excess: N={st3['n']} mean={st3['mean_pct']:+.3f}pp "
          f"down {n3}/{st3['n']} sign p (short)={sign_test(n3, st3['n']):.4f} "
          f"-> the short is WRONG-SIGNED in the cycle year we are in")

print("\n" + "=" * 130)
print("cost: IWM round trip ~1.5 bps + 1 day of borrow (~0.1 bp). Edge is the")
print("mean above, in bps:")
print("=" * 130)
sv = summarize(v)
print(f"   full cell {abs(100 * sv['mean_pct']):.1f} bps of edge vs ~2 bps cost;")
print(f"   drop-3-best {abs(100 * np.delete(v, order[:3]).mean() * 100):.1f} bps;")
mv = M[M['midterm']]['iwm_pct'].dropna().values / 100.0
print(f"   MIDTERM cell {100 * 100 * mv.mean():+.1f} bps -- the short PAYS "
      f"NEGATIVE {100 * 100 * mv.mean():+.1f} bps in midterm years, which is "
      f"the cycle 2026 is in.")
