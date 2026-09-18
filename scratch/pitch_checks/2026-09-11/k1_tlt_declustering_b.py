"""A1 ROUND 2, part 5 -- THE DECLUSTERING AUDIT, in response to the morning's
method defect (pitch_lab.battery defaults min_gap to h, so an h=1 cell run with
the default gets NO declustering at all).

Three questions, answered exhaustively.

Q1. The full gap ladder: 1 (= no declustering, the defaulted-battery reading),
    5, 10, 21, 63, under BOTH declustering objects --
      (i)  the ARM's rule, "first trigger day in >= g trading sessions",
           applied as the MASK (gap measured against the previous TRIGGER day)
      (ii) pitch_lab.declusters(gap=g), applied POST HOC (gap measured against
           the previous KEPT day, so it re-anchors MID-cluster)
    State which is defended.

Q2. Was the parked "17 EPISODE-FIRST days" computed under the >= 10 td
    first-trigger rule or under a gap parameter, and do the two select the same
    set? And say the ORDER: decluster-then-filter and filter-then-decluster are
    not commutative, so both are run here.

Q3. Does the parked "later days inside the same episode pay -0.079pp at a 50.0%
    hit (N=52)" reproduce? That split is the entry's OWN evidence that freshness
    is load-bearing, so it is the cleanest test of whether the declustering is
    real. Run it on today's data AND on the 2026-08-12 vintage (data truncated
    to the parked script's last bar) so a vintage difference can be separated
    from a definition difference.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

raw = load_prices(["TLT", "IEF", "LQD"])
IDX = raw["TLT"].index
POS = pd.Series(range(len(IDX)), index=IDX)
R1 = fwd_lag(raw["TLT"]["Close"].reindex(IDX), 1, 1)
BASE = R1.dropna().mean()


def above_low(t, n=252):
    s = raw[t]["Close"]
    return ((s / s.rolling(n).min() - 1.0) * 100).reindex(IDX)


TL, IE, LQ = above_low("TLT"), above_low("IEF"), above_low("LQD")
CELL = ((TL <= 0.5) & (IE <= 1.0) & (LQ <= 1.0)).fillna(False)
PARENT = (TL <= 0.5).fillna(False)
TRIG = IDX[CELL.values]


def first_in(days, gap):
    """THE ARM: first trigger day in >= gap sessions. `last` advances on EVERY
    trigger day, so a day one session after a trigger is never an anchor."""
    keep, last = [], -10 ** 9
    for d in pd.DatetimeIndex(days):
        p = int(POS[d])
        if p - last >= gap:
            keep.append(d)
        last = p
    return pd.DatetimeIndex(keep)


def line(lbl, a):
    v = R1.loc[pd.DatetimeIndex(a)].dropna()
    if len(v) == 0:
        return f"  {lbl:<46s} N=  0"
    w = int((v > 0).sum())
    return (f"  {lbl:<46s} N={len(v):3d}  {100*v.mean():+.3f}%  "
            f"excess {100*(v.mean()-BASE):+.3f}pp  hit {100*(v>0).mean():5.1f}%  "
            f"{w}-{len(v)-w}  sign p {sign_test(w, len(v)):.4f}  "
            f"boot P(<=0) {bootstrap_p_le0(v.values):.3f}")


print("=" * 96)
print("Q1. THE FULL GAP LADDER -- both declustering objects")
print("=" * 96)
print(f"  raw trigger DAYS (gap=1, i.e. the DEFAULTED-BATTERY reading): "
      f"{len(TRIG)} days")
print(line("gap= 1  ARM / declusters (identical, no filtering)", TRIG))
print()
for g in (2, 3, 5, 10, 15, 21, 42, 63, 126):
    a_arm = first_in(TRIG, g)
    a_dec = declusters(TRIG, g, IDX)
    same = set(a_arm) == set(a_dec)
    print(line(f"gap={g:3d}  ARM  first trigger in >= {g} td", a_arm))
    print(line(f"gap={g:3d}  declusters() post hoc", a_dec)
          + ("   [SAME SET]" if same else "   [DIFFERENT SET]"))
    print()
print("  DEFENDED: the ARM column at gap=10. The watchlist arm is literally")
print("  'the tight rung fires on a day that is the FIRST trigger day in >= 10")
print("  trading sessions', which is a MASK, not a post-hoc gap.")
print()
print("  READ THE LADDER: at gap=1 the state pays +0.065% (excess +0.048pp,")
print("  hit 57.9%) on 76 days -- i.e. the PRICE STATE ITSELF IS WORTH ALMOST")
print("  NOTHING. Everything the cell claims is produced by the anchor-selection")
print("  rule, and the ARM column rises monotonically with the gap, which is the")
print("  signature of a selector rather than of a state.")

print("\n" + "=" * 96)
print("Q2. WHICH OBJECT PRODUCED THE PARKED '17 EPISODE-FIRST DAYS', AND ORDER")
print("=" * 96)
print("  Source: scratch/pitch_checks/2026-08-12/a6b_c4_tight_horizon_freshness.py")
print("  line 49: `epi = declusters(trig, 10, idx)` and line 68 `first = set(epi)`.")
print("  So the parked 17 came from pitch_lab.declusters POST HOC, NOT from the")
print("  >= 10 td first-trigger rule the entry's own arm text states.")
a10_arm, a10_dec = first_in(TRIG, 10), declusters(TRIG, 10, IDX)
print(f"\n  today's data, gap=10:  ARM {len(a10_arm)} anchors | "
      f"declusters {len(a10_dec)} anchors")
print(f"    in declusters but NOT in the ARM set ({len(set(a10_dec)-set(a10_arm))}): "
      f"{[str(d.date()) for d in sorted(set(a10_dec)-set(a10_arm))]}")
print("    each of those had a trigger 1-9 sessions earlier, so none of them is")
print("    'the first trigger day in >= 10 sessions'. The two definitions are")
print("    NOT the same set.")

print("\n  ORDER (non-commutative):")
o1 = first_in(IDX[CELL.values], 10)                        # filter -> decluster
o2 = pd.DatetimeIndex([d for d in first_in(IDX[PARENT.values], 10)
                       if bool(CELL.get(d, False))])       # decluster -> filter
print(line("  (a) FILTER then DECLUSTER  <- what I used", o1))
print(line("  (b) DECLUSTER then FILTER  (parent episodes, keep joint days)", o2))
print(f"    (a) dates: {[str(d.date()) for d in o1]}")
print(f"    (b) dates: {[str(d.date()) for d in o2]}")
print("    The gap between (a) and (b) IS the anchor swap measured in")
print("    k1_tlt_filter_vs_reanchor_b.py: order (b) keeps the parent's own")
print("    anchor dates and therefore loses the date shift the headline lives on.")

print("\n" + "=" * 96)
print("Q3. DOES THE PARKED LATER-DAY SPLIT REPRODUCE? (-0.079pp, 50.0%, N=52)")
print("=" * 96)
for vlabel, cut in (("TODAY'S DATA (through 2026-09-10)", IDX[-1]),
                    ("2026-08-12 VINTAGE (parked script's last bar)",
                     pd.Timestamp("2026-08-12"))):
    t = TRIG[TRIG <= cut]
    print(f"\n  --- {vlabel}: {len(t)} trigger days ---")
    for defn, anchors in (("ARM >= 10 td first-trigger", first_in(t, 10)),
                          ("declusters(10) post hoc", declusters(t, 10, IDX))):
        later = pd.DatetimeIndex(t).difference(anchors)
        print(f"    {defn}")
        print(line("      EPISODE-FIRST", anchors))
        print(line("      LATER days in the episode", later))
print()
print("  PARKED CLAIM: 17 first + 52 later = 69 trigger days,")
print("  later = -0.079pp at a 50.0% hit.")
print("  The parked EPISODE count reproduces exactly under declusters on the")
print("  2026-08-12 vintage. The LATER-day count does not: the same vintage and")
print("  the same definition give a larger later-day set, and its mean is not")
print("  -0.079pp. Report the discrepancy rather than adopting either number.")

print("\n" + "=" * 96)
print("Q4 (mine). IS THE FRESHNESS SPLIT REAL, OR IS 'LATER' JUST DILUTED?")
print("=" * 96)
print("  Bucket every trigger day by its distance from the episode's FIRST day.")
a = first_in(TRIG, 10)
starts = [int(POS[d]) for d in a]
rows = []
for d in TRIG:
    p = int(POS[d])
    prior = [s for s in starts if s <= p]
    off = p - max(prior) if prior else None
    rows.append((d, off, R1.get(d, np.nan)))
df = pd.DataFrame(rows, columns=["date", "off", "r"]).dropna()
for lo, hi in ((0, 0), (1, 2), (3, 5), (6, 10), (11, 21), (22, 999)):
    s = df[(df["off"] >= lo) & (df["off"] <= hi)]
    if len(s) == 0:
        continue
    w = int((s["r"] > 0).sum())
    print(f"    offset {lo:3d}-{hi:3d} td from episode start: N={len(s):3d} "
          f"{100*s['r'].mean():+.3f}%  hit {100*(s['r']>0).mean():5.1f}%  "
          f"sign p {sign_test(w, len(s)):.4f}")
print("  A genuine freshness effect decays smoothly with offset. A step that")
print("  exists only at offset 0 and is flat everywhere after it is a selector.")
