"""C3 round 1 -- the vol complex across a TWO-PRINT-IN-TWO-SESSIONS cluster.

Watchlist 33 established that the pre-print short-vol cell is monotone in
RUNWAY (sessions of clear calendar from the print to the NEXT print):
<=1 -0.805%, >=2 +0.625%, >=3 +0.954%, >=4 +0.900%.  It disqualifies today
by name, because 2026-09-08 is the PPI k=-2 anchor at a runway of 1.

The question this script asks, which the runway framing implies and nobody
has run: **is a print PAIR with a structurally-zero runway a distinct
object with its own sign, or is it just the dead short-runway bucket?**

Step 0 settles the definitional half in one table: if "PPI with a print on
the very next session" is set-identical to "runway == 1", then the pair IS
the dead bucket and there is nothing else to say about the k=-2/h=1 form.

Step 1 asks the only version that could be different: an entry placed
BEFORE the first print and held THROUGH BOTH, so the position owns the
calendar CLEARING after the second print rather than the queue in front of
the first.  Today's geometry exactly: entry MOC 2026-09-08, PPI at entry+2,
CPI at entry+3, then 3 clear sessions to the FOMC on 2026-09-16.

Vehicles: SVXY (tradeable, -0.5x since Feb 2018) and short ^VIX
(unlevered, mechanism only -- never tradeable).

Kills:
  0. set identity with runway<=1 -> the pair is not a new object
  1. controls: own drift, all days, local +/-126td
  2. runway-AFTER-the-pair split: does the clear calendar pay?
  3. placebo anchor ladder
  4. leverage-era split (the -0.5x era is what trades) + midterm
  5. cost: SVXY is 8-10 bps all-in per the registry
  6. today's live state: dial 88.0, VIX rel-range band, and the standing
     "short vol is a levered equity bet" objection (corr 0.626-0.755).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 240)

px = close_panel(["SVXY", "^VIX", "SPY"])
cal = px["SPY"].dropna().index
pos = pd.Series(range(len(cal)), index=cal)
print(f"calendar {cal[0].date()} .. {cal[-1].date()}  N={len(cal)}")

KINDS = ("nfp", "cpi", "ppi", "fomc_decision")
EV = {k: load_events([k])["date"] for k in KINDS}
ALL_PRINTS = pd.DatetimeIndex(sorted(pd.concat(list(EV.values())).unique()))

# ------------------------------------------------------- runway machinery
rows = []
for kind in KINDS:
    p, kept = anchor_positions(cal, EV[kind], 0)
    for i, pp in enumerate(p):
        d = kept[i]
        nxt = ALL_PRINTS[ALL_PRINTS > d]
        if len(nxt) == 0:
            continue
        pn = pos.get(nxt[0])
        if pn is None:
            pn = int(cal.searchsorted(nxt[0]))
        rows.append({"print": d, "ppos": pp, "kind": kind,
                     "runway": int(pn - pp)})
PR = pd.DataFrame(rows).drop_duplicates("print").sort_values("print").reset_index(drop=True)

print("\n" + "=" * 104)
print("0. IS 'A PAIR' A NEW OBJECT, OR IS IT LITERALLY THE runway==1 BUCKET?")
print("=" * 104)
PR["pair_first"] = PR["runway"] == 1
print(PR.groupby("kind")["runway"].apply(
    lambda x: pd.Series({"n": len(x), "runway<=1": int((x <= 1).sum()),
                         "runway==1": int((x == 1).sum()),
                         "runway==0": int((x == 0).sum()),
                         "runway>=3": int((x >= 3).sum())})).unstack().to_string())
ppi_pair = PR[(PR.kind == "ppi") & (PR.runway == 1)]
print(f"\n  'PPI with another print the very next session': N={len(ppi_pair)}")
print(f"  'PPI at runway == 1'                          : N={len(ppi_pair)}")
print("  -> SET IDENTICAL BY CONSTRUCTION. A print pair IS a runway-1 anchor.")
print("     So the k=-2/h=1 pair cell cannot be anything but the dead bucket;")
print("     only a DIFFERENT ENTRY can make the pair a distinct object.")

# reproduce the dead bucket at the k=-2 / h=1 form the watchlist uses
svxy1 = fwd_lag(px["SVXY"].dropna(), 1, lag=1)
vix1 = -fwd_lag(px["^VIX"].dropna(), 1, lag=1)
print("\n  reproduce the watchlist's runway ladder (k=-2 anchor, h=1, ungated):")
r2 = []
for lbl, m in (("runway <= 1 (INCLUDES today's PPI)", PR.runway <= 1),
               ("runway == 1 (a pair, exactly)", PR.runway == 1),
               ("runway == 2", PR.runway == 2),
               ("runway >= 3 (clear calendar)", PR.runway >= 3)):
    a = cal[[p - 2 for p in PR.loc[m, "ppos"] if p - 2 >= 0]]
    for tag, s in (("SVXY", svxy1), ("shortVIX", vix1)):
        v = s.reindex(a).dropna().values
        r = summarize(v, f"{tag} | {lbl}")
        r2.append(r)
show(r2, "k=-2 anchor, h=1 (exit on the print session close)")

# ------------------------------------------------- 1. the different entry
print("\n" + "=" * 104)
print("1. THE ONLY FORM THAT COULD BE NEW: enter BEFORE the first print of")
print("   the pair and hold THROUGH BOTH, so the position owns the calendar")
print("   clearing AFTER the second print. Today: entry 2026-09-08,")
print("   PPI entry+2, CPI entry+3, then 3 clear sessions to the FOMC.")
print("=" * 104)
pair_pos = list(ppi_pair["ppos"].values)
# runway AFTER the second print of the pair
after = []
for pp in pair_pos:
    second = cal[pp + 1]
    nxt = ALL_PRINTS[ALL_PRINTS > second]
    if len(nxt) == 0:
        after.append(np.nan)
        continue
    pn = pos.get(nxt[0])
    if pn is None:
        pn = int(cal.searchsorted(nxt[0]))
    after.append(int(pn - (pp + 1)))
PAIRDF = pd.DataFrame({"ppos": pair_pos, "runway_after": after})
PAIRDF["anchor"] = [cal[p - 3] for p in PAIRDF["ppos"]]  # entry = anchor+1 = print-2
PAIRDF = PAIRDF.dropna()
print(f"  pair anchors N={len(PAIRDF)}  runway-after-the-pair distribution: "
      f"{dict(PAIRDF['runway_after'].astype(int).value_counts().sort_index())}")
print("  TODAY: runway after CPI 2026-09-11 to FOMC 2026-09-16 = 3 sessions.")

anch = pd.DatetimeIndex(PAIRDF["anchor"])
for tag, ser in (("SVXY", px["SVXY"].dropna()), ("shortVIX", px["^VIX"].dropna())):
    sgn = 1.0 if tag == "SVXY" else -1.0
    rows = []
    for h in (1, 2, 3, 4, 5, 6, 8):
        r = sgn * fwd_lag(ser, h, lag=1)
        base = r.dropna()
        v = r.reindex(anch).dropna().values
        if not len(v):
            continue
        s = summarize(v, f"h={h}")
        s["drift_pct"] = round(100 * base.mean(), 3)
        s["edge_pp"] = round(s["mean_pct"] - 100 * base.mean(), 3)
        s["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
        rows.append(s)
    show(rows, f"{tag}: entry = first-print - 2, held across BOTH prints "
               f"(h=2 exits on print 1, h=3 on print 2)")

print("\n  2. RUNWAY AFTER THE PAIR -- if the 'clear calendar' mechanism is")
print("     what pays, the post-pair runway must order the result.")
for tag, ser in (("SVXY", px["SVXY"].dropna()), ("shortVIX", px["^VIX"].dropna())):
    sgn = 1.0 if tag == "SVXY" else -1.0
    for h in (3, 4, 5):
        r = sgn * fwd_lag(ser, h, lag=1)
        rows = []
        for lbl, m in ((">=3 clear after pair (TODAY)", PAIRDF.runway_after >= 3),
                       ("==2 after", PAIRDF.runway_after == 2),
                       ("<=1 after", PAIRDF.runway_after <= 1)):
            a = pd.DatetimeIndex(PAIRDF.loc[m, "anchor"])
            v = r.reindex(a).dropna().values
            if not len(v):
                continue
            s = summarize(v, f"{lbl}")
            s["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
            rows.append(s)
        show(rows, f"{tag} h={h}, by runway AFTER the pair")

# ------------------------------------------------------ 3. placebo ladder
print("\n" + "=" * 104)
print("3. PLACEBO ANCHOR LADDER, k = firstprint-8 .. firstprint+3, h=3.")
print("=" * 104)
for tag, ser in (("SVXY", px["SVXY"].dropna()), ("shortVIX", px["^VIX"].dropna())):
    sgn = 1.0 if tag == "SVXY" else -1.0
    r = sgn * fwd_lag(ser, 3, lag=1)
    rows = []
    for k in range(-8, 4):
        a = cal[[p + k for p in pair_pos if 0 <= p + k < len(cal)]]
        v = r.reindex(a).dropna().values
        if len(v) < 5:
            continue
        rows.append({"k": k, "n": len(v), "mean_pct": round(100 * v.mean(), 3),
                     "hit": round(100 * (v > 0).mean(), 1),
                     "t": round(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), 2)})
    d = pd.DataFrame(rows).sort_values("mean_pct", ascending=False).reset_index(drop=True)
    tr = d.index[d["k"] == -3]
    rank = int(tr[0]) + 1 if len(tr) else -1
    print(f"\n  {tag}: TRUE ANCHOR k=-3 RANKS {rank} of {len(d)}")
    print(d.to_string(index=False))

# --------------------------------------------------- 4. eras / cost / live
print("\n" + "=" * 104)
print("4. LEVERAGE ERA (SVXY went -1x -> -0.5x 2018-02-28) + midterm + Sept.")
print("=" * 104)
for h in (3, 4, 5):
    r = fwd_lag(px["SVXY"].dropna(), h, lag=1)
    a = pd.DatetimeIndex(PAIRDF["anchor"]).intersection(r.dropna().index)
    v = r.reindex(a).values
    yrs = a.year
    rows = []
    for lbl, m in (("-1x era (pre 2018-02-28)", a < pd.Timestamp("2018-02-28")),
                   ("-0.5x era (LIVE)", a >= pd.Timestamp("2018-02-28")),
                   ("MIDTERM", (yrs % 4) == 2), ("SEPTEMBER", a.month == 9)):
        sub = v[m]
        if not len(sub):
            continue
        s = summarize(sub, lbl)
        s["sign_p"] = round(sign_test(int((sub > 0).sum()), len(sub)), 4)
        rows.append(s)
    show(rows, f"SVXY h={h}, pair anchor")

print("\n" + "=" * 104)
print("5. LIVE STATE + the standing 'short vol is a levered equity bet' charge")
print("=" * 104)
vix = px["^VIX"].dropna()
vix = vix[vix.index <= pd.Timestamp("2026-09-04")]   # 09-07 is a holiday stub
rng21 = (rolling_on_valid(vix, lambda x: x.rolling(21).max())
         - rolling_on_valid(vix, lambda x: x.rolling(21).min()))
REL = rolling_on_valid(rng21 / rolling_on_valid(vix, lambda x: x.rolling(21).mean()),
                       lambda x: x.rolling(252).rank(pct=True) * 100)
print(f"  VIX 21d rel-range pctile on 2026-09-04 = {REL.iloc[-1]:.2f}   "
      f"(watchlist 33's live band is (5,15]; (0,5] is its dead half)")
for h in (3, 4, 5):
    a = pd.DatetimeIndex(PAIRDF["anchor"])
    sv = fwd_lag(px["SVXY"].dropna(), h, lag=1).reindex(a)
    sp = fwd_lag(px["SPY"].dropna(), h, lag=1).reindex(a)
    d = pd.concat([sv, sp], axis=1).dropna()
    d.columns = ["svxy", "spy"]
    if len(d) < 5:
        continue
    beta = np.polyfit(d["spy"], d["svxy"], 1)
    corr = d["spy"].corr(d["svxy"])
    resid = d["svxy"] - (beta[0] * d["spy"] + beta[1])
    print(f"  h={h}: corr(SPY, SVXY) on pair anchors = {corr:+.3f}, beta "
          f"{beta[0]:.2f}, R2 {corr**2:.3f}; beta-neutral residual "
          f"{100*resid.mean():+.3f}% (n={len(d)}, hit {100*(resid>0).mean():.1f}%)")
print("  -> C1 measured SPY on this same anchor. If SVXY is mostly SPY, the")
print("     two are ONE position and C1's kill transfers.")
