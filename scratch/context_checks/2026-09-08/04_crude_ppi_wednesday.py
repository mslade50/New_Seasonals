"""Do either of tonight's two crude cells survive contact with today's tape?

The sweep handed us two cells that both preview the SAME session (Wed
2026-09-09) on the SAME instrument (CL=F):

  A. E:ppi   k=2 anchor  -- today is 2 td before Thursday's PPI.  n=310,
     h1 mean +0.321%, hit 55.8%, t=2.37, record 173-135.
  B. E:weekday_month     -- "Wednesdays in September".  n=110, h1 mean
     +0.569%, hit 57.3%, t=2.48, record 63-45.  A BARE 12x5 grid the engine
     fires every single evening, so it owes multiplicity honesty.

Three ways this dies, and this script tries all three:
  1. They are the same cell counted twice (overlap).
  2. The mean lives in two or three episodes (concentration).
  3. It does not hold in the state crude is ACTUALLY in tonight -- z10 +2.33,
     21d return in the 84th percentile, 18.6% over its 200d SMA.  The known
     control says extended crude has NO next-day edge (z10 >= 2: n=171,
     h1 mean -0.041%, t=-0.21), so the only version of these cells that can
     be published tomorrow is the extended-crude version of them.

Anchor convention is the product's: the anchor is the session BEFORE the one
being previewed, so h=1 is Wednesday's own close-to-close move (lag=0).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

ASOF = pd.Timestamp("2026-09-08")
ERA = "2018-01-01"

px = load_prices(["CL=F", "^GSPC"])
cl = px["CL=F"]["Close"].astype(float).dropna()
cl = cl[cl.index <= ASOF]
ref = pd.DatetimeIndex(px["^GSPC"].index)
ref = ref[ref <= ASOF]

FWD = {h: fwd_ret(cl, h) for h in (1, 5)}
VALID = {h: FWD[h].dropna().index for h in (1, 5)}

def engine_z10(close: pd.Series) -> pd.Series:
    """z10 EXACTLY as build_context_state._z10 / build_pitch_state._metrics_for
    define it: 10d return over 21d realised vol scaled to 10 days.  NOT
    pitch_lab.zscore, which scores the 10d return against its own trailing-year
    mean and sd and reads +1.14 on a tape the engine calls +2.33 (CLAUDE.md
    pins this: the trigger has to match the tape block or the two contradict)."""
    r10 = close.pct_change(10, fill_method=None)
    vol21 = close.pct_change(fill_method=None).rolling(21).std()
    return r10 / (vol21 * np.sqrt(10))


Z10 = engine_z10(cl)
RANK21 = pct_rank(cl, 21, 252)
Z10_LAB = zscore(cl, 10)          # kept only to show the two disagree

print("=" * 78)
print("0. LIVE STATE SANITY CHECK (must match the tape line in the prompt)")
print("=" * 78)
print(f"  CL=F last bar        {cl.index[-1].date()}  close {cl.iloc[-1]:.2f}")
print(f"  1d return            {100*(cl.iloc[-1]/cl.iloc[-2]-1):+.2f}%")
print(f"  21d return           {100*(cl.iloc[-1]/cl.iloc[-22]-1):+.2f}%")
print(f"  z10 (engine defn)    {Z10.iloc[-1]:+.2f}   "
      f"[pitch_lab.zscore would say {Z10_LAB.iloc[-1]:+.2f} -- not the tape's number]")
print(f"  21d pct rank (252d)  {RANK21.iloc[-1]:.1f}")
print(f"  vs 200d SMA          {100*(cl.iloc[-1]/cl.rolling(200).mean().iloc[-1]-1):+.1f}%")
print(f"  ^GSPC sessions in ref: {len(ref)}   CL=F sessions: {len(cl)}")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def anchors_before(ref_idx: pd.DatetimeIndex, prop: np.ndarray) -> pd.DatetimeIndex:
    """Engine primitive: sessions whose NEXT session has the property."""
    nxt = np.zeros(len(prop), dtype=bool)
    nxt[:-1] = np.asarray(prop, dtype=bool)[1:]
    return ref_idx[nxt]


def row(dates, h, label):
    d = pd.DatetimeIndex(dates).intersection(VALID[h])
    v = FWD[h].loc[d].values.astype(float)
    s = summarize(v, label)
    if s["n"] == 0:
        return s, d, v
    up, down = int((v > 0).sum()), int((v < 0).sum())
    s["record"] = f"{up}-{down}"
    s["sign_p"] = round(sign_test(max(up, down), len(v)), 4)
    s["sign_dir"] = "up" if up >= down else "down"
    base = FWD[h].dropna()
    s["edge_pp"] = round(s["mean_pct"] - 100 * base.mean(), 3)
    for k in ("worst_pct", "best_pct", "sd_pct"):
        s.pop(k, None)
    return s, d, v


def cell_block(name, anchors):
    print("\n" + "=" * 78)
    print(f"1. CELL: {name}   ({len(anchors)} raw anchors)")
    print("=" * 78)
    out = {}
    rows = []
    for h in (1, 5):
        s, d, v = row(anchors, h, f"CELL h={h}")
        out[h] = (d, v)
        rows.append(s)
        allv = FWD[h].dropna()
        c = summarize(allv.values, f"CTRL all days h={h}")
        upc, dnc = int((allv > 0).sum()), int((allv < 0).sum())
        c["record"] = f"{upc}-{dnc}"
        c["sign_p"] = round(sign_test(max(upc, dnc), len(allv)), 4)
        c["edge_pp"] = 0.0
        for k in ("worst_pct", "best_pct", "sd_pct"):
            c.pop(k, None)
        rows.append(c)
        loc = local_control(VALID[h], pd.DatetimeIndex(anchors), 126)
        lc, _, lv = row(loc, h, f"CTRL local +/-126td ex-trigger h={h}")
        rows.append(lc)
    show(rows, f"{name}: cell vs all-days vs local controls")

    for h in (1, 5):
        d, v = out[h]
        print(f"\n  -- h={h} era split (cut {ERA}) --")
        er = era_split(d, v, ERA)
        for e in er:
            for k in ("worst_pct", "best_pct", "sd_pct"):
                e.pop(k, None)
        show(er)
        for k in (2, 4):
            print(f"  h={h} concentration k={k}: {cluster_note(d, v, k)}")
        # what the mean looks like with the top-k days removed
        for k in (2, 4):
            order = np.argsort(-np.abs(v))[:k]
            keep = np.ones(len(v), dtype=bool)
            keep[order] = False
            print(f"  h={h} mean with top-{k} |moves| removed: "
                  f"{100*v[keep].mean():+.3f}%  (n={keep.sum()})")
    return out


# ---------------------------------------------------------------------------
# build the two anchor sets
# ---------------------------------------------------------------------------
ppi = load_events(["ppi"])["date"]
ppi = pd.DatetimeIndex(ppi[ppi <= ASOF + pd.Timedelta(days=400)])
pos, kept_ev = anchor_positions(ref, ppi, offset=-2)
PPI_A = pd.DatetimeIndex(sorted(set(ref[pos])))
print(f"\nPPI events considered: {len(ppi)}  ->  k2 anchors on the NYSE index: "
      f"{len(PPI_A)}   (today in set: {ASOF in PPI_A})")
print("  NOTE: today is EXCLUDED from both anchor sets on purpose -- the panel "
      "stops at today,\n  so anchor_positions drops the unrealised 2026-09-10 "
      "PPI and anchors_before has no\n  next session to look at. Today is still "
      "the live anchor for both cells; these are\n  the 310 / 110 PRIOR "
      "instances the brief would quote.")

sep_wed = ((ref.weekday.to_numpy() == 2) & (ref.month.to_numpy() == 9))
WED_A = anchors_before(ref, sep_wed)
print(f"September Wednesdays previewed: {int(sep_wed.sum())}  ->  anchors: "
      f"{len(WED_A)}   (today in set: {ASOF in WED_A})")

ppi_out = cell_block("A. CL=F, session 2 td before a PPI", PPI_A)
wed_out = cell_block("B. CL=F, Wednesdays in September", WED_A)


# ---------------------------------------------------------------------------
# 2. OVERLAP -- are these two cells or one cell counted twice?
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("2. OVERLAP between the two anchor sets")
print("=" * 78)
inter = PPI_A.intersection(WED_A)
print(f"  PPI-k2 anchors            : {len(PPI_A)}")
print(f"  Sep-Wednesday anchors     : {len(WED_A)}")
print(f"  intersection              : {len(inter)}  "
      f"({100*len(inter)/max(len(WED_A),1):.1f}% of the Sep-Wed cell, "
      f"{100*len(inter)/max(len(PPI_A),1):.1f}% of the PPI cell)")
print(f"  today ({ASOF.date()}) is in BOTH: {ASOF in inter}")
print("  intersection dates:", ", ".join(str(d.date()) for d in inter))

rows = []
for h in (1, 5):
    s, _, _ = row(inter, h, f"INTERSECTION only h={h}")
    rows.append(s)
    s, _, _ = row(PPI_A.difference(inter), h, f"PPI-k2 minus intersection h={h}")
    rows.append(s)
    s, _, _ = row(WED_A.difference(inter), h, f"Sep-Wed minus intersection h={h}")
    rows.append(s)
show(rows, "each cell with the shared dates REMOVED")

print("\n  Also: how much of PPI-k2 lands in September at all?")
rows = []
for h in (1, 5):
    sep_ppi = PPI_A[PPI_A.month.isin([9])]
    s, _, _ = row(sep_ppi, h, f"PPI-k2 anchors in Sept h={h}")
    rows.append(s)
    s, _, _ = row(PPI_A[~PPI_A.month.isin([9])], h, f"PPI-k2 anchors NOT Sept h={h}")
    rows.append(s)
show(rows, "PPI cell, September vs not")


# ---------------------------------------------------------------------------
# 3. CONDITIONING ON THE LIVE STATE -- the whole point
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("3. CONDITIONED ON CRUDE ALREADY BEING EXTENDED (tonight's actual state)")
print("=" * 78)

# the known control, reproduced
rows = []
for h in (1, 5):
    ext = cl.index[(Z10 >= 2).reindex(cl.index, fill_value=False).values]
    s, _, _ = row(ext, h, f"CTRL CL=F z10 >= 2, any day h={h}")
    rows.append(s)
    s, _, _ = row(cl.index[(RANK21 >= 80).reindex(cl.index, fill_value=False).values],
                  h, f"CTRL CL=F 21d rank >= 80, any day h={h}")
    rows.append(s)
show(rows, "unconditional 'extended crude' controls (should reproduce n=171, ~-0.04%)")

for name, anchors in [("A. PPI-k2", PPI_A), ("B. Sep-Wed", WED_A)]:
    for split_name, series, thr in [("z10", Z10, 1.0), ("21d pct rank", RANK21, 80.0)]:
        rows = []
        for h in (1, 5):
            hi = pd.DatetimeIndex([d for d in anchors
                                   if d in series.index and series.loc[d] >= thr])
            lo = pd.DatetimeIndex([d for d in anchors
                                   if d in series.index and series.loc[d] < thr])
            s, _, _ = row(hi, h, f">= {thr:g}  h={h}")
            rows.append(s)
            s, _, _ = row(lo, h, f"<  {thr:g}  h={h}")
            rows.append(s)
        show(rows, f"{name} split by {split_name} at the anchor "
                   f"(live: z10 +2.33 / rank 84 -> the '>=' row is tomorrow)")
        for r in rows:
            if r.get("n", 0) and r["n"] < 15:
                print(f"    ** {r['label']}: n={r['n']} -- UNDER 15, anecdote tier, "
                      f"quote the sign test or nothing **")

# tighter: the actual live state, z10 >= 2
print("\n  Tighter still: anchors where crude was as stretched as it is TONIGHT "
      "(z10 >= 2)")
rows = []
for name, anchors in [("A. PPI-k2", PPI_A), ("B. Sep-Wed", WED_A)]:
    for h in (1, 5):
        hi = pd.DatetimeIndex([d for d in anchors
                               if d in Z10.index and Z10.loc[d] >= 2])
        s, _, _ = row(hi, h, f"{name} & z10 >= 2  h={h}")
        rows.append(s)
show(rows, "the literal tomorrow-state cell")


# ---------------------------------------------------------------------------
# 4. MIDTERM YEARS (2026 is one)
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("4. MIDTERM-YEAR RESTRICTION (year %% 4 == 2)")
print("=" * 78)
rows = []
for name, anchors in [("A. PPI-k2", PPI_A), ("B. Sep-Wed", WED_A)]:
    for h in (1, 5):
        mid = pd.DatetimeIndex([d for d in anchors if d.year % 4 == 2])
        s, _, _ = row(mid, h, f"{name} midterm only h={h}")
        rows.append(s)
        oth = pd.DatetimeIndex([d for d in anchors if d.year % 4 != 2])
        s, _, _ = row(oth, h, f"{name} non-midterm h={h}")
        rows.append(s)
show(rows, "midterm split")

print("\n  Sep-Wed cell, per-year h=1 mean (is it a decade or a pattern?):")
d, v = wed_out[1]
byyr = pd.Series(100 * v, index=pd.DatetimeIndex(d).year).groupby(level=0)
tab = pd.DataFrame({"n": byyr.size(), "mean_pct": byyr.mean().round(2),
                    "sum_pp": byyr.sum().round(2)})
tab["midterm"] = [(y % 4 == 2) for y in tab.index]
print(tab.to_string())


# ---------------------------------------------------------------------------
# 5. MULTIPLICITY HONESTY for the bare weekday x month grid
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("5. MULTIPLICITY: the weekday x month grid is swept EVERY evening")
print("=" * 78)
print("  Scoring all 60 (weekday, month) cells for CL=F at h=1, then asking")
print("  where the September-Wednesday t-stat ranks among them.")
ts = []
for mo in range(1, 13):
    for wd in range(5):
        prop = ((ref.weekday.to_numpy() == wd) & (ref.month.to_numpy() == mo))
        a = anchors_before(ref, prop)
        d = pd.DatetimeIndex(a).intersection(VALID[1])
        v = FWD[1].loc[d].values.astype(float)
        if len(v) < 20:
            continue
        s = summarize(v, "")
        ts.append({"month": mo, "wd": wd, "n": s["n"],
                   "mean_pct": round(s["mean_pct"], 3), "t": round(s["t"], 2)})
grid = pd.DataFrame(ts).sort_values("t", ascending=False)
print(f"  cells scored: {len(grid)}   |t| >= 2: {(grid['t'].abs() >= 2).sum()}   "
      f"expected by chance at 5%: {0.05*len(grid):.1f}")
print("  top 6 by t:")
print(grid.head(6).to_string(index=False))
print("  bottom 3 by t:")
print(grid.tail(3).to_string(index=False))
me = grid[(grid["month"] == 9) & (grid["wd"] == 2)]
if len(me):
    rank = int((grid["t"] > me["t"].iloc[0]).sum()) + 1
    print(f"\n  September-Wednesday rank among {len(grid)} cells: #{rank}")
    print(f"  Bonferroni-ish: a nominal p from t={me['t'].iloc[0]:.2f} needs to "
          f"survive {len(grid)} looks.")
    from math import erfc, sqrt
    pnom = erfc(abs(me["t"].iloc[0]) / sqrt(2))
    print(f"  two-sided nominal p = {pnom:.4f}  ->  x{len(grid)} = "
          f"{min(1.0, pnom*len(grid)):.3f} after the family correction")

print("\nDONE.")
