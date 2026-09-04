"""C6b -- round 2 on the ONE cell that survived C6 round 1.

After the inception guard, exactly one region of the 210-cell grid has a pulse:
the rates/credit complex at h=1, IEF +0.228% t=3.99 (n=24) and LQD +0.218%
t=3.82 (n=24), and the placebo offset ladder ISOLATES k=0 (every neighbour
-0.09%..+0.11%).  That is the first JH ladder in this repo that does not die at
step one, so it gets a real round 2.

THE SUSPICION.  Jackson Hole is the last FRIDAY of August.  A lag=1 h=1 entry
therefore buys the MONDAY close and sells the TUESDAY close -- two sessions
after the speech, and almost always across the AUGUST MONTH TURN.  Month-end
duration-extension buying into the Agg rebalance plus a fresh-month bid is a
real, documented flow that has nothing to do with a Fed chair.  The C6.7 tdom
control anchored on the JH day's position, not the ENTRY day's, which is why it
could not see this.

Probes:
  1. where does the entry actually land?  weekday + trading-day-of-month of the
     entry and exit close, per anchor
  2. the correct month-turn control: same Mon->Tue span at the AUGUST turn in
     every year, JH or not; then at EVERY month's turn
  3. is rates/credit one object or four?  correlation of the four cells
  4. era split, and the 2022 hawkish shock specifically
  5. cost
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pitch_lab import (load_prices, load_events, fwd_lag, summarize, show,  # noqa: E402
                       sign_test)

TK = ["IEF", "LQD", "TLT", "^TNX", "AGG", "SHY"]
ASOF = pd.Timestamp("2026-08-27")
px = load_prices(TK)
S = {t: px[t]["Close"].dropna().loc[:ASOF] for t in px}
jh = pd.DatetimeIndex(load_events(["jackson_hole"])["date"])


def positions(idx, dates, offset=0):
    out, lo, hi = [], idx[0], idx[-1]
    for d in pd.DatetimeIndex(dates):
        if d < lo or d > hi:
            continue
        loc = idx.searchsorted(d)
        if loc >= len(idx):
            continue
        p = loc + offset
        if 0 <= p < len(idx):
            out.append(p)
    return out


def tdom_series(idx):
    s = pd.Series(0, index=idx)
    for _, g in pd.Series(idx, index=idx).groupby([idx.year, idx.month]):
        s.loc[g.index] = np.arange(1, len(g) + 1)
    return s


def tdom_from_end(idx):
    s = pd.Series(0, index=idx)
    for _, g in pd.Series(idx, index=idx).groupby([idx.year, idx.month]):
        s.loc[g.index] = np.arange(len(g), 0, -1)   # 1 == last session of month
    return s


# ---------------------------------------------------------------------------
print("=== C6b.1  WHERE DOES THE h=1 ENTRY ACTUALLY LAND? ===")
idx = S["IEF"].index
td, tde = tdom_series(idx), tdom_from_end(idx)
DAY = ["Mon", "Tue", "Wed", "Thu", "Fri"]
rows, straddle = [], 0
for p in positions(idx, jh):
    if p + 2 >= len(idx):
        continue
    d0, d1, d2 = idx[p], idx[p + 1], idx[p + 2]
    cross = (d1.month != d2.month)
    straddle += int(cross)
    rows.append({"JH": str(d0.date()), "jh_dow": DAY[d0.weekday()],
                 "entry": str(d1.date()), "entry_dow": DAY[d1.weekday()],
                 "entry_tdom_end": int(tde.iloc[p + 1]),
                 "exit": str(d2.date()), "exit_dow": DAY[d2.weekday()],
                 "crosses_month_turn": cross})
R = pd.DataFrame(rows)
print(R.to_string(index=False))
print(f"\n  JH weekday: {R['jh_dow'].value_counts().to_dict()}")
print(f"  ENTRY weekday: {R['entry_dow'].value_counts().to_dict()}")
print(f"  entry sessions-from-month-end (1 = last session of August): "
      f"{sorted(R['entry_tdom_end'].tolist())}")
print(f"  >>> the h=1 hold CROSSES THE AUGUST MONTH TURN in {straddle} of "
      f"{len(R)} anchors ({100*straddle/len(R):.0f}%)")

# ---------------------------------------------------------------------------
print("\n=== C6b.2  THE CORRECT CONTROL: same span at the month turn, no JH ===")
for t in ["IEF", "LQD", "TLT", "AGG"]:
    if t not in S:
        continue
    s = S[t]
    i = s.index
    tde_t = tdom_from_end(i)
    r = fwd_lag(s, 1, 1)
    jhv = np.asarray([r.iloc[p] for p in positions(i, jh)
                      if p < len(r) and not np.isnan(r.iloc[p])], float)
    rows = []
    rows.append(summarize(jhv, f"{t} JH anchor h=1"))
    # entry lands ME-k: anchor is the day whose D+1 sits k sessions from the end
    ent_k = pd.Series(tde_t.values, index=i).shift(-1)
    for k in (1, 2, 3, 4):
        m = (ent_k == k).values & r.notna().values
        rows.append(summarize(r[m].values, f"{t} ALL months, entry at ME-{k}"))
    # August only, entry at the same ME-k the JH anchors actually use
    aug = (i.month == 8)
    m_aug = aug & ((ent_k <= 3) & (ent_k >= 1)).values & r.notna().values
    rows.append(summarize(r[m_aug].values, f"{t} AUGUST, entry ME-1..ME-3"))
    m_any = ((ent_k <= 3) & (ent_k >= 1)).values & r.notna().values
    rows.append(summarize(r[m_any].values, f"{t} ALL months, entry ME-1..ME-3"))
    rows.append(summarize(r.dropna().values, f"{t} all days"))
    show(rows, f"{t}: JH label vs month-turn position")

# ---------------------------------------------------------------------------
print("\n=== C6b.3  the DOUBLE control: month-turn AND non-JH Augusts ===")
for t in ["IEF", "LQD"]:
    s = S[t]
    i = s.index
    tde_t = tdom_from_end(i)
    ent_k = pd.Series(tde_t.values, index=i).shift(-1)
    r = fwd_lag(s, 1, 1)
    jh_pos = set(positions(i, jh))
    is_jh = np.zeros(len(i), bool)
    for p in jh_pos:
        is_jh[p] = True
    m_turn = ((ent_k <= 3) & (ent_k >= 1)).values & r.notna().values
    rows = [summarize(r[m_turn & is_jh].values, f"{t} month-turn AND JH"),
            summarize(r[m_turn & ~is_jh].values, f"{t} month-turn, NOT JH"),
            summarize(r[~m_turn & is_jh].values, f"{t} JH but NOT month-turn"),
            summarize(r[~m_turn & ~is_jh].values, f"{t} neither")]
    show(rows, f"{t} 2x2: is the label or the position doing the work?")

# ---------------------------------------------------------------------------
print("\n=== C6b.4  rates + credit: one object or four? ===")
i = S["IEF"].index
cols = {}
for t in ["IEF", "LQD", "TLT", "^TNX"]:
    if t in S:
        cols[t] = fwd_lag(S[t], 1, 1).reindex(i)
D = pd.DataFrame(cols).dropna()
print("  daily h=1 forward-return correlation matrix (all days):")
print(D.corr().round(3).to_string())
print("\n  -> IEF/LQD/TLT/^TNX are ONE duration object measured four ways. "
      "Four 'confirmations' in C6.1 are one bet, and the 210-cell grid's "
      "effective width is far below 210.")

# ---------------------------------------------------------------------------
print("\n=== C6b.5  era split and the 2022 hawkish shock ===")
for t in ["IEF", "LQD"]:
    s = S[t]
    i = s.index
    r = fwd_lag(s, 1, 1)
    ps = positions(i, jh)
    v, dts = [], []
    for p in ps:
        if p < len(r) and not np.isnan(r.iloc[p]):
            v.append(r.iloc[p])
            dts.append(i[p])
    v = np.asarray(v, float)
    y = pd.DatetimeIndex(dts).year
    show([summarize(v[y < 2013], "pre-2013"), summarize(v[(y >= 2013) & (y < 2020)], "2013-2019"),
          summarize(v[y >= 2020], "2020+"),
          summarize(v[y % 4 == 2], "midterm"), summarize(v[y % 4 != 2], "non-midterm")],
         f"{t} h=1 JH era split")
    per = pd.Series(100 * v, index=y)
    print(f"  {t} by year: " + "  ".join(f"{yy}:{vv:+.2f}" for yy, vv in per.items()))
    w = int((v > 0).sum())
    print(f"  record {w}-{len(v)-w}  two-sided sign p = "
          f"{2*sign_test(max(w, len(v)-w), len(v)):.4f}")
    tot = v.sum()
    order = np.argsort(-np.abs(v))[:2]
    print(f"  top2 |moves| {[str(pd.Timestamp(dts[j]).date()) for j in order]} = "
          f"{100*v[order].sum():+.2f}pp of {100*tot:+.2f}pp total "
          f"({100*v[order].sum()/(100*tot)*100:.0f}%)")

print("\n=== C6b.6  cost ===")
for t, bps in [("IEF", 3.0), ("LQD", 5.0)]:
    s = S[t]
    i = s.index
    r = fwd_lag(s, 1, 1)
    v = np.asarray([r.iloc[p] for p in positions(i, jh)
                    if p < len(r) and not np.isnan(r.iloc[p])], float)
    print(f"  {t}: {100*v.mean()*100:.1f} bps vs ~{bps} bps round trip = "
          f"{100*v.mean()*100/bps:.1f}x  (need >= 5x)")
