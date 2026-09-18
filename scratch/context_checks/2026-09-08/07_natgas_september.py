"""One honest read on natural gas in this calendar window.

The sweep offered:
  E:seasonal_doy NG=F, "same trading day of year +/-2, Sep 09"
      all years n=25, h1 mean +1.351%, median +0.862%, record 18-7 up,
      sign p 0.0216;  midterm-only n=6, mean +1.974%, 5-1 up.
  E:weekday_month NG=F, "Wednesdays in September", h5 mean +2.940%.

NG=F is the single most concentration-prone instrument on the board -- one
2005 or 2021 September can be worth the entire mean -- and it is currently
61% below its 52-week high, which is not the state most of those Septembers
were in.  This script asks whether the cell is an EDGE (vs September, vs all
days) or just natural gas being natural gas, whether two years carry it, and
whether the deep-drawdown state changes it.

Anchor convention: the analogue of TODAY in each prior year, so h=1 is the
analogue of tomorrow (lag=0, close-to-close) -- the product's convention.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from seasonal_edge import seasonal_window_returns  # noqa: E402

ASOF = pd.Timestamp("2026-09-08")
ERA = "2018-01-01"
TOL = 2

px = load_prices(["NG=F", "^GSPC"])
frame = px["NG=F"]
frame = frame[frame.index <= ASOF]
ng = frame["Close"].astype(float).dropna()
ref = pd.DatetimeIndex(px["^GSPC"].index)
ref = ref[ref <= ASOF]

FWD = {h: fwd_ret(ng, h) for h in (1, 5)}
VALID = {h: FWD[h].dropna().index for h in (1, 5)}

# 52-week distance, via rolling_on_valid as the house rule requires
HI252 = rolling_on_valid(ng, lambda x: x.rolling(252).max())
DD52 = ng / HI252 - 1.0

print("=" * 78)
print("0. LIVE STATE SANITY CHECK")
print("=" * 78)
print(f"  NG=F last bar          {ng.index[-1].date()}  close {ng.iloc[-1]:.3f}")
print(f"  dist from 52w high     {100*DD52.iloc[-1]:+.1f}%   "
      f"(52w high {HI252.iloc[-1]:.3f})")
print(f"  21d return             {100*(ng.iloc[-1]/ng.iloc[-22]-1):+.1f}%")
print(f"  NG=F sessions: {len(ng)}   first bar {ng.index[0].date()}")


def stat(vals, label, base_h=None):
    v = np.asarray(vals, dtype=float)
    v = v[~np.isnan(v)]
    s = summarize(v, label)
    if s["n"] == 0:
        return s
    up, down = int((v > 0).sum()), int((v < 0).sum())
    s["record"] = f"{up}-{down}"
    s["sign_p"] = round(sign_test(max(up, down), len(v)), 4)
    s["sign_dir"] = "up" if up >= down else "down"
    if base_h is not None:
        s["edge_pp"] = round(s["mean_pct"] - 100 * FWD[base_h].dropna().mean(), 3)
    for k in ("sd_pct",):
        s.pop(k, None)
    return s


def row_dates(dates, h, label):
    d = pd.DatetimeIndex(dates).intersection(VALID[h])
    v = FWD[h].loc[d].values.astype(float)
    return stat(v, label, base_h=h), d, v


# ---------------------------------------------------------------------------
# 1. reproduce the seasonal doy cell exactly (engine function), then rebuild the
#    same picks by hand so we own the dates and can split them.
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("1. THE Sep-09 +/-2 TRADING-DAY-OF-YEAR CELL, reproduced")
print("=" * 78)
for h in (1, 5):
    for label, filt in (("all_years", None), ("midterm", 2)):
        s = seasonal_window_returns(frame, ASOF, h, cycle_phase_filter=filt,
                                    doy_tol=TOL)
        if not s or s.get("insufficient"):
            print(f"  h={h} {label}: insufficient (n={(s or {}).get('n')})")
            continue
        print(f"  h={h:<2} {label:<10} n={s['n']:<3} mean {100*s['mean']:+.3f}%  "
              f"median {100*s['median']:+.3f}%  record {s['n_up']}-{s['n_down']}  "
              f"sign p {sign_test(max(s['n_up'], s['n_down']), s['n']):.4f}")

# hand-built picks: one per prior year, trading-doy nearest today's, ties earliest
doy = pd.Series(ng.index.year, index=ng.index)
doy = pd.Series(doy.groupby(doy.values).cumcount().values + 1, index=ng.index)
target_doy = int(doy.iloc[-1])
print(f"\n  today's trading-day-of-year = {target_doy}  (tolerance +/-{TOL})")

picks = []
for y in sorted(set(ng.index.year)):
    if y >= ASOF.year:
        continue
    cand = doy[(ng.index.year == y) & (doy - target_doy).abs().le(TOL)]
    if cand.empty:
        continue
    picks.append(cand.index[(cand - target_doy).abs().values.argmin()])
PICKS = pd.DatetimeIndex(picks)
print(f"  hand-built picks: {len(PICKS)} years "
      f"({PICKS[0].date()} .. {PICKS[-1].date()})")

rows = []
for h in (1, 5):
    s, _, _ = row_dates(PICKS, h, f"doy cell h={h}")
    rows.append(s)
show(rows, "hand-built cell (must match the engine numbers above)")

print("\n  PER-YEAR values (this is where natural gas usually confesses):")
tab = []
for d in PICKS:
    r1 = FWD[1].get(d, np.nan)
    r5 = FWD[5].get(d, np.nan)
    tab.append({"year": d.year, "pick": str(d.date()),
                "doy": int(doy.loc[d]),
                "h1_pct": round(100 * r1, 2) if pd.notna(r1) else None,
                "h5_pct": round(100 * r5, 2) if pd.notna(r5) else None,
                "dd52_pct": round(100 * DD52.loc[d], 1)
                if pd.notna(DD52.loc[d]) else None,
                "midterm": d.year % 4 == 2})
print(pd.DataFrame(tab).to_string(index=False))


# ---------------------------------------------------------------------------
# 2. controls: is this an EDGE or just a level?
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("2. CONTROLS -- all-September and all-days")
print("=" * 78)
rows = []
for h in (1, 5):
    s, _, _ = row_dates(PICKS, h, f"doy cell (Sep 09 +/-2) h={h}")
    rows.append(s)
    sept = ng.index[ng.index.month == 9]
    s, _, _ = row_dates(sept, h, f"CTRL all September sessions h={h}")
    rows.append(s)
    s, _, _ = row_dates(sept[np.isin(sept.day, range(4, 15))], h,
                        f"CTRL Sept 4-14 sessions h={h}")
    rows.append(s)
    s = stat(FWD[h].dropna().values, f"CTRL all days h={h}", base_h=h)
    rows.append(s)
    loc = local_control(VALID[h], PICKS, 126)
    s, _, _ = row_dates(loc, h, f"CTRL local +/-126td ex-pick h={h}")
    rows.append(s)
show(rows, "cell vs controls")

print("\n  The control that matters is the MONTH, not all days. NG=F all-days "
      f"h1 = {100*FWD[1].dropna().mean():+.3f}%, but all-September h1 = "
      f"{100*FWD[1].loc[VALID[1][VALID[1].month == 9]].mean():+.3f}% and "
      f"all-September h5 = "
      f"{100*FWD[5].loc[VALID[5][VALID[5].month == 9]].mean():+.3f}%.\n  "
      "September is ALREADY the strong month for natural gas (injection season "
      "ending,\n  hurricane risk premium), so the edge_pp column vs all-days "
      "double-counts the month.\n  Read the 'vs all September' line instead.")


# ---------------------------------------------------------------------------
# 3. era split + concentration
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("3. ERA SPLIT AND CONCENTRATION")
print("=" * 78)
for h in (1, 5):
    _, d, v = row_dates(PICKS, h, "")
    er = era_split(d, v, ERA)
    for e in er:
        e.pop("sd_pct", None)
    show(er, f"h={h} era split (cut {ERA})")
    for k in (2, 3, 4):
        print(f"  h={h} concentration k={k}: {cluster_note(d, v, k)}")
    for k in (1, 2, 3):
        order = np.argsort(-np.abs(v))[:k]
        keep = np.ones(len(v), dtype=bool)
        keep[order] = False
        drop = [f"{pd.Timestamp(d[i]).year}:{100*v[i]:+.1f}%" for i in order]
        up = int((v[keep] > 0).sum())
        print(f"  h={h} drop top-{k} |moves| {drop} -> mean "
              f"{100*v[keep].mean():+.3f}%  n={keep.sum()}  record "
              f"{up}-{keep.sum()-up}  sign p "
              f"{sign_test(max(up, keep.sum()-up), int(keep.sum())):.4f}")


# ---------------------------------------------------------------------------
# 4. midterm-only: an anecdote by the tag rules, reported as one
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("4. MIDTERM-ONLY SPLIT -- ANECDOTE TIER (n < 15, no t-stat is honest)")
print("=" * 78)
mid = pd.DatetimeIndex([d for d in PICKS if d.year % 4 == 2])
non = pd.DatetimeIndex([d for d in PICKS if d.year % 4 != 2])
rows = []
for h in (1, 5):
    s, _, _ = row_dates(mid, h, f"midterm years h={h}")
    rows.append(s)
    s, _, _ = row_dates(non, h, f"non-midterm years h={h}")
    rows.append(s)
show(rows, "cycle split")
for h in (1, 5):
    _, d, v = row_dates(mid, h, "")
    up = int((v > 0).sum())
    print(f"  h={h} midterm picks: "
          + ", ".join(f"{pd.Timestamp(x).year} {100*r:+.1f}%"
                      for x, r in zip(d, v))
          + f"   record {up}-{len(v)-up}, exact sign p "
            f"{sign_test(max(up, len(v)-up), len(v)):.4f}")
print("\n  n=6 at h=1. Under the tag rules that is an ANECDOTE: a 5-1 record is "
      "sign p 0.109\n  two-sided-equivalent territory and 3 of those 6 years "
      "predate 2010. Do not lead with it.")


# ---------------------------------------------------------------------------
# 5. does the deep drawdown change the cell?
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("5. DOES BEING 61% BELOW THE 52-WEEK HIGH CHANGE IT?")
print("=" * 78)
print("  dd52 at each pick is in the per-year table above. Splitting the cell:")
rows = []
for h in (1, 5):
    for thr, lab in ((-0.40, "-40%"), (-0.50, "-50%"), (-0.60, "-60% = TODAY")):
        deep = pd.DatetimeIndex([d for d in PICKS
                                 if pd.notna(DD52.get(d)) and DD52.loc[d] <= thr])
        shal = pd.DatetimeIndex([d for d in PICKS
                                 if pd.notna(DD52.get(d)) and DD52.loc[d] > thr])
        s, _, _ = row_dates(deep, h, f"dd52 <= {lab}  h={h}")
        rows.append(s)
        s, _, _ = row_dates(shal, h, f"dd52 >  {lab}  h={h}")
        rows.append(s)
show(rows, "doy cell split by 52-week drawdown at the pick")
for r in rows:
    if r.get("n", 0) and r["n"] < 15:
        print(f"    ** {r['label']}: n={r['n']} -- UNDER 15, anecdote tier **")

print("\n  And the unconditional deep-drawdown control (any day, not just "
      "September):")
rows = []
for h in (1, 5):
    for thr, lab in ((-0.40, "-40%"), (-0.50, "-50%"), (-0.60, "-60%")):
        d = ng.index[(DD52 <= thr).reindex(ng.index, fill_value=False).values]
        s, _, _ = row_dates(d, h, f"any day, dd52 <= {lab}  h={h}")
        rows.append(s)
show(rows, "deep-drawdown natgas, unconditional")
print(f"\n  Sessions ever as deep as today ({100*DD52.iloc[-1]:.0f}%): "
      f"{int((DD52 <= DD52.iloc[-1]).sum())} of {int(DD52.notna().sum())}")


# ---------------------------------------------------------------------------
# 6. the companion weekday x month cell the sweep also flagged
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("6. COMPANION: NG=F 'Wednesdays in September' (E:weekday_month, h5 +2.94%)")
print("=" * 78)
prop = ((ref.weekday.to_numpy() == 2) & (ref.month.to_numpy() == 9))
nxt = np.zeros(len(prop), dtype=bool)
nxt[:-1] = prop[1:]
WED = ref[nxt]
rows = []
for h in (1, 5):
    s, _, _ = row_dates(WED, h, f"Sep Wednesdays h={h}")
    rows.append(s)
    sept_all = VALID[h][VALID[h].month == 9]
    s, _, _ = row_dates(sept_all, h, f"CTRL all September h={h}")
    rows.append(s)
    s, _, _ = row_dates(sept_all.difference(WED), h,
                        f"CTRL September, NON-Wednesday h={h}")
    rows.append(s)
    s = stat(FWD[h].dropna().values, f"CTRL all days h={h}", base_h=h)
    rows.append(s)
show(rows, "NG=F September Wednesdays -- with the WITHIN-MONTH control")
print("  The within-month row is the test: if September non-Wednesdays score the "
      "same,\n  the cell is the MONTH wearing a weekday label.")
for h in (1, 5):
    _, d, v = row_dates(WED, h, "")
    for k in (2, 4):
        print(f"  h={h} concentration k={k}: {cluster_note(d, v, k)}")
    order = np.argsort(-np.abs(v))[:4]
    keep = np.ones(len(v), dtype=bool)
    keep[order] = False
    print(f"  h={h} mean with top-4 |moves| removed: {100*v[keep].mean():+.3f}%"
          f"  (n={keep.sum()})")
    er = era_split(d, v, ERA)
    for e in er:
        e.pop("sd_pct", None)
    show(er, f"  h={h} era split")

# multiplicity for the bare grid, same as the crude drill
print("\n  MULTIPLICITY: the weekday x month grid is 60 cells swept nightly.")
ts = []
for mo in range(1, 13):
    for wd in range(5):
        p = ((ref.weekday.to_numpy() == wd) & (ref.month.to_numpy() == mo))
        n2 = np.zeros(len(p), dtype=bool)
        n2[:-1] = p[1:]
        a = pd.DatetimeIndex(ref[n2]).intersection(VALID[5])
        v = FWD[5].loc[a].values.astype(float)
        if len(v) < 20:
            continue
        s = summarize(v, "")
        ts.append({"month": mo, "wd": wd, "n": s["n"],
                   "mean_pct": round(s["mean_pct"], 3), "t": round(s["t"], 2)})
grid = pd.DataFrame(ts).sort_values("t", ascending=False)
print(f"  h=5 cells scored: {len(grid)}   |t| >= 2: {(grid['t'].abs() >= 2).sum()}"
      f"   expected at 5%: {0.05*len(grid):.1f}")
print(grid.head(5).to_string(index=False))
me = grid[(grid["month"] == 9) & (grid["wd"] == 2)]
if len(me):
    print(f"  September-Wednesday h=5 rank: "
          f"#{int((grid['t'] > me['t'].iloc[0]).sum()) + 1} of {len(grid)}, "
          f"t={me['t'].iloc[0]:.2f}")

print("\nDONE.")
