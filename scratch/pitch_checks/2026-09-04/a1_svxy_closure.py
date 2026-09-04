"""C1 round 1 — long SVXY from the last close before a >=3 calendar-day market
closure, on a clear calendar.

Blockers run here, in this order, because each one can end it:
  B2  holiday anchor vs payroll-print anchor (post-NFP vol is already dead)
  B3  runway / clear-calendar control (the 2026-09-03 finding)
  MECH the 3-day closure against the ordinary 2-day weekend  <-- the whole thesis
  B1  placebo anchor ladder k=-8..+8
  B8  leverage regime split at 2018-02-28
  B10 midterm cross
  B6/B7/B9 controls, cost, tail
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

pd.set_option("display.width", 200)

PRINT_KINDS = ("nfp", "cpi", "ppi", "fomc_decision")
COST_BPS = 8.0

# ---------------------------------------------------------------- calendar
prices = load_prices(["SVXY", "SPY", "^VIX"])
cal = prices["SPY"].index
px = pd.DataFrame({t: prices[t]["Close"].reindex(cal) for t in prices})
all_dates = px.index

gap = pd.Series(
    np.append((all_dates[1:] - all_dates[:-1]).days, np.nan), index=all_dates)
closure = gap - 1.0          # calendar days with NO session after this close
print(f"calendar {all_dates[0].date()} .. {all_dates[-1].date()}  n={len(all_dates)}")
print("closure-day histogram (non-session calendar days after each close):")
print(closure.value_counts().sort_index().to_string())

# runway: trading sessions from this close to the next scheduled macro print
ev = load_events(list(PRINT_KINDS))["date"]
pos = pd.Series(range(len(all_dates)), index=all_dates)
ev_pos = np.array(sorted({int(all_dates.searchsorted(d)) for d in ev
                          if all_dates[0] <= d <= all_dates[-1]}))
runway = np.full(len(all_dates), 999.0)
for i in range(len(all_dates)):
    j = int(np.searchsorted(ev_pos, i, side="right"))
    if j < len(ev_pos):
        runway[i] = ev_pos[j] - i
runway = pd.Series(runway, index=all_dates)

nfp_days = set(load_events(["nfp"])["date"])
is_print = pd.Series([d in set(ev) for d in all_dates], index=all_dates)
is_nfp = pd.Series([d in nfp_days for d in all_dates], index=all_dates)

TODAY_RUNWAY = 3            # 2026-09-04 close -> PPI 2026-09-10
CLEAR = runway >= 3

m_close3 = closure >= 3
m_close2 = closure == 2      # the ordinary weekend
m_close1 = closure == 1      # a midweek holiday (Thanksgiving Wed, July-3 etc)

print(f"\nanchors: closure>=3 {int(m_close3.sum())}, closure==2 {int(m_close2.sum())}, "
      f"closure==1 {int(m_close1.sum())}")
svxy_ok = px["SVXY"].notna()
print(f"  of which inside SVXY history: >=3 {int((m_close3 & svxy_ok).sum())}, "
      f"==2 {int((m_close2 & svxy_ok).sum())}")
print("  closure>=3 anchors that are ALSO a scheduled print:",
      int((m_close3 & is_print).sum()), " of which NFP:", int((m_close3 & is_nfp).sum()))


def cell(mask, h, legs, label, lag=1, min_gap=None):
    ret = vehicle_ret(px, legs, h, lag)
    valid = ret.notna()
    d = all_dates[mask.reindex(all_dates, fill_value=False).values & valid.values]
    if len(d) == 0:
        return {"label": label, "n": 0}, pd.DatetimeIndex([]), np.array([])
    epi = declusters(d, min_gap or max(h, 3), all_dates)
    v = ret.loc[epi].values
    r = summarize(v, label)
    return r, epi, v


SVXY = [("SVXY", 1.0)]
VIXS = [("^VIX", -1.0)]

# ============================================================ B2 + MECH + B3
print("\n" + "=" * 78)
print("PART 1 — the pattern, and the two anchors it is confounded with")
print("=" * 78)
for h in (1, 2, 3):
    rows = []
    ret = vehicle_ret(px, SVXY, h, 1)
    valid = ret.notna()
    rows.append(cell(m_close3, h, SVXY, f"h={h} closure>=3 (ALL)")[0])
    rows.append(cell(m_close3 & CLEAR, h, SVXY, f"h={h} closure>=3 & clear (the pitch)")[0])
    rows.append(cell(m_close3 & ~CLEAR, h, SVXY, f"h={h} closure>=3 & NOT clear")[0])
    rows.append(cell(m_close2, h, SVXY, f"h={h} closure==2 ordinary weekend")[0])
    rows.append(cell(m_close2 & CLEAR, h, SVXY, f"h={h} closure==2 & clear")[0])
    rows.append(cell(CLEAR & (closure <= 1), h, SVXY, f"h={h} clear, NO closure at all")[0])
    rows.append(summarize(ret[valid].values, f"h={h} CTRL-b all days"))
    show(rows, f"SVXY h={h}")

print("\n--- MECHANISM: does the EXTRA calendar day pay?  (h=1, clear only) ---")
for lbl, m in [("closure==1 (midweek hol)", m_close1 & CLEAR),
               ("closure==2 (weekend)", m_close2 & CLEAR),
               ("closure==3 (3-day wknd)", (closure == 3) & CLEAR),
               ("closure>=4", (closure >= 4) & CLEAR)]:
    r, epi, v = cell(m, 1, SVXY, lbl)
    if r["n"]:
        w = int((v > 0).sum())
        print(f"  {lbl:26s} n={r['n']:4d}  mean {r['mean_pct']:+.3f}%  "
              f"med {r['median_pct']:+.3f}%  hit {r['hit']:.1f}%  t {r['t']:+.2f}  "
              f"record {w}-{r['n']-w} sign p {sign_test(w, r['n']):.4f}")

# ============================================================ B2 separation
print("\n" + "=" * 78)
print("PART 2 (B2) — holiday work that the payroll print does not do")
print("=" * 78)
for h in (1, 2, 3):
    rows = [
        cell(m_close3 & ~is_print & CLEAR, h, SVXY, "closure>=3, NOT a print, clear")[0],
        cell(m_close3 & is_print & CLEAR, h, SVXY, "closure>=3 AND a print, clear")[0],
        cell(is_nfp & ~m_close3 & CLEAR, h, SVXY, "NFP print, NOT holiday-adj, clear")[0],
        cell(is_print & ~m_close3 & CLEAR, h, SVXY, "any print, NOT holiday-adj, clear")[0],
    ]
    show(rows, f"B2 separation, SVXY h={h}")

# ============================================================ B3 runway ctrl
print("\n" + "=" * 78)
print("PART 3 (B3) — runway as the control variable.  Today's runway = 3.")
print("=" * 78)
for h in (1, 3):
    rows = []
    for rl, rm in [("runway<=1", runway <= 1), ("runway==2", runway == 2),
                   ("runway==3", runway == 3), ("runway>=4", runway >= 4)]:
        rows.append(cell(m_close3 & rm, h, SVXY, f"closure>=3 & {rl}")[0])
        rows.append(cell(m_close2 & rm, h, SVXY, f"weekend   & {rl}")[0])
    show(rows, f"runway x closure, SVXY h={h}")

print("\n--- the exact live cell vs its no-holiday twin (h=1 and h=3) ---")
for h in (1, 3):
    a = cell(m_close3 & (runway == 3), h, SVXY, f"h={h} closure>=3, runway==3 (LIVE)")
    b = cell(m_close2 & (runway == 3), h, SVXY, f"h={h} weekend,    runway==3")
    c = cell((closure <= 1) & (runway == 3), h, SVXY, f"h={h} no closure, runway==3")
    show([a[0], b[0], c[0]], f"live-cell twins h={h}")
    if a[0]["n"] and b[0]["n"]:
        d = a[2].mean() - b[2].mean()
        se = np.sqrt(a[2].var(ddof=1)/len(a[2]) + b[2].var(ddof=1)/len(b[2]))
        print(f"  holiday-minus-weekend at runway==3, h={h}: {100*d:+.3f}%  welch t {d/se:+.2f}")

# ============================================================ B1 placebo
print("\n" + "=" * 78)
print("PART 4 (B1) — placebo anchor ladder, k = -8..+8 sessions")
print("=" * 78)
base_anchor = all_dates[(m_close3 & CLEAR).values]
posmap = pd.Series(range(len(all_dates)), index=all_dates)
for h in (1, 3):
    ret = vehicle_ret(px, SVXY, h, 1)
    ladder = []
    for k in range(-8, 9):
        idx = []
        for d in base_anchor:
            p = posmap.get(d)
            if p is None:
                continue
            q = p + k
            if 0 <= q < len(all_dates):
                idx.append(all_dates[q])
        idx = pd.DatetimeIndex(idx)
        idx = idx[ret.reindex(idx).notna().values]
        epi = declusters(idx, max(h, 3), all_dates)
        v = ret.loc[epi].values
        if len(v):
            ladder.append({"k": k, "n": len(v), "mean_pct": 100*v.mean(),
                           "hit": 100*(v > 0).mean(), "t": summarize(v)["t"]})
    L = pd.DataFrame(ladder).sort_values("mean_pct", ascending=False)
    L["rank"] = range(1, len(L)+1)
    print(f"\n  h={h} ladder ranked by mean:")
    print(L.round(3).to_string(index=False))
    print(f"  TRUE ANCHOR k=0 rank = {int(L.loc[L.k == 0, 'rank'].iloc[0])} of {len(L)}")

# ============================================================ B8 leverage
print("\n" + "=" * 78)
print("PART 5 (B8) — SVXY leverage regime break 2018-02-28")
print("=" * 78)
for h in (1, 3):
    r, epi, v = cell(m_close3 & CLEAR, h, SVXY, f"h={h} all")
    m = pd.DatetimeIndex(epi) < pd.Timestamp("2018-02-28")
    rows = [summarize(v[m], f"h={h} pre-2018-02-28 (-1x)"),
            summarize(v[~m], f"h={h} post-2018-02-28 (-0.5x, TRADEABLE)")]
    show(rows, f"leverage split h={h}")
    if (~m).sum():
        vv = v[~m]
        w = int((vv > 0).sum())
        print(f"  post-change record {w}-{len(vv)-w}, sign p {sign_test(w, len(vv)):.4f}, "
              f"bootstrap P(mean<=0) {bootstrap_p_le0(vv):.3f}")
        print(f"  {cluster_note(pd.DatetimeIndex(epi)[~m], vv)}")

# ============================================================ B10 midterm
print("\n" + "=" * 78)
print("PART 6 (B10) — midterm cross (2026 is midterm)")
print("=" * 78)
for h in (1, 3):
    r, epi, v = cell(m_close3 & CLEAR, h, SVXY, "")
    yr = pd.DatetimeIndex(epi).year
    mt = (yr % 4) == 2
    show([summarize(v[mt], f"h={h} midterm"), summarize(v[~mt], f"h={h} non-midterm")],
         f"midterm split h={h}")

# ============================================================ Labor Day only
print("\n" + "=" * 78)
print("PART 7 — the LITERAL live anchor: Labor Day eve only")
print("=" * 78)
lab = []
for d in all_dates[m_close3.values]:
    nd = all_dates[posmap[d] + 1]
    if d.month == 9 or (d.month == 8 and nd.month == 9):
        if nd.month == 9 and nd.day <= 8:
            lab.append(d)
lab = pd.DatetimeIndex(lab)
print("Labor Day eves:", ", ".join(str(d.date()) for d in lab))
for h in (1, 3):
    ret = vehicle_ret(px, SVXY, h, 1)
    idx = lab[ret.reindex(lab).notna().values]
    v = ret.loc[idx].values
    if len(v):
        w = int((v > 0).sum())
        print(f"  SVXY h={h}: n={len(v)} mean {100*v.mean():+.3f}% hit {100*(v>0).mean():.1f}% "
              f"record {w}-{len(v)-w} sign p {sign_test(w, len(v)):.4f} "
              f"worst {100*v.min():+.2f}% best {100*v.max():+.2f}%")
        print("   ", ", ".join(f"{d.date()}:{100*x:+.2f}" for d, x in zip(idx, v)))

# ============================================================ ^VIX mirror
print("\n" + "=" * 78)
print("PART 8 — short-^VIX mirror (mechanism read, not tradeable)")
print("=" * 78)
for h in (1, 3):
    rows = [cell(m_close3 & CLEAR, h, VIXS, f"h={h} closure>=3 & clear")[0],
            cell(m_close2 & CLEAR, h, VIXS, f"h={h} weekend & clear")[0],
            cell(m_close3 & CLEAR & ~is_print, h, VIXS, f"h={h} closure>=3 clear NOT print")[0]]
    ret = vehicle_ret(px, VIXS, h, 1)
    rows.append(summarize(ret.dropna().values, f"h={h} CTRL-b all days"))
    show(rows, f"short ^VIX h={h}")

# ============================================================ cost + controls
print("\n" + "=" * 78)
print("PART 9 — cost, local control, tail")
print("=" * 78)
for h in (1, 3):
    r, epi, v = cell(m_close3 & CLEAR, h, SVXY, "")
    ret = vehicle_ret(px, SVXY, h, 1)
    loc = local_control(all_dates[ret.notna().values], pd.DatetimeIndex(epi))
    print(f"\n h={h}: episode mean {100*v.mean():+.3f}% = {100*v.mean()*100:.1f} bps "
          f"vs {COST_BPS} bps cost -> {100*v.mean()*100/COST_BPS:.1f}x")
    show([summarize(v, f"h={h} COND"),
          summarize(ret.loc[loc].values, f"h={h} CTRL-c local +/-126td")],
         f"local control h={h}")
    print(f"  {cluster_note(pd.DatetimeIndex(epi), v)}")
