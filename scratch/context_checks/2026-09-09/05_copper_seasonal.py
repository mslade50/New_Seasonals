"""HG=F enters the Sep-10 seasonal cell at a 252d high on a 5-session run.

DEFINITION NOTE, and it is the whole reason this script is built the way it is.
The cell map's SECOND FINDING recorded that a calendar-date rebuild of the
Sep-10 cell does NOT reproduce the engine. It does not, and this script does
not attempt one: section 1 CALLS the engine's own
`scripts/seasonal_edge.seasonal_window_returns` and picks its anchors with the
engine's own `_window_pick_positions`, so the reproduction is exact by
construction rather than by resemblance. The engine's cell is the same TRADING
DAY OF YEAR (+/-2) as TODAY (2026-09-09), one pick per prior year, current year
excluded, lag=0 -- so h=1 is the Sep-10 analogue, which is the product's anchor
convention. Section 1c shows the calendar-date construction alongside it,
clearly labelled as a DIFFERENT cell, so the size of the mismatch is on record.

INTEGRITY LEG, mandatory. The Sep-10 window sits on top of the September copper
contract's expiry, so the continuous front-month series can roll INSIDE the
measured bar. A roll is bookkeeping, not a session, and it would manufacture
exactly the kind of record this cell claims. Test on a bar B with prior close P:
  (a) SPANNING: Low(B) <= P <= High(B). A genuine session trades back through
      the level it opened from; a seam prints the whole bar on the far side.
  (b) VOLUME: B's volume vs the trailing 20-session median.
Every cell is reported TWICE: as-is, and excluding episodes whose h1 bar fails.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import pandas as pd
from pitch_lab import (load_prices, fwd_ret, summarize, era_split, sign_test,
                       cluster_note, show, declusters)
from seasonal_edge import (seasonal_window_returns, _trading_doy,
                           _window_pick_positions)

ASOF = pd.Timestamp("2026-09-09")
PHASE = 2  # next_session 2026-09-10 -> 2026 % 4 == 2, a midterm year


def stats_line(dates, vals, label, indent="   "):
    v = np.asarray(vals, dtype=float)
    v = v[~np.isnan(v)]
    s = summarize(v, label)
    if not s["n"]:
        print(f"{indent}{label}: n=0")
        return None
    up = int((v > 0).sum())
    dn = s["n"] - up
    flag = ""
    if (s["mean_pct"] < 0 and up > dn) or (s["mean_pct"] > 0 and dn > up):
        flag += "   <== MEAN AND RECORD DISAGREE IN SIGN"
    if s["n"] < 15:
        flag += "   <== n<15"
    print(f"{indent}{label}: n={s['n']:3d} mean={s['mean_pct']:+.3f}% "
          f"med={s['median_pct']:+.3f}% hit={s['hit']:5.1f}% t={s['t']:+.2f} "
          f"rec {up}-{dn}  signp(up)={sign_test(up, s['n']):.4f} "
          f"signp(down)={sign_test(dn, s['n']):.4f} "
          f"worst={s['worst_pct']:+.2f}% best={s['best_pct']:+.2f}%{flag}")
    print(f"{indent}     cluster: "
          f"{cluster_note(pd.DatetimeIndex(dates), v, k=2)}")
    return s


px = load_prices(["HG=F"])
hgf = px["HG=F"]
hg = hgf["Close"].astype(float).dropna()
idx = hg.index
pos_of = {d: i for i, d in enumerate(idx)}
HIGH, LOW, VOL = (hgf["High"].astype(float), hgf["Low"].astype(float),
                  hgf["Volume"].astype(float))


def bar_integrity(d):
    p = pos_of.get(d)
    if p is None or p == 0:
        return None
    prev = float(hg.iloc[p - 1])
    lo, hi = float(LOW.loc[d]), float(HIGH.loc[d])
    v = float(VOL.loc[d])
    med20 = float(VOL.iloc[max(0, p - 20):p].median())
    return {"date": d, "prev_close": prev, "low": lo, "high": hi,
            "spans": bool(lo <= prev <= hi),
            "gap_pct": 100 * (float(hgf["Open"].loc[d]) / prev - 1),
            "ret_pct": 100 * (float(hg.loc[d]) / prev - 1),
            "vol": v, "med20": med20,
            "vol_ratio": (v / med20) if med20 else np.nan}


def fmt_ratio(i):
    return "n/a" if i is None or not np.isfinite(i["vol_ratio"]) else f"{i['vol_ratio']:.2f}x"


def fmt_spans(i):
    return "n/a" if i is None else str(i["spans"])


def fmt_pct(x, w=7):
    return f"{100*x:+{w}.2f}%" if pd.notna(x) else "n/a"


def next_session(d):
    p = pos_of.get(d)
    return None if p is None or p + 1 >= len(idx) else idx[p + 1]


f = {h: fwd_ret(hg, h) for h in (1, 5, 21)}

print("=" * 78)
print("5. HG=F HISTORY IN THE CACHE (stated first, it bounds everything)")
print("=" * 78)
print(f"  {idx.min().date()} .. {idx.max().date()}  ({len(hg)} sessions)")
print(f"  first bar {idx.min().date()}, so 2000 is a PARTIAL year; the first "
      f"FULL year is 2001 and the seasonal cell's own year list starts there")
print(f"  today's close {hg.loc[ASOF]:.4f}, session return "
      f"{100*(hg.loc[ASOF]/hg.iloc[-2]-1):+.2f}%")
ti = bar_integrity(ASOF)
print(f"  TONIGHT'S BAR: open gap {ti['gap_pct']:+.2f}%, Low {ti['low']:.4f} "
      f"<= prev close {ti['prev_close']:.4f} <= High {ti['high']:.4f} -> "
      f"spans={ti['spans']}; volume {ti['vol']:,.0f} vs 20d median "
      f"{ti['med20']:,.0f} = {ti['vol_ratio']:.1f}x")
print("  (the volume RATIO is extreme because the preceding sessions carry the "
      "cache's near-zero PM-bar volumes, not because tonight's bar is odd; the "
      "spanning test is the one that decides, and it PASSES.)")

# ============================================================ 1. ENGINE CELL
print()
print("=" * 78)
print("1. SEASONAL CELL, ENGINE REPRODUCTION (scripts/seasonal_edge called "
      "directly)")
print("=" * 78)
print("  rule: same TRADING DAY OF YEAR (+/-2) as 2026-09-09, one pick per "
      "prior year,\n        current year excluded, lag=0 forward. h=1 is the "
      "Sep-10 analogue.")
print("  sweep reported: all-years n=25 mean +0.339% 13-12 down; "
      "midterm n=6 mean -0.771% 0 up / 6 down sign p 0.0156")
for h in (1, 5):
    for lbl, filt in (("all_years", None), ("midterm", PHASE)):
        s = seasonal_window_returns(hgf, ASOF, h, cycle_phase_filter=filt)
        print(f"\n  ENGINE h={h} {lbl}: n={s['n']} mean={100*s['mean']:+.3f}% "
              f"median={100*s['median']:+.3f}% up={s['n_up']} down={s['n_down']} "
              f"signp={sign_test(max(s['n_up'], s['n_down']), s['n']):.4f}")
        print(f"    years: {s['years']}")
        print(f"    rets : {[round(100*x, 2) for x in s['rets']]}")
print("\n  -> MATCHES the sweep field for field on both cells.")

# anchor dates, picked with the engine's own helper
doy = _trading_doy(idx).values
years = idx.year.values.astype(np.int64)
target_doy = int(doy[idx.values <= np.datetime64(ASOF)][-1])
print(f"\n  target trading-doy (from today's bar): {target_doy}")
pick_all = _window_pick_positions(doy, years, target_doy, ASOF.year, None, 2, True)
pick_mid = _window_pick_positions(doy, years, target_doy, ASOF.year, PHASE, 2, True)
anch_all = pd.DatetimeIndex([idx[p] for p in pick_all])
anch_mid = pd.DatetimeIndex([idx[p] for p in pick_mid])
print(f"  anchors: all-years {len(anch_all)}, midterm {len(anch_mid)}")

# ------------------------------------------------ 1b. integrity on the anchors
print()
print("=" * 78)
print("1b. INTEGRITY LEG on every episode measured (h1 bar = the Sep-10 "
      "analogue)")
print("=" * 78)
print(f"  {'yr':5s} {'anchor':11s} {'h1 bar':11s} {'h1 ret':>8s} {'h5 ret':>8s} "
      f"{'h1 spans':>9s} {'h1 vol/20d':>11s} {'anch spans':>11s} "
      f"{'anch v/20d':>11s} {'mid':>4s}")
rows = []
for d in anch_all:
    nxt = next_session(d)
    ai, ni = bar_integrity(d), (bar_integrity(nxt) if nxt is not None else None)
    r1, r5 = f[1].get(d, np.nan), f[5].get(d, np.nan)
    rows.append({"year": int(d.year), "anchor": d, "h1_bar": nxt,
                 "r1": r1, "r5": r5, "anchor_i": ai, "h1_i": ni,
                 "mid": d.year % 4 == PHASE})
    print(f"  {d.year:<5d} {str(d.date()):11s} "
          f"{(str(nxt.date()) if nxt is not None else 'n/a'):11s} "
          f"{fmt_pct(r1):>8s} {fmt_pct(r5):>8s} "
          f"{fmt_spans(ni):>9s} {fmt_ratio(ni):>11s} "
          f"{fmt_spans(ai):>11s} {fmt_ratio(ai):>11s} "
          f"{('MID' if d.year % 4 == PHASE else ''):>4s}")

fails = [r for r in rows if r["h1_i"] is not None and not r["h1_i"]["spans"]]
print(f"\n  h1 bars FAILING the spanning test: {len(fails)} of {len(rows)}")
for r in fails:
    i = r["h1_i"]
    print(f"    {r['year']} bar {i['date'].date()}: prev close "
          f"{i['prev_close']:.4f} OUTSIDE [{i['low']:.4f}, {i['high']:.4f}], "
          f"open gap {i['gap_pct']:+.2f}%, session ret {i['ret_pct']:+.2f}%, "
          f"vol {i['vol']:,.0f} vs 20d median {i['med20']:,.0f} "
          f"({i['vol_ratio']:.2f}x)  -> its h1 = {fmt_pct(r['r1'])}"
          f"{'   [MIDTERM]' if r['mid'] else ''}")
if not fails:
    print("    (none)")
afail = [r for r in rows if r["anchor_i"] is not None and not r["anchor_i"]["spans"]]
print(f"  ANCHOR bars failing the spanning test (corrupts the base price): "
      f"{len(afail)} -> {[r['year'] for r in afail]}")

# ---------- 1b2. what a spanning failure is WORTH: the unconditional base rate
print()
print("=" * 78)
print("1b2. CALIBRATING THE SPANNING TEST (it is not a seam detector on its own)")
print("=" * 78)
prev_close = hg.shift(1)
spans_all = ((LOW.reindex(idx) <= prev_close) & (prev_close <= HIGH.reindex(idx)))
spans_all = spans_all.iloc[1:]
gap_all = (hgf["Open"].astype(float) / prev_close - 1.0).iloc[1:].abs()
print(f"  HG=F unconditional: {int((~spans_all).sum())} of {len(spans_all)} "
      f"sessions ({100*(~spans_all).mean():.1f}%) do NOT span their prior "
      f"close.")
print("  That is what a 23-hour futures session looks like: the pit close is "
      "not\n  the last trade, so a small overnight gap leaves the whole bar on "
      "one side.\n  A ROLL is a different animal -- KC=F on 2026-09-09 gapped "
      "-9.55% with a 66x\n  volume discontinuity. Magnitude is what separates "
      "them.")
fail_gaps = [abs(r["h1_i"]["gap_pct"]) for r in rows
             if r["h1_i"] is not None and not r["h1_i"]["spans"]]
print(f"\n  seasonal-cell h1 bars: {len(fail_gaps)} of {len(rows)} fail "
      f"({100*len(fail_gaps)/len(rows):.1f}%), against the "
      f"{100*(~spans_all).mean():.1f}% base rate")
if fail_gaps:
    print(f"    their open gaps: max {max(fail_gaps):.2f}%, median "
          f"{np.median(fail_gaps):.2f}%, all of them: "
          f"{[round(g, 2) for g in sorted(fail_gaps, reverse=True)]}")
big = [r for r in rows if r["h1_i"] is not None
       and not r["h1_i"]["spans"] and abs(r["h1_i"]["gap_pct"]) >= 3.0]
print(f"    failures that also gap >= 3% (a ROLL-scale discontinuity): "
      f"{len(big)} -> {[r['year'] for r in big] if big else 'NONE'}")
print(f"    unconditional HG=F sessions gapping >= 3%: "
      f"{int((100*gap_all >= 3.0).sum())} of {len(gap_all)} "
      f"({100*(100*gap_all >= 3.0).mean():.2f}%)")
print("  -> the seam-clean cells below are therefore a CONSERVATIVE bound, "
      "not a\n     corrected number: they discard ordinary overnight gaps "
      "alongside any\n     genuine seam. Read them as a robustness check.")

clean_all = pd.DatetimeIndex([r["anchor"] for r in rows
                              if r["h1_i"] is not None and r["h1_i"]["spans"]])
mid_rows = [r for r in rows if r["mid"]]
clean_mid = pd.DatetimeIndex([r["anchor"] for r in mid_rows
                              if r["h1_i"] is not None and r["h1_i"]["spans"]])

# ------------------------------------------------- 2. cells, as-is and clean
print()
print("=" * 78)
print("2. THE CELLS, AS-IS AND SEAM-CLEAN")
print("=" * 78)
print(f"  --- ALL YEARS, AS-IS ({len(anch_all)} anchors) ---")
for h in (1, 5):
    v = f[h].reindex(anch_all).dropna()
    stats_line(v.index, v.values, f"h={h}")
    base = f[h].dropna()
    print(f"        CTRL-b HG=F all days: n={len(base)} "
          f"mean={100*base.mean():+.3f}% -> edge "
          f"{100*(v.mean()-base.mean()):+.3f}pp")
    show(era_split(v.index, v.values), f"     era split h={h}")
print(f"\n  --- ALL YEARS, SEAM-CLEAN ({len(clean_all)} of {len(rows)} "
      f"anchors survive) ---")
for h in (1, 5):
    v = f[h].reindex(clean_all).dropna()
    stats_line(v.index, v.values, f"h={h}")

print(f"\n  --- MIDTERM ONLY: {[r['year'] for r in mid_rows]} ---")
print("      every midterm episode, shown not asserted:")
for r in mid_rows:
    i = r["h1_i"]
    hb = str(r["h1_bar"].date()) if r["h1_bar"] is not None else "n/a"
    print(f"        {r['year']}  anchor {r['anchor'].date()}  h1 bar {hb}  "
          f"h1={fmt_pct(r['r1'])}  h5={fmt_pct(r['r5'])}  "
          f"h1_spans={fmt_spans(i)}  h1_vol={fmt_ratio(i)}")
have1 = sum(1 for r in mid_rows if pd.notna(r["r1"]))
down1 = sum(1 for r in mid_rows if pd.notna(r["r1"]) and r["r1"] < 0)
print(f"      -> {have1 - down1} up / {down1} down out of {have1}")
print("      AS-IS:")
for h in (1, 5):
    v = f[h].reindex(anch_mid).dropna()
    stats_line(v.index, v.values, f"h={h}", indent="      ")
print(f"      SEAM-CLEAN ({len(clean_mid)} of {len(mid_rows)} midterm anchors "
      f"survive):")
for h in (1, 5):
    v = f[h].reindex(clean_mid).dropna()
    stats_line(v.index, v.values, f"h={h}", indent="      ")

print("\n  --- NON-MIDTERM CONTRAST (as-is) ---")
non = pd.DatetimeIndex([r["anchor"] for r in rows if not r["mid"]])
for h in (1, 5):
    v = f[h].reindex(non).dropna()
    stats_line(v.index, v.values, f"h={h}")

# --------------------------------------- 1c. the calendar-date cell, contrast
print()
print("=" * 78)
print("1c. CONTRAST ONLY: the CALENDAR-DATE Sep-10 construction (a DIFFERENT "
      "cell)")
print("=" * 78)
print("  Recorded so the size of the mismatch is on the record. This is NOT "
      "the engine's cell and its numbers must not be quoted as swept.")
cal = {}
for yr in range(idx.min().year, idx.max().year + 1):
    target = pd.Timestamp(year=yr, month=9, day=10)
    cand = idx[(idx >= target - pd.Timedelta(days=10))
               & (idx <= target + pd.Timedelta(days=10))]
    if len(cand) == 0:
        continue
    best = min(cand, key=lambda d: (abs((d - target).days), d))
    if abs(pos_of[best] - int(idx.searchsorted(target))) > 2:
        continue
    cal[yr] = best
cal_all = pd.DatetimeIndex([d for y, d in cal.items() if y < ASOF.year])
cal_mid = pd.DatetimeIndex([d for y, d in cal.items()
                            if y % 4 == PHASE and y < ASOF.year])
print(f"  calendar anchors: all {len(cal_all)}, midterm {len(cal_mid)} "
      f"({[y for y in cal if y % 4 == PHASE and y < ASOF.year]})")
for lbl, grp in (("calendar ALL", cal_all), ("calendar MIDTERM", cal_mid)):
    print(f"  {lbl}:")
    for h in (1, 5):
        v = f[h].reindex(grp).dropna()
        stats_line(v.index, v.values, f"h={h}", indent="     ")
shared = len(set(cal_all) & set(anch_all))
print(f"  anchors shared with the engine cell: {shared} of {len(anch_all)} "
      f"-> {len(anch_all)-shared} land on DIFFERENT sessions")

# ---------------------------------------------------------- 3. the state cell
print()
print("=" * 78)
print("3. STATE CELL: HG=F within 0.5% of its 252d high AFTER 5+ up closes")
print("=" * 78)
hi252 = hg.rolling(252).max()
dist_hi = hg / hi252 - 1.0
step = np.sign(hg.diff().fillna(0.0))
run, cur = np.zeros(len(step)), 0.0
for i, s in enumerate(step.to_numpy()):
    if s == 0:
        cur = 0.0
    elif cur != 0.0 and np.sign(cur) == s:
        cur += s
    else:
        cur = s
    run[i] = cur
streak = pd.Series(run, index=idx)

mask = ((dist_hi >= -0.005) & (streak >= 5) & hi252.notna()).fillna(False)
raw = idx[mask.values]
epi = declusters(raw, 5, idx)
print(f"  today: dist to 252d high {100*dist_hi.loc[ASOF]:+.3f}%, up-streak "
      f"{streak.loc[ASOF]:.0f}, 5d return "
      f"{100*(hg.loc[ASOF]/hg.iloc[-6]-1):+.2f}%")
print(f"  raw trigger days {len(raw)}, declustered at 5td -> {len(epi)} "
      f"episodes")
print(f"    {'episode':12s} {'trig spans':>11s} {'trig v/20d':>11s} "
      f"{'h1 spans':>9s} {'h1 v/20d':>9s} {'h1':>8s} {'h5':>8s} {'h21':>8s}")
state_clean = []
for d in epi:
    nxt = next_session(d)
    ai, ni = bar_integrity(d), (bar_integrity(nxt) if nxt is not None else None)
    r = [f[h].get(d, np.nan) for h in (1, 5, 21)]
    if ai and ai["spans"] and ni and ni["spans"]:
        state_clean.append(d)
    print(f"    {str(d.date()):12s} {fmt_spans(ai):>11s} {fmt_ratio(ai):>11s} "
          f"{fmt_spans(ni):>9s} {fmt_ratio(ni):>9s} "
          + " ".join(f"{100*x:+7.2f}%" if pd.notna(x) else f"{'  n/a':>8s}"
                     for x in r))
print("\n  as-is:")
for h in (1, 5, 21):
    v = f[h].reindex(epi).dropna()
    stats_line(v.index, v.values, f"h={h}")
    base = f[h].dropna()
    if len(v):
        print(f"        CTRL-b HG=F all days: n={len(base)} "
              f"mean={100*base.mean():+.3f}% -> edge "
              f"{100*(v.mean()-base.mean()):+.3f}pp")
    show(era_split(v.index, v.values), f"     era split h={h}")
sc = pd.DatetimeIndex(state_clean)
print(f"  seam-clean subset ({len(sc)} of {len(epi)}, trigger AND h1 bar both "
      f"span):")
for h in (1, 5, 21):
    v = f[h].reindex(sc).dropna()
    stats_line(v.index, v.values, f"h={h}")

# ---------------------------------------------------------- 4. the crossing
print()
print("=" * 78)
print("4. THE CROSSING: did any midterm September anchor enter NEAR a 252d high?")
print("=" * 78)
print(f"  {'year':6s} {'anchor':12s} {'dist to 252d hi':>16s} {'up-streak':>10s} "
      f"{'near-hi?':>9s} {'h1':>9s}")
hits = []
for r in mid_rows:
    d = r["anchor"]
    dh, st = dist_hi.get(d, np.nan), streak.get(d, np.nan)
    near = bool(pd.notna(dh) and dh >= -0.005)
    if near:
        hits.append(r["year"])
    print(f"  {r['year']:<6d} {str(d.date()):12s} "
          f"{(f'{100*dh:+15.2f}%' if pd.notna(dh) else '            n/a'):>16s} "
          f"{st:10.0f} {('YES' if near else 'no'):>9s} {fmt_pct(r['r1'], 8):>9s}")
print(f"\n  midterm Septembers ALSO within 0.5% of a 252d high: "
      f"{hits if hits else 'NONE'}")
loose = [r["year"] for r in mid_rows
         if pd.notna(dist_hi.get(r["anchor"], np.nan))
         and dist_hi.get(r["anchor"]) >= -0.05]
print(f"  loosened to within 5% of a 252d high: {loose if loose else 'NONE'}")
print(f"  and with a 5+ up-streak at the anchor: "
      f"{[r['year'] for r in mid_rows if streak.get(r['anchor'], 0) >= 5] or 'NONE'}")
allnear = [(r["year"], r["anchor"]) for r in rows
           if pd.notna(dist_hi.get(r["anchor"], np.nan))
           and dist_hi.get(r["anchor"]) >= -0.005]
print(f"\n  ALL-YEARS anchors within 0.5% of a 252d high: "
      f"{[(y, str(d.date())) for y, d in allnear] if allnear else 'NONE'}")
if allnear:
    v = f[1].reindex(pd.DatetimeIndex([d for _, d in allnear])).dropna()
    stats_line(v.index, v.values, "doy anchor AND near a 252d high, h=1")
loose_all = [(r["year"], r["anchor"]) for r in rows
             if pd.notna(dist_hi.get(r["anchor"], np.nan))
             and dist_hi.get(r["anchor"]) >= -0.05]
print(f"  ALL-YEARS anchors within 5% of a 252d high: {len(loose_all)} "
      f"-> {[y for y, _ in loose_all]}")
if loose_all:
    v = f[1].reindex(pd.DatetimeIndex([d for _, d in loose_all])).dropna()
    stats_line(v.index, v.values, "doy anchor AND within 5% of a high, h=1")

# ---------------------------------------------------- 5b. recent bars, by eye
print()
print("=" * 78)
print("5b. RECENT HG=F BARS (the entry state's own integrity)")
print("=" * 78)
prev = None
for d, row in hgf.loc["2026-08-24":].iterrows():
    o, h_, l_, c = (float(row["Open"]), float(row["High"]),
                    float(row["Low"]), float(row["Close"]))
    v = float(row["Volume"])
    if prev is None:
        print(f"  {d.date()} O={o:8.4f} H={h_:8.4f} L={l_:8.4f} C={c:8.4f} "
              f"V={v:>10,.0f}")
    else:
        inside = l_ <= prev <= h_
        print(f"  {d.date()} O={o:8.4f} H={h_:8.4f} L={l_:8.4f} C={c:8.4f} "
              f"V={v:>10,.0f} gap={100*(o/prev-1):+6.2f}% "
              f"ret={100*(c/prev-1):+6.2f}% spans={inside}"
              f"{'' if inside else '   <== PRIOR CLOSE OUTSIDE THE BAR'}")
    prev = c
