"""Drill 01 (2026-09-09): PPI landing the session immediately BEFORE CPI.

This week PPI prints 2026-09-10 (Thu) and CPI 2026-09-11 (Fri), i.e. PPI is
exactly 1 trading day BEFORE CPI. The normal BLS ordering is CPI first, PPI a
day or two later. Question: how rare is the inversion, and does the PPI session
behave differently when it carries the tape into a next-day CPI?

Conventions: fwd_ret is lag=0 close-to-close (this is CONTEXT, not an entry).
Anchor for "the PPI session's own move" is the session BEFORE the PPI, h=1.
^TNX and ^VIX moves are percent changes in the INDEX LEVEL (yield / vol), not
price returns: +x% on ^TNX means the 10y yield rose x%, i.e. bonds sold off.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SUBJECTS = ["^GSPC", "^TNX", "IEF", "^VIX"]
ERA_CUT = "2018-01-01"

px = load_prices(SUBJECTS)
CAL = px["^GSPC"].index  # master equity trading calendar (2000-01-03 .. today)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def cell(dates, vals, label):
    """summarize + W-L record + exact sign test. FRACTIONS in."""
    v = np.asarray(vals, dtype=float)
    v = v[~np.isnan(v)]
    s = summarize(v, label)
    if s["n"] == 0:
        return s
    w = int((v > 0).sum())
    losses = int((v < 0).sum())
    flat = int((v == 0).sum())
    s["record"] = f"{w}-{losses}" + (f" ({flat} flat)" if flat else "")
    s["sign_p"] = sign_test(w, len(v))
    return s


def report(dates, vals, label, show_cluster=True):
    d = pd.DatetimeIndex(dates)
    v = np.asarray(vals, dtype=float)
    s = cell(d, v, label)
    if s["n"] == 0:
        print(f"  {label}: EMPTY")
        return s
    small = "   *** n<15, SMALL ***" if s["n"] < 15 else ""
    print(f"  {label}: n={s['n']}  mean={s['mean_pct']:+.4f}%  "
          f"median={s['median_pct']:+.4f}%  hit={s['hit']:.1f}%  "
          f"t={s['t']:+.3f}  record {s['record']}  sign_p={s['sign_p']:.4f}  "
          f"sd={s['sd_pct']:.3f}%  worst={s['worst_pct']:+.2f}%  "
          f"best={s['best_pct']:+.2f}%{small}")
    for e in era_split(d, v, ERA_CUT):
        if e["n"]:
            print(f"      era {e['label']}: n={e['n']} mean={e['mean_pct']:+.4f}% "
                  f"hit={e['hit']:.1f}% t={e['t']:+.3f}")
        else:
            print(f"      era {e['label']}: n=0")
    if show_cluster:
        print(f"      cluster: {cluster_note(d, v)}")
    return s


def session_move(tkr, event_dates, h=1, offset=-1):
    """(event_dates_kept, anchor_dates, values) for the h-session close-to-close
    move measured from `offset` sessions relative to each event date, on the
    instrument's OWN dropna'd calendar."""
    s = px[tkr]["Close"].dropna()
    idx = s.index
    pos, kept = anchor_positions(idx, event_dates, offset)
    r = fwd_ret(s, h)
    vals = r.iloc[pos].to_numpy()
    anch = pd.DatetimeIndex([idx[p] for p in pos])
    m = ~np.isnan(vals)
    return kept[m], anch[m], vals[m]


def welch(a, b):
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    b = np.asarray(b, float); b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return (a.mean() - b.mean()) / se if se > 0 else np.nan


# ---------------------------------------------------------------------------
# 1. how rare is "PPI exactly 1 td before CPI"?
# ---------------------------------------------------------------------------
ev = load_events(["cpi", "ppi"])
cpi_dates = pd.DatetimeIndex(ev.loc[ev["event"] == "cpi", "date"])
ppi_dates = pd.DatetimeIndex(ev.loc[ev["event"] == "ppi", "date"])

print("=" * 78)
print("DRILL 01 -- PPI landing 1 trading day BEFORE CPI")
print("=" * 78)
print(f"macro_events.csv: {len(cpi_dates)} CPI dates "
      f"({cpi_dates.min().date()}..{cpi_dates.max().date()}), "
      f"{len(ppi_dates)} PPI dates "
      f"({ppi_dates.min().date()}..{ppi_dates.max().date()})")
print(f"calendar = ^GSPC sessions {CAL[0].date()}..{CAL[-1].date()} "
      f"(n={len(CAL)}).  NOTE: the events file begins 2000-01-05, so "
      f"'since 1999' is in practice 2000+.")

cpi_pos, cpi_kept = anchor_positions(CAL, cpi_dates, 0)
ppi_pos, ppi_kept = anchor_positions(CAL, ppi_dates, 0)
cpi_pos = np.asarray(cpi_pos)
print(f"\nmapped onto the calendar: {len(cpi_kept)} CPI, {len(ppi_kept)} PPI "
      f"(events outside the price index are dropped -- incl. this week's "
      f"2026-09-10 PPI / 2026-09-11 CPI, which are the live case)")

rows = []
for p, d in zip(ppi_pos, ppi_kept):
    diffs = p - cpi_pos                      # >0 : PPI AFTER cpi ; <0 : BEFORE
    j = int(np.argmin(np.abs(diffs)))        # ties -> earliest CPI
    rows.append({"ppi": d, "cpi": cpi_kept[j], "offset_td": int(diffs[j]),
                 "cal_days": int((d - cpi_kept[j]).days)})
gap = pd.DataFrame(rows)

print("\n--- 1. distribution of PPI position relative to its NEAREST CPI ---")
print("    offset_td = trading days from CPI to PPI. NEGATIVE = PPI BEFORE CPI.")
vc = gap["offset_td"].value_counts().sort_index()
tot = len(gap)
for off, n in vc.items():
    if off < 0:
        lbl = f"PPI {abs(off)} td BEFORE CPI"
    elif off == 0:
        lbl = "PPI SAME DAY as CPI"
    else:
        lbl = f"PPI {off} td AFTER CPI"
    print(f"    offset {off:+3d}  {lbl:<26s}  n={n:4d}   {100*n/tot:5.1f}%")
print(f"    total {tot}")

INV = gap["offset_td"] == -1
n_inv = int(INV.sum())
print(f"\n    >>> 'PPI exactly 1 td BEFORE CPI' : n={n_inv} of {tot} PPI prints "
      f"= {100*n_inv/tot:.1f}%")
print(f"    >>> PPI at or before CPI (offset <= 0): "
      f"{int((gap['offset_td'] <= 0).sum())} = "
      f"{100*(gap['offset_td'] <= 0).mean():.1f}%")
print(f"    >>> PPI after CPI (offset >= 1, the normal ordering): "
      f"{int((gap['offset_td'] >= 1).sum())} = "
      f"{100*(gap['offset_td'] >= 1).mean():.1f}%")

inv_dates = pd.DatetimeIndex(gap.loc[INV, "ppi"])
print("\n    most recent 10 inverted-ordering PPI dates (ppi -> its cpi):")
for _, r in gap.loc[INV].tail(10).iterrows():
    print(f"      PPI {r['ppi'].date()} ({r['ppi'].day_name()[:3]})  ->  "
          f"CPI {r['cpi'].date()} ({r['cpi'].day_name()[:3]})   "
          f"cal gap {r['cal_days']}d")
print("\n    ALL inverted dates:",
      ", ".join(str(d.date()) for d in inv_dates))
print("    by year:", dict(pd.Series(inv_dates.year).value_counts().sort_index()))

norm_dates = pd.DatetimeIndex(gap.loc[~INV, "ppi"])

# --- 1b. the ordering is NOT era-stable. BLS re-sequenced the releases. ------
print("\n--- 1b. ordering by era (the full-history share above is misleading) ---")
gap["yr"] = pd.DatetimeIndex(gap["ppi"]).year
by_yr = gap.groupby("yr").agg(
    n_ppi=("offset_td", "size"),
    n_inv=("offset_td", lambda s: int((s == -1).sum())),
    n_before=("offset_td", lambda s: int((s < 0).sum())),
    n_after=("offset_td", lambda s: int((s > 0).sum())),
)
by_yr["inv_share_pct"] = (100 * by_yr["n_inv"] / by_yr["n_ppi"]).round(1)
print(by_yr.to_string())
for cut in (2013, 2016, 2019, 2020, 2021):
    sub = gap[gap["yr"] >= cut]
    n_i = int((sub["offset_td"] == -1).sum())
    n_a = int((sub["offset_td"] > 0).sum())
    print(f"    {cut}+ : {len(sub)} PPI prints, "
          f"1td-BEFORE-CPI {n_i} ({100*n_i/len(sub):.1f}%), "
          f"AFTER-CPI {n_a} ({100*n_a/len(sub):.1f}%)")
print("    -> the inversion is the MAJORITY ordering pre-2019 and RARE after; "
      "quote the era, not the pooled share.")

# ---------------------------------------------------------------------------
# 2 + 3. the PPI session's own move: inverted cell vs everything else
# ---------------------------------------------------------------------------
print("\n--- 2/3. the PPI SESSION's own close-to-close move "
      "(anchor = session before PPI, h=1, lag=0) ---")
print("    ^TNX / ^VIX numbers are pct changes in the INDEX LEVEL "
      "(+ = yield up / vol up).")

edge = {}
for tkr in SUBJECTS:
    print(f"\n  ##### {tkr} #####")
    e_i, a_i, v_i = session_move(tkr, inv_dates, 1, -1)
    e_n, a_n, v_n = session_move(tkr, norm_dates, 1, -1)
    s_i = report(e_i, v_i, f"INVERTED (PPI 1td before CPI)")
    s_n = report(e_n, v_n, f"NORMAL   (every other PPI)   ")
    if s_i["n"] and s_n["n"]:
        d_mean = s_i["mean_pct"] - s_n["mean_pct"]
        d_hit = s_i["hit"] - s_n["hit"]
        wt = welch(v_i, v_n)
        edge[tkr] = (d_mean, d_hit, wt)
        print(f"      EDGE inverted - normal: mean {d_mean:+.4f}pp   "
              f"hit {d_hit:+.1f}pp   welch t {wt:+.3f}")

print("\n  --- edge summary (inverted minus normal, PPI-session move) ---")
for tkr, (dm, dh, wt) in edge.items():
    print(f"    {tkr:<6s} mean {dm:+.4f}pp   hit {dh:+.1f}pp   welch t {wt:+.3f}")

# ---------------------------------------------------------------------------
# 4. h=2 (PPI session + CPI session) and the reversal count
# ---------------------------------------------------------------------------
print("\n--- 4. h=2 from the same anchor: PPI session + CPI session together ---")
for tkr in ["^GSPC", "^TNX"]:
    print(f"\n  ##### {tkr} #####")
    e2, a2, v2 = session_move(tkr, inv_dates, 2, -1)
    report(e2, v2, "INVERTED h=2 (PPI day + CPI day)")
    e2n, a2n, v2n = session_move(tkr, norm_dates, 2, -1)
    report(e2n, v2n, "NORMAL   h=2 (PPI day + next day)")

    # per-session decomposition on the inverted cell
    s = px[tkr]["Close"].dropna()
    idx = s.index
    pos, kept = anchor_positions(idx, inv_dates, -1)
    r1 = fwd_ret(s, 1)
    d1, d2, pairs = [], [], []
    for p, d in zip(pos, kept):
        if p + 1 >= len(idx):
            continue
        x1 = r1.iloc[p]
        x2 = r1.iloc[p + 1]
        if np.isnan(x1) or np.isnan(x2):
            continue
        d1.append(x1); d2.append(x2); pairs.append(d)
    d1 = np.asarray(d1); d2 = np.asarray(d2)
    rev = int(((d1 > 0) & (d2 < 0)).sum() + ((d1 < 0) & (d2 > 0)).sum())
    same = int(((d1 > 0) & (d2 > 0)).sum() + ((d1 < 0) & (d2 < 0)).sum())
    zero = len(d1) - rev - same
    print(f"      session-1 (PPI day) mean {100*d1.mean():+.4f}%  "
          f"session-2 (CPI day) mean {100*d2.mean():+.4f}%")
    print(f"      REVERSAL: session 2 opposite sign to session 1 on "
          f"{rev} of {len(d1)} ({100*rev/len(d1):.1f}%); same sign {same}; "
          f"exact-zero cases {zero}")
    print(f"      sign test on reversal count: sign_p(rev>={rev} of {len(d1)}) "
          f"= {sign_test(rev, len(d1)):.4f}")
    if len(d1) > 2:
        c = np.corrcoef(d1, d2)[0, 1]
        print(f"      corr(session1, session2) = {c:+.3f}")
    mod = [(dd, x1, x2) for dd, x1, x2 in zip(pairs, d1, d2)
           if dd >= pd.Timestamp("2019-01-01")]
    print(f"      per-episode, MODERN ERA 2019+ only (PPI date | ppi-day % | "
          f"cpi-day %), n={len(mod)}:")
    for dd, x1, x2 in mod:
        flag = "  REV" if x1 * x2 < 0 else ""
        print(f"        {dd.date()}  {100*x1:+7.3f}  {100*x2:+7.3f}{flag}")
    if mod:
        m1 = np.array([x for _, x, _ in mod]); m2 = np.array([y for _, _, y in mod])
        mr = int((m1 * m2 < 0).sum())
        print(f"        2019+ session-1 mean {100*m1.mean():+.4f}%  "
              f"session-2 mean {100*m2.mean():+.4f}%  "
              f"reversals {mr}/{len(mod)}")

# ---------------------------------------------------------------------------
# 5. September / midterm subsets of the inverted cell
# ---------------------------------------------------------------------------
print("\n--- 5. subsets of the INVERTED cell (labelled small where n<15) ---")
sep = inv_dates[inv_dates.month == 9]
mid = inv_dates[inv_dates.year % 4 == 2]
print(f"    September inverted PPIs: n={len(sep)} -> "
      f"{[str(d.date()) for d in sep]}")
print(f"    midterm-year inverted PPIs (year%4==2): n={len(mid)} -> "
      f"{[str(d.date()) for d in mid]}")
modern = inv_dates[inv_dates >= pd.Timestamp("2019-01-01")]
print(f"    MODERN-ERA inverted PPIs (2019+, the current release regime): "
      f"n={len(modern)} -> {[str(d.date()) for d in modern]}")
for lbl, sub in [("SEPTEMBER", sep), ("MIDTERM", mid), ("MODERN 2019+", modern)]:
    if len(sub) < 4:
        print(f"\n  {lbl}: n={len(sub)} < 4, NOT REPORTED per spec")
        continue
    print(f"\n  ##### {lbl} subset of the inverted cell (n={len(sub)}) #####")
    for tkr in SUBJECTS:
        e, a, v = session_move(tkr, sub, 1, -1)
        report(e, v, f"{tkr} PPI-session h=1", show_cluster=False)

print("\ndone.")
