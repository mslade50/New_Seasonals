"""D5 round 2 -- LONG TLT into the CPI print, UNGATED parent cell.

Provenance: c3_movefloor_cpi.py killed the GATED versions (MOVE floor / TLT
52w floor) and in doing so reported the ungated parent at "+0.193%, edge
+0.143pp, sign p 0.001, boot 0.021" for h=3. That number was measured with
anchor offset -2 and lag=1, i.e. ENTRY AT THE CLOSE OF THE SESSION BEFORE THE
PRINT (k=1). Today is Monday 2026-08-10, freshest bar Friday 2026-08-07, CPI
lands Wednesday 2026-08-12. A pitch composed this morning enters MOC TONIGHT,
which is k=2. So the reported cell is not the cell today can trade.

Everything here is re-derived from scratch. k is defined ONLY by the entry
session: entry at the close of (print - k), exit h sessions later.

  k=2  -> entry Mon 2026-08-10 close  (what today can execute)
  k=1  -> entry Tue 2026-08-11 close  (needs tomorrow's pitch, not today's)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

TICKERS = ["TLT", "IEF", "LQD", "HYG", "^TNX", "SPY"]
px = close_panel(TICKERS)
tl = px["TLT"].dropna()
idx = tl.index
c = tl.values
N = len(c)

ev = load_events()
CPI = ev[ev.event == "cpi"]["date"]


def print_positions(dates, lo_pad=8, hi_pad=14):
    """Index position of each print session, with padding so k/h stay valid."""
    out = []
    for x in dates:
        p = int(idx.searchsorted(x, "left"))
        if lo_pad <= p < N - hi_pad:
            out.append(p)
    return np.array(sorted(set(out)))


cpi_p = print_positions(CPI)
print("=" * 96)
print("SETUP")
print("=" * 96)
print(f"TLT bars {idx[0].date()} .. {idx[-1].date()}  N={N}")
print(f"CPI print sessions usable: {len(cpi_p)}  "
      f"{idx[cpi_p[0]].date()} .. {idx[cpi_p[-1]].date()}")
# sanity: is the CPI date itself a trading day?
bad = sum(1 for x in CPI if int(idx.searchsorted(x, "left")) < N
          and idx[int(idx.searchsorted(x, "left"))] != x)
print(f"CPI dates that are NOT trading days (mapped forward): {bad}")


def cell(pos, k, h, series=None):
    """Return (values, anchor dates) for entry at close p-k, exit p-k+h."""
    s = c if series is None else series
    v, d = [], []
    for p in pos:
        e0, e1 = p - k, p - k + h
        if e0 < 0 or e1 >= len(s):
            continue
        v.append(s[e1] / s[e0] - 1.0)
        d.append(idx[p])
    return np.array(v), pd.DatetimeIndex(d)


def base_h(h, series=None):
    s = c if series is None else series
    r = s[h:] / s[:-h] - 1.0
    return r


def line(v, lbl, ctrl):
    if len(v) == 0:
        return {"cell": lbl, "N": 0}
    w = int((v > 0).sum())
    sd = v.std(ddof=1) if len(v) > 1 else np.nan
    return {"cell": lbl, "N": len(v), "mean_pct": round(100 * v.mean(), 3),
            "edge_pp": round(100 * (v.mean() - ctrl), 3),
            "hit": round(100 * w / len(v), 1),
            "t": round(v.mean() / (sd / np.sqrt(len(v))), 2) if sd > 0 else np.nan,
            "sign_p": round(sign_test(w, len(v)), 4),
            "boot": round(bootstrap_p_le0(v), 3)}


# ---------------------------------------------------------------- A. k x h
print("\n" + "=" * 96)
print("A. k x h GRID, long TLT, entry at close of (print - k), exit +h sessions")
print("   ctrl = TLT's own unconditional h-session drift, full history")
print("=" * 96)
rows = []
for k in range(1, 6):
    for h in (1, 2, 3, 5, 10):
        v, d = cell(cpi_p, k, h)
        r = line(v, f"k={k} h={h}", base_h(h).mean())
        r["exit_rel_print"] = f"p{-k + h:+d}"
        rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
for h in (1, 2, 3, 5, 10):
    print(f"  ctrl all-days h={h}: {100*base_h(h).mean():+.4f}%")

# ---------------------------------------------------------------- B. battery
print("\n" + "=" * 96)
print("B. FULL BATTERY at the two cells that matter")
print("=" * 96)
for k, h in [(2, 3), (1, 3)]:
    m = pd.Series(False, index=idx)
    for p in cpi_p:
        m.iloc[p - k - 1] = True   # mask date; battery enters at lag=1 -> p-k
    battery(px.loc[idx], m, [("TLT", 1.0)], h=h,
            title=f"LONG TLT into CPI, entry close(print-{k}), h={h}"
                  f"  [{'TODAY-EXECUTABLE' if k == 2 else 'reported by c3'}]",
            cost_bps=2.0, lag=1, min_gap=10, event_kinds=("ppi",))

# ------------------------------------------------- C. session decomposition
print("\n" + "=" * 96)
print("C. SESSION-BY-SESSION decomposition (each column = ONE session's TLT move)")
print("   is this a pre-print duration bid, the print session, or post drift?")
print("=" * 96)
daily = c[1:] / c[:-1] - 1.0
ctrl_d = daily.mean()
rows = []
for off, nm in [(-4, "p-5 -> p-4"), (-3, "p-4 -> p-3"), (-2, "p-3 -> p-2"),
                (-1, "p-2 -> p-1"), (0, "p-1 -> p  (THE PRINT)"),
                (1, "p   -> p+1"), (2, "p+1 -> p+2"), (3, "p+2 -> p+3")]:
    v = np.array([c[p + off] / c[p + off - 1] - 1.0 for p in cpi_p])
    d = pd.DatetimeIndex([idx[p] for p in cpi_p])
    mid = (d.year % 4) == 2
    rows.append({"session": nm, "N": len(v),
                 "all_pct": round(100 * v.mean(), 4),
                 "pre2018": round(100 * v[d.year < 2018].mean(), 4),
                 "y2018p": round(100 * v[d.year >= 2018].mean(), 4),
                 "midterm": round(100 * v[mid].mean(), 4),
                 "nonmid": round(100 * v[~mid].mean(), 4),
                 "hit_all": round(100 * (v > 0).mean(), 1),
                 "hit_18p": round(100 * (v[d.year >= 2018] > 0).mean(), 1),
                 "sign_p": round(sign_test(int((v > 0).sum()), len(v)), 4)})
print(pd.DataFrame(rows).to_string(index=False))
print(f"  ctrl: TLT unconditional daily mean {100*ctrl_d:+.4f}%")
print("\n  Cumulative build of the k=2,h=3 window (entry p-2 close -> exit p+1 close):")
for off, nm in [(-1, "p-2 -> p-1"), (0, "p-1 -> p (PRINT)"), (1, "p -> p+1")]:
    v = np.array([c[p + off] / c[p + off - 1] - 1.0 for p in cpi_p])
    print(f"    {nm:24s} {100*v.mean():+.4f}%  (ctrl {100*ctrl_d:+.4f}%, "
          f"excess {100*(v.mean()-ctrl_d):+.4f}pp)")
print("  Cumulative build of the k=1,h=3 window (entry p-1 close -> exit p+2 close):")
for off, nm in [(0, "p-1 -> p (PRINT)"), (1, "p -> p+1"), (2, "p+1 -> p+2")]:
    v = np.array([c[p + off] / c[p + off - 1] - 1.0 for p in cpi_p])
    print(f"    {nm:24s} {100*v.mean():+.4f}%  (ctrl {100*ctrl_d:+.4f}%, "
          f"excess {100*(v.mean()-ctrl_d):+.4f}pp)")

# ---------------------------------------------------------------- D. era
print("\n" + "=" * 96)
print("D. ERA. year histogram, drop-best-year, 2020-22 fence test  [k=2, h=3]")
print("=" * 96)
K, H = 2, 3
v, d = cell(cpi_p, K, H)
ctrl = base_h(H).mean()
byyr = pd.DataFrame({"y": d.year, "v": v}).groupby("y")["v"].agg(
    ["count", "mean", "sum"])
byyr["mean_pct"] = (100 * byyr["mean"]).round(3)
byyr["sum_pp"] = (100 * byyr["sum"]).round(2)
byyr["pos"] = pd.DataFrame({"y": d.year, "v": v}).groupby("y")["v"].apply(
    lambda s: int((s > 0).sum()))
print(byyr[["count", "mean_pct", "sum_pp", "pos"]].to_string())
pos_years = int((byyr["mean"] > 0).sum())
print(f"\n  positive years: {pos_years}/{len(byyr)}")
best = byyr["sum"].idxmax()
worst = byyr["sum"].idxmin()
print(f"  best year {best} contributes {100*byyr.loc[best,'sum']:+.2f}pp of "
      f"{100*v.sum():+.2f}pp total ({100*byyr.loc[best,'sum']/v.sum():.0f}%)")
rows = [line(v, "ALL", ctrl),
        line(v[d.year != best], f"drop best year ({best})", ctrl),
        line(v[d.year != worst], f"drop worst year ({worst})", ctrl),
        line(v[d.year < 2018], "pre-2018", ctrl),
        line(v[d.year >= 2018], "2018+", ctrl),
        line(v[(d.year >= 2020) & (d.year <= 2022)], "2020-2022 ONLY", ctrl),
        line(v[~((d.year >= 2020) & (d.year <= 2022))], "ex 2020-2022", ctrl),
        line(v[(d.year >= 2018) & ~((d.year >= 2020) & (d.year <= 2022))],
             "2018+ ex 2020-2022", ctrl),
        line(v[d.year >= 2023], "2023+", ctrl)]
print()
print(pd.DataFrame(rows).to_string(index=False))
print(f"\n  {cluster_note(d, v, k=2)}")
top5 = np.argsort(-np.abs(v))[:5]
print("  top5 |moves|: " + ", ".join(
    f"{d[i].date()} {100*v[i]:+.2f}%" for i in top5))
print(f"  top5 share of total: {100*v[top5].sum()/v.sum():.0f}%")

# ---------------------------------------------------------------- E. midterm
print("\n" + "=" * 96)
print("E. MIDTERM split (2026 is a midterm year)")
print("=" * 96)
for K2, H2 in [(2, 3), (1, 3), (2, 5), (2, 1)]:
    v2, d2 = cell(cpi_p, K2, H2)
    ctrl2 = base_h(H2).mean()
    mid = (d2.year % 4) == 2
    rows = [line(v2, f"k={K2} h={H2} ALL", ctrl2),
            line(v2[mid], f"k={K2} h={H2} MIDTERM", ctrl2),
            line(v2[~mid], f"k={K2} h={H2} non-midterm", ctrl2)]
    print(pd.DataFrame(rows).to_string(index=False))
    print()

# ---------------------------------------------------------------- F. placebo
print("=" * 96)
print("F. PLACEBO ANCHORS -- is this 'duration into any macro print'?  [k=2,h=3]")
print("=" * 96)
rows = []
for kind in ["cpi", "ppi", "nfp", "fomc_decision", "fomc_minutes", "opex",
             "vix_expiry"]:
    pos = print_positions(ev[ev.event == kind]["date"])
    v2, d2 = cell(pos, K, H)
    r = line(v2, kind, ctrl)
    mid = (d2.year % 4) == 2
    r["mid_mean"] = round(100 * v2[mid].mean(), 3) if mid.sum() else np.nan
    r["y18p"] = round(100 * v2[d2.year >= 2018].mean(), 3)
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
print(f"  ctrl all-days h={H}: {100*ctrl:+.4f}%")

print("\n  Same placebo table at k=1, h=3 (the reported cell):")
rows = []
for kind in ["cpi", "ppi", "nfp", "fomc_decision", "opex"]:
    pos = print_positions(ev[ev.event == kind]["date"])
    v2, d2 = cell(pos, 1, 3)
    rows.append(line(v2, kind, ctrl))
print(pd.DataFrame(rows).to_string(index=False))

# ---------------------------------------------------- G. mid-month artifact
print("\n" + "=" * 96)
print("G. MID-MONTH CALENDAR ARTIFACT? CPI lands mid-month.")
print("=" * 96)
dom = np.array([x.day for x in idx])
cpi_entry_pos = set(int(p - K) for p in cpi_p)
cpi_window_pos = set()
for p in cpi_p:
    for j in range(p - K, p - K + H + 1):
        cpi_window_pos.add(int(j))
print("  CPI print day-of-month distribution: "
      f"min {min(idx[p].day for p in cpi_p)} med "
      f"{int(np.median([idx[p].day for p in cpi_p]))} max "
      f"{max(idx[p].day for p in cpi_p)}")
rh = base_h(H)
rows = []
allpos = np.arange(len(rh))
for lo, hi, nm in [(1, 7, "dom 1-7"), (8, 13, "dom 8-13"), (14, 20, "dom 14-20"),
                   (21, 31, "dom 21-31")]:
    m = (dom[:len(rh)] >= lo) & (dom[:len(rh)] <= hi)
    rows.append(line(rh[m], nm + " (all entries)", ctrl))
# entries in the same dom band as CPI entries, excluding the CPI window entirely
entry_doms = sorted(set(idx[p - K].day for p in cpi_p))
lo_e, hi_e = np.percentile(entry_doms, [10, 90])
band = (dom[:len(rh)] >= lo_e) & (dom[:len(rh)] <= hi_e)
notcpi = np.array([i not in cpi_window_pos for i in allpos])
rows.append(line(rh[band], f"CPI-entry dom band [{lo_e:.0f},{hi_e:.0f}] all", ctrl))
rows.append(line(rh[band & notcpi],
                 f"SAME dom band, CPI window REMOVED", ctrl))
rows.append(line(rh[np.array([i in cpi_entry_pos for i in allpos])],
                 "CPI entries only (day-level check)", ctrl))
print(pd.DataFrame(rows).to_string(index=False))

# ------------------------------------------------- H. cross-instrument
print("\n" + "=" * 96)
print("H. CROSS-INSTRUMENT coherence  [k=2, h=3]")
print("=" * 96)
rows = []
for t in ["TLT", "IEF", "LQD", "HYG", "SPY"]:
    s = px[t].dropna()
    # remap positions onto this ticker's index
    pos_t = []
    for p in cpi_p:
        q = int(s.index.searchsorted(idx[p], "left"))
        if 2 <= q < len(s) - 6 and s.index[q] == idx[p]:
            pos_t.append(q)
    sv = s.values
    v2 = np.array([sv[q - K + H] / sv[q - K] - 1.0 for q in pos_t])
    d2 = pd.DatetimeIndex([s.index[q] for q in pos_t])
    b = sv[H:] / sv[:-H] - 1.0
    r = line(v2, t, b.mean())
    r["ctrl_pct"] = round(100 * b.mean(), 3)
    r["y18p"] = round(100 * v2[d2.year >= 2018].mean(), 3)
    rows.append(r)
# ^TNX: yield. long duration => yield should FALL
s = px["^TNX"].dropna()
pos_t = [int(s.index.searchsorted(idx[p], "left")) for p in cpi_p]
pos_t = [q for q in pos_t if 2 <= q < len(s) - 6]
sv = s.values
v2 = np.array([sv[q - K + H] / sv[q - K] - 1.0 for q in pos_t])
b = sv[H:] / sv[:-H] - 1.0
r = line(v2, "^TNX yield chg (want NEGATIVE)", b.mean())
r["ctrl_pct"] = round(100 * b.mean(), 3)
rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))

# ------------------------------------------------- I. today's state
print("\n" + "=" * 96)
print("I. DOES THE TRIGGER POPULATION LOOK LIKE TODAY?")
print("=" * 96)
dist_lo = (tl / tl.rolling(252).min() - 1.0)
dist_hi = (tl / tl.rolling(252).max() - 1.0)
r21 = pct_rank(tl, 21)
r63 = pct_rank(tl, 63)
z10 = zscore(tl, 10)
today = {"dist_52w_low_pct": 100 * dist_lo.iloc[-1],
         "dist_52w_high_pct": 100 * dist_hi.iloc[-1],
         "rank21": r21.iloc[-1], "rank63": r63.iloc[-1],
         "z10": z10.iloc[-1]}
print("  TODAY (Fri 2026-08-07 close, the anchor bar):")
for k2, val in today.items():
    print(f"    {k2:20s} {val:.2f}")
ent = pd.DatetimeIndex([idx[p - K] for p in cpi_p])
pop = pd.DataFrame({"dist_52w_low_pct": 100 * dist_lo.reindex(ent).values,
                    "rank21": r21.reindex(ent).values,
                    "rank63": r63.reindex(ent).values,
                    "z10": z10.reindex(ent).values})
print("\n  Where today sits in the CPI-entry population (percentile of the "
      "trigger days):")
for col, tv in [("dist_52w_low_pct", today["dist_52w_low_pct"]),
                ("rank21", today["rank21"]), ("rank63", today["rank63"]),
                ("z10", today["z10"])]:
    s2 = pop[col].dropna()
    print(f"    {col:20s} today {tv:7.2f}  -> {100*(s2 <= tv).mean():5.1f} "
          f"pctile of trigger days  (pop median {s2.median():.2f})")

print("\n  Conditional sub-cells that DESCRIBE TODAY (not gates I am imposing "
      "-- the question is whether today's tape sits where the parent works):")
sub = {
    "TLT within 3% of 52w low": (dist_lo.reindex(ent).values <= 0.03),
    "TLT within 5% of 52w low": (dist_lo.reindex(ent).values <= 0.05),
    "TLT rank21 < 35": (r21.reindex(ent).values < 35),
    "TLT rank63 < 35": (r63.reindex(ent).values < 35),
    "TLT >10% below 52w high": (dist_hi.reindex(ent).values <= -0.10),
}
rows = [line(v, "ALL CPI entries", ctrl)]
for lbl, m in sub.items():
    m = np.asarray(m, dtype=bool)
    rows.append(line(v[m], lbl, ctrl))
    mid = ((d.year % 4) == 2) & m
    rows.append(line(v[mid], f"  ^ AND midterm", ctrl))
print(pd.DataFrame(rows).to_string(index=False))

# ------------------------------------------------- J. cost
print("\n" + "=" * 96)
print("J. COST")
print("=" * 96)
for K2, H2 in [(2, 3), (1, 3), (2, 1), (2, 5)]:
    v2, _ = cell(cpi_p, K2, H2)
    e = 100 * 100 * (v2.mean() - base_h(H2).mean())
    m = 100 * 100 * v2.mean()
    print(f"  k={K2} h={H2}: raw {m:+.1f} bps, edge-over-drift {e:+.1f} bps, "
          f"TLT round trip ~2.5 bps -> {m/2.5:.0f}x raw / {e/2.5:.0f}x edge")
