"""C2 round 1 — the index into the Labor Day closure, and the post-Labor-Day short.

Both sides are calendar-known in advance, so both are stated in TRADEABLE form:
an order placed MOC at the close k sessions from the Labor Day eve, held h
sessions. k=0 is "long into the closure" (the pitch). k=+1 is the first
post-Labor-Day close (the folk-claim short).

Blockers: placebo ladder over k; the arbitraged-fossil split pre-2013 /
2013-2017 / 2018+; the broader pre-holiday family so the Labor Day gate can be
attributed (run it WITHOUT the gate); the calendar-month control for the
"September weakness starts after Labor Day" claim; midterm cross; per-year
histogram; cost.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

pd.set_option("display.width", 230)

COST = {"SPY": 2.0, "IWM": 3.0}
prices = load_prices(["SPY", "IWM", "QQQ", "^GSPC", "^RUT"])
cal = prices["SPY"].index
px = pd.DataFrame({t: prices[t]["Close"].reindex(cal) for t in prices})
all_dates = px.index
posmap = pd.Series(range(len(all_dates)), index=all_dates)

gap_days = pd.Series(np.append((all_dates[1:] - all_dates[:-1]).days, np.nan),
                     index=all_dates)
closure = gap_days - 1.0

eves3 = all_dates[(closure >= 3).values]
labor = pd.DatetimeIndex([d for d in eves3
                          if all_dates[posmap[d] + 1].month == 9
                          and all_dates[posmap[d] + 1].day <= 8])
other3 = eves3.difference(labor)
plain = all_dates[(closure == 0).values]
print(f"Labor Day eves n={len(labor)}  {labor[0].date()} .. {labor[-1].date()}")
print(f"other >=3-day closure eves n={len(other3)}; plain sessions n={len(plain)}")


def seg(tkr, dates, a, b, sign=1.0):
    c = px[tkr].values
    out, keep = [], []
    for d in dates:
        p = posmap.get(d)
        if p is None or p + a < 0 or p + b >= len(all_dates):
            continue
        x, y = c[p + a], c[p + b]
        if not (np.isfinite(x) and np.isfinite(y)) or x == 0:
            continue
        out.append(sign * (y / x - 1.0))
        keep.append(d)
    return pd.DatetimeIndex(keep), np.array(out)


def line(tag, dates, vals, cost_bps=None):
    if len(vals) == 0:
        print(f"  {tag:46s} n=0")
        return
    w = int((vals > 0).sum())
    s = summarize(vals)
    extra = ""
    if cost_bps:
        extra = f" | {100*vals.mean()*100:+.1f}bps = {100*vals.mean()*100/cost_bps:+.1f}x cost"
    print(f"  {tag:46s} n={len(vals):4d} mean {s['mean_pct']:+7.3f}% "
          f"med {s['median_pct']:+7.3f}% hit {s['hit']:5.1f}% t {s['t']:+6.2f} "
          f"rec {w}-{len(vals)-w} signp {sign_test(w, len(vals)):.4f} "
          f"worst {s['worst_pct']:+7.2f}%{extra}")


# ==================================================================== side A
print("\n" + "=" * 104)
print("SIDE A — LONG into the Labor Day closure. Entry MOC on the eve (k=0), held h sessions.")
print("=" * 104)
for tkr in ("SPY", "IWM"):
    print(f"\n--- {tkr} ---")
    for h in (1, 2, 3, 5):
        d, v = seg(tkr, labor, 0, h)
        line(f"h={h} LABOR DAY eve -> +h (the pitch)", d, v, COST[tkr])
        dg, vg = seg(tkr, labor, 0, 1)
        do, vo = seg(tkr, other3, 0, h)
        dp, vp = seg(tkr, plain, 0, h)
        line(f"h={h}   gate off: any >=3d closure eve", do, vo)
        line(f"h={h}   CTRL any plain session", dp, vp)
    print("   pure closure gap (eve close -> first post close):")
    d, v = seg(tkr, labor, 0, 1)
    line("     Labor Day gap", d, v)
    d2, v2 = seg(tkr, other3, 0, 1)
    line("     other 3-day closure gaps", d2, v2)
    d3, v3 = seg(tkr, all_dates[(closure == 2).values], 0, 1)
    line("     ordinary weekend gaps", d3, v3)
    d4, v4 = seg(tkr, plain, 0, 1)
    line("     plain overnight", d4, v4)

# ==================================================================== ladder
print("\n" + "=" * 104)
print("BLOCKER 1 — placebo ladder over ENTRY offset k from the Labor Day eve")
print("=" * 104)
for tkr in ("SPY", "IWM"):
    for h in (1, 3):
        rows = []
        for k in range(-8, 9):
            d, v = seg(tkr, labor, k, k + h)
            if len(v) < 20:
                continue
            s = summarize(v)
            rows.append({"k": k, "n": len(v), "mean_pct": s["mean_pct"],
                         "hit": s["hit"], "t": s["t"]})
        L = pd.DataFrame(rows).sort_values("mean_pct", ascending=False)
        L["rank"] = range(1, len(L) + 1)
        r0 = int(L.loc[L.k == 0, "rank"].iloc[0])
        print(f"\n {tkr} h={h}: long. k=0 (pitched entry) rank {r0} of {len(L)}")
        print(L.round(3).to_string(index=False))

# ==================================================================== eras
print("\n" + "=" * 104)
print("BLOCKER 4 — arbitraged-fossil split: pre-2013 / 2013-2017 / 2018+")
print("=" * 104)
for tkr in ("SPY", "IWM"):
    print(f"\n--- {tkr} long from the Labor Day eve ---")
    for h in (1, 3):
        d, v = seg(tkr, labor, 0, h)
        for lbl, m in [("pre-2013", pd.DatetimeIndex(d).year < 2013),
                       ("2013-2017", (pd.DatetimeIndex(d).year >= 2013) &
                        (pd.DatetimeIndex(d).year < 2018)),
                       ("2018+", pd.DatetimeIndex(d).year >= 2018)]:
            line(f"h={h} {lbl}", d[m], v[m])
        print("      per-year:", ", ".join(
            f"{x.year}:{100*y:+.2f}" for x, y in zip(d, v)))

# ============================================================== side B short
print("\n" + "=" * 104)
print("SIDE B — SHORT the index from the FIRST post-Labor-Day close (entry k=+1), h sessions.")
print("   (this is anchor=eve with the lab's lag=1, so it is the convention-clean form)")
print("=" * 104)
for tkr in ("SPY", "IWM"):
    print(f"\n--- short {tkr} ---")
    for h in (3, 5, 7, 10):
        d, v = seg(tkr, labor, 1, 1 + h, sign=-1.0)
        line(f"h={h} SHORT from first post-LD close", d, v, COST[tkr])
        do, vo = seg(tkr, other3, 1, 1 + h, sign=-1.0)
        line(f"h={h}   gate off: any >=3d closure", do, vo)
        dp, vp = seg(tkr, plain, 1, 1 + h, sign=-1.0)
        line(f"h={h}   CTRL short any plain session", dp, vp)
    print("   per-year, h=10:")
    d, v = seg(tkr, labor, 1, 11, sign=-1.0)
    print("     ", ", ".join(f"{x.year}:{100*y:+.2f}" for x, y in zip(d, v)))
    for lbl, m in [("pre-2013", pd.DatetimeIndex(d).year < 2013),
                   ("2013-2017", (pd.DatetimeIndex(d).year >= 2013) &
                    (pd.DatetimeIndex(d).year < 2018)),
                   ("2018+", pd.DatetimeIndex(d).year >= 2018)]:
        line(f"   h=10 {lbl}", d[m], v[m])

# ================================================== calendar-month attribution
print("\n" + "=" * 104)
print("BLOCKER: is 'September weakness AFTER Labor Day' the Labor Day boundary, or just September?")
print("=" * 104)
for tkr in ("SPY", "IWM"):
    r10 = px[tkr].shift(-10) / px[tkr] - 1.0
    sep = px.index.month == 9
    # sessions in Sept before vs after the Labor Day boundary of that year
    lab_by_year = {d.year: posmap[d] for d in labor}
    before, after = [], []
    for i, d in enumerate(all_dates):
        if d.month != 9 or d.year not in lab_by_year:
            continue
        (after if i > lab_by_year[d.year] else before).append(d)
    for lbl, dd in [("Sept sessions BEFORE Labor Day", pd.DatetimeIndex(before)),
                    ("Sept sessions AFTER Labor Day", pd.DatetimeIndex(after))]:
        v = r10.reindex(dd).dropna().values
        line(f"{tkr} fwd10 {lbl}", dd, -v)      # short side
    v = r10[px.index.month != 9].dropna().values
    line(f"{tkr} fwd10 SHORT all non-Sept days", all_dates, -v)
    v = r10.dropna().values
    line(f"{tkr} fwd10 SHORT all days", all_dates, -v)

# ==================================================================== midterm
print("\n" + "=" * 104)
print("BLOCKER 10 — midterm cross (2026 is midterm)")
print("=" * 104)
for tkr in ("SPY", "IWM"):
    for lab_k, hh, sgn, nm in [(0, 3, 1.0, "LONG from eve h=3"),
                               (1, 10, -1.0, "SHORT from first post-LD close h=10")]:
        d, v = seg(tkr, labor, lab_k, lab_k + hh, sign=sgn)
        yr = pd.DatetimeIndex(d).year
        print(f"\n {tkr} {nm}")
        line("   midterm", d[(yr % 4) == 2], v[(yr % 4) == 2])
        line("   non-midterm", d[(yr % 4) != 2], v[(yr % 4) != 2])
