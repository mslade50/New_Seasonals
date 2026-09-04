"""C11 round 1 - the index over the first week of SEPTEMBER in a MIDTERM year.
Today is trading day 2 of September 2026 and 2026 is a midterm year (2026%4==2).

Anchor convention, stated because the prompt's tdom-2 and the pitch's own entry
convention are not the same trade.  The signal date is 2026-09-01, which is
trading day 1 of September; the tradeable order is the 2026-09-02 MOC, i.e.
lag=1 from a TDOM-1 anchor.  A TDOM-2 anchor with lag=1 enters TOMORROW.  Both
are measured, and the whole tdom 1..8 ladder is measured beside them because a
seasonal cell with six observations is exactly where an anchor placebo ladder
earns its keep.

  V0. Population: how many midterm Septembers does the cache actually hold?
  V1. The cell: tdom anchor x horizon, midterm Septembers only, exact sign
      test, by-year record.
  V2. GATE ATTRIBUTION on the cycle leg: is this September weakness (the main
      effect everybody knows) rather than anything about midterms?
  V3. PLACEBO ANCHOR LADDER across tdom 1..8 - does the true anchor spike?
  V4. MULTIPLICITY: the 12 months x 4 cycle classes grid actually walked, with
      a permutation max-of-K, plus the horizon dimension.
  V5. Vehicle (^GSPC vs SPY) and the direction the expression would take.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 260)

BAR = pd.Timestamp("2026-09-01")
px = close_panel(["^GSPC", "SPY", "IWM"])
px = px[px.index <= BAR].dropna(subset=["^GSPC"])
D = px.index
print(f"panel {D[0].date()} .. {D[-1].date()}  n={len(D)}")
print(f"today's session {D[-1].date()} is trading day "
      f"{int((pd.DatetimeIndex([d for d in D if d.year==2026 and d.month==9]) <= D[-1]).sum())}"
      f" of September 2026; the ENTRY session 2026-09-02 is trading day 2")


def tdom_index(dates: pd.DatetimeIndex) -> pd.Series:
    """1-based trading day of month for every session in the index."""
    s = pd.Series(dates, index=dates)
    return s.groupby([dates.year, dates.month]).cumcount() + 1


TDOM = pd.Series(tdom_index(D).values, index=D)


def anchors(month: int, tdom: int, cycle: str = "all") -> pd.DatetimeIndex:
    m = (pd.DatetimeIndex(D).month == month) & (TDOM.values == tdom)
    d = D[m]
    if cycle == "midterm":
        d = d[(pd.DatetimeIndex(d).year % 4) == 2]
    elif cycle == "pres":
        d = d[(pd.DatetimeIndex(d).year % 4) == 0]
    elif cycle == "pre":
        d = d[(pd.DatetimeIndex(d).year % 4) == 3]
    elif cycle == "post":
        d = d[(pd.DatetimeIndex(d).year % 4) == 1]
    return pd.DatetimeIndex(d)


def stats(dates, h, tkr="^GSPC", lag=1):
    s = fwd_lag(px[tkr], h, lag)
    v = s.reindex(pd.DatetimeIndex(dates)).dropna()
    if len(v) == 0:
        return None
    a = v.values
    w = int((a > 0).sum())
    base = s.dropna()
    return {"n": len(a), "mean_pct": 100 * a.mean(),
            "median_pct": 100 * float(np.median(a)),
            "rec": f"{w}-{len(a)-w}", "hit": 100 * w / len(a),
            "worst_pct": 100 * a.min(), "best_pct": 100 * a.max(),
            "ctrl_pct": 100 * base.mean(),
            "excess_pp": 100 * (a.mean() - base.mean()),
            "sign_p_coin": sign_test(w, len(a)),
            "sign_p_base": sign_test(w, len(a), float((base > 0).mean())),
            "se_pp": 100 * a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else np.nan,
            "vals": a, "dates": pd.DatetimeIndex(v.index)}


# ------------------------------------------------------------- V0. population
print("\n" + "=" * 100)
print("V0. POPULATION - how many midterm Septembers does the cache hold?")
print("=" * 100)
sep_mid = sorted({d.year for d in D if d.month == 9 and d.year % 4 == 2})
print(f"  midterm Septembers in the cache: {sep_mid}  (N={len(sep_mid)}, and "
      f"2026 is the LIVE one, so at most {len(sep_mid)-1} are measurable)")
print(f"  all Septembers: {len(sorted({d.year for d in D if d.month == 9}))}")

# --------------------------------------------------------------- V1. the cell
print("\n" + "=" * 100)
print("V1. THE CELL - midterm September, by anchor tdom and horizon")
print("=" * 100)
for tk in ("^GSPC", "SPY"):
    for tdom in (1, 2):
        rows = []
        for h in (1, 2, 3, 5, 7, 10):
            a = anchors(9, tdom, "midterm")
            r = stats(a, h, tk)
            if r is None:
                continue
            rows.append({"label": f"{tk} SEP midterm tdom{tdom} h={h}",
                         **{k: v for k, v in r.items()
                            if k not in ("vals", "dates")}})
        show(rows, f"{tk}, anchor = trading day {tdom} of September, "
                   f"midterm years, lag=1")
    # the by-year record at the two horizons the pitch would use
    for h in (3, 5):
        r = stats(anchors(9, 1, "midterm"), h, tk)
        if r is None:
            continue
        print(f"\n  {tk} tdom1 h={h} BY YEAR: "
              + "  ".join(f"{d.year} {100*v:+.2f}%"
                          for d, v in zip(r["dates"], r["vals"])))
        print(f"    mean {r['mean_pct']:+.3f}%  record {r['rec']}  "
              f"exact sign p vs coin {r['sign_p_coin']:.4f}  vs the "
              f"instrument's own up-rate {r['sign_p_base']:.4f}")

# --------------------------------------------------------- V2. gate attribution
print("\n" + "=" * 100)
print("V2. GATE ATTRIBUTION on the CYCLE leg - is this just 'September'?")
print("=" * 100)
for h in (3, 5, 10):
    rows = []
    for cyc, lbl in [("all", "September, ALL years (the main effect)"),
                     ("midterm", "September, MIDTERM only (the pitch)"),
                     ("pres", "September, presidential"),
                     ("pre", "September, pre-election"),
                     ("post", "September, post-election")]:
        r = stats(anchors(9, 1, cyc), h)
        if r:
            rows.append({"label": lbl, **{k: v for k, v in r.items()
                                          if k not in ("vals", "dates")}})
    # the non-September midterm control and the all-month control
    mid_all = pd.DatetimeIndex([d for d in D if TDOM[d] == 1
                                and d.year % 4 == 2 and d.month != 9])
    r = stats(mid_all, h)
    rows.append({"label": "MIDTERM, every month EXCEPT September",
                 **{k: v for k, v in r.items() if k not in ("vals", "dates")}})
    allm = pd.DatetimeIndex([d for d in D if TDOM[d] == 1 and d.month != 9])
    r = stats(allm, h)
    rows.append({"label": "ALL years, every month except September",
                 **{k: v for k, v in r.items() if k not in ("vals", "dates")}})
    show(rows, f"^GSPC tdom1, lag=1, h={h}")
    a = stats(anchors(9, 1, "midterm"), h)
    b = stats(anchors(9, 1, "all"), h)
    nm = pd.DatetimeIndex([d for d in anchors(9, 1, "all")
                           if d.year % 4 != 2])
    c = stats(nm, h)
    se = np.sqrt(a["vals"].var(ddof=1) / a["n"]
                 + c["vals"].var(ddof=1) / c["n"])
    print(f"    h={h}: midterm Sept {a['mean_pct']:+.3f}% (N={a['n']}) vs "
          f"NON-midterm Sept {c['mean_pct']:+.3f}% (N={c['n']}) -> the cycle "
          f"gate adds {a['mean_pct']-c['mean_pct']:+.3f}pp, welch t "
          f"{(a['vals'].mean()-c['vals'].mean())/se:+.2f}")
    print(f"          all-September {b['mean_pct']:+.3f}% (N={b['n']}); "
          f"the MAIN EFFECT excess over all-months is "
          f"{b['mean_pct'] - stats(allm, h)['mean_pct']:+.3f}pp")

# ------------------------------------------------------ V3. placebo anchor ladder
print("\n" + "=" * 100)
print("V3. PLACEBO ANCHOR LADDER - tdom 1..8, midterm September, ^GSPC")
print("=" * 100)
for h in (3, 5, 10):
    rows = []
    for t in range(1, 9):
        r = stats(anchors(9, t, "midterm"), h)
        if r:
            rows.append({"label": f"tdom {t}" + ("  <- TRUE (today's entry)"
                                                 if t == 1 else
                                                 ("  <- prompt's anchor" if t == 2 else "")),
                         **{k: v for k, v in r.items()
                            if k not in ("vals", "dates")}})
    show(rows, f"anchor ladder h={h}")
    means = [r["mean_pct"] for r in rows]
    true_i = 0
    rank = sum(1 for m in means if m > means[true_i]) + 1
    print(f"    h={h}: the TRUE anchor (tdom 1) ranks {rank} of {len(means)} "
          f"offsets; ladder mean {np.mean(means):+.3f}%, true "
          f"{means[true_i]:+.3f}%, true minus ladder-mean "
          f"{means[true_i]-np.mean(means):+.3f}pp")

# --------------------------------------------------------- V4. multiplicity
print("\n" + "=" * 100)
print("V4. MULTIPLICITY - the 12 months x 4 cycle classes grid actually walked")
print("=" * 100)
for h in (3, 5, 10):
    cells = []
    for mth in range(1, 13):
        for cyc in ("midterm", "pres", "pre", "post"):
            r = stats(anchors(mth, 1, cyc), h)
            if r is None or r["n"] < 4 or not np.isfinite(r["se_pp"]):
                continue
            cells.append({"cell": f"m{mth:02d}/{cyc}", "n": r["n"],
                          "mean_pct": round(r["mean_pct"], 3),
                          "excess_pp": round(r["excess_pp"], 3),
                          "se_pp": round(r["se_pp"], 3)})
    df = pd.DataFrame(cells)
    obs_row = df[df["cell"] == "m09/midterm"].iloc[0]
    print(f"\n  h={h}: grid of {len(df)} cells (12 months x 4 cycle classes, "
          f"min N=4)")
    print("  most NEGATIVE 6 (the pitch is a SHORT, so the tail that matters):")
    print(df.sort_values("excess_pp").head(6).to_string(index=False))
    print(f"  the pitched cell m09/midterm: excess {obs_row['excess_pp']:+.3f}pp, "
          f"rank {int((df['excess_pp'] < obs_row['excess_pp']).sum())+1} of "
          f"{len(df)} from the negative end")
    rng = np.random.default_rng(42)
    nulls = rng.normal(0.0, df["se_pp"].values[None, :], size=(20000, len(df)))
    nmin = nulls.min(axis=1)
    print(f"    permutation P(min-of-{len(df)} <= {obs_row['excess_pp']:+.3f}) "
          f"= {float((nmin <= obs_row['excess_pp']).mean()):.4f}   null median "
          f"worst {np.median(nmin):+.3f}pp")
    print(f"    (charging the horizon dimension too: the grid is really "
          f"{len(df)} x 6 horizons x 8 anchors = {len(df)*6*8} cells)")

# ------------------------------------------------------------- V5. vehicle
print("\n" + "=" * 100)
print("V5. VEHICLE and EXPRESSION")
print("=" * 100)
for tk in ("^GSPC", "SPY", "IWM"):
    row = []
    for h in (3, 5, 10):
        r = stats(anchors(9, 1, "midterm"), h, tk)
        if r:
            row.append(f"h={h} {r['mean_pct']:+.3f}% ({r['rec']})")
    print(f"  {tk:6s}: " + "   ".join(row))
r = stats(anchors(9, 1, "midterm"), 5)
if r:
    print(f"\n  A SHORT of the h=5 cell earns {-r['mean_pct']:+.3f}% per "
          f"episode against a ~4 bp SPY round trip = "
          f"{-r['mean_pct']*100/4.0:+.1f}x cost, on N={r['n']} and a "
          f"{r['rec']} record.")
