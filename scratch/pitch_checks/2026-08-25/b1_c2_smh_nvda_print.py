"""C2 round 1 - Long SMH into the NVDA print with semis at a one-year relative low.

Anchor definition (stated once, load-bearing):
  P = the session on whose close NVDA reports (after the bell).
  Entry convention is lag=1, so the SIGNAL date D = P-2 sessions:
  entry MOC at close D+1 = P-1 (the session BEFORE the print), exit h after.
    h=1 -> exit at close P            = pure PRE-print hold, out before the release
    h=2 -> exit at close P+1          = straddles the reaction gap
    h>=3-> straddle + post-print drift
  POST-print anchor: D = P, entry MOC at close P+1 (after the reaction), exit h.

Today: D = 2026-08-24, entry MOC 2026-08-25, NVDA prints 2026-08-26 AMC.
That is exactly the h=1 pre-print / h>=2 straddle configuration above.

Registry collision priced explicitly: 2026-08-14 "SMH into the NVDA print"
(c4_nvda_print_runup.py) and 2026-08-19 "Short the semis complex into the
August NVDA print" (b1_c4_semis_short.py).
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change
import numpy as np
import pandas as pd

TK = ["SMH", "SPY", "QQQ", "NVDA", "AVGO", "TXN", "AMD", "INTC", "MU",
      "AMKR", "ON", "POWI"]
px = close_panel(TK)
idx = px.index
pos = pd.Series(range(len(idx)), index=idx)

EARN = pd.read_parquet(ROOT / "data" / "earnings_calendar.parquet")
EARN["date"] = pd.to_datetime(EARN["date"])


def print_sessions(ticker: str) -> pd.DatetimeIndex:
    """Announcement dates mapped onto the price calendar (exact or next open)."""
    d = EARN[EARN["ticker"] == ticker]["date"].sort_values().unique()
    out = []
    for x in d:
        x = pd.Timestamp(x)
        loc = idx.searchsorted(x)
        if loc >= len(idx):
            continue
        out.append(idx[loc])
    return pd.DatetimeIndex(sorted(set(out)))


def anchor_dates(prints: pd.DatetimeIndex, offset: int) -> pd.DatetimeIndex:
    """Signal date D = print session + offset. offset=-2 is the live shape."""
    out = []
    for p in prints:
        q = pos.get(p)
        if q is None:
            continue
        j = q + offset
        if 0 <= j < len(idx):
            out.append(idx[j])
    return pd.DatetimeIndex(sorted(set(out)))


# ---------------------------------------------------------------- state series
r63 = _valid_pct_change(px["SMH"], 63) - _valid_pct_change(px["SPY"], 63)
r5 = _valid_pct_change(px["SMH"], 5) - _valid_pct_change(px["SPY"], 5)
sprd63_pit = rolling_on_valid(r63, lambda x: x.rolling(252).rank(pct=True) * 100.0)
sprd5_pit = rolling_on_valid(r5, lambda x: x.rolling(252).rank(pct=True) * 100.0)
rank63_smh = pct_rank(px["SMH"], 63)          # 08-14's gate definition
sma200 = rolling_on_valid(px["SPY"], lambda x: x.rolling(200).mean())
above200 = (px["SPY"] > sma200)
smh_above200 = px["SMH"] > rolling_on_valid(px["SMH"], lambda x: x.rolling(200).mean())

print("=== today's state (2026-08-24) ===")
for lbl, s in [("SMH-SPY 63d spread", r63), ("  its PIT252 pctile", sprd63_pit),
               ("SMH-SPY 5d spread", r5), ("  its PIT252 pctile", sprd5_pit),
               ("SMH 63d rank (08-14 gate)", rank63_smh)]:
    print(f"  {lbl:28s} {s.iloc[-1]:8.3f}")
print(f"  SPY above 200d               {bool(above200.iloc[-1])}")
print(f"  SMH above own 200d           {bool(smh_above200.iloc[-1])}")

NV = print_sessions("NVDA")
NV = NV[NV <= idx[-1]]
print(f"\nNVDA print sessions on the price calendar: {len(NV)} "
      f"({NV[0].date()} .. {NV[-1].date()}); since 2013 = "
      f"{(NV >= '2013-01-01').sum()}, since 2020 = {(NV >= '2020-01-01').sum()}")

# --------------------------------------------------------- 1. the print cell
LEGS = [("SMH", 1.0)]
COST = 5.0   # SMH round trip, two-way, stated assumption


def cell(dates, h, label, lag=1, legs=LEGS):
    ret = vehicle_ret(px, legs, h, lag)
    d = pd.DatetimeIndex(dates).intersection(ret.dropna().index)
    if len(d) == 0:
        return {"label": label, "n": 0}, np.array([]), d
    v = ret.loc[d].values
    r = summarize(v, label)
    return r, v, d


print("\n" + "=" * 78)
print("1. THE PRINT CELL, ungated -- both anchors, horizon by horizon")
print("=" * 78)
rows = []
for h in (1, 2, 3, 5, 7, 10):
    r, v, d = cell(anchor_dates(NV, -2), h, f"PRE-anchor(D=P-2) h={h}")
    ret = vehicle_ret(px, LEGS, h, 1)
    r["ctl_all_pct"] = round(100 * ret.dropna().mean(), 3)
    r["edge_pp"] = round(r["mean_pct"] - r["ctl_all_pct"], 3)
    rows.append(r)
for h in (1, 2, 3, 5, 7, 10):
    r, v, d = cell(anchor_dates(NV, 0), h, f"POST-anchor(D=P) h={h}")
    ret = vehicle_ret(px, LEGS, h, 1)
    r["ctl_all_pct"] = round(100 * ret.dropna().mean(), 3)
    r["edge_pp"] = round(r["mean_pct"] - r["ctl_all_pct"], 3)
    rows.append(r)
show(rows, "SMH around the NVDA print (all prints, no relative-low gate)")

# ------------------------------------------------- 2. does the low gate filter
print("\n" + "=" * 78)
print("2. GATE ATTRIBUTION -- does the one-year relative low do anything?")
print("=" * 78)
for h in (1, 2, 3, 5):
    rows = []
    A = anchor_dates(NV, -2)
    for lbl, gate in [
            ("ALL prints (no gate)", None),
            ("spread63 PIT <= 10 (live=0.0)", sprd63_pit <= 10),
            ("spread63 PIT <= 20", sprd63_pit <= 20),
            ("spread63 PIT <= 25", sprd63_pit <= 25),
            ("spread63 PIT  > 25 (complement)", sprd63_pit > 25),
            ("spread63 PIT >= 75", sprd63_pit >= 75),
            ("SMH rank63 < 25 (08-14 gate)", rank63_smh < 25),
            ("SMH -15%+ off own 52w hi", (px["SMH"] / rolling_on_valid(px["SMH"], lambda x: x.rolling(252).max()) - 1) <= -0.15),
    ]:
        d = A if gate is None else A.intersection(px.index[gate.reindex(idx, fill_value=False).values])
        r, v, dd = cell(d, h, lbl)
        if r["n"]:
            wins = int((v > 0).sum())
            r["sign_p"] = round(sign_test(wins, len(v)), 4)
        rows.append(r)
    show(rows, f"h={h}  gate walk on the PRE anchor (D=P-2)")

# ------------------------------------------------------- 3. placebo offset ladder
print("\n" + "=" * 78)
print("3. OFFSET PLACEBO LADDER -- relocate the anchor -10..+5 sessions")
print("=" * 78)
for h in (1, 3):
    rows = []
    for k in range(-10, 6):
        A = anchor_dates(NV, -2 + k)
        r, v, d = cell(A, h, f"offset {k:+d}" + ("  <-- TRUE" if k == 0 else ""))
        rows.append(r)
    show(rows, f"h={h} ungated ladder (all NVDA prints)")
    m = [x["mean_pct"] for x in rows if x.get("n")]
    tru = [x for x in rows if "TRUE" in x["label"]][0]["mean_pct"]
    rk = 1 + sum(1 for x in m if x > tru)
    print(f"  -> true anchor ranks {rk} of {len(m)}; ladder mean {np.mean(m):+.3f}%, "
          f"true {tru:+.3f}%, true-minus-ladder {tru - np.mean(m):+.3f}pp")

    rows = []
    for k in range(-10, 6):
        A = anchor_dates(NV, -2 + k)
        A = A.intersection(px.index[(sprd63_pit <= 25).reindex(idx, fill_value=False).values])
        r, v, d = cell(A, h, f"offset {k:+d}" + ("  <-- TRUE" if k == 0 else ""))
        rows.append(r)
    show(rows, f"h={h} GATED ladder (spread63 PIT<=25)")
    m = [x["mean_pct"] for x in rows if x.get("n")]
    tru = [x for x in rows if "TRUE" in x["label"]][0].get("mean_pct", np.nan)
    if m and not np.isnan(tru):
        rk = 1 + sum(1 for x in m if x > tru)
        print(f"  -> true anchor ranks {rk} of {len(m)}; ladder mean {np.mean(m):+.3f}%, "
              f"true {tru:+.3f}%, true-minus-ladder {tru - np.mean(m):+.3f}pp")

# ------------------------------------------ 4. NVDA-specific or generic semis?
print("\n" + "=" * 78)
print("4. NVDA-SPECIFIC OR GENERIC BIG-SEMI PRINT? (vehicle = SMH throughout)")
print("=" * 78)
for h in (1, 3):
    rows = []
    for t in ["NVDA", "AVGO", "TXN", "AMD", "INTC", "MU"]:
        P = print_sessions(t)
        P = P[P <= idx[-1]]
        A = anchor_dates(P, -2)
        r, v, d = cell(A, h, f"{t} prints (N_prints={len(P)})")
        if r.get("n"):
            r["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
        rows.append(r)
        Ag = A.intersection(px.index[(sprd63_pit <= 25).reindex(idx, fill_value=False).values])
        rg, vg, dg = cell(Ag, h, f"  {t} x spread63 PIT<=25")
        rows.append(rg)
    # pooled non-NVDA
    allP = []
    for t in ["AVGO", "TXN", "AMD", "INTC", "MU"]:
        P = print_sessions(t)
        allP.extend(list(anchor_dates(P[P <= idx[-1]], -2)))
    A = pd.DatetimeIndex(sorted(set(allP)))
    r, v, d = cell(A, h, "POOLED non-NVDA big semis")
    rows.append(r)
    show(rows, f"h={h} same rule on other large semi prints")

# ------------------------------------------------ 5. month / era falsification
print("\n" + "=" * 78)
print("5. THE MECHANISM INSIDE ITS OWN WINDOW -- print month and era")
print("=" * 78)
for h in (1, 3):
    A = anchor_dates(NV, -2)
    ret = vehicle_ret(px, LEGS, h, 1)
    A = A.intersection(ret.dropna().index)
    v = ret.loc[A].values
    mon = pd.DatetimeIndex([idx[pos[d] + 2] for d in A]).month
    rows = []
    for mgrp, lbl in [((2,), "Feb prints"), ((5,), "May prints"),
                      ((8,), "Aug prints  <-- LIVE"), ((11,), "Nov prints")]:
        m = np.isin(mon, mgrp)
        r = summarize(v[m], lbl)
        if r["n"]:
            r["sign_p"] = round(sign_test(int((v[m] > 0).sum()), int(m.sum())), 4)
        rows.append(r)
    rows += era_split(A, v, "2013-01-01")
    rows += era_split(A, v, "2020-01-01")
    show(rows, f"h={h} by print month and era (ungated)")
    # August x 2020+
    m = (np.isin(mon, [8])) & (A >= pd.Timestamp("2020-01-01"))
    r = summarize(v[m], "Aug x 2020+")
    show([r], f"h={h} the live cell's own slice")

# ------------------------------------------------------- 6. full battery, live def
print("\n" + "=" * 78)
print("6. BATTERY on the pitched form: PRE anchor x spread63 PIT<=25")
print("=" * 78)
A = anchor_dates(NV, -2).intersection(
    px.index[(sprd63_pit <= 25).reindex(idx, fill_value=False).values])
mask = pd.Series(idx.isin(A), index=idx)
variants = {
    "PIT<=10": pd.Series(idx.isin(anchor_dates(NV, -2).intersection(
        px.index[(sprd63_pit <= 10).reindex(idx, fill_value=False).values])), index=idx),
    "PIT<=25": mask,
    "PIT<=50": pd.Series(idx.isin(anchor_dates(NV, -2).intersection(
        px.index[(sprd63_pit <= 50).reindex(idx, fill_value=False).values])), index=idx),
    "no gate": pd.Series(idx.isin(anchor_dates(NV, -2)), index=idx),
}
battery(px, mask, LEGS, 3, "C2 SMH pre-NVDA-print x relative low", COST,
        variants=variants, min_gap=5)

# ------------------------------------------------- 7. tape over-selection + cost
print("\n" + "=" * 78)
print("7. TAPE OVER-SELECTION, COST, BOOK OVERLAP")
print("=" * 78)
base_a200 = 100 * above200.dropna().mean()
for lbl, A in [("all NVDA prints (D=P-2)", anchor_dates(NV, -2)),
               ("gated PIT<=25", anchor_dates(NV, -2).intersection(
                   px.index[(sprd63_pit <= 25).reindex(idx, fill_value=False).values])),
               ("gated PIT<=10", anchor_dates(NV, -2).intersection(
                   px.index[(sprd63_pit <= 10).reindex(idx, fill_value=False).values]))]:
    a = above200.reindex(A).dropna()
    s = smh_above200.reindex(A).dropna()
    print(f"  {lbl:26s} SPY>200d on {100*a.mean():5.1f}% of {len(a)} days "
          f"(base {base_a200:.1f}%) | SMH>own200d {100*s.mean():5.1f}%  "
          f"[TODAY SPY {bool(above200.iloc[-1])} / SMH {bool(smh_above200.iloc[-1])}]")

print("\n  book overlap: staged OLV semis longs AMKR/ON/POWI vs SMH")
rr = px[["SMH", "AMKR", "ON", "POWI", "NVDA"]].pct_change()
for h in (1, 3, 5):
    win = rr.rolling(h).sum().dropna()
    win = win[win.index >= "2020-01-01"]
    line = []
    for t in ["AMKR", "ON", "POWI", "NVDA"]:
        c = win["SMH"].corr(win[t])
        b = np.polyfit(win["SMH"], win[t], 1)[0]
        line.append(f"{t} corr {c:.3f} beta {b:.2f}")
    print(f"   h={h} (2020+): " + " | ".join(line))
print("  NOTE: a long SMH ticket adds the same factor the scan already staged "
      "in three single names.")
