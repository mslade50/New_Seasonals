"""A12 round 1: the pre-collision window into an FOMC decision that is ALSO a
`^VIX` expiry. 2026-09-16 is both; today (2026-09-11) is the k=3 rung.

MECHANISM HONESTY, stated before any number. The story is that a `^VIX`
settlement whose print is fixed by a decision-day move disturbs the ordinary
pre-expiry pin. This repo holds NO dealer-gamma history, NO options open
interest history and NO futures positioning history. The only positioning data
that exist -- data/option_surface_history.parquet and
data/option_positioning_history.parquet -- begin 2026-08-05, i.e. three
collisions' worth of nothing. The mechanism is therefore NOT MEASURABLE here
and has NOT been verified. What is measurable is the price consequence, plus
two proxies for the pin story: realised vol across the window and the
`^VIX`/`^VIX3M` term ratio, both against ordinary expiry weeks.

Geometry: anchor at collision - (k+1) td, entry lag=1 MOC (= collision - k),
exit MOC ON the collision date, so h=k. Today is k=3.

Sections:
  1. the collision calendar
  2. the k = 2..6 LADDER, SPY then IWM, direction measured not assumed
  3. GATE ATTRIBUTION: collision vs all-FOMC vs FOMC-without-expiry vs
     expiry-without-FOMC, at every rung
  4. MIDTERM split (2026 is midterm; the registry says midterm inverts the
     Lucca-Moench pre-FOMC drift)
  5. concentration: every episode date, year histogram, drop-best-2
  6. quad witching at +5 td: how many historical episodes shared it
  7. the pin proxies (realised vol, VIX/VIX3M) vs ordinary expiry weeks
  8. placebo ladder k=-5..+5 on the k=3 geometry
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

ev = load_events()
vx = pd.DatetimeIndex(ev[ev["event"] == "vix_expiry"]["date"])
fo = pd.DatetimeIndex(ev[ev["event"] == "fomc_decision"]["date"])
qw = pd.DatetimeIndex(ev[ev["event"] == "quad_witching"]["date"])
op = pd.DatetimeIndex(ev[ev["event"] == "opex"]["date"])
coll = vx.intersection(fo)
nocoll = fo.difference(vx)
vx_only = vx.difference(fo)

TK = ["SPY", "IWM", "^VIX", "^VIX3M"]
px_d = load_prices(TK)
idx = px_d["SPY"]["Close"].index
panel = pd.DataFrame({t: px_d[t]["Close"].reindex(idx) for t in TK})
LAG = 1

print("=== 1. THE COLLISION CALENDAR ===")
print("vix_expiry n=%d  fomc_decision n=%d  collisions n=%d (%.1f%% of FOMCs)"
      % (len(vx), len(fo), len(coll), 100 * len(coll) / len(fo)))
print("dates:", ", ".join(str(d.date()) for d in coll))
print("by month:", pd.Series(1, index=coll).groupby(coll.month).sum().to_dict())
print("midterm collisions:", [str(d.date()) for d in coll if d.year % 4 == 2])


def cell(anchors, veh, offset, h, label=""):
    pos, kept = anchor_positions(idx, anchors, offset)
    d = pd.DatetimeIndex(idx[pos])
    r = fwd_lag(panel[veh], h, LAG)
    val = r.dropna().index
    d = d.intersection(val)
    if len(d) == 0:
        return {"label": label, "n": 0}, d, np.array([])
    v = r.loc[d].values
    base = r.loc[val]
    w = int((v > 0).sum())
    return ({"label": label, "n": len(v),
             "mean_pct": round(100 * v.mean(), 3),
             "base_pct": round(100 * base.mean(), 3),
             "edge_pct": round(100 * (v.mean() - base.mean()), 3),
             "median_pct": round(100 * np.median(v), 3),
             "hit": round(100 * w / len(v), 1),
             "record": "%d-%d" % (w, len(v) - w),
             "sign_p": round(sign_test(w, len(v)), 4),
             "t": round(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), 2) if len(v) > 1 else np.nan,
             "worst_pct": round(100 * v.min(), 2),
             "best_pct": round(100 * v.max(), 2)}, d, v)


print("\n=== 2+3. THE k LADDER with GATE ATTRIBUTION (entry = collision - k, "
      "exit MOC ON the collision) ===")
for veh in ["SPY", "IWM"]:
    for k in range(2, 7):
        rows = []
        for nm, anc in [("COLLISION (live cell)", coll),
                        ("ALL fomc_decision", fo),
                        ("FOMC, NO expiry (parent A)", nocoll),
                        ("vix_expiry, NO FOMC (parent B)", vx_only)]:
            r, _, _ = cell(anc, veh, -(k + 1), k, nm)
            rows.append(r)
        show(rows, "%s  k=%d" % (veh, k))
        a, b = rows[0], rows[2]
        if a.get("n") and b.get("n"):
            print("  >> collision gate over the FOMC parent: %+.3fpp  "
                  "(collision edge %+.3f vs FOMC-no-expiry edge %+.3f)"
                  % (a["edge_pct"] - b["edge_pct"], a["edge_pct"], b["edge_pct"]))

print("\n=== 4. MIDTERM SPLIT at every rung (2026 IS MIDTERM) ===")
mt = pd.DatetimeIndex([d for d in coll if d.year % 4 == 2])
nm_ = pd.DatetimeIndex([d for d in coll if d.year % 4 != 2])
mt_f = pd.DatetimeIndex([d for d in fo if d.year % 4 == 2])
nm_f = pd.DatetimeIndex([d for d in fo if d.year % 4 != 2])
for veh in ["SPY", "IWM"]:
    for k in range(2, 7):
        rows = [cell(mt, veh, -(k + 1), k, "collision MIDTERM (LIVE)")[0],
                cell(nm_, veh, -(k + 1), k, "collision non-midterm")[0],
                cell(mt_f, veh, -(k + 1), k, "ALL FOMC midterm")[0],
                cell(nm_f, veh, -(k + 1), k, "ALL FOMC non-midterm")[0]]
        show(rows, "%s k=%d cycle split" % (veh, k))

print("\n=== 5. CONCENTRATION on the live rung k=3 ===")
for veh in ["SPY", "IWM"]:
    r, d, v = cell(coll, veh, -4, 3, "collision k=3")
    print("\n%s k=3 -- every episode:" % veh)
    for dd, vv in zip(d, v):
        ev_date = coll[np.argmin(np.abs((coll - dd).days))]
        print("   entry %s -> collision %s : %+.3f%%  (midterm=%s)"
              % (dd.date(), ev_date.date(), 100 * vv, ev_date.year % 4 == 2))
    print("  year histogram:", pd.Series(1, index=d).groupby(d.year).sum().to_dict())
    print("  ", cluster_note(d, v))
    order = np.argsort(-v)
    print("  drop-best-2: mean %+.3f%% -> %+.3f%% on n=%d"
          % (100 * v.mean(), 100 * np.delete(v, order[:2]).mean(), len(v) - 2))
    print("  bootstrap P(mean<=0) = %.3f" % bootstrap_p_le0(v))

print("\n=== 6. QUAD WITCHING at +5 td: how many episodes shared the "
      "configuration live today? ===")
pos = pd.Series(range(len(idx)), index=idx)
shared = []
for d in coll:
    p = pos.get(d)
    if p is None or p + 5 >= len(idx):
        continue
    nxt = idx[p + 5]
    shared.append((str(d.date()), str(nxt.date()), bool((qw == nxt).any()),
                   int(min(abs((qw - d).days)))))
print("collision, +5td session, is that quad witching?, |days to nearest QW|")
for s in shared:
    print("  ", s)
print("share with QW exactly at +5td: %d of %d"
      % (sum(1 for s in shared if s[2]), len(shared)))

print("\n=== 7. PIN PROXIES vs ordinary expiry weeks (NOT the mechanism, a "
      "proxy for it) ===")
spy = panel["SPY"]
rr = spy.pct_change()
rv5 = rolling_on_valid(rr, lambda x: x.rolling(5).std() * np.sqrt(252) * 100)
term = panel["^VIX"] / panel["^VIX3M"]


def win_stat(anchors, series, k=3, label=""):
    """value of `series` measured ON the collision/expiry date, and the change
    over the k-session run-in."""
    p, kept = anchor_positions(idx, anchors, 0)
    on, chg = [], []
    for q in p:
        if q - k < 0:
            continue
        a, b = series.iloc[q - k], series.iloc[q]
        if np.isnan(a) or np.isnan(b):
            continue
        on.append(b)
        chg.append(b - a)
    if not on:
        return {"label": label, "n": 0}
    return {"label": label, "n": len(on), "on_date_mean": round(float(np.mean(on)), 3),
            "run_in_change_mean": round(float(np.mean(chg)), 3),
            "chg_median": round(float(np.median(chg)), 3),
            "chg_pos_share": round(100 * float(np.mean(np.array(chg) > 0)), 1)}


for nm, ser in [("realised vol 5d ann %", rv5), ("VIX/VIX3M term ratio", term)]:
    show([win_stat(coll, ser, 3, "collision"),
          win_stat(vx_only, ser, 3, "expiry, no FOMC"),
          win_stat(nocoll, ser, 3, "FOMC, no expiry")], nm)

print("\n=== 8. PLACEBO LADDER k=-5..+5 on the k=3 geometry (anchor shift) ===")
for veh in ["SPY", "IWM"]:
    rows = []
    for s in range(-5, 6):
        rows.append(cell(coll, veh, -4 + s, 3, "shift %+d" % s)[0])
    show(rows, "%s placebo ladder" % veh)
    df = pd.DataFrame(rows).dropna(subset=["mean_pct"])
    df["rank"] = df["mean_pct"].rank(ascending=False)
    tr = df[df["label"] == "shift +0"]
    print("  TRUE anchor ranks %s of %d by mean (long reading); "
          "%s of %d by -mean (short reading)"
          % (tr["rank"].iloc[0], len(df),
             df["mean_pct"].rank(ascending=True)[tr.index[0]], len(df)))
