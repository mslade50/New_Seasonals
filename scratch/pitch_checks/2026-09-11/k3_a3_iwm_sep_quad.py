"""A3 KILL CHECK - short IWM into September quad witching, midterm year.

Pre-specified by the candidate: anchor on the September quad-witching date,
ENTER lag=1 MOC at k trading sessions BEFORE it, exit MOC on the quad session.
So the SIGNAL close sits at qw-(k+1), entry at qw-k, hold h=k. Scan k=2..8 and
report the WHOLE ladder. Vehicle IWM, direction SHORT; long is the contrast.

LIVE: signal 2026-09-10 = qw-6, entry 2026-09-11 = qw-5, quad 2026-09-18.
So the live rung is k=5, h=5, and the FOMC decision 2026-09-16 is INSIDE.

Attacks run here:
  1. the whole k ladder, both directions, against IWM's own h-matched drift
  2. inception / proxy: does ^RUT extend the sample? (cache starts 2000)
  3. midterm vs non-midterm split
  4. FOMC-inside-the-window split -- the LIVE configuration
  5. placebo anchor ladder j=-5..+5 around the live rung
  6. month ladder: the same run-in into EVERY month's opex/quad
  7. concentration, era split, sign test, cost
"""
import sys
from pathlib import Path

ROOTP = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOTP))
import numpy as np                                        # noqa: E402
import pandas as pd                                       # noqa: E402
from pitch_lab import *  # noqa: E402,F403

TK = ["IWM", "SPY", "^RUT", "QQQ", "DIA", "XLF", "XLI", "EEM", "EFA"]
px = close_panel(TK)
cal = px["SPY"].dropna().index
pos = pd.Series(range(len(cal)), index=cal)

LIVE_K = 5          # entry 2026-09-11 is 5 sessions before the 09-18 quad
COST_BPS = 4.0      # IWM MOC both ways + ~0.3%/yr borrow over a week


def sep_quads(upto=None):
    ev = load_events(["quad_witching"])
    s = ev[ev["date"].dt.month == 9]["date"]
    if upto is not None:
        s = s[s <= upto]
    return pd.DatetimeIndex(sorted(s))


def anchor_for(quad, k, extra=0):
    """SIGNAL date for entry at qw-k (+ placebo shift `extra`)."""
    loc = int(cal.searchsorted(quad))
    if loc >= len(cal) or cal[loc] != quad:
        return None
    p = loc - (k + 1) + extra
    if 0 <= p < len(cal):
        return cal[p]
    return None


QUADS = sep_quads(cal[-1])
print("=" * 78)
print("SANITY")
print(f"  cache calendar {cal[0].date()} .. {cal[-1].date()}")
print(f"  September quad witchings resolvable on the calendar: {len(QUADS)}")
print(f"  {[str(d.date()) for d in QUADS]}")
for t in ["IWM", "^RUT", "SPY"]:
    s = px[t].dropna()
    print(f"  {t:<6} first bar {s.index[0].date()}  (cache floor is 2000)")
print("  -> ^RUT starts 2000-01-03, the cache floor, so the index proxy buys")
print("     ONE extra September (2000) over IWM at best. It does NOT extend")
print("     the sample; the cache is the binding constraint, not inception.")

q26 = pd.Timestamp("2026-09-18")
print(f"\n  LIVE CHECK: signal {anchor_for(q26, LIVE_K)} should be 2026-09-10"
      if anchor_for(q26, LIVE_K) is not None else "\n  LIVE: 2026 quad off-cal")
# 2026 quad is in the future relative to the cache, resolve by hand
loc26 = int(cal.searchsorted(pd.Timestamp("2026-09-10")))
print(f"  2026-09-10 is calendar position {loc26}; entry is the next session")
print(f"  2026-09-11 -> 2026-09-18 is {LIVE_K} sessions (11,14,15,16,17,18)")


# ------------------------------------------------------------------ 1. ladder
def cell(k, legs, extra=0, quads=None, month=9):
    """Return (signal dates, forward returns) for the qw-k entry, h=k."""
    qs = quads if quads is not None else QUADS
    ret = vehicle_ret(px, legs, k, 1)
    ds, vals, meta = [], [], []
    for q in qs:
        a = anchor_for(q, k, extra)
        if a is None:
            continue
        v = ret.get(a, np.nan)
        if np.isnan(v):
            continue
        ds.append(a)
        vals.append(v)
        meta.append(q)
    return pd.DatetimeIndex(ds), np.asarray(vals), pd.DatetimeIndex(meta)


print("\n" + "=" * 78)
print("1. THE WHOLE k LADDER (entry at qw-k, exit at the quad close, h=k)")
print("   direction as PITCHED = SHORT IWM. long shown as the honest contrast.")
rows = []
for k in range(2, 9):
    d, v, q = cell(k, [("IWM", 1.0)])
    ret_all = vehicle_ret(px, [("IWM", 1.0)], k, 1).dropna()
    short = -v
    w = int((short > 0).sum())
    rows.append({
        "k": k, "N": len(v),
        "SHORT_pct": round(100 * short.mean(), 3),
        "LONG_pct": round(100 * v.mean(), 3),
        "IWM_drift_pct": round(100 * ret_all.mean(), 3),
        "SHORT_edge_pp": round(100 * (short.mean() + ret_all.mean()), 3),
        "hit_short": round(100 * (short > 0).mean(), 1),
        "t_short": round(short.mean() / (short.std(ddof=1) / np.sqrt(len(short))), 2),
        "signp_short": round(sign_test(w, len(short)), 4),
        "worst_short": round(100 * short.min(), 2),
        "LIVE": "<== live rung" if k == LIVE_K else "",
    })
print(pd.DataFrame(rows).to_string(index=False))
print(f"\n  cost bound: {COST_BPS} bps round trip -> need >= {5*COST_BPS} bps "
      f"= {5*COST_BPS/100:.2f}% mean to clear 5x")

# ------------------------------------------------------ 2. the live rung, full
print("\n" + "=" * 78)
print(f"2. THE LIVE RUNG k={LIVE_K} (h={LIVE_K}) IN FULL, SHORT IWM")
d5, v5, q5 = cell(LIVE_K, [("IWM", 1.0)])
s5 = -v5
ret_all = vehicle_ret(px, [("IWM", 1.0)], LIVE_K, 1)
base = ret_all.dropna()
show([summarize(s5, f"SHORT IWM qw-{LIVE_K} (N={len(s5)})"),
      summarize(-base.values, "CTRL short IWM all days"),
      summarize(v5, "LONG IWM same anchors"),
      summarize(base.values, "CTRL long IWM all days")],
     "live rung vs unconditional drift")
print("  per-year table:")
tab = pd.DataFrame({"signal": [str(x.date()) for x in d5],
                    "quad": [str(x.date()) for x in q5],
                    "year": q5.year,
                    "midterm": (q5.year % 4 == 2),
                    "short_pct": np.round(100 * s5, 2)})
# FOMC inside the window?
fomc = load_events(["fomc_decision"])["date"]
ins = []
for a in d5:
    p = int(pos[a])
    lo, hi = cal[p + 1], cal[min(p + 1 + LIVE_K, len(cal) - 1)]
    ins.append(bool(((fomc > lo) & (fomc <= hi)).any()))
tab["fomc_in"] = ins
print(tab.to_string(index=False))
print(f"  concentration: {cluster_note(d5, s5)}")

# ------------------------------------------------------------- 3. midterm
print("\n" + "=" * 78)
print("3. MIDTERM SPLIT (2026 is midterm, year%4==2)")
mid = np.asarray(tab["midterm"].values, dtype=bool)
show([summarize(s5[mid], f"SHORT midterm (N={int(mid.sum())})"),
      summarize(s5[~mid], f"SHORT non-midterm (N={int((~mid).sum())})")],
     f"k={LIVE_K}")
for k in (3, 5, 8):
    d, v, q = cell(k, [("IWM", 1.0)])
    m = (q.year % 4 == 2)
    print(f"  k={k}: SHORT midterm {100*(-v[m]).mean():+.3f}% (N={m.sum()}) | "
          f"non-mid {100*(-v[~m]).mean():+.3f}% (N={(~m).sum()})")

# --------------------------------------------------- 4. FOMC-in-window split
print("\n" + "=" * 78)
print("4. FOMC-INSIDE-THE-WINDOW SPLIT -- THIS IS THE LIVE CONFIGURATION")
print("   (FOMC decision 2026-09-16 sits inside the 09-11 -> 09-18 hold)")
f = np.asarray(ins, dtype=bool)
show([summarize(s5[f], f"SHORT, FOMC IN window (N={int(f.sum())})"),
      summarize(s5[~f], f"SHORT, FOMC OUT (N={int((~f).sum())})"),
      summarize(v5[f], f"LONG, FOMC IN (N={int(f.sum())})"),
      summarize(v5[~f], f"LONG, FOMC OUT (N={int((~f).sum())})")],
     f"k={LIVE_K}")
wi = int((s5[f] > 0).sum())
print(f"  SHORT with FOMC in window record {wi}-{int(f.sum())-wi}, "
      f"sign p(short wins) = {sign_test(wi, int(f.sum())):.4f}, "
      f"sign p(LONG wins) = {sign_test(int(f.sum())-wi, int(f.sum())):.4f}")
print("  the same split across the whole k ladder:")
for k in range(2, 9):
    d, v, q = cell(k, [("IWM", 1.0)])
    ff = []
    for a in d:
        p = int(pos[a])
        lo, hi = cal[p + 1], cal[min(p + 1 + k, len(cal) - 1)]
        ff.append(bool(((fomc > lo) & (fomc <= hi)).any()))
    ff = np.asarray(ff, bool)
    print(f"   k={k}: SHORT fomc-IN {100*(-v[ff]).mean():+.3f}% "
          f"(N={ff.sum()}, hit {100*((-v[ff])>0).mean():.0f}%) | "
          f"fomc-OUT {100*(-v[~ff]).mean():+.3f}% (N={(~ff).sum()}, "
          f"hit {100*((-v[~ff])>0).mean():.0f}%)")

# ------------------------------------------------- 5. placebo anchor ladder
print("\n" + "=" * 78)
print(f"5. PLACEBO ANCHOR LADDER at h={LIVE_K}: slide the anchor j=-5..+5")
print("   (an anchor that does not outrank its own placebos is momentum)")
pl = []
for j in range(-5, 6):
    d, v, q = cell(LIVE_K, [("IWM", 1.0)], extra=j)
    pl.append({"j": j, "N": len(v),
               "SHORT_pct": round(100 * (-v).mean(), 3),
               "hit": round(100 * ((-v) > 0).mean(), 1),
               "TRUE": "<== true anchor" if j == 0 else ""})
P = pd.DataFrame(pl)
print(P.to_string(index=False))
order = P.sort_values("SHORT_pct", ascending=False).reset_index(drop=True)
rk = int(order.index[order["j"] == 0][0]) + 1
print(f"  TRUE ANCHOR RANKS {rk} OF {len(P)} on the short side "
      f"(1 = best short).")

# ------------------------------------------------------------ 6. month ladder
print("\n" + "=" * 78)
print(f"6. MONTH LADDER: the same qw-{LIVE_K} run-in into EVERY month's")
print("   opex (quad_witching is Mar/Jun/Sep/Dec; opex is all 12)")
opex = load_events(["opex"])
ml = []
for m in range(1, 13):
    qs = pd.DatetimeIndex(sorted(opex[opex["date"].dt.month == m]["date"]))
    qs = qs[qs <= cal[-1]]
    d, v, q = cell(LIVE_K, [("IWM", 1.0)], quads=qs)
    if len(v) == 0:
        continue
    ml.append({"month": m, "N": len(v),
               "SHORT_pct": round(100 * (-v).mean(), 3),
               "hit": round(100 * ((-v) > 0).mean(), 1)})
M = pd.DataFrame(ml)
print(M.to_string(index=False))
mo = M.sort_values("SHORT_pct", ascending=False).reset_index(drop=True)
print(f"  September ranks {int(mo.index[mo['month']==9][0])+1} of {len(M)} "
      f"months on the short side.")
print(f"  all-months pooled SHORT mean = "
      f"{M['SHORT_pct'].mul(M['N']).sum()/M['N'].sum():+.3f}%")

# ------------------------------------------------------- 7. era + pair + ^RUT
print("\n" + "=" * 78)
print("7. ERA SPLIT, THE IWM/SPY PAIR, AND THE ^RUT PROXY")
show(era_split(d5, s5), f"SHORT IWM k={LIVE_K} episode era split")
dp, vp, qp = cell(LIVE_K, [("IWM", -1.0), ("SPY", 1.0)])
show([summarize(vp, f"SHORT IWM / LONG SPY pair (N={len(vp)})"),
      summarize(s5, "SHORT IWM outright"),
      summarize(-cell(LIVE_K, [("SPY", 1.0)])[1], "SHORT SPY outright")],
     "is it a small-cap story at all?")
dr, vr, qr = cell(LIVE_K, [("^RUT", 1.0)])
show([summarize(-vr, f"SHORT ^RUT index (N={len(vr)})"),
      summarize(s5, f"SHORT IWM etf (N={len(s5)})")],
     "proxy contrast (same cache floor, so ~same sample)")

print("\n" + "=" * 78)
print("DONE")
