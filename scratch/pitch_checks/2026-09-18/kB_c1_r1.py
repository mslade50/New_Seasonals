"""c1 KILL CHECK round 1 - long IWM from the quad-witching close when small caps
arrive washed out (IWM z10 <= -1 at the SIGNAL close, lag-1 entry = the quad close).

Pre-specified by the event-sleeve prereg (2026-08-06, T3): "skip when IWM z10
(10-session return over vol21*sqrt(10), lag-1) < -1. Washed-out tapes into quad
expiry BOUNCE". This tests the long side of that exception as a standalone trade.

Signal D = quad session - 1 (z10 measured on D's close), entry MOC on the quad
close, exit close quad+h. Live: D = 2026-09-17, entry 2026-09-18 (Sep quad).

Attacks:
  1. cell vs all quads ungated, the discarded complement, generic z10<=-1 any day,
     own drift, local +/-126td
  2. month rows: September; all four quads; monthly (non-quad) opex; all opex
  3. placebo offset ladder: shift D by k=-5..+5, re-evaluate the gate at the
     shifted D, rank k=0
  4. FOMC-inside-the-hold split, era split, midterm split, to-quarter-end exit
  5. rows SPY, QQQ, EEM, EFA on their own gates
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

TK = ["IWM", "SPY", "QQQ", "EEM", "EFA"]
px = close_panel(TK)
cal = px["SPY"].dropna().index
px = px.reindex(cal)
pos = pd.Series(range(len(cal)), index=cal)
HS = [1, 2, 3, 5, 8, 10]


def z10_sleeve(s: pd.Series) -> pd.Series:
    c = s.dropna()
    vol21 = c.pct_change().rolling(21).std()
    return (c.pct_change(10) / (vol21 * np.sqrt(10))).reindex(s.index)


Z = {t: z10_sleeve(px[t]) for t in TK}
ZPL = {t: zscore(px[t], 10) for t in TK}
print("LIVE 2026-09-17: IWM z10 sleeve-def %.3f | pitch_lab.zscore %.3f" %
      (Z["IWM"].loc["2026-09-17"], ZPL["IWM"].loc["2026-09-17"]))
for t in TK:
    print(f"  {t}: sleeve z10 {Z[t].loc['2026-09-17']:+.3f}  pl z10 {ZPL[t].loc['2026-09-17']:+.3f}")


def expiry_session(d):
    """Quad/opex date -> the actual expiry session (holiday Friday -> prior session)."""
    loc = int(cal.searchsorted(d))
    if loc < len(cal) and cal[loc] == d:
        return loc
    return loc - 1  # e.g. Good Friday: expiry moved to Thursday


def anchors(kind, months=None):
    ev = load_events([kind])
    out = []
    for d in ev["date"]:
        if d < cal[0] or d > cal[-1]:
            continue
        if months and d.month not in months:
            continue
        q = expiry_session(d)
        if cal[q] != d:
            print(f"  note: {kind} {d.date()} not a session -> expiry {cal[q].date()}")
        out.append(q)
    return sorted(set(out))


def stats(vals, dates=None, label=""):
    v = np.asarray(vals, float)
    ok = ~np.isnan(v)
    v = v[ok]
    r = summarize(v, label)
    if r["n"]:
        w = int((v > 0).sum())
        r["rec"] = f"{w}-{len(v)-w}"
        r["sign_p"] = round(sign_test(w, len(v)), 4)
        tot = v.sum()
        top2 = np.sort(v)[::-1][:2].sum()
        r["top2_share"] = round(100 * top2 / tot, 0) if tot > 0 else np.nan
        if len(v) > 2:
            r["drop2_pct"] = round(100 * np.sort(v)[::-1][2:].mean(), 3)
    return r


def ret_from_q(t, q, h):
    if q + h >= len(cal):
        return np.nan
    return px[t].iloc[q + h] / px[t].iloc[q] - 1.0


def cell_rows(t, qs, gate_t, thr, h, label):
    """qs: expiry session positions. D = q-1. gate on gate_t's z10 at D."""
    g, c, a = [], [], []
    for q in qs:
        D = q - 1
        z = Z[gate_t].iloc[D]
        r = ret_from_q(t, q, h)
        if np.isnan(r) or np.isnan(z):
            continue
        a.append((cal[q], r))
        (g if z <= thr else c).append((cal[q], r))
    return g, c, a


def dvals(lst):
    return np.array([r for _, r in lst]), pd.DatetimeIndex([d for d, _ in lst])


QUADS = anchors("quad_witching")
SEPQ = [q for q in QUADS if cal[q].month == 9]
OPEX = anchors("opex")
MON = [q for q in OPEX if q not in QUADS]
print(f"\nquads {len(QUADS)}, Sep quads {len(SEPQ)}, opex {len(OPEX)}, monthly non-quad {len(MON)}")

# ---------------------------------------------------------------- 1. main cells
for fam_name, fam in (("ALL QUADS", QUADS), ("SEPTEMBER QUAD", SEPQ),
                      ("MONTHLY non-quad OPEX", MON), ("ALL OPEX", OPEX)):
    rows = []
    for h in HS:
        g, c, a = cell_rows("IWM", fam, "IWM", -1.0, h, "")
        gv, gd = dvals(g)
        cv, _ = dvals(c)
        av, _ = dvals(a)
        rg = stats(gv, gd, f"h={h} GATED z<=-1")
        rc = stats(cv, None, f"h={h} complement z>-1")
        ra = stats(av, None, f"h={h} ungated")
        rows += [rg, rc, ra]
    show(rows, f"1. IWM long from the {fam_name} close, gate IWM z10(D) <= -1")

# own drift + generic reversal + local control at the key horizons
ivals = {}
for h in HS:
    ivals[h] = fwd_lag(px["IWM"], h, lag=1)
rows = []
zmask = (Z["IWM"] <= -1.0)
for h in HS:
    r = ivals[h]
    allv = r.dropna()
    gen_days = cal[zmask.values & r.notna().values]
    gen_ep = declusters(gen_days, h, cal)
    rows.append(summarize(allv.values, f"h={h} own drift all days"))
    rows.append(summarize(r.loc[gen_days].values, f"h={h} generic z<=-1 ANY day (day-lvl N={len(gen_days)})"))
    rows.append(stats(r.loc[gen_ep].values, None, f"h={h} generic z<=-1 declustered"))
show(rows, "1b. controls: own drift, generic IWM z10<=-1 on any day (lag-1 entry)")

# gated quad anchors as D dates for local control
for h in (5, 8, 10):
    g, c, a = cell_rows("IWM", QUADS, "IWM", -1.0, h, "")
    Ds = pd.DatetimeIndex([cal[pos[d] - 1] for d, _ in g])
    r = ivals[h]
    loc = local_control(cal[r.notna().values], Ds)
    print(f"  h={h}: quad gated mean {100*np.mean([x for _, x in g]):+.3f}% | local +/-126td ctrl "
          f"{100*r.loc[loc].mean():+.3f}% | generic-z local ctrl (z<=-1 days within +/-126 of anchors, ex-anchors) "
          f"{100*r.loc[loc.intersection(cal[zmask.values])].mean():+.3f}%")

# ---------------------------------------------------------------- 2. episodes list (all quads gated)
print("\n2. gated quad episodes (IWM, D z10, h=1,3,5,8,10 and to-quarter-end), FOMC in hold(h=8)")
fomc = load_events(["fomc_decision"])["date"]
rows = []
for q in QUADS:
    D = q - 1
    z = Z["IWM"].iloc[D]
    if np.isnan(z) or z > -1.0:
        continue
    d = cal[q]
    # quarter's last session
    me = cal[(cal.year == d.year) & (cal.month == d.month)][-1]
    hq = pos[me] - q
    rr = {h: ret_from_q("IWM", q, h) for h in (1, 3, 5, 8, 10)}
    rqe = ret_from_q("IWM", q, hq)
    lo, hi = cal[q], cal[min(q + 8, len(cal) - 1)]
    fin = bool(((fomc > lo) & (fomc <= hi)).any())
    rows.append({"quad": str(d.date()), "z10": round(z, 2), "plz": round(ZPL["IWM"].iloc[D], 2),
                 **{f"h{h}": round(100 * v, 2) for h, v in rr.items()},
                 "toQE": round(100 * rqe, 2), "hQE": hq, "fomc_in8": fin,
                 "midterm": d.year % 4 == 2})
show(rows)

# ---------------------------------------------------------------- 3. offset ladder
print("\n3. placebo offset ladder: D shifted by k, gate re-evaluated at shifted D, entry D+k+1")
for fam_name, fam in (("ALL QUADS", QUADS), ("SEP QUAD", SEPQ), ("ALL OPEX", OPEX)):
    for h in (3, 5, 8, 10):
        res = []
        for k in range(-5, 6):
            vals = []
            for q in fam:
                D = q - 1 + k
                if D < 0 or D + 1 + h >= len(cal):
                    continue
                z = Z["IWM"].iloc[D]
                if np.isnan(z) or z > -1.0:
                    continue
                vals.append(px["IWM"].iloc[D + 1 + h] / px["IWM"].iloc[D + 1] - 1)
            res.append((k, len(vals), 100 * np.mean(vals) if vals else np.nan))
        ms = [m for _, _, m in res]
        m0 = res[5][2]
        rank = 1 + sum(1 for m in ms if m > m0)
        print(f"  {fam_name:10s} h={h:2d}: " + " ".join(f"k{k:+d}:{m:+.2f}({n})" for k, n, m in res)
              + f"  -> k=0 rank {rank} of 11")

# ---------------------------------------------------------------- 4. splits on all-quad gated cell
print("\n4. splits, ALL quads gated, IWM")
for h in (3, 5, 8, 10):
    g, c, a = cell_rows("IWM", QUADS, "IWM", -1.0, h, "")
    gv, gd = dvals(g)
    fl = event_in_window(pd.DatetimeIndex([cal[pos[d] - 1] for d in gd]), cal, h, lag=1,
                         kinds=("fomc_decision",))
    mid = np.array([d.year % 4 == 2 for d in gd])
    pre = np.array([d.year < 2018 for d in gd])
    show([stats(gv[fl], None, f"h={h} FOMC in hold"), stats(gv[~fl], None, f"h={h} FOMC out"),
          stats(gv[pre], None, f"h={h} pre-2018"), stats(gv[~pre], None, f"h={h} 2018+"),
          stats(gv[mid], None, f"h={h} midterm"), stats(gv[~mid], None, f"h={h} non-midterm")])
    # the same splits for the ungated all-quad
    av, ad = dvals(a)
    fla = event_in_window(pd.DatetimeIndex([cal[pos[d] - 1] for d in ad]), cal, h, lag=1,
                          kinds=("fomc_decision",))
    print(f"   ungated all quads h={h}: FOMC-in {100*np.nanmean(av[fla]):+.3f}% (n={fla.sum()}) "
          f"FOMC-out {100*np.nanmean(av[~fla]):+.3f}% (n={(~fla).sum()})")

# ---------------------------------------------------------------- 5. other vehicles
print("\n5. rows on other vehicles, own z10 gate and IWM gate, ALL quads")
rows = []
for t in ["SPY", "QQQ", "EEM", "EFA", "IWM"]:
    for h in (5, 8, 10):
        g, c, a = cell_rows(t, QUADS, t, -1.0, h, "")
        gv, _ = dvals(g)
        cv, _ = dvals(c)
        r = stats(gv, None, f"{t} own-gate h={h}")
        r["compl_pct"] = round(100 * np.nanmean(cv), 3) if len(cv) else np.nan
        dr = fwd_lag(px[t], h, 1).dropna()
        r["drift_pct"] = round(100 * dr.mean(), 3)
        rows.append(r)
        g2, c2, _ = cell_rows(t, QUADS, "IWM", -1.0, h, "")
        gv2, _ = dvals(g2)
        rows.append(stats(gv2, None, f"{t} IWM-gate h={h}"))
show(rows)

# ---------------------------------------------------------------- 6. tail risk in window
print("\n6. worst path inside h=8 for gated quads (min cumulative from entry)")
for q in QUADS:
    D = q - 1
    z = Z["IWM"].iloc[D]
    if np.isnan(z) or z > -1.0 or q + 8 >= len(cal):
        continue
    path = px["IWM"].iloc[q + 1:q + 9].values / px["IWM"].iloc[q] - 1
    print(f"  {cal[q].date()} min {100*path.min():+.2f}% final {100*path[-1]:+.2f}%")
