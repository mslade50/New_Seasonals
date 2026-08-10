"""C4 round 3 -- price the cell that ACTUALLY FIRES TODAY.

Correction found in round 2b: the freshest bar is 2026-08-07 and CPI is
2026-08-12.  Sessions: 08-07 (D) -> 08-10 (D+1, today, the MOC entry) ->
08-11 (D+2) -> 08-12 (D+3 = CPI) -> 08-13 (D+4 = PPI).  So D sits THREE
trading days before the print: the live cell is k=3, not the k=2 cell whose
t=2.22 headline came out of round 2.  k=2 is TOMORROW's trade, not today's.

This script prices k=3 on its own terms: horizon scan, era, cycle, LOYO,
concentration, placebos, definition neighbours, entry form, loser paths.
Anything that only works at k=2 is not evidence for what is pitched today.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

LAG, K = 1, 3
P = close_panel(["GDX", "GLD"]).dropna(subset=["GDX"])
g, gl = P["GDX"], P["GLD"]
idx = g.index
rk5 = pct_rank(g, 5)
ASOF = idx[-1]
cpi = load_events(["cpi"])["date"]


def anchor_k(kind: str, k: int) -> pd.DatetimeIndex:
    ev = load_events([kind])["date"]
    out = []
    for d in ev:
        p = int(np.searchsorted(idx.values, np.datetime64(d)))
        if p - k < 0 or p >= len(idx):
            continue
        out.append(idx[p - k])
    return pd.DatetimeIndex(sorted(set(out)))


A3 = anchor_k("cpi", K)
print(f"ASOF {ASOF.date()}; is ASOF a CPI-{K} anchor? {ASOF in A3}")
print(f"GDX rank5 today = {rk5.loc[ASOF]:.1f}")
THR = (rk5 >= 80.0).fillna(False)

# ------------------------------------------------------------- horizon scan
print("\n### horizon scan -- pick h FROM the table, do not assume it ###")
rows = []
for h in (1, 2, 3, 4, 5, 7, 10):
    fw = fwd_lag(g, h, LAG)
    ok = fw.notna()
    t = pd.DatetimeIndex(A3).intersection(idx[THR.values & ok.values])
    e = declusters(t, h, idx[ok.values])
    v = fw.loc[e].values
    r = summarize(v, f"k=3 h={h}")
    r["ctrl_all"] = round(100 * fw[ok].mean(), 3)
    r["edge"] = round(r["mean_pct"] - 100 * fw[ok].mean(), 3)
    r["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
    r["boot_p"] = round(bootstrap_p_le0(v), 3)
    # what the hold covers for today
    r["exit_for_today"] = ["", "08-11", "08-12 CPI", "08-13 PPI", "08-14",
                           "08-17", "08-19", "08-24"][min(h, 7)] if h <= 10 else ""
    rows.append(r)
show(rows, "k=3 horizon scan (entry = today's MOC)")

# ------------------------------------------------------------- pick h and drill
for H in (3, 5):
    print(f"\n{'='*70}\n### DRILL: k=3, rank5>=80, h={H} ###")
    fw = fwd_lag(g, H, LAG)
    ok = fw.notna()
    VALID = idx[ok.values]
    T = pd.DatetimeIndex(A3).intersection(idx[THR.values & ok.values])
    e = declusters(T, H, VALID)
    v = fw.loc[e].values
    print(f"N={len(v)} mean={100*v.mean():+.3f}% med={100*np.median(v):+.3f}% "
          f"hit={100*(v>0).mean():.1f}% t={v.mean()/(v.std(ddof=1)/np.sqrt(len(v))):.2f} "
          f"sign_p={sign_test(int((v>0).sum()), len(v)):.4f} boot_p={bootstrap_p_le0(v):.4f}")
    print(f"  own drift (all days) = {100*fw[ok].mean():+.3f}%  -> edge "
          f"{100*(v.mean()-fw[ok].mean()):+.3f}pp")
    print(f"  ALL CPI-3 anchors (no thrust) = "
          f"{100*fw.loc[pd.DatetimeIndex(A3).intersection(VALID)].mean():+.3f}%")
    print(f"  thrust NOT on CPI-3          = "
          f"{100*fw.loc[declusters(idx[THR.values & ok.values].difference(A3), H, VALID)].mean():+.3f}%")
    print("  concentration:", cluster_note(e, v, k=2))
    o = np.argsort(v)
    print(f"  drop-2-best {100*np.delete(v, o[-2:]).mean():+.3f}%  "
          f"drop-2-worst {100*np.delete(v, o[:2]).mean():+.3f}%")
    print(f"  worst 3:", [(str(pd.Timestamp(e[i]).date()), round(100*v[i], 2)) for i in o[:3]])
    show(era_split(e, v), "era")
    mt = np.array([d.year % 4 == 2 for d in e])
    show([summarize(v[mt], f"MIDTERM (N={int(mt.sum())})"), summarize(v[~mt], "non-midterm")],
         "cycle -- 2026 IS midterm")
    yrs = pd.DatetimeIndex(e).year
    loyo = [(int(y), round(100 * v[yrs.values != y].mean(), 3)) for y in sorted(set(yrs))]
    print("  LOYO:", loyo)
    print("  LOYO min:", min(x[1] for x in loyo), " n_years:", len(loyo),
          " positive years:", int((pd.Series(v).groupby(yrs.values).mean() > 0).sum()),
          "/", len(set(yrs)))
    # gold decomposition
    fwl = fwd_lag(gl, H, LAG)
    okb = fw.notna() & fwl.notna()
    beta = np.polyfit(fwl[okb].values, fw[okb].values, 1)[0]
    resid = fw - beta * fwl
    show([summarize(v, "GDX cell"), summarize(fwl.loc[e].values, "GLD same days"),
          summarize(resid.loc[e].values, f"GDX - {beta:.2f}*GLD residual"),
          summarize(resid[okb].values, "residual all days")],
         f"gold decomposition (beta={beta:.2f})")
    # definition neighbours at THIS k
    rows = []
    for thr in (60.0, 70.0, 80.0, 85.0, 90.0, 95.0):
        m = (rk5 >= thr).fillna(False)
        t2 = pd.DatetimeIndex(A3).intersection(idx[m.values & ok.values])
        e2 = declusters(t2, H, VALID)
        v2 = fw.loc[e2].values
        r = summarize(v2, f"rank5>={thr:.0f}")
        r["sign_p"] = round(sign_test(int((v2 > 0).sum()), len(v2)), 4)
        rows.append(r)
    show(rows, f"definition neighbours at k=3, h={H}")
    # month-position placebo at this k
    dom = pd.Series(0, index=idx)
    cur, lm = 0, None
    for i, d in enumerate(idx):
        if lm != (d.year, d.month):
            cur, lm = 0, (d.year, d.month)
        cur += 1
        dom.iloc[i] = cur
    tp = dom.loc[T]
    lo, hi = int(tp.quantile(.10)), int(tp.quantile(.90))
    matched = pd.DatetimeIndex(idx[THR.values & ok.values & ((dom >= lo) & (dom <= hi)).values]).difference(A3)
    show([summarize(v, "the cell"),
          summarize(fw.loc[declusters(matched, H, VALID)].values,
                    f"month-pos matched bd {lo}-{hi}, thrust, NOT CPI-3")],
         "month-position placebo")
    # disjoint event placebo
    rows = []
    for kind in ("ppi", "nfp", "opex"):
        a = pd.DatetimeIndex(anchor_k(kind, K)).intersection(idx[THR.values & ok.values]).difference(A3)
        e3 = declusters(a, H, VALID)
        rows.append(summarize(fw.loc[e3].values, f"{kind}-{K} ex-CPI x thrust"))
    rows.append(summarize(v, "CPI-3 x thrust"))
    show(rows, "disjoint anchor placebo")

# ------------------------------------------------------------- entry form
print(f"\n{'='*70}\n### entry form: MOC vs close-anchored LIMIT, WHOLE variants ###")
H = 3
px = load_prices(["GDX"])["GDX"]
atr = wilder_atr(px, 14)
fwok = fwd_lag(g, H, LAG).notna()
T = pd.DatetimeIndex(A3).intersection(idx[THR.values & fwok.values])
e = declusters(T, H, idx[fwok.values])
pos = pd.Series(range(len(idx)), index=idx)
for k_atr in (0.0, 0.25, 0.5):
    fills, rets = 0, []
    for d in e:
        p = pos[d]
        if p + 1 + H >= len(idx):
            continue
        anchor = g.iloc[p]           # signal close
        lim = anchor - k_atr * atr.iloc[p]
        entry_day = p + 1            # the MOC session
        if k_atr == 0.0:
            fill = g.iloc[entry_day]
        else:
            lo_ = px["Low"].iloc[entry_day]
            if lo_ > lim:
                continue             # unfilled -> variant simply does not trade
            fill = min(lim, px["Open"].iloc[entry_day])
        fills += 1
        rets.append(g.iloc[entry_day + H] / fill - 1.0)
    rets = np.array(rets)
    r = summarize(rets, f"limit close-{k_atr}ATR (1-day window)" if k_atr else "MOC at D+1")
    r["fill_rate"] = round(100 * fills / len(e), 1)
    show([r], f"entry variant k={k_atr} ATR")

# ------------------------------------------------------------- loser paths
print(f"\n{'='*70}\n### loser paths: where does a bad episode go? (k=3, h=3) ###")
H = 3
fw = fwd_lag(g, H, LAG)
ok = fw.notna()
T = pd.DatetimeIndex(A3).intersection(idx[THR.values & ok.values])
e = declusters(T, H, idx[ok.values])
paths = episode_paths(P, e, [("GDX", 1.0)], H, LAG)
v = fw.loc[e].values
lose = paths.loc[[d for i, d in enumerate(e) if v[i] < 0]]
print(f"losing episodes: {len(lose)}/{len(e)}")
print((100 * lose).round(2).to_string())
print("\nday-1 stats over ALL episodes:", summarize(paths[1].values, "day1"))
print("worst day-1 among losers:", round(100 * lose[1].min(), 2), "%")
print("P(day1 < 0 | episode loses) =",
      round(100 * (lose[1] < 0).mean(), 1), "%")
print("P(episode loses | day1 < 0) =",
      round(100 * (v[paths[1].values < 0] < 0).mean(), 1), "%   n_day1_neg =",
      int((paths[1].values < 0).sum()))
print(f"\ncost: GDX ~2.5 bps/side -> 5 bps rt; edge {100*v.mean():.1f} bps -> "
      f"{100*v.mean()*100/5:.0f}x")
