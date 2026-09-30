"""C5B round 2: long GLD when GLD >= 15% under its 252 high and DX 21d rank >= 85.
Decluster + concentration (by signed value), definition neighbours, era/regime,
gate attribution incl. filter_vs_reanchor, GLD's own 21d washout as the rival
explanation, DX reversal as the mechanism test, GC=F translation, live state.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

gp = close_panel(["GLD"]).dropna()
idx = gp.index
g = gp["GLD"]
dxs = close_panel(["DX-Y.NYB"])["DX-Y.NYB"].dropna()
gcs = close_panel(["GC=F"])["GC=F"].dropna()


def dxrank(n: int) -> pd.Series:
    return pct_rank(dxs, n).reindex(idx, method="ffill", limit=2)


dxr = dxrank(21)
gdd = 100 * (1 - g / g.rolling(252).max())
gr21 = pct_rank(g, 21)
above200 = g > g.rolling(200).mean()
gate = (gdd >= 15) & (dxr >= 85)
print("LIVE last 12 sessions:")
print(pd.DataFrame({"GLD": g, "dd": gdd, "dx_r21": dxr, "gld_r21": gr21, "gate": gate,
                    "above200": above200}).tail(12).round(2).to_string())
first_on = gate[gate & ~gate.shift(1, fill_value=False)].index
print("gate switched on (last 3):", [str(d.date()) for d in first_on[-3:]])


def ep(ret, m, gap):
    s = idx[m.reindex(idx, fill_value=False).values & ret.notna().values]
    e = declusters(s, gap, idx)
    return e, ret.loc[e].values


def line(v, lbl, base):
    r = summarize(v, lbl)
    if len(v):
        w = int((v > 0).sum())
        r["rec"] = f"{w}-{len(v)-w}"
        r["sign_p"] = round(sign_test(w, len(v)), 4)
        r["xs_pp"] = round(r["mean_pct"] - 100 * base, 3)
    return r


for h in (5, 10):
    ret = fwd_lag(g, h)
    base = ret.mean()
    print(f"\n==================== h={h}  GLD all-days drift {100*base:+.3f}% ====================")
    # 1. decluster + concentration by signed value
    rows = []
    for gap in (h, 10, 21, 42):
        e, v = ep(ret, gate, gap)
        rows.append(line(v, f"min_gap {gap} (N={len(e)})", base))
    show(rows, "1. decluster stability")
    e, v = ep(ret, gate, h)
    srt = np.sort(v)[::-1]
    print(f"  top-2 by VALUE {100*srt[:2].sum():+.2f}pp of {100*v.sum():+.2f}pp "
          f"({100*srt[:2].sum()/v.sum():.0f}%); drop-best-2 mean {100*srt[2:].mean():+.3f}%, "
          f"drop-best-3 {100*srt[3:].mean():+.3f}%")
    yrs = pd.Series(v, index=e).groupby(e.year).agg(["sum", "count"])
    print("  by year (sum pp / n):", {int(y): (round(100*r['sum'], 2), int(r['count'])) for y, r in yrs.iterrows()})
    ex08 = e.year != 2008
    show([line(v[ex08], f"ex-2008 (N={ex08.sum()})", base),
          line(v[e.year >= 2018], "2018+", base), line(v[e.year >= 2021], "2021+", base),
          line(v[e.year == 2026], "2026 only", base)], "   era cuts")
    # 2. definition neighbours
    rows = []
    for dd in (10, 12.5, 15, 17.5, 20, 25):
        for dr in (80, 85, 90, 95):
            e2, v2 = ep(ret, (gdd >= dd) & (dxr >= dr), h)
            r = line(v2, f"dd>={dd} dx21>={dr}", base)
            rows.append({k: r.get(k) for k in ("label", "n", "mean_pct", "hit", "rec", "sign_p", "xs_pp")})
    show(rows, "2a. dd x DX-rank grid (charged: 24 cells)")
    rows = []
    for n in (10, 15, 42, 63):
        e2, v2 = ep(ret, (gdd >= 15) & (dxrank(n) >= 85), h)
        rows.append(line(v2, f"dd>=15 & DX r{n}>=85 (N={len(e2)})", base))
    uup = close_panel(["UUP"])["UUP"].dropna()
    ur = pct_rank(uup, 21).reindex(idx, method="ffill", limit=2)
    e2, v2 = ep(ret, (gdd >= 15) & (ur >= 85), h)
    rows.append(line(v2, f"dd>=15 & UUP r21>=85 (N={len(e2)})", base))
    show(rows, "2b. dollar lookback / proxy neighbours")
    # dose response in dd bands under the DX gate
    rows = []
    for lo, hi in ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 100)):
        e2, v2 = ep(ret, (gdd >= lo) & (gdd < hi) & (dxr >= 85), h)
        rows.append(line(v2, f"dd [{lo},{hi}) & dx>=85", base))
    show(rows, "2c. dd dose response under DX>=85 (live 20.67)")
    # 3. rival: GLD's own 21d washout
    rows = []
    for lbl, m in (("joint & gld r21<=15", gate & (gr21 <= 15)),
                   ("joint & gld r21>15", gate & (gr21 > 15)),
                   ("gld r21<=15 & dd>=15, no DX gate", (gr21 <= 15) & (gdd >= 15)),
                   ("gld r21<=15 & dx>=85, no dd gate", (gr21 <= 15) & (dxr >= 85)),
                   ("joint & GLD>200d", gate & above200), ("joint & GLD<200d", gate & ~above200)):
        e2, v2 = ep(ret, m, h)
        rows.append(line(v2, f"{lbl} (N={len(e2)})", base))
    show(rows, "3. rival explanations / regime")
    # 4. mechanism: does the dollar revert over the same hold?
    dxf = fwd_lag(dxs, h).reindex(idx)
    e, v = ep(ret, gate, h)
    dv = dxf.reindex(e).values
    ok = ~np.isnan(dv)
    print(f"  DX fwd over hold on episodes: mean {100*np.nanmean(dv):+.3f}% (all-days "
          f"{100*dxf.mean():+.3f}%); corr(GLD, DX fwd) {np.corrcoef(v[ok], dv[ok])[0,1]:+.2f}; "
          f"GLD mean when DX fell {100*v[ok][dv[ok]<0].mean():+.3f}% (n={(dv[ok]<0).sum()}) "
          f"vs DX rose {100*v[ok][dv[ok]>=0].mean():+.3f}% (n={(dv[ok]>=0).sum()})")
    # 5. filter vs re-anchor: parent DX>=85 episodes, child joint
    pe = declusters(idx[(dxr >= 85).values], h, idx)
    ce = declusters(idx[gate.values], h, idx)
    pm = pd.Series(False, index=idx); pm.loc[pe] = True
    cm = pd.Series(False, index=idx); cm.loc[ce] = True
    fr = filter_vs_reanchor(ret, pm, cm, idx, 21, f"parent DX>=85 -> child +dd>=15, h={h}")
    if fr["shifts"]:
        rn = reanchor_null(ret, [a for a, _, _ in fr["pairs"]], fr["shifts"], idx,
                           ret.reindex(ce).mean())
        print("  reanchor_null:", {k: round(v, 4) if isinstance(v, float) else v for k, v in rn.items()})
    # GC=F translation
    gcr = fwd_lag(gcs, h).reindex(idx)
    e, _ = ep(ret, gate, h)
    gv = gcr.reindex(e).dropna().values
    print(f"  GC=F on same episodes: {line(gv, 'GC=F', gcr.mean())}")
