"""kD round 1: c10 LONG dollar MOC on the FOMC decision close, h=3.

Search byproduct of kA's c5 (short dollar after the decision close, killed
wrong-signed). Alignment reuses kA: returns on the vehicle's OWN index, entry
close on decision session D (lag=0 on D == pitch_lab eve anchor + lag=1),
exit close D+h.

Controls: own drift same span, all days, local +/-126td ex-trigger, and the
mandatory tdom-matched control (same trading-day-of-month bucket, FOMC
windows p-5..p+5 removed), plus month+tdom matched.

Regime proxies (stated):
  regime126  = kA's classifier on the EVE: ^IRX < 0.30 -> zirp, else 126td
               ^IRX change > +25bp hike / < -25bp cut / else flat
  decision   = ^IRX change from D-15 to D+15 td: > +15bp hike, < -15bp cut,
               else hold (zirp eve -> hold). Bills price ahead; this is a
               'direction of the policy window' proxy, not the actual vote.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

H = 3
pxd = load_prices(["DX-Y.NYB", "UUP", "^IRX"])
irx = pxd["^IRX"]["Close"].dropna()
fomc = load_events(["fomc_decision"])["date"]
fomc = pd.DatetimeIndex(fomc[fomc <= pd.Timestamp("2026-09-15")])


def tdom_of(idx):
    ym = pd.Series(idx.year * 100 + idx.month, index=idx)
    return ym.groupby(ym.values).cumcount().values + 1


def classify(d):
    e = irx.index[irx.index < d][-1]
    ir = irx[e]
    ic = irx - irx.shift(126)
    if ir < 0.30:
        r126 = "zirp"
    elif ic[e] > 0.25:
        r126 = "hike"
    elif ic[e] < -0.25:
        r126 = "cut"
    else:
        r126 = "flat"
    p = irx.index.get_loc(e) + 1
    lo, hi = max(0, p - 15), min(len(irx) - 1, p + 15)
    ch = irx.iloc[hi] - irx.iloc[lo]
    dec = "hold" if ir < 0.30 else ("hike" if ch > 0.15 else ("cut" if ch < -0.15 else "hold"))
    return r126, dec


CLS = {d: classify(d) for d in fomc}


def build(tkr, h, w=1.0):
    s = pxd[tkr]["Close"].dropna()
    idx = s.index
    td = tdom_of(idx)
    mon = idx.month.values
    pos = pd.Series(np.arange(len(idx)), index=idx)
    miss = [d for d in fomc if d >= idx[0] and d not in pos.index]
    dd = [d for d in fomc if d in pos.index]
    dp = np.array([pos[d] for d in dd])
    excl = np.zeros(len(idx), bool)
    for p in dp:
        excl[max(0, p - 5):p + 6] = True
    r = (w * (s.shift(-h) / s - 1.0)).values
    ok = ~np.isnan(r)
    bucket = {j: np.nanmean(r[(td == j) & ~excl & ok]) for j in np.unique(td)}
    keep = [i for i, p in enumerate(dp) if ok[p]]
    dd = [dd[i] for i in keep]
    dp = dp[keep]
    v = r[dp]
    x = np.array([r[p] - bucket[td[p]] for p in dp])
    xm = []
    for p in dp:
        m = (mon == mon[p]) & (td == td[p]) & ~excl & ok
        xm.append(r[p] - np.nanmean(r[m]) if m.sum() >= 8 else np.nan)
    span = (idx >= dd[0]) & (idx <= dd[-1]) & ok
    loc = local_control(idx[ok], pd.DatetimeIndex(dd))
    ctl = {"own_drift_span": 100 * np.nanmean(r[span]), "all_days": 100 * np.nanmean(r[ok]),
           "local126": 100 * np.nanmean(pd.Series(r, index=idx).loc[loc].values),
           "all_days_exFOMC": 100 * np.nanmean(r[ok & ~excl])}
    return pd.DatetimeIndex(dd), v, x, np.array(xm), ctl, miss


def line(lbl, v, x=None):
    wn = int((v > 0).sum())
    o = summarize(v, lbl)
    o["sign_p"] = sign_test(wn, len(v))
    if x is not None and len(x):
        we = int((x > 0).sum())
        o["tdomX"] = 100 * x.mean()
        o["X_t"] = x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 else np.nan
        o["X_rec"] = f"{we}-{len(x) - we}"
        o["X_sign_p"] = sign_test(we, len(x))
    return o


for tkr in ("DX-Y.NYB", "UUP"):
    dd, v, x, xm, ctl, miss = build(tkr, H)
    print(f"\n{'=' * 100}\n{tkr} LONG, entry decision close D, h={H}\n{'=' * 100}")
    print("decisions missing from index:", [str(m.date()) for m in miss])
    show([line(f"COND N={len(v)}", v, x)], "1. existence")
    print("  controls (mean %):", {k: round(val, 4) for k, val in ctl.items()})
    print(f"  tdom-matched excess {100 * x.mean():+.3f}%  month+tdom excess "
          f"{100 * np.nanmean(xm):+.3f}%  bootstrap P(mean<=0) raw {bootstrap_p_le0(v):.4f} "
          f"tdomX {bootstrap_p_le0(x):.4f}")
    print("  " + cluster_note(dd, v))
    print(f"  worst window {100 * v.min():.2f}% on {dd[int(np.argmin(v))].date()}")
    yrs = dd.year.values
    eras = [("pre-2008", yrs < 2008), ("2008-2017", (yrs >= 2008) & (yrs < 2018)),
            ("2018+", yrs >= 2018)]
    show([line(n, v[m], x[m]) for n, m in eras if m.any()], "2. era split")
    show([line("midterm", v[yrs % 4 == 2], x[yrs % 4 == 2]),
          line("non-midterm", v[yrs % 4 != 2], x[yrs % 4 != 2])], "3. midterm")
    r126 = np.array([CLS[d][0] for d in dd])
    dec = np.array([CLS[d][1] for d in dd])
    show([line(f"regime126={g}", v[r126 == g], x[r126 == g]) for g in ("hike", "cut", "flat", "zirp")
          if (r126 == g).any()], "4a. regime (126d ^IRX on eve)")
    show([line(f"decision={g}", v[dec == g], x[dec == g]) for g in ("hike", "cut", "hold")
          if (dec == g).any()], "4b. decision proxy (^IRX D-15..D+15)")
    sep = dd.month.values == 9
    show([line("September decisions", v[sep], x[sep]), line("SEP meetings (*)", v[np.array(
        ["*" in str(load_events(['fomc_decision']).set_index('date').loc[d, 'detail']) for d in dd])],
        None)], "5. September / projection meetings")
    if tkr == "DX-Y.NYB":
        dx_edge_bp = 100 * 100 * x.mean()
        print(f"\n6. cost: DX futures ~1bp RT; tdomX {dx_edge_bp:.1f}bp raw {1e4 * v.mean():.1f}bp "
              f"-> {dx_edge_bp / 1.0:.1f}x cost")
    else:
        print(f"\n6. UUP own prices: raw {1e4 * v.mean():.1f}bp, own drift h3 {100 * ctl['own_drift_span']:.1f}bp, "
              f"tdomX {1e4 * x.mean():.1f}bp; UUP cost ~ spread 1bp x2 + ER 0.77%/yr x 3/252 = "
              f"{2 + 0.0077 * 3 / 252 * 1e4:.1f}bp (ER already in prices, spread not)")
    ew = event_in_window(dd, pxd[tkr]["Close"].dropna().index, H, lag=0,
                         kinds=("quad_witching",))
    show([line(f"quad witch IN hold N={int(ew.sum())}", v[ew], x[ew]),
          line(f"quad witch OUT N={int((~ew).sum())}", v[~ew], x[~ew])], "7. quad witching in hold")
    ew2 = event_in_window(dd, pxd[tkr]["Close"].dropna().index, H, lag=0, kinds=("cpi", "nfp"))
    show([line(f"CPI/NFP IN hold N={int(ew2.sum())}", v[ew2], x[ew2]),
          line(f"CPI/NFP OUT N={int((~ew2).sum())}", v[~ew2], x[~ew2])], "8. US data print in hold")
    if tkr == "DX-Y.NYB":
        DX = (dd, v, x)

ev = load_events(None)
print("\nevents 2026-09-16..2026-09-22:\n",
      ev.query("date >= '2026-09-16' and date <= '2026-09-22'")[["date", "event", "detail"]].to_string())

# close-timing sanity: DX-Y.NYB vs UUP same-day vs lagged daily correlation
a = pxd["DX-Y.NYB"]["Close"].pct_change()
b = pxd["UUP"]["Close"].pct_change()
j = pd.concat([a, b], axis=1, keys=["dx", "uup"]).dropna()
print(f"\nclose timing: corr(dx_t, uup_t) {j.dx.corr(j.uup):.3f}  corr(dx_t, uup_t+1) "
      f"{j.dx.corr(j.uup.shift(-1)):.3f}  corr(dx_t+1, uup_t) {j.dx.shift(-1).corr(j.uup):.3f}")
