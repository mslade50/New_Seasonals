"""c11 round 1 (2026-09-21): long LQD against beta*IEF (duration-hedged IG
spread proxy) from QE-7 (today's 09-21 close; quarter-end 09-30 at +7) through
QE+10 / QE+15, every quarter; September as the live row.

Mechanism (pre-specified): the pre-earnings issuance blackout dries up IG
primary supply from late in the quarter's last month into early earnings
season, so spreads tighten into the drought and widen when supply resumes.

Hedge: static, beta = trailing-252 OLS slope of LQD daily returns on IEF daily
returns, estimated through the SIGNAL close (entry-1), held fixed over the hold.
Vehicle return per $1 of LQD = r_LQD - beta * r_IEF (entry close -> exit close).

Falsifications here:
  (a) offset ladder over the whole quarter cycle (entry QE+k, k=-60..+5) at the
      same holds; the heavy-issuance windows are inside the ladder
  (b) own drift (every-day entries, same holds)
  (c) era split pre-2008 / 2008-2017 / 2018+ and midterm split
  (d) HYG against beta*IEF (2007+)
  (e) cost (two-leg ETF round trip ~5 bps)
  (f) equity content: regress the window spread return on SPY's window return
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
BAR = pd.Timestamp("2026-09-18")

px = close_panel(["LQD", "IEF", "HYG", "SPY"]).loc[:BAR]
px = px[px["SPY"].notna()]
idx = px.index
ser = pd.Series(range(len(idx)), index=idx)
dr = px / px.shift(1) - 1.0


def trail_beta(y: str, x: str, n: int = 252) -> pd.Series:
    cov = dr[y].rolling(n, min_periods=200).cov(dr[x])
    var = dr[x].rolling(n, min_periods=200).var()
    return cov / var


BETA = {"LQD": trail_beta("LQD", "IEF"), "HYG": trail_beta("HYG", "IEF")}

# quarter-end sessions
qe = [idx[(idx.year == y) & (idx.month == m)][-1]
      for y in range(2002, 2027) for m in (3, 6, 9, 12)
      if len(idx[(idx.year == y) & (idx.month == m)])]
qe = [d for d in qe if d < BAR]
QEP = [int(ser[d]) for d in qe]
print(f"live check: 2026-09-21 -> 2026-09-30 = {len(pd.bdate_range('2026-09-22','2026-09-30'))} sessions (QE-7 entry)")


def spread_ret(leg: str, p0: int, h: int) -> float:
    """entry close p0 -> exit close p0+h, static beta from the signal close p0-1."""
    if p0 - 1 < 0 or p0 + h >= len(idx):
        return np.nan
    b = BETA[leg].iloc[p0 - 1]
    a, c = px[leg], px["IEF"]
    if np.isnan(b) or np.isnan(a.iloc[p0]) or np.isnan(a.iloc[p0 + h]):
        return np.nan
    return (a.iloc[p0 + h] / a.iloc[p0] - 1.0) - b * (c.iloc[p0 + h] / c.iloc[p0] - 1.0)


def plain(t: str, p0: int, h: int) -> float:
    if p0 + h >= len(idx):
        return np.nan
    return px[t].iloc[p0 + h] / px[t].iloc[p0] - 1.0


def cell(leg: str, k: int, h: int, months=None):
    rows = []
    for d, q in zip(qe, QEP):
        if months and d.month not in months:
            continue
        p0 = q + k
        v = spread_ret(leg, p0, h)
        if not np.isnan(v):
            rows.append((idx[p0], v, d.year))
    if not rows:
        return pd.DatetimeIndex([]), np.array([]), np.array([])
    dts, vals, yrs = zip(*rows)
    return pd.DatetimeIndex(dts), np.array(vals), np.array(yrs)


def stat(vals, label, base=None):
    s = summarize(vals, label)
    if s["n"]:
        w = int((vals > 0).sum())
        s["rec"] = f"{w}-{len(vals)-w}"
        s["sign_p"] = round(sign_test(w, len(vals)), 4)
        if base is not None:
            s["edge_pp"] = s["mean_pct"] - 100 * np.nanmean(base)
    return s


def alldays(leg: str, h: int) -> np.ndarray:
    out = [spread_ret(leg, p, h) for p in range(1, len(idx) - h)]
    return np.array([v for v in out if not np.isnan(v)])


print("\nlive hedge ratios at the 09-18 signal close: "
      f"LQD/IEF beta {BETA['LQD'].iloc[-1]:.3f}, HYG/IEF beta {BETA['HYG'].iloc[-1]:.3f}")

for leg in ["LQD", "HYG"]:
    print(f"\n######## {leg} - beta*IEF ########")
    for h in (5, 10, 17, 22):
        base = alldays(leg, h)
        dts, v, yrs = cell(leg, -7, h)
        dS, vS, _ = cell(leg, -7, h, months=[9])
        rows = [stat(base, f"own drift all days h={h}"),
                stat(v, f"QE-7 all quarters h={h}", base),
                stat(vS, f"QE-7 SEPTEMBER only h={h}", base)]
        show(rows, f"{leg} h={h}")
        if h in (17, 22):
            print(f"  concentration (all quarters): {cluster_note(dts, v)}")
            eras = []
            for lo, hi, lbl in [("2000", "2008-01-01", "pre-2008"), ("2008-01-01", "2018-01-01", "2008-2017"),
                                ("2018-01-01", "2030", "2018+")]:
                m = (dts >= lo) & (dts < hi)
                eras.append(stat(v[m], lbl, base))
            eras.append(stat(v[yrs % 4 == 2], "midterm years", base))
            eras.append(stat(v[yrs % 4 != 2], "non-midterm", base))
            show(eras, f"  {leg} h={h} era / cycle split")
            cost = 5.0
            print(f"  cost: episode mean {100*v.mean()*100:.1f} bps vs ~{cost} bps two-leg round trip "
                  f"-> {v.mean()*1e4/cost:.1f}x; edge over drift {(v.mean()-base.mean())*1e4:.1f} bps "
                  f"-> {(v.mean()-base.mean())*1e4/cost:.1f}x")

    # (a) offset ladder over the whole quarter cycle
    for h in (17, 22):
        lad = []
        for k in range(-62, 6):
            dts, v, _ = cell(leg, k, h)
            lad.append({"k": k, "n": len(v), "mean_bps": 1e4 * v.mean(),
                        "hit": 100 * (v > 0).mean()})
        L = pd.DataFrame(lad)
        live = L.loc[L.k == -7].iloc[0]
        rk = int((L.mean_bps > live.mean_bps).sum()) + 1
        print(f"\n  (a) {leg} ladder h={h}: QE-7 = {live.mean_bps:+.1f} bps hit {live.hit:.0f}% "
              f"ranks {rk} of {len(L)} (k=-62..+5); ladder median {L.mean_bps.median():+.1f} bps, "
              f"max {L.mean_bps.max():+.1f} at k={int(L.loc[L.mean_bps.idxmax(),'k'])}, "
              f"min {L.mean_bps.min():+.1f} at k={int(L.loc[L.mean_bps.idxmin(),'k'])}")
        # heavy-issuance comparison windows: first half of the quarter's middle month
        # (~QE-42..-35) and post-earnings weeks (~QE+20..+30, i.e. next quarter's QE-43..-33)
        heavy = L[(L.k >= -45) & (L.k <= -35)].mean_bps.mean()
        print(f"      heavy-issuance entries QE-45..-35 average {heavy:+.1f} bps; blackout-side entries "
              f"QE-10..-4 average {L[(L.k >= -10) & (L.k <= -4)].mean_bps.mean():+.1f} bps")
        # print a compressed ladder
        print("      " + "  ".join(f"{int(r.k)}:{r.mean_bps:+.0f}" for r in L.itertuples() if r.k % 3 == 2 or r.k == -7))

# (f) equity content
print("\n######## (f) equity content of the LQD spread proxy ########")
for h in (17, 22):
    dts, v, _ = cell("LQD", -7, h)
    spy = np.array([plain("SPY", int(ser[d]), h) for d in dts])
    b, a = np.polyfit(spy, v, 1)
    r2 = np.corrcoef(spy, v)[0, 1] ** 2
    print(f"  h={h}: spread = {1e4*a:+.1f} bps + {b:.3f} x SPY  (R2 {r2:.3f}); "
          f"SPY on these windows {100*spy.mean():+.3f}%")
    base_pairs = [(spread_ret("LQD", p, h), plain("SPY", p, h)) for p in range(260, len(idx) - h)]
    bp = np.array([x for x in base_pairs if not np.isnan(x[0]) and not np.isnan(x[1])])
    bb, ba = np.polyfit(bp[:, 1], bp[:, 0], 1)
    print(f"        all-days fit: spread = {1e4*ba:+.1f} bps + {bb:.3f} x SPY; cell residual vs that line "
          f"{1e4*np.mean(v - (ba + bb*spy)):+.1f} bps (t {np.mean(v-(ba+bb*spy))/(np.std(v-(ba+bb*spy),ddof=1)/np.sqrt(len(v))):+.2f})")

# September episode table
print("\n######## September rows, LQD - b*IEF, QE-7 entry ########")
rows = []
for d, q in zip(qe, QEP):
    if d.month != 9:
        continue
    p0 = q - 7
    rows.append({"entry": idx[p0].date(), "beta": round(BETA["LQD"].iloc[p0-1], 3),
                 "h17_bps": round(1e4*spread_ret("LQD", p0, 17), 1),
                 "h22_bps": round(1e4*spread_ret("LQD", p0, 22), 1),
                 "LQD_h17": round(100*plain("LQD", p0, 17), 2),
                 "IEF_h17": round(100*plain("IEF", p0, 17), 2),
                 "HYGsp_h17_bps": round(1e4*spread_ret("HYG", p0, 17), 1) if p0 - 1 > 0 else np.nan,
                 "SPY_h17": round(100*plain("SPY", p0, 17), 2)})
print(pd.DataFrame(rows).to_string(index=False))
