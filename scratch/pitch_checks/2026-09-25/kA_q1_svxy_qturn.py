"""kA Q1 round 1: long SVXY across the quarter turn, QE-3 close -> QE+1 close
(4 sessions, exits before NFP), charged against SPY at the era beta.
Pre-specified sign: LONG SVXY residual (turn-of-quarter inflows + dealer/bank
balance-sheet reset soften implied vol across the turn).
Entry is the QE-3 close ITSELF (calendar signal known in advance; today's order
is MOC 09-25 = QE-3), so windows are close(QE-3) -> close(QE+1).
Eras: SVXY 2018-03+ (-0.5x) is the tradeable cell; 2011-10..2018-02 reported on a
synthetic -0.5x series (pre-break daily returns halved). ^VIX back to 1990 as the
mechanism check (spot VIX, not tradeable). Controls: ordinary month-ends at the
same offsets, all 4-session windows, and the offset ladder / placebo rank."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

BREAK = pd.Timestamp("2018-02-28")
POST = pd.Timestamp("2018-03-01")
TK = ["SVXY", "SPY", "^VIX", "^VIX3M"]
raw = load_prices(TK)
IDX = raw["SPY"].index
px = close_panel([t for t in TK if t in raw]).reindex(IDX)
r_sv = px["SVXY"].pct_change()
r_adj = r_sv.where(IDX >= BREAK, 0.5 * r_sv)
S = (1 + r_adj.fillna(0)).cumprod()
S[IDX <= px["SVXY"].first_valid_index()] = np.nan
px["S05"] = S
rs = px["SPY"].pct_change()
ok = (IDX >= POST) & rs.notna().values & r_sv.notna().values
B = np.polyfit(rs[ok].values, r_sv[ok].values, 1)[0]
okp = (IDX < BREAK) & (IDX >= pd.Timestamp("2011-10-10")) & rs.notna().values & r_adj.notna().values
BP = np.polyfit(rs[okp].values, r_adj[okp].values, 1)[0]
print(f"daily beta SVXY/SPY 2018-03+ {B:.3f}; synthetic -0.5x pre-break {BP:.3f}")
a = wilder_atr(raw["SVXY"]["High"].to_numpy(), raw["SVXY"]["Low"].to_numpy(), raw["SVXY"]["Close"].to_numpy())
print(f"SVXY {raw['SVXY']['Close'].iloc[-1]:.2f} Wilder-14 ATR {a[-1]:.3f} ({100*a[-1]/raw['SVXY']['Close'].iloc[-1]:.2f}%)")

s = pd.Series(IDX, index=IDX)
me = s.groupby([IDX.year, IDX.month]).max()
me = pd.DatetimeIndex(me.values)
me = me[me < IDX[-1] - pd.Timedelta(days=3)]  # completed months only
qe = me[me.month.isin([3, 6, 9, 12])]
ome = me[~me.month.isin([3, 6, 9, 12])]
pos = pd.Series(range(len(IDX)), index=IDX)
KE, KX = -3, 1


def wret(col, anchors, ke=KE, kx=KX, beta=None):
    out = {}
    c = px[col].values
    spy = px["SPY"].values
    for d in anchors:
        p = pos[d]
        if p + ke < 0 or p + kx >= len(IDX):
            continue
        a0, a1 = c[p + ke], c[p + kx]
        if np.isnan(a0) or np.isnan(a1):
            continue
        r = a1 / a0 - 1
        if beta is not None:
            r -= beta * (spy[p + kx] / spy[p + ke] - 1)
        out[IDX[p + ke]] = r
    return pd.Series(out, dtype=float)


def rr(x, lab):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    r = summarize(x, lab)
    if r["n"]:
        w = int((x > 0).sum())
        r["rec"] = f"{w}-{len(x)-w}"
        r["sign_p"] = round(sign_test(w, len(x)), 4)
    return r


def all_windows(col, beta=None, lo=None, hi=None):
    c = px[col]
    r = c.shift(-4) / c - 1
    if beta is not None:
        r = r - beta * (px["SPY"].shift(-4) / px["SPY"] - 1)
    m = r.notna()
    if lo is not None:
        m &= IDX >= lo
    if hi is not None:
        m &= IDX < hi
    return r[m]


post_q = qe[qe >= POST]
pre_q = qe[(qe < BREAK) & (qe >= pd.Timestamp("2011-11-01"))]
post_m = ome[ome >= POST]
pre_m = ome[(ome < BREAK) & (ome >= pd.Timestamp("2011-11-01"))]
rows = []
for lab, col, beta, qa, ma, lo, hi in [
        ("SVXY resid 2018-03+", "SVXY", B, post_q, post_m, POST, None),
        ("SVXY outright 2018-03+", "SVXY", None, post_q, post_m, POST, None),
        ("SYNTH -0.5x resid 2011-18", "S05", BP, pre_q, pre_m, pd.Timestamp("2011-10-10"), BREAK),
        ("SYNTH -0.5x outright 2011-18", "S05", None, pre_q, pre_m, pd.Timestamp("2011-10-10"), BREAK),
        ("SPY 2018-03+", "SPY", None, post_q, post_m, POST, None)]:
    rows += [rr(wret(col, qa, beta=beta).values, f"{lab} | QUARTER-END"),
             rr(wret(col, ma, beta=beta).values, f"{lab} | ordinary month-end"),
             rr(all_windows(col, beta, lo, hi).values, f"{lab} | ALL 4-session windows")]
show([{k: r.get(k) for k in ("label", "n", "mean_pct", "median_pct", "hit", "t", "rec", "sign_p", "worst_pct", "best_pct")}
      for r in rows], "Q1 QE-3 close -> QE+1 close (pre-specified: LONG, positive = pays)")

# ^VIX mechanism (negative VIX change = pays), full history and eras
vrows = []
for lab, lo, hi in [("1990+", None, None), ("pre-2013", None, pd.Timestamp("2013-01-01")),
                    ("2013+", pd.Timestamp("2013-01-01"), None), ("2018-03+", POST, None)]:
    def cut(a):
        m = np.ones(len(a), dtype=bool)
        if lo is not None:
            m &= a >= lo
        if hi is not None:
            m &= a < hi
        return a[m]
    qa, ma = cut(qe), cut(ome)
    vrows += [rr(-wret("^VIX", qa).values, f"-dVIX% {lab} | QUARTER-END"),
              rr(-wret("^VIX", ma).values, f"-dVIX% {lab} | ordinary ME"),
              rr(-all_windows("^VIX", None, lo, hi).values, f"-dVIX% {lab} | ALL windows")]
show([{k: r.get(k) for k in ("label", "n", "mean_pct", "median_pct", "hit", "t", "rec", "sign_p")} for r in vrows],
     "^VIX change across the turn, sign flipped so positive = vol softens (pays)")
if "^VIX3M" in px:
    ts = px["^VIX"] / px["^VIX3M"]
    px["TS"] = ts
    q7 = qe[qe >= pd.Timestamp("2008-01-01")]
    m7 = ome[ome >= pd.Timestamp("2008-01-01")]

    def dlev(anchors):
        out = []
        for d in anchors:
            p = pos[d]
            if p + KX < len(IDX):
                out.append(ts.iloc[p + KX] - ts.iloc[p + KE])
        return np.array(out)
    dq, dm = dlev(q7), dlev(m7)
    dall = (ts.shift(-4) - ts).dropna()
    print(f"\nVIX/VIX3M change QE-3->QE+1: quarter {np.nanmean(dq):+.4f} (N={np.isfinite(dq).sum()}, "
          f"down {100*np.nanmean(dq < 0):.0f}%), ordinary ME {np.nanmean(dm):+.4f}, all windows {dall.mean():+.4f}")

# episode table 2018-03+
res = wret("SVXY", post_q, beta=B)
spy = wret("SPY", post_q)
sv = wret("SVXY", post_q)
vx = wret("^VIX", post_q)
print("\n2018-03+ quarter-turn episodes (entry QE-3 date): resid / SVXY / SPY / dVIX %")
for d in res.index:
    print(f"  {d.date()}  resid {100*res[d]:+.2f}  SVXY {100*sv[d]:+.2f}  SPY {100*spy[d]:+.2f}  VIX {100*vx[d]:+.1f}")
print("  ", cluster_note(res.index, res.values))
show(era_split(res.index, res.values, "2022-01-01"), "resid era split inside 2018-03+ (2018-21 / 2022+)")
sep = res.index.month == 9
show([rr(res.values[sep], "resid September QE"), rr(res.values[~sep], "resid other QE")], "September vs other")
mid = np.isin(res.index.year, [2018, 2022, 2026])
show([rr(res.values[mid], "resid midterm years"), rr(res.values[~mid], "resid other years")], "midterm split")
