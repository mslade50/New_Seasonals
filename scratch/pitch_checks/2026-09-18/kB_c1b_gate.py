"""c1 round 2 - the washout-into-opex long on IWM.

Round 1 (kB_c1_r1.py): all quads gated z10<=-1 go 10-1 at h=8 (+3.655%) vs the
complement -0.003%; all opex gated 30-10 (+2.200%); offset ladder k=0 rank 1.
But the live anchor is a SEPTEMBER quad (gated N=1, ungated 7-19 at -1.602%)
and today's washout is IWM-only (SPY z10 -0.18, ^VIX 15.44 after a crush).

Round 2 attacks:
  A. per-episode state table (SPY z10, ^VIX, IWM vs 200d, below 52w high)
  B. regime splits: broad washout (SPY z10<=-1) vs IWM-only; VIX>=20 vs <20;
     IWM above/below its 200d; z10 dose bins
  C. definition neighbours: threshold -0.75/-1.25/-1.5, z5, pitch_lab.zscore,
     10d-return rank <= 10, pre-window vol (vol21 at D-10)
  D. concentration: signed top-2, drop-best-2, year shares, drop famous lows
  E. September within-month dose response (all 26 Sep quads)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

TK = ["IWM", "SPY", "^VIX"]
px = close_panel(TK)
cal = px["SPY"].dropna().index
px = px.reindex(cal)
px["^VIX"] = px["^VIX"].ffill(limit=2)
pos = pd.Series(range(len(cal)), index=cal)


def z_sleeve(s, n=10, vol_lag=0):
    c = s.dropna()
    vol21 = c.pct_change().rolling(21).std().shift(vol_lag)
    return (c.pct_change(n) / (vol21 * np.sqrt(n))).reindex(s.index)


Z = z_sleeve(px["IWM"])
ZS = z_sleeve(px["SPY"])
Z5 = z_sleeve(px["IWM"], 5)
ZPRE = z_sleeve(px["IWM"], 10, vol_lag=10)
ZPL = zscore(px["IWM"], 10)
R10 = pct_rank(px["IWM"], 10)
SMA200 = px["IWM"].rolling(200).mean()
HI252 = px["IWM"].rolling(252).max()


def expiry_session(d):
    loc = int(cal.searchsorted(d))
    if loc < len(cal) and cal[loc] == d:
        return loc
    return loc - 1


def anchors(kind):
    ev = load_events([kind])["date"]
    return sorted({expiry_session(d) for d in ev if cal[0] <= d <= cal[-1]})


QUADS = anchors("quad_witching")
OPEX = anchors("opex")


def r_q(q, h, t="IWM"):
    return np.nan if q + h >= len(cal) else px[t].iloc[q + h] / px[t].iloc[q] - 1


def stats(v, label):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    r = summarize(v, label)
    if r["n"]:
        w = int((v > 0).sum())
        r["rec"] = f"{w}-{len(v)-w}"
        r["sign_p"] = round(sign_test(w, len(v)), 4)
        tot = v.sum()
        srt = np.sort(v)[::-1]
        r["top2_sh"] = round(100 * srt[:2].sum() / tot, 0) if tot > 0 else np.nan
        r["drop2"] = round(100 * srt[2:].mean(), 3) if len(v) > 2 else np.nan
    return r


# live state
d0 = pd.Timestamp("2026-09-17")
print(f"LIVE D=2026-09-17: IWM z10 {Z[d0]:+.3f} z5 {Z5[d0]:+.3f} zpre {ZPRE[d0]:+.3f} "
      f"pl {ZPL[d0]:+.3f} r10 {R10[d0]:.1f} | SPY z10 {ZS[d0]:+.3f} | VIX {px['^VIX'][d0]:.2f} | "
      f"IWM vs 200d {100*(px['IWM'][d0]/SMA200[d0]-1):+.2f}% | below 252 hi {100*(1-px['IWM'][d0]/HI252[d0]):.2f}%")

# ------------------------------------------------------------- A. state table
H = 8
rows = []
for fam, qs in (("quad", QUADS), ("opex", OPEX)):
    for q in qs:
        D = q - 1
        if np.isnan(Z.iloc[D]) or Z.iloc[D] > -1.0:
            continue
        if fam == "opex" and q in QUADS:
            continue
        rows.append({"fam": fam, "exp": cal[q].date(), "z10": round(Z.iloc[D], 2),
                     "spy_z": round(ZS.iloc[D], 2), "vix": round(px["^VIX"].iloc[D], 1),
                     "vs200": round(100 * (px["IWM"].iloc[D] / SMA200.iloc[D] - 1), 1),
                     "offhi": round(100 * (1 - px["IWM"].iloc[D] / HI252.iloc[D]), 1),
                     "h5": round(100 * r_q(q, 5), 2), "h8": round(100 * r_q(q, 8), 2),
                     "h10": round(100 * r_q(q, 10), 2)})
T = pd.DataFrame(rows)
print("\nA. gated episodes, quad first then monthly opex")
print(T.to_string(index=False))

# ------------------------------------------------------------- B. regime splits
print("\nB. regime splits of the gated cells (h=5/8/10)")
for fam in ("quad", "all"):
    sub = T if fam == "all" else T[T.fam == "quad"]
    for h in ("h5", "h8", "h10"):
        v = sub[h].values / 100
        splits = {
            "broad SPY z<=-1": sub.spy_z <= -1, "IWM-only SPY z>-1": sub.spy_z > -1,
            "SPY z > -0.5 (today -0.18)": sub.spy_z > -0.5,
            "VIX>=20": sub.vix >= 20, "VIX<20": sub.vix < 20,
            "IWM above 200d": sub.vs200 > 0, "IWM below 200d": sub.vs200 <= 0,
            "z in (-1.25,-1]": sub.z10 > -1.25, "z in (-1.5,-1.25]": (sub.z10 <= -1.25) & (sub.z10 > -1.5),
            "z <= -1.5": sub.z10 <= -1.5,
            "offhi < 10%": sub.offhi < 10,
            "LIVE-LIKE: SPY z>-0.5 & VIX<20 & above 200d": (sub.spy_z > -0.5) & (sub.vix < 20) & (sub.vs200 > 0),
        }
        show([stats(v[m.values], f"{fam} {h} {k} (n={int(m.sum())})") for k, m in splits.items()])

# ------------------------------------------------------------- C. neighbours
print("\nC. definition neighbours (h=8), all quads and all opex; complement in brackets")
gates = {
    "z10<=-0.75": Z <= -0.75, "z10<=-1.0": Z <= -1.0, "z10<=-1.25": Z <= -1.25,
    "z10<=-1.5": Z <= -1.5, "z5<=-1": Z5 <= -1, "zpre(vol@D-10)<=-1": ZPRE <= -1,
    "pitch_lab z<=-1": ZPL <= -1, "r10 rank<=10": R10 <= 10, "r10 rank<=20": R10 <= 20,
}
for fam, qs in (("quads", QUADS), ("opex", OPEX)):
    rows = []
    for gname, g in gates.items():
        for h in (5, 8, 10):
            vin = [r_q(q, h) for q in qs if g.iloc[q - 1]]
            vout = [r_q(q, h) for q in qs if not g.iloc[q - 1]]
            r = stats(vin, f"{fam} {gname} h={h}")
            r["compl"] = round(100 * np.nanmean(vout), 3)
            rows.append(r)
    show(rows)

# ------------------------------------------------------------- D. concentration
print("\nD. concentration, all quads gated z10<=-1")
for h in (5, 8, 10):
    q_in = [q for q in QUADS if Z.iloc[q - 1] <= -1 and q + h < len(cal)]
    v = np.array([r_q(q, h) for q in q_in])
    yrs = pd.Series(v, index=[cal[q].year for q in q_in]).groupby(level=0).sum()
    print(f"  h={h}: total {100*v.sum():+.2f}pp, best year {yrs.idxmax()} {100*yrs.max():+.2f}pp "
          f"({100*yrs.max()/v.sum():.0f}%)")
    famous = {"2001-09-21", "2020-03-20", "2018-12-21"}
    keep = [q for q in q_in if str(cal[q].date()) not in famous]
    vk = np.array([r_q(q, h) for q in keep])
    print(f"     drop 2001-09/2018-12/2020-03 famous lows: {stats(vk, '')}")

# ------------------------------------------------------------- E. September dose response
print("\nE. September quads: z10 at D vs forward h (all 26), spearman + bottom-6 list")
sq = [q for q in QUADS if cal[q].month == 9 and q + 10 < len(cal)]
zz = np.array([Z.iloc[q - 1] for q in sq])
for h in (5, 8, 10):
    f = np.array([r_q(q, h) for q in sq])
    rho, p = spearmanr(zz, f)
    order = np.argsort(zz)
    b6 = f[order[:6]]
    rest = f[order[6:]]
    print(f"  h={h}: rho {rho:+.3f} (p {p:.3f}); 6 most washed Sept {100*b6.mean():+.3f}% "
          f"({int((b6>0).sum())}-{int((b6<=0).sum())}) vs other 20 {100*rest.mean():+.3f}%")
for i in np.argsort(zz)[:8]:
    q = sq[i]
    print(f"   {cal[q].date()} z10 {zz[i]:+.2f} spy_z {ZS.iloc[q-1]:+.2f} vix {px['^VIX'].iloc[q-1]:.1f} "
          f"h5 {100*r_q(q,5):+.2f} h8 {100*r_q(q,8):+.2f} h10 {100*r_q(q,10):+.2f}")

# same within each quad month: dose response
print("\nE2. spearman z10(D) vs h=8, by quad month and pooled")
for m in (3, 6, 9, 12, None):
    qs = [q for q in QUADS if (m is None or cal[q].month == m) and q + 8 < len(cal)]
    zz = np.array([Z.iloc[q - 1] for q in qs])
    f = np.array([r_q(q, 8) for q in qs])
    ok = ~np.isnan(zz)
    rho, p = spearmanr(zz[ok], f[ok])
    print(f"  month {m}: n={ok.sum()} rho {rho:+.3f} p {p:.3f}")

# the IWM-SPY relative washout at quads: IWM z10 <= -1 AND SPY z10 > -0.5
print("\nF. relative washout at quads/opex: IWM z<=-1 & SPY z>-0.5 ; and IWM-SPY spread h=8")
for fam, qs in (("quads", QUADS), ("opex", OPEX)):
    sel = [q for q in qs if Z.iloc[q - 1] <= -1 and ZS.iloc[q - 1] > -0.5 and q + 8 < len(cal)]
    for q in sel:
        print(f"  {fam} {cal[q].date()} IWM z {Z.iloc[q-1]:+.2f} SPY z {ZS.iloc[q-1]:+.2f} "
              f"IWM h8 {100*r_q(q,8):+.2f} SPY h8 {100*r_q(q,8,'SPY'):+.2f}")
