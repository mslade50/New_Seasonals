"""C8 round 2 — definition neighbours, concentration, regime split, and the
measured collision with the 2026-08-11 QQQ/SPY pitch.

Round 1 (c3) already showed the naked SPY leg beats the pair at every horizon
on every trigger form. This walks the definition so the kill is not a single
parameterisation, splits the eras and the cycle, and measures the registry's
"index pairs are not interchangeable re-skins" claim on TODAY's pair.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BAR = pd.Timestamp("2026-08-21")
SECT9 = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
NAMES = sorted(set(["SPY", "QQQ", "IWM", "DIA", "XLV"] + SECT9))
px_all = load_prices(NAMES)
spy = px_all["SPY"]["Close"].dropna()
CAL = spy.index[spy.index <= BAR]
px = pd.DataFrame({t: px_all[t]["Close"] for t in NAMES}).reindex(CAL)


def d52(t):
    c = px_all[t]["Close"].dropna()
    c = c[c.index <= BAR]
    return (c / c.rolling(252).max() - 1.0).reindex(CAL)


spy_d, qqq_d, xlk_d = d52("SPY"), d52("QQQ"), d52("XLK")
gap = spy_d - qqq_d
r63_spy = pct_rank(px_all["SPY"]["Close"].dropna(), 63).reindex(CAL)
r63_qqq = pct_rank(px_all["QQQ"]["Close"].dropna(), 63).reindex(CAL)
cnt = pd.Series(0.0, index=CAL)
for s in [x for x in SECT9 if x != "XLK"]:
    dd = d52(s)
    cnt += (dd >= -0.0025).fillna(False).astype(float)

PAIR = [("SPY", 1.0), ("QQQ", -1.0)]
GAP_LIVE = float(gap.iloc[-1])
d_spy, d_qqq = px["SPY"].pct_change(), px["QQQ"].pct_change()
ok = d_spy.notna() & d_qqq.notna()
BETA = float(np.polyfit(d_qqq[ok], d_spy[ok], 1)[0])


def epi_row(mask, h, label, legs=PAIR, min_gap=10):
    ret = vehicle_ret(px, legs, h, 1)
    valid = ret.dropna().index
    t = CAL[mask.reindex(CAL, fill_value=False).values].intersection(valid)
    if len(t) == 0:
        return {"label": label, "n": 0}
    epi = declusters(t, min_gap, valid)
    r = summarize(ret.loc[epi].values, label)
    r["n_days"] = len(t)
    r["edge_vs_alldays_pp"] = round(r["mean_pct"] - 100 * ret.loc[valid].mean(), 3)
    r["x_cost"] = round(r["mean_pct"] * 100 / 4.0, 1)
    return r


print("=" * 100)
print("A. DEFINITION NEIGHBOURS — four independent directions, pair h=5")
print("=" * 100)
print(f"  (all-days pair drift h=5 = {100*vehicle_ret(px, PAIR, 5, 1).mean():+.3f}%; "
      f"cost 4.0 bps; today gap {100*GAP_LIVE:+.2f}pp)")

rows = []
for g in (0.005, 0.01, 0.02, GAP_LIVE, 0.04, 0.06, 0.08):
    rows.append(epi_row((gap >= g - 1e-9) & (spy_d > -0.03), 5,
                        f"gap>={100*g:.2f}pp & SPY>-3%"))
show(rows, "A1. the SPY-minus-QQQ 52w-high gap threshold")

rows = []
for lo in (-0.01, -0.02, -0.03, -0.05, -0.10, -1.0):
    rows.append(epi_row((gap >= GAP_LIVE - 1e-9) & (spy_d > lo), 5,
                        f"gap>=2.72pp & SPY>{100*lo:.0f}%"))
show(rows, "A2. how near its high the index must be")

rows = []
for x in (-0.01, -0.03, -0.05, -0.08, -0.12):
    rows.append(epi_row((xlk_d <= x) & (cnt >= 2) & (spy_d > -0.03), 5,
                        f"XLK<={100*x:.0f}% off high & >=2 sectors at high"))
show(rows, "A3. how far tech must be off its own high")

rows = []
for k in (1, 2, 3, 4):
    rows.append(epi_row((xlk_d <= -0.03) & (cnt >= k) & (spy_d > -0.03), 5,
                        f"XLK<=-3% & >={k} of other-8 at a high"))
show(rows, "A4. how much of the rest of the tape must be at a high")

rows = []
for lo, hi in ((10, 20), (0, 20), (0, 30), (0, 40)):
    for sp in (25, 40):
        rows.append(epi_row((r63_qqq <= hi) & (r63_qqq >= lo) & (r63_spy >= sp), 5,
                            f"r63 QQQ in [{lo},{hi}] & SPY>={sp}"))
show(rows, "A5. the rank form")

print("\n" + "=" * 100)
print("B. HORIZON x TRIGGER: sign stability of the PAIR and of the beta-neutral residual")
print("=" * 100)
TRIG = {
    "A gap>=2.72pp & SPY>-3%": (gap >= GAP_LIVE - 1e-9) & (spy_d > -0.03),
    "B r63 QQQ<=20 & SPY>=25": (r63_qqq <= 20) & (r63_spy >= 25),
    "C XLK<=-3% & >=2 sectors at high & SPY>-3%": (xlk_d <= -0.03) & (cnt >= 2) & (spy_d > -0.03),
}
for lbl, m in TRIG.items():
    out = []
    for h in (1, 2, 3, 4, 5, 7, 10):
        a = epi_row(m, h, "pair")
        b = epi_row(m, h, "resid", legs=[("SPY", 1.0), ("QQQ", -BETA)])
        out.append({"h": h, "n_epi": a.get("n"),
                    "pair_pct": round(a.get("mean_pct", np.nan), 3),
                    "pair_hit": round(a.get("hit", np.nan), 1),
                    "pair_xcost": a.get("x_cost"),
                    f"resid(SPY-{BETA:.2f}QQQ)_pct": round(b.get("mean_pct", np.nan), 3),
                    "resid_edge_pp": b.get("edge_vs_alldays_pp")})
    print(f"\n  --- {lbl}")
    print(pd.DataFrame(out).to_string(index=False))

print("\n" + "=" * 100)
print("C. CONCENTRATION + DROP-TWO + CYCLE, on the two forms that are live today")
print("=" * 100)
for lbl in ("A gap>=2.72pp & SPY>-3%", "C XLK<=-3% & >=2 sectors at high & SPY>-3%"):
    m = TRIG[lbl]
    for h in (5, 10):
        ret = vehicle_ret(px, PAIR, h, 1)
        valid = ret.dropna().index
        t = CAL[m.reindex(CAL, fill_value=False).values].intersection(valid)
        epi = declusters(t, 10, valid)
        v = ret.loc[epi].values
        print(f"\n  --- {lbl} | h={h} | N_epi={len(epi)}")
        print(f"      {cluster_note(epi, v)}")
        order = np.argsort(-np.abs(v))
        keep = np.ones(len(v), bool)
        keep[order[:2]] = False
        mid = np.array([d.year % 4 == 2 for d in epi])
        show([summarize(v, "all"), summarize(v[keep], "drop-top-2 by |R|"),
              summarize(v[mid], f"MIDTERM (N={int(mid.sum())})"),
              summarize(v[~mid], f"non-midterm (N={int((~mid).sum())})")], "")

print("\n" + "=" * 100)
print("D. VEHICLE SWAP: is SPY-vs-QQQ even the best expression of 'not tech'?")
print("=" * 100)
m = TRIG["C XLK<=-3% & >=2 sectors at high & SPY>-3%"]
rows = []
for name, legs, cost in (("SPY - QQQ", PAIR, 4.0),
                         ("DIA - QQQ", [("DIA", 1.0), ("QQQ", -1.0)], 5.0),
                         ("IWM - QQQ", [("IWM", 1.0), ("QQQ", -1.0)], 5.0),
                         ("XLV - XLK", [("XLV", 1.0), ("XLK", -1.0)], 8.0),
                         ("SPY naked long", [("SPY", 1.0)], 1.5),
                         ("QQQ naked long", [("QQQ", 1.0)], 2.0)):
    r = epi_row(m, 5, name, legs=legs)
    r["x_cost"] = round(r.get("mean_pct", np.nan) * 100 / cost, 1)
    rows.append(r)
show(rows, "trigger C, h=5, episode level")

print("\n" + "=" * 100)
print("E. REGISTRY COLLISION, measured: correlation with the DIA/SPY residual")
print("=" * 100)
print("  2026-08-13 registry: the DIA/SPY residual is -0.363 to -0.442 correlated")
print("  with the 2026-08-11 QQQ/SPY pitch at h=1/3/5. C8 is the OPPOSITE side of")
print("  that pitch, so it should be POSITIVELY correlated with DIA/SPY:")
for h in (1, 3, 5, 10):
    c8 = vehicle_ret(px, PAIR, h, 1)
    beta_ds = float(np.polyfit(px["SPY"].pct_change()[ok], px["DIA"].pct_change()[ok], 1)[0])
    dia_resid = vehicle_ret(px, [("DIA", 1.0), ("SPY", -beta_ds)], h, 1)
    j = c8.notna() & dia_resid.notna()
    print(f"    h={h:2d}: corr(C8 pair, DIA-{beta_ds:.2f}*SPY residual) = "
          f"{c8[j].corr(dia_resid[j]):+.3f}")
print("\n  The 2026-08-11 idea (QQQ long / SPY short, MOC, h<=10) graded B and")
print("  realized +0.099R. C8 is that trade with the sign reversed, 9 trading")
print("  days later. Its own fingerprint does NOT collide (side is in the hash),")
print("  but the structural object is identical.")
