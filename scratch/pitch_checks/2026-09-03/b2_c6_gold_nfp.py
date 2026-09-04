"""C6 KILL CHECK -- long gold into the payrolls print after a 5-day metal washout
with the miners still bid.

Live: GLD r5 = 9.1 (-4.40%), GDX r21 = 94.4, NEM +28.07%/21d, and GLD is
-18.78% below its 52-week high.

Order of attack is dictated by the registry: the 2026-08-21 GLD teardown says
the metal's OWN DRAWDOWN STATE is load-bearing, so that split runs FIRST on
this cell's own parent. If the live drawdown bucket is wrong-signed, stop.

  1. GLD drawdown split on the washout parent  <- run first, most likely kill
  2. count the joint state (washout x miners-bid) and its NFP cross
  3. battery on whatever is countable, vs GLD's own drift
  4. gate attribution: does the miner-bid leg filter?
  5. midterm split + era split
  6. beta of GDX on GLD (reported so no pair is credited without it)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import (load_prices, load_events, anchor_positions, battery,
                       summarize, show, sign_test, declusters, pct_rank,
                       rolling_on_valid, bootstrap_p_le0)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)

TK = ["GLD", "GDX", "NEM", "SLV", "SPY"]
raw = load_prices(TK)
px = pd.DataFrame({t: raw[t]["Close"] for t in TK}).dropna(subset=["GLD"])
cal = px.index
gld = px["GLD"]
r5 = pct_rank(gld, 5)
gdx_r21 = pct_rank(px["GDX"], 21).ffill(limit=3)
nem_r21 = pct_rank(px["NEM"], 21).ffill(limit=3)
hi252 = rolling_on_valid(gld, lambda x: x.rolling(252).max())
dd = gld / hi252 - 1.0                      # today -0.1878

print("cal %s .. %s" % (cal[0].date(), cal[-1].date()))
print("live 2026-09-02: GLD r5 %.1f  GDX r21 %.1f  NEM r21 %.1f  GLD dd from 52wh %.2f%%"
      % (r5.iloc[-1], gdx_r21.iloc[-1], nem_r21.iloc[-1], 100 * dd.iloc[-1]))

nfp = load_events(["nfp"])["date"]
pos, _ = anchor_positions(cal, nfp, -2)
anchor = pd.DatetimeIndex([cal[i] for i in pos])


def ret_from_entry(s, h):
    return s.shift(-h) / s - 1.0


WASH = (r5 <= 15)                            # today 9.1
MINE = (gdx_r21 >= 90)                       # today 94.4
JOINT = WASH & MINE
ANCH = pd.Series(cal.isin(anchor), index=cal)

# ==========================================================================
print("\n" + "=" * 78)
print("ATTACK 1 -- GLD DRAWDOWN SPLIT on the washout parent (registry 2026-08-21)")
print("  today's bucket is dd <= -10%%: GLD is -18.78%% below its 52-week high")
print("=" * 78)
for h in (1, 2, 3, 5, 10):
    rows = []
    for name, m in (("washout only (r5<=15)", WASH),
                    ("washout x miners bid", JOINT)):
        r = ret_from_entry(gld, h)
        days = cal[(m.fillna(False)).values & r.notna().values]
        epi = declusters(days, max(h, 5), cal)
        deep = pd.DatetimeIndex([d for d in epi if dd.loc[d] <= -0.10])
        shal = pd.DatetimeIndex([d for d in epi if dd.loc[d] > -0.10])
        for lbl, sel in ((f"{name} DEEP dd<=-10% (LIVE)", deep),
                         (f"{name} shallow dd>-10%", shal),
                         (f"{name} ALL episodes", epi)):
            v = r.loc[sel].dropna()
            s = summarize(v.values, f"h={h} {lbl}")
            if s["n"]:
                s["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
                s["edge_pp"] = round(100 * (v.mean() - r.dropna().mean()), 3)
            rows.append(s)
    rows.append(summarize(ret_from_entry(gld, h).dropna().values,
                          f"h={h} GLD all-days drift"))
    show(rows, f"GLD drawdown split, h={h}")

# ==========================================================================
print("\n" + "=" * 78)
print("ATTACK 2 -- COUNT FIRST: the joint state and its NFP cross")
print("=" * 78)
for lbl, m in (("washout r5<=15", WASH), ("miners GDX r21>=90", MINE),
               ("JOINT", JOINT), ("JOINT & NFP k=-2 anchor", JOINT & ANCH),
               ("JOINT & NFP & dd<=-10% (LIVE)", JOINT & ANCH & (dd <= -0.10))):
    days = cal[m.fillna(False).values]
    epi = declusters(days, 5, cal)
    yrs = dict(pd.Series(pd.DatetimeIndex(epi).year).value_counts().sort_index())
    print("  %-32s days=%4d  episodes=%3d  episode years=%s"
          % (lbl, len(days), len(epi), yrs))
    if 0 < len(epi) <= 12:
        print("      episode dates:", ", ".join(str(d.date()) for d in epi))

# ==========================================================================
print("\n" + "=" * 78)
print("ATTACK 3 -- battery on the countable forms, GLD long")
print("=" * 78)
variants = {"r5<=10 (tighter)": (r5 <= 10) & MINE,
            "r5<=15 x GDXr21>=95": WASH & (gdx_r21 >= 95),
            "r5<=15 x GDXr21>=80": WASH & (gdx_r21 >= 80),
            "washout alone": WASH,
            "miners alone": MINE,
            "JOINT & dd<=-10% (LIVE bucket)": JOINT & (dd <= -0.10)}
for h in (1, 3, 5):
    battery(px, JOINT, [("GLD", 1.0)], h=h,
            title="C6 GLD long, 5d washout x miners bid (NO event gate)",
            cost_bps=6.0, variants=variants if h == 1 else None,
            lag=1, min_gap=5, event_kinds=("nfp",))

# ==========================================================================
print("\n" + "=" * 78)
print("ATTACK 4 -- does the NFP cross survive at all? GLD pre-print, unconditional")
print("=" * 78)
for h in (1, 2, 3):
    r = ret_from_entry(gld, h)
    ent = pd.DatetimeIndex([cal[i + 1] for i in pos if i + 1 < len(cal)])
    c = r.loc[ent].dropna()
    w = int((c > 0).sum())
    print("  GLD h=%d, entry k=-1 (last session before the print): %+.3f%% n=%d "
          "hit %.1f%% record %d-%d sign p %.4f  vs own drift %+.3f%% -> %+.3fpp"
          % (h, 100 * c.mean(), len(c), 100 * (c > 0).mean(), w, len(c) - w,
             sign_test(w, len(c)), 100 * r.dropna().mean(),
             100 * (c.mean() - r.dropna().mean())))
    deep = pd.DatetimeIndex([d for d in ent if dd.get(d, np.nan) <= -0.10])
    v = r.loc[deep].dropna()
    if len(v):
        print("     of those, the LIVE dd<=-10%% bucket: %+.3f%% n=%d hit %.1f%% sign p %.4f"
              % (100 * v.mean(), len(v), 100 * (v > 0).mean(),
                 sign_test(int((v > 0).sum()), len(v))))
    wash_ent = pd.DatetimeIndex([d for d in ent if bool(WASH.get(d, False))])
    v2 = r.loc[wash_ent].dropna()
    if len(v2):
        print("     of those, WASHOUT r5<=15 at the anchor: %+.3f%% n=%d hit %.1f%% sign p %.4f"
              % (100 * v2.mean(), len(v2), 100 * (v2 > 0).mean(),
                 sign_test(int((v2 > 0).sum()), len(v2))))

# ==========================================================================
print("\n" + "=" * 78)
print("ATTACK 5 -- midterm + era split on the joint state")
print("=" * 78)
for h in (1, 3, 5):
    r = ret_from_entry(gld, h)
    days = cal[JOINT.fillna(False).values & r.notna().values]
    epi = declusters(days, max(h, 5), cal)
    v = r.loc[epi].dropna()
    rows = []
    for lbl, sel in (("MIDTERM", [d.year % 4 == 2 for d in v.index]),
                     ("non-midterm", [d.year % 4 != 2 for d in v.index]),
                     ("pre-2018", [d.year < 2018 for d in v.index]),
                     ("2018+", [d.year >= 2018 for d in v.index])):
        x = v[sel]
        s = summarize(x.values, f"h={h} {lbl}")
        if s["n"]:
            s["sign_p"] = round(sign_test(int((x > 0).sum()), len(x)), 4)
        rows.append(s)
    show(rows, f"joint-state episodes, splits h={h}")

# ==========================================================================
print("\n" + "=" * 78)
print("ATTACK 6 -- GDX-on-GLD beta (reported so no pair is credited without it)")
print("=" * 78)
for h in (1, 3, 5):
    a = ret_from_entry(px["GDX"], h)
    b = ret_from_entry(gld, h)
    both = pd.concat([a, b], axis=1).dropna()
    both.columns = ["GDX", "GLD"]
    beta = np.polyfit(both["GLD"], both["GDX"], 1)[0]
    resid = both["GDX"] - beta * both["GLD"]
    days = cal[JOINT.fillna(False).values]
    epi = declusters(days, max(h, 5), cal)
    v = resid.reindex(epi).dropna()
    print("  h=%d beta(GDX~GLD)=%.3f  joint-state beta-neutral miner residual "
          "%+.3fpp n=%d hit %.1f%% sign p %.4f"
          % (h, beta, 100 * v.mean(), len(v), 100 * (v > 0).mean(),
             sign_test(int((v > 0).sum()), len(v)) if len(v) else np.nan))
print("\nDONE.")
