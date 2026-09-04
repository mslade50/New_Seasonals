"""C8 KILL CHECK -- investment grade at its 252-day low while high yield holds
near its 252-day high, crossed with the payrolls print.

Live: LQD +0.12% above its 252d low, IEF +0.36% above its, TLT +1.12% above its,
HYG -0.79% off its 252d HIGH.

COUNT FIRST (2026-08-07 rule, and watchlist 1 / 26 both park on the count).
Then, and only then, ask whether the EVENT cross can rescue it -- it cannot add
episodes, only subtract, so the count is the whole question. Also settle the
standing debt from watchlist 1: any surviving form must show a CREDIT-SPECIFIC
residual (LQD regressed on IEF), not duration wearing a credit label.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import (load_prices, load_events, anchor_positions, battery,
                       summarize, show, sign_test, declusters, rolling_on_valid,
                       bootstrap_p_le0)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 230)

TK = ["LQD", "HYG", "IEF", "TLT", "SPY"]
raw = load_prices(TK)
px = pd.DataFrame({t: raw[t]["Close"] for t in TK}).dropna(subset=["LQD", "HYG"])
cal = px.index


def prox_low(s):
    lo = rolling_on_valid(s, lambda x: x.rolling(252).min())
    return s / lo - 1.0          # 0.000 = AT the low


def prox_high(s):
    hi = rolling_on_valid(s, lambda x: x.rolling(252).max())
    return s / hi - 1.0          # 0.000 = AT the high, negative below


lqd_lo = prox_low(px["LQD"])
ief_lo = prox_low(px["IEF"])
tlt_lo = prox_low(px["TLT"])
hyg_hi = prox_high(px["HYG"])
print("cal %s .. %s   (HYG inception bounds the joint state)" % (cal[0].date(), cal[-1].date()))
print("live 2026-09-02: LQD +%.2f%% above 252d low | IEF +%.2f%% | TLT +%.2f%% | "
      "HYG %.2f%% off 252d high"
      % (100 * lqd_lo.iloc[-1], 100 * ief_lo.iloc[-1], 100 * tlt_lo.iloc[-1],
         100 * hyg_hi.iloc[-1]))

nfp = load_events(["nfp"])["date"]
pos, _ = anchor_positions(cal, nfp, -2)
anchor = pd.DatetimeIndex([cal[i] for i in pos])
ANCH = pd.Series(cal.isin(anchor), index=cal)


def ret_from_entry(s, h):
    return s.shift(-h) / s - 1.0


# ==========================================================================
print("\n" + "=" * 78)
print("ATTACK 1 -- COUNT FIRST. Rung ladder, days / declustered episodes / years")
print("=" * 78)
RUNGS = [
    ("LIVE tight:  LQD<=0.5% HYG>=-1.0%", (lqd_lo <= 0.005) & (hyg_hi >= -0.010)),
    ("watchlist-26 tight + IEF leg",
     (lqd_lo <= 0.015) & (ief_lo <= 0.015) & (hyg_hi >= -0.0025)),
    ("watchlist-1:  LQD<=2%  HYG>=-0.5%", (lqd_lo <= 0.02) & (hyg_hi >= -0.005)),
    ("wide-1:       LQD<=2%  HYG>=-1%", (lqd_lo <= 0.02) & (hyg_hi >= -0.01)),
    ("wide-2:       LQD<=3%  HYG>=-2%", (lqd_lo <= 0.03) & (hyg_hi >= -0.02)),
    ("wide-3:       LQD<=5%  HYG>=-3%", (lqd_lo <= 0.05) & (hyg_hi >= -0.03)),
    ("LQD-low leg alone (<=2%)", (lqd_lo <= 0.02)),
    ("HYG-high leg alone (>=-1%)", (hyg_hi >= -0.01)),
]
count_rows = []
for lbl, m in RUNGS:
    days = cal[m.fillna(False).values]
    epi = declusters(days, 21, cal)
    yrs = dict(pd.Series(pd.DatetimeIndex(epi).year).value_counts().sort_index())
    dyrs = dict(pd.Series(pd.DatetimeIndex(days).year).value_counts().sort_index())
    n2018 = dyrs.get(2018, 0)
    count_rows.append({"rung": lbl, "days": len(days), "episodes": len(epi),
                       "days_2018": n2018,
                       "pct_days_2018": round(100 * n2018 / max(len(days), 1), 1),
                       "epi_years_exNaN": len(yrs)})
    print("  %-38s days=%4d episodes=%3d  episode years=%s" % (lbl, len(days), len(epi), yrs))
    if 0 < len(epi) <= 14:
        print("      episodes:", ", ".join(str(d.date()) for d in epi))
show(count_rows, "count ladder")

print("\n  --- the EVENT cross (it can only SUBTRACT episodes) ---")
for lbl, m in RUNGS:
    days = cal[(m & ANCH).fillna(False).values]
    epi = declusters(days, 21, cal)
    print("  %-38s  x NFP k=-2 anchor -> days=%3d episodes=%2d %s"
          % (lbl, len(days), len(epi),
             ", ".join(str(d.date()) for d in epi) if len(epi) <= 8 else ""))

# ==========================================================================
print("\n" + "=" * 78)
print("ATTACK 2 -- the credit-specific residual: LQD regressed on IEF")
print("  (watchlist 1's standing note; a surviving form MUST show one)")
print("=" * 78)
for h in (1, 3, 5, 10):
    a = ret_from_entry(px["LQD"], h)
    b = ret_from_entry(px["IEF"], h)
    both = pd.concat([a, b], axis=1).dropna()
    both.columns = ["LQD", "IEF"]
    beta = np.polyfit(both["IEF"], both["LQD"], 1)[0]
    resid = both["LQD"] - beta * both["IEF"]
    print("  h=%2d beta(LQD~IEF)=%.3f   residual all-days %+.4fpp" % (h, beta, 100 * resid.mean()))
    for lbl, m in RUNGS[:6]:
        days = cal[m.fillna(False).values]
        epi = declusters(days, 21, cal)
        v = resid.reindex(epi).dropna()
        if len(v) == 0:
            print("       %-38s n=0" % lbl)
            continue
        print("       %-38s residual %+.4fpp n=%2d hit %.0f%% sign p %.4f"
              % (lbl, 100 * v.mean(), len(v), 100 * (v > 0).mean(),
                 sign_test(int((v > 0).sum()), len(v))))

# ==========================================================================
print("\n" + "=" * 78)
print("ATTACK 3 -- what the widened rungs actually pay (all four vehicles)")
print("=" * 78)
for lbl, m in [RUNGS[3], RUNGS[4], RUNGS[5]]:
    days = cal[m.fillna(False).values]
    for h in (1, 3, 5, 10):
        epi = declusters(days, 21, cal)
        rows = []
        for tkr in ("LQD", "HYG", "TLT", "IEF", "SPY"):
            r = ret_from_entry(px[tkr], h)
            v = r.reindex(epi).dropna()
            s = summarize(v.values, f"{tkr} h={h}")
            if s["n"]:
                s["drift_pct"] = round(100 * r.dropna().mean(), 3)
                s["edge_pp"] = round(100 * (v.mean() - r.dropna().mean()), 3)
                s["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
            rows.append(s)
        show(rows, f"{lbl}  h={h}  (episodes={len(epi)})")

# ==========================================================================
print("\n" + "=" * 78)
print("ATTACK 4 -- gate attribution: does the conjunction add anything?")
print("=" * 78)
for h in (3, 10):
    for tkr in ("LQD", "TLT", "SPY"):
        r = ret_from_entry(px[tkr], h)
        base = r.dropna().mean()
        out = []
        for lbl, m in (("LQD-low alone", RUNGS[6][1]), ("HYG-high alone", RUNGS[7][1]),
                       ("conjunction wide-1", RUNGS[3][1]), ("conjunction wide-2", RUNGS[4][1])):
            epi = declusters(cal[m.fillna(False).values], 21, cal)
            v = r.reindex(epi).dropna()
            out.append("%s %+.3f%% (n=%d)" % (lbl, 100 * v.mean(), len(v)))
        print("  %s h=%2d  drift %+.3f%% |  %s" % (tkr, h, 100 * base, "  |  ".join(out)))
print("\nDONE.")
