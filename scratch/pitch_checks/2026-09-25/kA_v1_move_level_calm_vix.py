"""kA V1 round 1: rate vol at a one-year LEVEL extreme under calm equity vol.
Cell: ^MOVE level pctile >= 95 AND ^VIX level pctile <= 30 (trailing 252 valid
obs, surface convention: share of the window <= today). Pre-specified sign:
equity vol catches up -> SHORT SPY, ^VIX UP, SHORT the SVXY residual
(SVXY - b*SPY, b estimated ex-ante on 2018-03+ daily data). Lag 1, h=1..10.
Mandatory contrasts: VIX-low alone (mean reversion of a cheap VIX is not a
MOVE finding), MOVE-high alone, MOVE-high with VIX not low."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

POST = pd.Timestamp("2018-03-01")
TK = ["SVXY", "SPY", "^VIX", "^MOVE"]
raw = load_prices(TK)
IDX = raw["SPY"].index
px = close_panel(TK).reindex(IDX)
move, vix = px["^MOVE"], px["^VIX"]


def lvl_pct(s, n=252):
    return rolling_on_valid(s, lambda x: x.rolling(n).apply(lambda w: (w <= w[-1]).mean() * 100, raw=True))


mp = lvl_pct(move)
vp = lvl_pct(vix)
print(f"LIVE {IDX[-1].date()}: MOVE {move.iloc[-1]:.2f} pct {mp.iloc[-1]:.1f}; VIX {vix.iloc[-1]:.2f} pct {vp.iloc[-1]:.1f}")
for t in ("SPY", "SVXY"):
    a = wilder_atr(raw[t]["High"].to_numpy(), raw[t]["Low"].to_numpy(), raw[t]["Close"].to_numpy())
    print(f"  {t} {raw[t]['Close'].iloc[-1]:.2f} Wilder-14 ATR {a[-1]:.3f} ({100*a[-1]/raw[t]['Close'].iloc[-1]:.2f}%)")

rs = px["SPY"].pct_change()
rsv = px["SVXY"].pct_change()
ok = (IDX >= POST) & rs.notna().values & rsv.notna().values
B = np.polyfit(rs[ok].values, rsv[ok].values, 1)[0]
print(f"SVXY daily beta on SPY 2018-03+: {B:.3f}")

cell = ((mp >= 95) & (vp <= 30)).fillna(False)
mhi = (mp >= 95).fillna(False)
vlo = (vp <= 30).fillna(False)
span = move.first_valid_index()
print(f"cell days {int(cell.sum())} (2018-03+ {int((cell & (IDX >= POST)).sum())}); MOVE>=95 {int(mhi.sum())}; "
      f"VIX<=30 in MOVE span {int((vlo & (IDX >= span)).sum())}")
cd = IDX[cell.values]
print("cell days by year:", pd.Series(1, index=cd.year).groupby(level=0).sum().to_dict())
print("cell first dates of runs (gap>=21):", [str(x.date()) for x in declusters(cd, 21, IDX)])

COST_ETF = 5.0
pxs = px.loc[span:].copy()
m_s = cell.loc[span:]
for h in (2, 5, 10):
    battery(pxs, m_s, [("SPY", -1.0)], h, f"V1 SHORT SPY | MOVE>=95 & VIX<=30 (decluster 21)",
            cost_bps=COST_ETF, min_gap=21,
            variants={"MOVE>=95 alone": mhi, "VIX<=30 alone": vlo, "MOVE>=95 & VIX>30": mhi & ~vlo,
                      "MOVE>=90 & VIX<=30": (mp >= 90) & vlo, "MOVE>=97.5 & VIX<=30": (mp >= 97.5) & vlo,
                      "MOVE>=95 & VIX<=20": mhi & (vp <= 20), "MOVE>=95 & VIX<=40": mhi & (vp <= 40)},
            event_kinds=("nfp", "cpi", "fomc_decision"))

pxp = px.loc[POST:].copy()
for h in (2, 5, 10):
    battery(pxp, cell.loc[POST:], [("SVXY", -1.0), ("SPY", B)], h,
            f"V1 SHORT SVXY resid (b={B:.2f}) | MOVE>=95 & VIX<=30, 2018-03+ (decluster 21)",
            cost_bps=3.5, min_gap=21,
            variants={"MOVE>=95 alone": mhi, "VIX<=30 alone": vlo, "MOVE>=95 & VIX>30": mhi & ~vlo},
            event_kinds=("nfp", "cpi", "fomc_decision"))


# compact ladder: SPY short, VIX change, SVXY residual short; cell vs gates; episodes declustered 21
def ep_row(ret, mask, lab, era=None, gap=21):
    okk = ret.notna()
    if era is not None:
        okk &= era
    ds = IDX[(mask.reindex(IDX, fill_value=False) & okk).values]
    e = declusters(ds, gap, IDX)
    x = ret.loc[e].values
    r = summarize(x, lab)
    if r["n"]:
        w = int((x > 0).sum())
        r["rec"] = f"{w}-{len(x)-w}"
        r["sign_p"] = round(sign_test(w, len(x)), 4)
    return r


inspan = pd.Series(IDX >= span, index=IDX)
post = pd.Series(IDX >= POST, index=IDX)
allm = pd.Series(True, index=IDX)
for h in (1, 3, 5, 10):
    sspy = vehicle_ret(px, [("SPY", -1.0)], h)
    fv = fwd_lag(vix, h, 1)
    ssv = vehicle_ret(px, [("SVXY", -1.0), ("SPY", B)], h)
    rows = []
    for nm, ret, era in [("shortSPY", sspy, inspan), ("VIXchg", fv, inspan), ("shortSVXYres", ssv, post)]:
        for lab, m in [("CELL", cell), ("VIX<=30 alone", vlo), ("MOVE>=95 alone", mhi),
                       ("MOVE>=95 & VIX>30", mhi & ~vlo), ("ALL days (gap1)", allm)]:
            rows.append(ep_row(ret, m, f"{nm} | {lab}", era, gap=1 if lab.startswith("ALL") else 21))
    show([{k: r.get(k) for k in ("label", "n", "mean_pct", "median_pct", "hit", "t", "rec", "sign_p", "worst_pct")}
          for r in rows], f"h={h}: pre-specified side positive = thesis pays (episodes, decluster 21)")
