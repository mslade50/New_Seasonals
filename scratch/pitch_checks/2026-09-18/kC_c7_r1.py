"""c7 round 1: LONG ITB the session after TLT >= +1% while ITB UNDERREACTS
(ITB d1 < beta_pit(ITB on TLT, trailing 252 through D-1) x TLT d1) and ITB sits
within 10% of its 252-session closing low. Entry MOC D+1 (lag=1).
Signal 2026-09-17: TLT +1.11%, ITB +0.49%, ITB 3.73% above its low.

1. LIVE verify (does the underreaction gate actually fire today?)
2. battery h=1,3,5 on ITB
3. LAG PROFILE lag 0/1/2 at h=1..5 (catch-up on D+1 is not capturable)
4. SPY-beta-hedged residual (is the 'catch-up' just the market rising?)
5. rows: XHB, XLRE, KRE, XLU (each with its OWN underreaction + near-low gate)
6. gate attribution: TLT>=1% alone -> ITB; + underreact; + near-low; complements
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kC_common import *  # noqa

GAP = 5
COST = 6.0
SECT = ["ITB", "XHB", "XLRE", "KRE", "XLU", "XLF", "XLE", "XLK", "XLB", "XLP"]
px = panel(["SPY", "TLT", "IEF"] + SECT, "SPY")
r1 = {t: dret(px[t]) for t in px.columns}
tlt1 = r1["TLT"]
beta_t = {t: pit_beta(r1[t], tlt1) for t in SECT}
beta_s = {t: pit_beta(r1[t], r1["SPY"]) for t in SECT}
nearlow = {t: (px[t] / rolling_on_valid(px[t], lambda x: x.rolling(252).min()) - 1.0) for t in SECT}


def trig_for(t, thr=0.01, low=0.10, under=True):
    m = tlt1 >= thr
    if under:
        m = m & (r1[t] < beta_t[t] * tlt1)
    if low is not None:
        m = m & (nearlow[t] <= low)
    return m.fillna(False)


last = px.index[-1]
print(f"panel {px.index[0].date()}..{last.date()}")
print(f"LIVE {last.date()}: TLT {100*tlt1.iloc[-1]:+.2f}%")
for t in ["ITB", "XHB", "XLRE", "KRE", "XLU"]:
    print(f"  {t}: d1 {100*r1[t].iloc[-1]:+.2f}%  beta_TLT {beta_t[t].iloc[-1]:+.3f} -> "
          f"implied {100*beta_t[t].iloc[-1]*tlt1.iloc[-1]:+.2f}%  underreact={bool(r1[t].iloc[-1] < beta_t[t].iloc[-1]*tlt1.iloc[-1])}"
          f"  above-low {100*nearlow[t].iloc[-1]:.2f}%  fired={bool(trig_for(t).iloc[-1])}")
trig = trig_for("ITB")
print(f"ITB day-level triggers {int(trig.sum())}  (TLT>=1% days: {int((tlt1>=0.01).sum())})")
print("ITB beta_TLT history (yearly mean):", {int(y): round(v, 2) for y, v in beta_t['ITB'].groupby(px.index.year).mean().items()})

L = [("ITB", 1.0)]
for h in (1, 3, 5):
    battery(px, trig, L, h, f"c7 LONG ITB after TLT>=1% underreaction near low h={h}", COST, min_gap=GAP,
            variants={"TLT >= 0.75%": trig_for("ITB", 0.0075),
                      "TLT >= 1.25%": trig_for("ITB", 0.0125),
                      "near-low 5%": trig_for("ITB", low=0.05),
                      "near-low 15%": trig_for("ITB", low=0.15),
                      "no near-low gate": trig_for("ITB", low=None),
                      "no underreact gate": trig_for("ITB", under=False)})

print("\n" + "=" * 100)
print("3. LAG PROFILE, long ITB (episode, gap 5)")
rows = []
for h in (1, 2, 3, 5):
    for lag in (0, 1, 2):
        s, _, _ = cellstats(px, trig, L, h, f"h={h} lag={lag}", GAP, lag)
        rows.append(s)
show(rows)

print("\n" + "=" * 100)
print("4. SPY-BETA-HEDGED RESIDUAL of ITB (beta trailing 252 through D-1), lag 0 and 1")
rows = []
for h in (1, 3, 5):
    for lag in (0, 1):
        ri = vehicle_ret(px, [("ITB", 1.0)], h, lag)
        rs = vehicle_ret(px, [("SPY", 1.0)], h, lag)
        res = ri - beta_s["ITB"] * rs
        days = px.index[trig.values & res.notna().values]
        epi = declusters(days, GAP, px.index)
        v = res.loc[epi].values
        s = summarize(v, f"resid h={h} lag={lag}")
        s["ctl_pct"] = 100 * res.dropna().mean()
        s["rec"] = f"{int((v>0).sum())}-{int((v<=0).sum())}"
        s["p_coin"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
        spy_leg = rs.loc[epi].values
        s["spy_leg_pct"] = 100 * np.nanmean(spy_leg)
        rows.append(s)
show(rows)

print("\n" + "=" * 100)
print("5. ROWS: each sector with its OWN underreaction + near-low(10%) gate; lag 0 vs 1")
rows = []
for t in ["ITB", "XHB", "XLRE", "KRE", "XLU"]:
    m = trig_for(t)
    for h in (1, 3, 5):
        for lag in (0, 1):
            s, _, _ = cellstats(px, m, [(t, 1.0)], h, f"{t} h={h} lag={lag}", GAP, lag)
            rows.append({k: s.get(k) for k in ("label", "n", "mean_pct", "ctl_pct", "edge_pp", "rec", "p_base", "t")})
show(rows)

print("\n" + "=" * 100)
print("6. GATE ATTRIBUTION, long ITB lag=1 (episode means, gap 5)")
par = (tlt1 >= 0.01).fillna(False)
und = (r1["ITB"] < beta_t["ITB"] * tlt1).fillna(False)
low = (nearlow["ITB"] <= 0.10).fillna(False)
cells = {"parent TLT>=1%": par, "+underreact": par & und, "+underreact ANTI (overreact)": par & ~und,
         "+near-low": par & low, "+near-low ANTI (far from low)": par & ~low,
         "cell (both)": par & und & low, "underreact, NOT near low": par & und & ~low,
         "near low, NOT underreact": par & ~und & low}
rows = []
for lbl, m in cells.items():
    for h in (1, 3, 5):
        s, _, _ = cellstats(px, m, L, h, f"{lbl} h={h}", GAP)
        rows.append({k: s.get(k) for k in ("label", "n", "mean_pct", "ctl_pct", "edge_pp", "rec", "p_base", "t")})
show(rows)
