"""C4 round 1 - long OIH against short XOP at a 63-day services-vs-E&P extreme.

Live premise (surface map): OIH-XOP 63d spread -18.52pp, PIT trailing-252
pctile 0.4; OIH -9.96% off its 52w high vs XLE -1.00% and XOP -1.74%.

Round-1 obligations discharged here:
  0. PREMISE re-derivation (2026-08-24 rule: print the thing it is NAMED after)
     + PIT vs full-sample percentile gap.
  0b. FACTOR STRUCTURE of {XLE, XOP, OIH, USO}: PC1 share, participation ratio,
      effective N, pairwise correlations, and P(XLE thrusting | pair extreme).
  1. battery() on the equal-dollar pair, h=5, three trigger rungs.
  2. BETA-NEUTRAL residual with a POINT-IN-TIME trailing-252d beta (the
     2026-08-10 rule), reported beside the equal-dollar spread.
  3. LEG ATTRIBUTION (2026-08-07/-08-19): what each leg earns against its OWN
     drift over the same span, and whether the NAKED LONG OIH beats the pair.
  4. cost, era, concentration, JH-in-window tail.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change

import numpy as np
import pandas as pd

BAR = pd.Timestamp("2026-08-24")
NAMES = ["OIH", "XOP", "XLE", "USO", "SPY"]

px_all = load_prices(NAMES)
spy = px_all["SPY"]["Close"].dropna()
CAL = spy.index[spy.index <= BAR]
px = pd.DataFrame({t: px_all[t]["Close"] for t in NAMES}).reindex(CAL)

for t in NAMES:
    s = px_all[t]["Close"].dropna()
    print(f"  {t}: {s.index[0].date()} .. {s.index[-1].date()}  n={len(s)}")

# ----------------------------------------------------------------- 0. premise
print("\n" + "=" * 100)
print("0. PREMISE re-derivation")
print("=" * 100)


def rn(t, n):
    return _valid_pct_change(px_all[t]["Close"].dropna(), n).reindex(CAL)


def dist_hi(t, look=252):
    c = px_all[t]["Close"].dropna()
    return rolling_on_valid(c, lambda x: x / x.rolling(look).max() - 1.0).reindex(CAL)


sp63 = rn("OIH", 63) - rn("XOP", 63)
sp63_pit = rolling_on_valid(sp63, lambda x: x.rolling(252).rank(pct=True) * 100)
print(f"  OIH r63 {100*rn('OIH',63).iloc[-1]:+.2f}%   XOP r63 "
      f"{100*rn('XOP',63).iloc[-1]:+.2f}%   spread {100*sp63.iloc[-1]:+.2f}pp")
print(f"  spread PIT trailing-252 pctile = {sp63_pit.iloc[-1]:.2f}   "
      f"FULL-SAMPLE pctile = {100*(sp63 <= sp63.iloc[-1]).mean():.2f}   "
      f"(gap {100*(sp63 <= sp63.iloc[-1]).mean() - sp63_pit.iloc[-1]:+.1f}pt)")
for t in ["OIH", "XOP", "XLE", "USO"]:
    print(f"  {t}: off 52wh {100*dist_hi(t).iloc[-1]:+.2f}%   r5 rank "
          f"{pct_rank(px_all[t]['Close'].dropna(),5).reindex(CAL).iloc[-1]:.1f}"
          f"   r63 rank {pct_rank(px_all[t]['Close'].dropna(),63).reindex(CAL).iloc[-1]:.1f}")

# ------------------------------------------------------- 0b. factor structure
print("\n" + "=" * 100)
print("0b. FACTOR STRUCTURE of the complex (2026-08-24 rule)")
print("=" * 100)
COMPLEX = ["XLE", "XOP", "OIH", "USO"]
rets = pd.DataFrame({t: px[t].pct_change() for t in COMPLEX}).dropna()
C = rets.corr()
print(C.round(3).to_string())
off = C.values[np.triu_indices(len(COMPLEX), 1)]
ev = np.linalg.eigvalsh(C.values)[::-1]
pr = ev.sum() ** 2 / (ev ** 2).sum()
print(f"  mean pairwise corr {off.mean():.3f}   PC1 share {100*ev[0]/ev.sum():.1f}%"
      f"   participation ratio {pr:.2f} effective of {len(COMPLEX)}")
print(f"  OIH~XOP daily corr {C.loc['OIH','XOP']:.3f}   "
      f"OIH~XLE {C.loc['OIH','XLE']:.3f}   XOP~XLE {C.loc['XOP','XLE']:.3f}")

# ------------------------------------------------------------- triggers
LIVE = float(sp63_pit.iloc[-1])
TRIG = {
    "A PIT63 spread <= 1": sp63_pit <= 1.0,
    "B PIT63 spread <= 2.5": sp63_pit <= 2.5,
    "C PIT63 spread <= 5": sp63_pit <= 5.0,
    "D PIT63 <=2.5 & OIH off-52wh <= -8%": (sp63_pit <= 2.5) & (dist_hi("OIH") <= -0.08),
    "E PIT63 <=2.5 & XLE off-52wh >= -3%": (sp63_pit <= 2.5) & (dist_hi("XLE") >= -0.03),
}
print("\n  trigger day counts:")
for k, m in TRIG.items():
    m = m.reindex(CAL, fill_value=False).fillna(False)
    print(f"    {k:45s} n_days={int(m.sum()):4d}  last={m[m].index[-1].date() if m.any() else 'never'}"
          f"  live={bool(m.iloc[-1])}")

MAIN = TRIG["B PIT63 spread <= 2.5"].reindex(CAL, fill_value=False).fillna(False)

# ------------------------------------------------------------ 1. battery
battery(px, MAIN, [("OIH", 1.0), ("XOP", -1.0)], h=5,
        title="C4 equal-dollar OIH long / XOP short, PIT63 spread <= 2.5",
        cost_bps=6.0, variants=TRIG,
        event_kinds=("jackson_hole", "fomc_decision", "cpi"))

# ---------------------------------------------- 2. beta-neutral PIT residual
print("\n" + "=" * 100)
print("2. BETA-NEUTRAL residual, POINT-IN-TIME trailing-252d beta")
print("=" * 100)
ro, rx = px["OIH"].pct_change(), px["XOP"].pct_change()
cov = ro.rolling(252).cov(rx)
var = rx.rolling(252).var()
beta = (cov / var).reindex(CAL)
print(f"  live beta OIH-on-XOP (252d) = {beta.iloc[-1]:.3f}   "
      f"median over history {beta.median():.3f}  range "
      f"[{beta.min():.2f}, {beta.max():.2f}]")


def resid_ret(h, lag=1):
    a = fwd_lag(px["OIH"], h, lag)
    b = fwd_lag(px["XOP"], h, lag)
    return a - beta * b


rows = []
for h in (1, 2, 3, 5, 10):
    eq = vehicle_ret(px, [("OIH", 1.0), ("XOP", -1.0)], h)
    rs = resid_ret(h)
    ol = fwd_lag(px["OIH"], h)
    valid = rs.notna() & eq.notna()
    trig = CAL[MAIN.values & valid.values]
    epi = declusters(trig, h, CAL[valid.values])
    for lbl, ser in (("equal-dollar", eq), ("beta-neutral", rs), ("naked OIH", ol)):
        r = summarize(ser.loc[epi].values, f"h={h} {lbl}")
        base = ser[valid].mean()
        r["ctl_all_pct"] = round(100 * base, 3)
        r["edge_pct"] = round(r["mean_pct"] - 100 * base, 3)
        rows.append(r)
show(rows, "2. equal-dollar vs beta-neutral vs naked long, episode level")

# --------------------------------------------------------- 3. leg attribution
print("\n" + "=" * 100)
print("3. LEG ATTRIBUTION - which leg pays, against its OWN drift?")
print("=" * 100)
for h in (3, 5, 10):
    eq = vehicle_ret(px, [("OIH", 1.0), ("XOP", -1.0)], h)
    valid = eq.notna()
    trig = CAL[MAIN.values & valid.values]
    epi = declusters(trig, h, CAL[valid.values])
    span = (CAL >= trig[0]) & (CAL <= trig[-1]) & valid.values
    oih = fwd_lag(px["OIH"], h)
    xop = fwd_lag(px["XOP"], h)
    o_c, x_c = oih[span].mean(), xop[span].mean()
    o_t, x_t = oih.loc[epi].mean(), xop.loc[epi].mean()
    print(f"  h={h:2d} N_epi={len(epi):3d}   "
          f"LONG OIH leg {100*o_t:+.3f}% vs own drift {100*o_c:+.3f}% -> "
          f"contrib {100*(o_t-o_c):+.3f}pp   |   "
          f"SHORT XOP leg {100*(-x_t):+.3f}% vs {100*(-x_c):+.3f}% -> "
          f"contrib {100*(x_c-x_t):+.3f}pp")
    tot = (o_t - o_c) + (x_c - x_t)
    if abs(tot) > 1e-12:
        print(f"        long-leg share of the spread's excess = "
              f"{100*(o_t-o_c)/tot:+.0f}%   short-leg share = {100*(x_c-x_t)/tot:+.0f}%")

# ------------------------------------------- 4. P(XLE thrusting | pair extreme)
print("\n" + "=" * 100)
print("4. Is the pair a disguised XLE/crude bet?")
print("=" * 100)
h = 5
eq = vehicle_ret(px, [("OIH", 1.0), ("XOP", -1.0)], h)
valid = eq.notna()
trig = CAL[MAIN.values & valid.values]
epi = declusters(trig, h, CAL[valid.values])
xle_fwd = fwd_lag(px["XLE"], h)
uso_fwd = fwd_lag(px["USO"], h)
sub = pd.DataFrame({"pair": eq.loc[epi], "xle": xle_fwd.loc[epi],
                    "uso": uso_fwd.loc[epi]}).dropna()
if len(sub) > 5:
    b_xle = np.polyfit(sub["xle"], sub["pair"], 1)
    print(f"  regress pair fwd on XLE fwd: slope {b_xle[0]:+.3f}  "
          f"intercept (residual alpha) {100*b_xle[1]:+.3f}%  corr "
          f"{sub['pair'].corr(sub['xle']):+.3f}   N={len(sub)}")
    b_uso = np.polyfit(sub["uso"].fillna(0), sub["pair"], 1)
    print(f"  regress pair fwd on USO fwd: slope {b_uso[0]:+.3f}  "
          f"intercept {100*b_uso[1]:+.3f}%  corr {sub['pair'].corr(sub['uso']):+.3f}")
print(f"  P(XLE r5 rank >= 80 | trigger) = "
      f"{(pct_rank(px_all['XLE']['Close'].dropna(),5).reindex(CAL).loc[trig] >= 80).mean():.3f}"
      f"   base {(pct_rank(px_all['XLE']['Close'].dropna(),5).reindex(CAL) >= 80).mean():.3f}")
sma200 = rolling_on_valid(spy, lambda x: x.rolling(200).mean()).reindex(CAL)
above = (px["SPY"] > sma200)
print(f"  TAPE over-selection: SPY above 200d on {100*above.loc[trig].mean():.1f}% of "
      f"trigger days vs base {100*above[valid].mean():.1f}%")

# ------------------------------------------------------ 5. lookback neighbours
print("\n" + "=" * 100)
print("5. DEFINITION NEIGHBOURS - lookback 21/42/63/126 at the same rung")
print("=" * 100)
rows = []
for n in (21, 42, 63, 126):
    sp = rn("OIH", n) - rn("XOP", n)
    pit = rolling_on_valid(sp, lambda x: x.rolling(252).rank(pct=True) * 100)
    for h in (3, 5, 10):
        m = (pit <= 2.5).reindex(CAL, fill_value=False).fillna(False)
        ser = vehicle_ret(px, [("OIH", 1.0), ("XOP", -1.0)], h)
        v = ser.notna()
        t = CAL[m.values & v.values]
        if len(t) == 0:
            continue
        e = declusters(t, h, CAL[v.values])
        r = summarize(ser.loc[e].values, f"lb={n} h={h}")
        r["live_pit"] = round(float(pit.iloc[-1]), 1)
        rows.append(r)
show(rows, "5. lookback neighbours (episodes, PIT<=2.5)")
print("\nDONE C4 round 1")
