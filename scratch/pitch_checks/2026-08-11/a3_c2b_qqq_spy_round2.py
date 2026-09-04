"""C2 round 2. Round 1 killed h=3 on beta-neutrality (hit 52.4%, sign p 0.21)
but h=1 SURVIVED it (+0.115% beta-neutral, hit 59.8%, t=2.76, sign p 0.0003).
So round 2 goes after the h=1 cell specifically:

  R2-a  RESIDUAL DIRECTION. Is the "market-neutral" spread actually a levered
        long-equity bet on a CPI relief rally? Split by the sign of SPY's own
        print-session move and regress the spread on it.
  R2-b  HEDGE-RATIO FRAGILITY. 63/126/252d beta, vol-ratio hedge, and what
        each of them would set TODAY (QQQ realised vol 25.6 vs SPY 14.0 =
        1.83, while the 126d beta is 1.48 -- a 24% hedge disagreement).
  R2-c  IS CPI SPECIAL? Same construction on NFP, PPI, FOMC anchors.
  R2-d  TODAY'S ACTUAL BUCKET. rel63 rank is 1.6, not "<=30". Measure the
        extreme-laggard cell, and the midterm subset (2026 is midterm).
  R2-e  HONEST COST at closing-auction prices.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, load_events, fwd_lag, declusters, summarize, sign_test,
    bootstrap_p_le0, pct_rank,
)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)

px = close_panel(["SPY", "QQQ"]).dropna()
all_dates = px.index
pos = pd.Series(np.arange(len(all_dates)), index=all_dates)
rs, rq = px["SPY"].pct_change(), px["QQQ"].pct_change()


def beta_win(w):
    return rq.rolling(w).cov(rs) / rs.rolling(w).var()


BETAS = {f"beta{w}": beta_win(w) for w in (63, 126, 252)}
BETAS["volratio63"] = (rq.rolling(63).std() / rs.rolling(63).std())
BETAS["volratio21"] = (rq.rolling(21).std() / rs.rolling(21).std())

today = all_dates[-1]
print("hedge ratios TODAY (%s):" % today.date(),
      {k: round(float(v.loc[today]), 3) for k, v in BETAS.items()})
print("   -> the constructions disagree by %.0f%% on today's hedge; the number "
      "you short is not a fact, it is a choice."
      % (100 * (BETAS['volratio21'].loc[today] / BETAS['beta126'].loc[today] - 1)))


def anchors(kind, offset, gap=5):
    ev = load_events([kind])
    out = []
    for d in pd.DatetimeIndex(sorted(ev["date"].unique())):
        j = all_dates.searchsorted(d) - offset
        if 0 <= j < len(all_dates):
            out.append(all_dates[j])
    a = declusters(pd.DatetimeIndex(sorted(set(out))), gap, all_dates)
    return a[a.isin(all_dates)]


CPI = anchors("cpi", 2)
H = 1
q1, s1 = fwd_lag(px["QQQ"], H, 1), fwd_lag(px["SPY"], H, 1)


def cell(name, ser, dates, extra=""):
    v = ser.reindex(dates).dropna()
    st = summarize(v.values)
    if not st["n"]:
        return None
    wins = int((v.values > 0).sum())
    return dict(cell=name, N=st["n"], mean=round(st["mean_pct"], 3),
                hit=round(st["hit"], 1), t=round(st["t"], 2),
                signp=round(sign_test(wins, st["n"]), 4),
                worst=round(st["worst_pct"], 2), note=extra)


print("\n" + "=" * 110)
print("R2-a  RESIDUAL DIRECTION: does the beta-neutral spread pay when SPY FALLS?")
print("=" * 110)
bn = q1 - BETAS["beta126"] * s1
eqd = q1 - s1
spy_up = (s1.reindex(CPI) > 0)
sel_up = pd.DatetimeIndex(spy_up[spy_up.fillna(False)].index)
sel_dn = pd.DatetimeIndex(spy_up[(~spy_up.fillna(True))].index)
rows = [cell("beta-neutral ALL trigger days", bn, CPI),
        cell("beta-neutral | SPY UP on the print", bn, sel_up),
        cell("beta-neutral | SPY DOWN on the print", bn, sel_dn),
        cell("equal-$ | SPY UP on the print", eqd, sel_up),
        cell("equal-$ | SPY DOWN on the print", eqd, sel_dn)]
print(pd.DataFrame([r for r in rows if r]).to_string(index=False))
print(f"\n  base rate: SPY closes UP on {100*spy_up.mean():.1f}% of these print "
      f"sessions (N={int(spy_up.notna().sum())}); unconditional SPY up-day rate "
      f"{100*(rs>0).mean():.1f}%")

xv = s1.reindex(CPI).dropna()
yv = bn.reindex(xv.index)
ok = yv.notna()
b_resid, a_resid = np.polyfit(xv[ok].values, yv[ok].values, 1)
corr = np.corrcoef(xv[ok].values, yv[ok].values)[0, 1]
print(f"  regression of the BETA-NEUTRAL spread on SPY's own print-session "
      f"return: slope {b_resid:+.3f} (residual beta), intercept "
      f"{100*a_resid:+.4f}%, corr {corr:+.2f}")
print(f"  -> at SPY's mean print-session move ({100*xv.mean():+.3f}%), the "
      f"direction component is {100*b_resid*xv.mean():+.4f}% of the "
      f"{100*yv[ok].mean():+.4f}% total "
      f"({100*b_resid*xv.mean()/yv[ok].mean():.0f}%).")

print("\n" + "=" * 110)
print("R2-b  HEDGE-RATIO FRAGILITY: same cell under five hedge definitions")
print("=" * 110)
rows = []
for k, b in BETAS.items():
    rows.append(cell(f"QQQ - {k}*SPY", q1 - b * s1, CPI,
                     f"today's ratio {float(b.loc[today]):.3f}"))
rows.append(cell("QQQ - 1.00*SPY (equal $)", eqd, CPI, "today's ratio 1.000"))
rows.append(cell("QQQ - 1.83*SPY (today's 21d vol ratio, frozen)",
                 q1 - 1.83 * s1, CPI, "what today's tape implies"))
print(pd.DataFrame([r for r in rows if r]).to_string(index=False))

print("\n" + "=" * 110)
print("R2-c  IS CPI SPECIAL? identical construction, other scheduled prints")
print("=" * 110)
rows = []
for kind, off in (("cpi", 2), ("ppi", 2), ("nfp", 2), ("fomc_decision", 2),
                  ("fomc_minutes", 2)):
    try:
        a = anchors(kind, off)
    except Exception:
        continue
    if len(a) < 10:
        continue
    rows.append(cell(f"{kind} eve, beta-neutral", bn, a))
    rows.append(cell(f"{kind} eve, equal-$", eqd, a))
# and the null: all days
rows.append(cell("ALL DAYS beta-neutral", bn, all_dates))
rows.append(cell("ALL DAYS equal-$", eqd, all_dates))
print(pd.DataFrame([r for r in rows if r]).to_string(index=False))

print("\n" + "=" * 110)
print("R2-d  TODAY'S ACTUAL BUCKET (rel63 rank 1.6, not '<=30') + midterm")
print("=" * 110)
rel = px["QQQ"] / px["SPY"]
rel21, rel63 = pct_rank(rel, 21), pct_rank(rel, 63)
print(f"  today: rel21 {rel21.loc[today]:.1f}, rel63 {rel63.loc[today]:.1f}")
rows = []
for lbl, m in (("rel63 <= 10 (TODAY = 1.6)", rel63.reindex(CPI) <= 10),
               ("rel63 <= 5", rel63.reindex(CPI) <= 5),
               ("rel21 <= 10 (TODAY = 6.0)", rel21.reindex(CPI) <= 10),
               ("rel63 > 10", rel63.reindex(CPI) > 10)):
    sel = pd.DatetimeIndex(m[m.fillna(False)].index)
    rows.append(cell(lbl + " | beta-neut", bn, sel))
    rows.append(cell(lbl + " | equal-$", eqd, sel))
mid = pd.DatetimeIndex([d for d in CPI if d.year % 4 == 2])
non = pd.DatetimeIndex([d for d in CPI if d.year % 4 != 2])
rows.append(cell("MIDTERM years only (2026 is one) | beta-neut", bn, mid))
rows.append(cell("non-midterm | beta-neut", bn, non))
aug = pd.DatetimeIndex([d for d in CPI if d.month == 8])
rows.append(cell("AUGUST CPI only (this print) | beta-neut", bn, aug))
rows.append(cell("AUGUST CPI only | equal-$", eqd, aug))
print(pd.DataFrame([r for r in rows if r]).to_string(index=False))

# joint: today's exact state = QQQ deep laggard AND SPY at its 52w high
hi_s = px["SPY"] / px["SPY"].rolling(252).max() - 1.0
joint = (rel63.reindex(CPI) <= 20) & (hi_s.reindex(CPI) >= -0.01)
sel = pd.DatetimeIndex(joint[joint.fillna(False)].index)
print(f"\n  JOINT today-state (QQQ rel63 rank <= 20 AND SPY within 1% of its "
      f"52w high): N={len(sel)}")
r = cell("joint | beta-neut", bn, sel)
r2 = cell("joint | equal-$", eqd, sel)
print(pd.DataFrame([x for x in (r, r2) if x]).to_string(index=False))
if len(sel):
    print("  dates:", ", ".join(str(d.date()) for d in sel))
    vv = bn.reindex(sel).dropna()
    if len(vv) >= 3:
        print(f"  bootstrap P(mean<=0) = {bootstrap_p_le0(vv.values):.3f}")

print("\n" + "=" * 110)
print("R2-e  HONEST COST. SPY/QQQ close in the auction; spread is ~0.13 bps of")
print("      price each. Assume 0.5 bps all-in per leg round trip (auction")
print("      print + commission) = 1.0 bps for the pair; 1.5 bps/leg = 3.0.")
print("=" * 110)
v = bn.reindex(CPI).dropna()
edge = 100 * v.mean() * 100
for c in (1.0, 2.0, 3.0):
    print(f"   edge {edge:.1f} bps / {c:.1f} bps cost = {edge/c:.1f}x "
          f"({'PASSES' if edge/c >= 5 else 'FAILS'} the 5x bar)")
print(f"   per-episode sd is {summarize(v.values)['sd_pct']:.2f}% = "
      f"{summarize(v.values)['sd_pct']*100:.0f} bps, i.e. {summarize(v.values)['sd_pct']*100/edge:.0f}x "
      f"the mean. Worst single print session {summarize(v.values)['worst_pct']:.2f}%.")
