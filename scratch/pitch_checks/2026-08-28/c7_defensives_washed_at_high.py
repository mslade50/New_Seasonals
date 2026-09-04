"""C7 -- XLP, XLU and XLRE all at bottom-quintile 21d return ranks while SPY
closes within 1% of its trailing-252 high.  16 days in history.

Mechanism claimed: a maximal risk-on rotation with the index at a high leaves
the defensive complex mechanically oversold against a benchmark that cannot keep
outrunning it, so the complex mean-reverts.

Required probes (all round 2, run together with round 1 because N is tiny):
 (i)   does the SPY-near-high clause do any work over all-three-under-20 (188d)?
 (ii)  is the breadth count monotone 1-of-3 / 2-of-3 / 3-of-3?
 (iii) episode count, years, cluster_note
 (iv)  basket-vs-SPY leg attribution + beta-neutral residual
 (v)   TLT loading -- is this really a rates call?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pitch_lab import (load_prices, pct_rank, fwd_lag, summarize, show,  # noqa: E402
                       declusters, local_control, cluster_note, sign_test,
                       bootstrap_p_le0)

DEF = ["XLP", "XLU", "XLRE"]
ALL = DEF + ["SPY", "TLT"]
ASOF = pd.Timestamp("2026-08-27")
HS = (1, 2, 3, 5, 7, 10, 21)

px = load_prices(ALL)
S = {t: px[t]["Close"].dropna().loc[:ASOF] for t in ALL}
for t in ALL:
    print(f"  {t:<5} history {S[t].index[0].date()} .. {S[t].index[-1].date()}  "
          f"({len(S[t])} sessions)")

# single-instrument ranks computed on each instrument's OWN series
rk21 = {t: pct_rank(S[t], 21) for t in DEF}
spy = S["SPY"]
spy_dist = spy / spy.rolling(252).max() - 1.0

cal = S["XLRE"].index  # the binding calendar (XLRE inception 2015-10)
R = pd.DataFrame({t: rk21[t].reindex(cal) for t in DEF}).dropna()
sd = spy_dist.reindex(R.index)
under = (R <= 20)
count = under.sum(axis=1)
near_hi = (sd >= -0.01)

print(f"\ncommon calendar {R.index[0].date()} .. {R.index[-1].date()} "
      f"({len(R)} sessions)")
print(f"  3-of-3 under 20th pct           : {int((count==3).sum())} days")
print(f"  3-of-3 AND SPY within 1% of high: {int(((count==3)&near_hi).sum())} days")
print(f"  live today: {R.iloc[-1].round(1).to_dict()}  SPY dist {100*sd.iloc[-1]:+.2f}%")

# equal-weight basket total-return series on the common calendar
basket_lvl = None
for t in DEF:
    s = S[t].reindex(cal).ffill()
    n = s / s.iloc[0]
    basket_lvl = n / 3 if basket_lvl is None else basket_lvl + n / 3
BK = basket_lvl.dropna()
PX = pd.DataFrame({"BASKET": BK, "SPY": spy.reindex(BK.index).ffill(),
                   "TLT": S["TLT"].reindex(BK.index).ffill()}).dropna()


def stats(mask: pd.Series, legs, h, label, min_gap=None):
    r = None
    for tk, w in legs:
        rr = fwd_lag(PX[tk], h, 1)
        r = w * rr if r is None else r + w * rr
    m = mask.reindex(PX.index, fill_value=False).values & r.notna().values
    dts = PX.index[m]
    epi = declusters(dts, min_gap or h, PX.index)
    return summarize(r.loc[epi].values, f"{label} (epi N={len(epi)})"), dts, epi, r


MASK3 = (count == 3).reindex(PX.index, fill_value=False)
MASKJ = ((count == 3) & near_hi).reindex(PX.index, fill_value=False)

# --------------------------------------------------------------------------
print("\n=== C7.1  LONG BASKET: joint cell vs gate-attribution controls ===")
for h in HS:
    rows = []
    for lbl, mk in [("JOINT 3of3 + SPY near high", MASKJ),
                    ("3of3 only (drop SPY clause)", MASK3),
                    ("SPY near high only", near_hi.reindex(PX.index, fill_value=False))]:
        s0, dts, epi, r = stats(mk, [("BASKET", 1.0)], h, lbl)
        s0["n_days"] = len(dts)
        rows.append(s0)
    rall = fwd_lag(PX["BASKET"], h, 1)
    rows.append(summarize(rall.dropna().values, "CTRL-b all days"))
    _, dtsJ, _, _ = stats(MASKJ, [("BASKET", 1.0)], h, "x")
    loc = local_control(PX.index[rall.notna().values], dtsJ)
    rows.append(summarize(rall.loc[loc].values, "CTRL-c local +/-126td"))
    show(rows, f"h={h}")

# --------------------------------------------------------------------------
print("\n=== C7.2  breadth monotonicity (long basket, SPY-near-high held ON) ===")
for h in (3, 5, 10):
    rows = []
    for k in (1, 2, 3):
        mk = ((count == k) & near_hi).reindex(PX.index, fill_value=False)
        s0, dts, epi, _ = stats(mk, [("BASKET", 1.0)], h, f"{k}-of-3")
        s0["n_days"] = len(dts)
        rows.append(s0)
    show(rows, f"breadth count, h={h}")

print("\n=== C7.2b breadth monotonicity WITHOUT the SPY clause ===")
for h in (3, 5, 10):
    rows = []
    for k in (1, 2, 3):
        mk = (count == k).reindex(PX.index, fill_value=False)
        s0, dts, epi, _ = stats(mk, [("BASKET", 1.0)], h, f"{k}-of-3")
        s0["n_days"] = len(dts)
        rows.append(s0)
    show(rows, f"breadth count (no SPY gate), h={h}")

# --------------------------------------------------------------------------
print("\n=== C7.3  episodes, years, concentration on the joint cell ===")
for h in (3, 5, 10):
    s0, dts, epi, r = stats(MASKJ, [("BASKET", 1.0)], h, "joint")
    v = r.loc[epi].values
    print(f"  h={h}: {len(dts)} days -> {len(epi)} episodes "
          f"{[str(d.date()) for d in epi]}")
    if len(v) > 1:
        w = int((v > 0).sum())
        print(f"    mean {100*v.mean():+.3f}%  record {w}-{len(v)-w}  "
              f"sign p={sign_test(max(w, len(v)-w), len(v)):.3f}  "
              f"bootstrap P(mean<=0)={bootstrap_p_le0(v):.3f}")
        print(f"    {cluster_note(epi, v)}")

# --------------------------------------------------------------------------
print("\n=== C7.4  PAIR form: long basket / short SPY, leg attribution ===")
# measured beta of the basket on SPY over the common calendar
br = PX["BASKET"].pct_change()
sr = PX["SPY"].pct_change()
ok = br.notna() & sr.notna()
beta = float(np.polyfit(sr[ok], br[ok], 1)[0])
print(f"  measured basket beta on SPY (daily, full common calendar) = {beta:.3f}")
print(f"  -> an EQUAL-DOLLAR pair carries {beta-1:+.3f} units of net SPY beta")
for h in (3, 5, 10):
    rows = []
    s_b, _, epi, rb = stats(MASKJ, [("BASKET", 1.0)], h, "leg A: long basket")
    s_s, _, _, rs = stats(MASKJ, [("SPY", -1.0)], h, "leg B: short SPY")
    s_p, _, _, rp = stats(MASKJ, [("BASKET", 1.0), ("SPY", -1.0)], h, "equal-$ pair")
    s_n, _, _, rn = stats(MASKJ, [("BASKET", 1.0), ("SPY", -beta)], h,
                          f"beta-neutral (short {beta:.2f}x SPY)")
    rows += [s_b, s_s, s_p, s_n]
    # unconditional pair control
    rpa = (fwd_lag(PX["BASKET"], h, 1) - beta * fwd_lag(PX["SPY"], h, 1)).dropna()
    rows.append(summarize(rpa.values, "CTRL beta-neutral, ALL days"))
    rp_all = (fwd_lag(PX["BASKET"], h, 1) - fwd_lag(PX["SPY"], h, 1)).dropna()
    rows.append(summarize(rp_all.values, "CTRL equal-$ pair, ALL days"))
    show(rows, f"pair legs, h={h}")

# --------------------------------------------------------------------------
print("\n=== C7.5  is it a rates call?  TLT loading + TLT's own forward return ===")
tr = PX["TLT"].pct_change()
ok2 = br.notna() & tr.notna() & sr.notna()
X = np.column_stack([sr[ok2].values, tr[ok2].values, np.ones(int(ok2.sum()))])
coef, *_ = np.linalg.lstsq(X, br[ok2].values, rcond=None)
print(f"  basket daily returns ~ SPY + TLT:  b_SPY={coef[0]:.3f}  b_TLT={coef[1]:.3f}")
for h in (3, 5, 10):
    _, dtsJ, epiJ, _ = stats(MASKJ, [("BASKET", 1.0)], h, "x")
    rt = fwd_lag(PX["TLT"], h, 1)
    a = summarize(rt.loc[epiJ].dropna().values, f"TLT fwd on joint episodes h={h}")
    b = summarize(rt.dropna().values, f"TLT all days h={h}")
    show([a, b], f"TLT forward, h={h}")

# --------------------------------------------------------------------------
print("\n=== C7.6  definition neighbours ===")
print("  (a) XLP+XLU only -- drops XLRE, extends history to XLU/XLP inception")
cal2 = S["XLU"].index
R2 = pd.DataFrame({t: pct_rank(S[t], 21).reindex(cal2) for t in ["XLP", "XLU"]}).dropna()
sd2 = spy_dist.reindex(R2.index)
b2 = None
for t in ["XLP", "XLU"]:
    s = S[t].reindex(cal2).ffill()
    n = s / s.iloc[0]
    b2 = n / 2 if b2 is None else b2 + n / 2
PX2 = pd.DataFrame({"BASKET": b2, "SPY": spy.reindex(cal2).ffill()}).dropna()
m2 = ((R2 <= 20).all(axis=1) & (sd2 >= -0.01)).reindex(PX2.index, fill_value=False)
print(f"    2-name joint cell: {int(m2.sum())} days over "
      f"{PX2.index[0].date()}..{PX2.index[-1].date()}")
for h in (3, 5, 10):
    r = fwd_lag(PX2["BASKET"], h, 1)
    dts = PX2.index[m2.values & r.notna().values]
    epi = declusters(dts, h, PX2.index)
    rows = [summarize(r.loc[epi].values, f"2-name joint h={h} (epi {len(epi)})"),
            summarize(r.dropna().values, f"all days h={h}")]
    rp = (fwd_lag(PX2["BASKET"], h, 1) - fwd_lag(PX2["SPY"], h, 1))
    rows.append(summarize(rp.loc[epi].dropna().values, f"2-name pair vs SPY h={h}"))
    rows.append(summarize(rp.dropna().values, f"pair ALL days h={h}"))
    show(rows, "")

print("\n  (b) rank-threshold neighbours on the 3-name joint (h=5, long basket)")
rows = []
for thr in (10, 15, 20, 25, 30):
    mk = ((R <= thr).all(axis=1) & near_hi).reindex(PX.index, fill_value=False)
    s0, dts, epi, _ = stats(mk, [("BASKET", 1.0)], 5, f"rank<={thr}")
    s0["n_days"] = len(dts)
    rows.append(s0)
show(rows, "rank threshold")

rows = []
for nh in (0.005, 0.01, 0.02, 0.03, 0.05):
    mk = ((count == 3) & (sd >= -nh)).reindex(PX.index, fill_value=False)
    s0, dts, epi, _ = stats(mk, [("BASKET", 1.0)], 5, f"SPY within {100*nh:.1f}%")
    s0["n_days"] = len(dts)
    rows.append(s0)
show(rows, "SPY near-high threshold (h=5, long basket)")

# --------------------------------------------------------------------------
print("\n=== C7.7  era / midterm split on the joint cell (h=5, long basket) ===")
s0, dts, epi, r = stats(MASKJ, [("BASKET", 1.0)], 5, "joint")
v = r.loc[epi].values
yrs = pd.DatetimeIndex(epi).year
show([summarize(v[yrs % 4 == 2], "midterm yrs"),
      summarize(v[yrs % 4 != 2], "non-midterm")], "midterm split")
print(f"  episode years: {sorted(yrs.tolist())}")
print("\n  cost bar: 3 sector ETFs ~6bps each on an equal-weight basket ~= 6 bps "
      "round trip for the basket; the pair doubles it to ~8-10 bps. Need >=5x.")
