"""A2 round 1 - the five-day flush inside a top-decile 63-day trend.

Pre-specified cell: pct_rank(close,5,252) <= 5 AND pct_rank(close,63,252) >= 85
on the signal close, long the instrument, lag=1 MOC entry.

FAMILY declared BEFORE any measurement (surface map / handover text): the nine
SPDR sectors plus SMH IBB XBI IHI KRE ITA XME XRT XHB IYR OIH. All 20 are in
master_prices (checked separately), so nothing is dropped.

Live: IBB r5 0.4 / r63 91.3 is the only clean instance.

Round-1 obligations answered here:
 1. pattern vs real controls (battery on the DEFENDED instrument, IBB)
 2. N / worst window / era
 4. standalone worth after cost
 5. cost (IBB ~3-4 bp round trip)
 6. tomorrow-specific: CPI is on the ENTRY session (k=0), FOMC +3, quad +5
Plus the two attacks that matter most, run in round 1 because they are cheap:
 - gate attribution on the 63-day leg (dose response across r63 buckets)
 - pooled family fixed-effect / Cochran Q / I-squared
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

ASOF = pd.Timestamp("2026-09-10")

FAMILY = ["XLK", "XLF", "XLV", "XLY", "XLP", "XLI", "XLE", "XLU", "XLB",
          "SMH", "IBB", "XBI", "IHI", "KRE", "ITA", "XME", "XRT", "XHB",
          "IYR", "OIH"]
DEFENDED = "IBB"
H = 5
COST_BPS = 4.0

px = load_prices(FAMILY)
panel = pd.DataFrame({t: px[t]["Close"] for t in FAMILY})
# each ticker's own calendar is honoured by rolling_on_valid inside pct_rank
R5 = pd.DataFrame({t: pct_rank(panel[t], 5) for t in FAMILY})
R63 = pd.DataFrame({t: pct_rank(panel[t], 63) for t in FAMILY})
R21 = pd.DataFrame({t: pct_rank(panel[t], 21) for t in FAMILY})

print("=" * 78)
print("0. LIVE STATE on the 2026-09-10 close (r5 <= 5 AND r63 >= 85)")
live = []
for t in FAMILY:
    r5, r63 = R5[t].loc[:ASOF].iloc[-1], R63[t].loc[:ASOF].iloc[-1]
    flag = "  <== FIRES" if (r5 <= 5 and r63 >= 85) else ""
    live.append((t, round(r5, 1), round(r63, 1), flag))
for row in live:
    print(f"  {row[0]:<5} r5={row[1]:>5} r63={row[2]:>5}{row[3]}")

# ---------------------------------------------------------------------------
# 1. the defended single-name cell: full battery
# ---------------------------------------------------------------------------
ibb = pd.DataFrame({DEFENDED: panel[DEFENDED].dropna()})
m_ibb = ((R5[DEFENDED] <= 5) & (R63[DEFENDED] >= 85)).reindex(ibb.index,
                                                              fill_value=False)
variants = {
    "r5<=2 & r63>=85": (R5[DEFENDED] <= 2) & (R63[DEFENDED] >= 85),
    "r5<=10 & r63>=85": (R5[DEFENDED] <= 10) & (R63[DEFENDED] >= 85),
    "r5<=15 & r63>=85": (R5[DEFENDED] <= 15) & (R63[DEFENDED] >= 85),
    "r5<=5 & r63>=75": (R5[DEFENDED] <= 5) & (R63[DEFENDED] >= 75),
    "r5<=5 & r63>=90": (R5[DEFENDED] <= 5) & (R63[DEFENDED] >= 90),
    "r5<=5, NO r63 gate": (R5[DEFENDED] <= 5),
    "r5<=5 & r63<50 (COMPLEMENT)": (R5[DEFENDED] <= 5) & (R63[DEFENDED] < 50),
}
variants = {k: v.reindex(ibb.index, fill_value=False) for k, v in variants.items()}
battery(ibb, m_ibb, [(DEFENDED, 1.0)], H,
        f"A2 DEFENDED: {DEFENDED} r5<=5 & r63>=85", COST_BPS,
        variants=variants, event_kinds=("cpi", "fomc_decision"))

print("\n  IBB horizon scan (episodes, gap=h):")
sig_ibb = ibb.index[m_ibb.values]
show(horizon_scan(ibb, sig_ibb, [(DEFENDED, 1.0)], hs=(1, 2, 3, 5, 7, 10)),
     "IBB horizon scan")

# ---------------------------------------------------------------------------
# 2. POOLED family - stack every (ticker, date) trigger
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print(f"2. POOLED FAMILY (N={len(FAMILY)}), h={H}, lag=1")

FW = {t: fwd_lag(panel[t].dropna(), H, 1) for t in FAMILY}


def cell(tkr, mask_fn):
    """(dates, vals) for a ticker under a boolean-mask function of r5/r21/r63."""
    m = mask_fn(tkr)
    f = FW[tkr]
    idx = f.dropna().index
    m = m.reindex(idx, fill_value=False)
    d = idx[m.values]
    return d, f.loc[d].values


def pooled(mask_fn, label):
    rows, all_d, all_v, all_t = [], [], [], []
    for t in FAMILY:
        d, v = cell(t, mask_fn)
        if len(v) == 0:
            continue
        all_d.extend(list(d))
        all_v.extend(list(v))
        all_t.extend([t] * len(v))
    v = np.asarray(all_v, float)
    if len(v) == 0:
        print(f"  {label}: NO TRIGGERS")
        return None
    # date-clustered t: average across tickers within a date first
    s = pd.Series(v, index=pd.DatetimeIndex(all_d))
    daily = s.groupby(level=0).mean()
    t_clu = daily.mean() / (daily.std(ddof=1) / np.sqrt(len(daily))) if len(daily) > 1 else np.nan
    print(f"  {label:<42} N={len(v):>5} days={len(daily):>4} "
          f"mean={100*v.mean():+.3f}% hit={100*(v>0).mean():.1f}% "
          f"t_dateclust={t_clu:+.2f}")
    return pd.DataFrame({"ticker": all_t, "date": all_d, "ret": v})


def mk(r5_max=None, r63_min=None, r63_max=None):
    def f(t):
        m = pd.Series(True, index=R5.index)
        if r5_max is not None:
            m &= (R5[t] <= r5_max)
        if r63_min is not None:
            m &= (R63[t] >= r63_min)
        if r63_max is not None:
            m &= (R63[t] < r63_max)
        return m.fillna(False)
    return f


base = pooled(lambda t: pd.Series(True, index=R5.index), "ALL DAYS (pool baseline)")
join = pooled(mk(5, 85), "JOIN r5<=5 & r63>=85  [DEFENDED]")
print()
print("  --- 4. GATE ATTRIBUTION on the 63-day leg ---")
flush = pooled(mk(5, None), "r5<=5 alone (parent)")
for lo, hi in [(0, 10), (10, 25), (25, 50), (50, 75), (75, 85), (85, 101)]:
    pooled(mk(5, lo, hi), f"   r5<=5 & r63 in [{lo},{hi})")
print()
pooled(mk(None, 85), "r63>=85 alone (trend leg)")
pooled(mk(5, None, 85), "r5<=5 & r63<85 (DISCARDED COMPLEMENT)")

print()
print("  --- threshold ladder r5 x r63 (pooled, date-clustered t) ---")
for a in (2, 5, 10, 15):
    for b in (75, 85, 90):
        pooled(mk(a, b), f"   r5<={a:<2} & r63>={b}")

# ---------------------------------------------------------------------------
# 3. per-member excess, fixed effect, Cochran Q, I-squared
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("3. PER-MEMBER EXCESS vs own drift (fixed effect / Cochran Q / I^2)")
mem = []
for t in FAMILY:
    d, v = cell(t, mk(5, 85))
    f = FW[t].dropna()
    if len(v) < 2:
        mem.append((t, len(v), np.nan, np.nan, np.nan))
        continue
    exc = v.mean() - f.mean()
    se = np.sqrt(v.var(ddof=1) / len(v) + f.var(ddof=1) / len(f))
    mem.append((t, len(v), 100 * exc, 100 * se, exc / se))
md = pd.DataFrame(mem, columns=["ticker", "n", "excess_pct", "se_pct", "t"])
print(md.round(3).to_string(index=False))
ok = md.dropna(subset=["excess_pct"])
w = 1.0 / (ok["se_pct"] ** 2)
fe = float((w * ok["excess_pct"]).sum() / w.sum())
fe_se = float(np.sqrt(1.0 / w.sum()))
Q = float((w * (ok["excess_pct"] - fe) ** 2).sum())
dfq = len(ok) - 1
I2 = max(0.0, 100 * (Q - dfq) / Q) if Q > 0 else 0.0
from scipy import stats as _st
print(f"\n  fixed-effect common excess = {fe:+.3f}% (se {fe_se:.3f}, "
      f"t {fe/fe_se:+.2f})")
print(f"  Cochran Q = {Q:.2f} on {dfq} df, p = {1-_st.chi2.cdf(Q, dfq):.4f}, "
      f"I^2 = {I2:.1f}%")
print(f"  members with POSITIVE excess: {int((ok['excess_pct']>0).sum())} of {len(ok)}")
if DEFENDED in set(ok["ticker"]):
    r = ok.set_index("ticker").loc[DEFENDED]
    rank = int((ok["excess_pct"] > r["excess_pct"]).sum()) + 1
    print(f"  {DEFENDED} excess {r['excess_pct']:+.3f}% ranks {rank} of {len(ok)}")

# ---------------------------------------------------------------------------
# 4. is it just short-term reversal wearing a sector label?
#    long the family's WORST 5-day performer on the same dates
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("4. GENERIC CROSS-SECTIONAL REVERSAL CONTROL")
r5_raw = pd.DataFrame({t: panel[t] / panel[t].shift(5) - 1.0 for t in FAMILY})
fwd_all = pd.DataFrame({t: FW[t] for t in FAMILY})
trig_dates = pd.DatetimeIndex(sorted(set(join["date"]))) if join is not None else pd.DatetimeIndex([])
gen, defd = [], []
for d in trig_dates:
    if d not in r5_raw.index:
        continue
    row = r5_raw.loc[d].dropna()
    fr = fwd_all.loc[d].dropna()
    common = row.index.intersection(fr.index)
    if len(common) < 10:
        continue
    worst = row[common].idxmin()
    gen.append(fr[worst])
    defd.append(join[(join["date"] == d)]["ret"].mean())
gen, defd = np.asarray(gen, float), np.asarray(defd, float)
show([summarize(defd, "DEFENDED cell (date-averaged)"),
      summarize(gen, "GENERIC worst-5d-in-family, same dates")],
     "generic reversal on the identical dates")
if len(gen) > 2:
    dif = defd - gen
    print(f"  paired difference = {100*dif.mean():+.3f}% "
          f"t={dif.mean()/(dif.std(ddof=1)/np.sqrt(len(dif))):+.2f} "
          f"record {(dif>0).sum()}-{(dif<=0).sum()}")

# ---------------------------------------------------------------------------
# 5. IBB episode dates - is the flush an FDA/policy tape or a market tape?
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("5. IBB EPISODES with the market tape on the same day (SPY 5d)")
spy = load_prices(["SPY"])["SPY"]["Close"]
epi = declusters(sig_ibb, H, ibb.index)
r5s = spy / spy.shift(5) - 1.0
fwd_i = FW[DEFENDED]
rows = []
for d in epi:
    rows.append({
        "date": str(d.date()),
        "IBB_5d_pct": round(100 * float(panel[DEFENDED].loc[d] /
                                        panel[DEFENDED].shift(5).loc[d] - 1), 2),
        "SPY_5d_pct": round(100 * float(r5s.reindex([d]).iloc[0]), 2),
        "fwd5_pct": round(100 * float(fwd_i.reindex([d]).iloc[0]), 2)
        if d in fwd_i.index else np.nan,
    })
print(pd.DataFrame(rows).to_string(index=False))
