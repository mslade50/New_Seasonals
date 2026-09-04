"""Adversarial verification of dial-design.md claims.

Independent recompute, no reuse of the researcher's code.
Conventions taken from the md's stated setup:
- trades: non-OVS, signal >= 2016-08-01, joined merge_asof backward tol 7d
  to each dial's trailing-SMA daily score; require all three 10d-MA dials non-NaN.
- significance: Welch t between monthly mean R of flagged vs unflagged trades
  (grouped by signal month).
- LOYO: drop each signal-year, recompute t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

FRAG = pd.read_parquet("data/rd2_fragility.parquet")
TRADES = pd.read_parquet("data/backtest_trades_full.parquet")

DIALS = ["5d", "21d", "63d"]


def smooth(col: str, w: int) -> pd.Series:
    return FRAG[col].rolling(w, min_periods=1).mean()


def base_trades() -> pd.DataFrame:
    t = TRADES[TRADES["Strategy"] != "Overbot Vol Spike"].copy()
    t = t[t["Signal Date"] >= "2016-08-01"].sort_values("Signal Date")
    return t


def join_dial(t: pd.DataFrame, col: str, w: int, name: str) -> pd.DataFrame:
    s = smooth(col, w).rename(name).to_frame().reset_index()
    s.columns = ["Date", name]
    s = s.dropna()
    out = pd.merge_asof(
        t.sort_values("Signal Date"), s.sort_values("Date"),
        left_on="Signal Date", right_on="Date",
        direction="backward", tolerance=pd.Timedelta("7D"),
    ).drop(columns=["Date"])
    return out


def monthly_t(df: pd.DataFrame, flag: pd.Series) -> tuple[float, float, int, float, float]:
    """Welch t on monthly mean R, flagged vs unflagged."""
    d = df.copy()
    d["flag"] = flag.values
    d["mo"] = d["Signal Date"].dt.to_period("M")
    hi = d[d["flag"]].groupby("mo")["R_Multiple"].mean()
    lo = d[~d["flag"]].groupby("mo")["R_Multiple"].mean()
    if len(hi) < 2 or len(lo) < 2:
        return np.nan, np.nan, int(d["flag"].sum()), np.nan, np.nan
    t, p = stats.ttest_ind(hi, lo, equal_var=False)
    return t, p, int(d["flag"].sum()), d.loc[d["flag"], "R_Multiple"].mean(), d.loc[~d["flag"], "R_Multiple"].mean()


def loyo(df: pd.DataFrame, flag: pd.Series) -> tuple[float, float]:
    d = df.copy()
    d["flag"] = flag.values
    ts = []
    for y in sorted(d["Signal Date"].dt.year.unique()):
        sub = d[d["Signal Date"].dt.year != y]
        t, _, _, _, _ = monthly_t(sub, sub["flag"])
        if not np.isnan(t):
            ts.append(t)
    return min(ts), max(ts)


# ---------- build the joined table (all three dials, 10d MA) ----------
t0 = base_trades()
for c in DIALS:
    t0 = join_dial(t0, c, 10, f"ma10_{c}")
tj = t0.dropna(subset=[f"ma10_{c}" for c in DIALS]).copy()
print(f"N non-OVS 2016-08+ with all 3 dials (10d MA): {len(tj)}  (claimed 1136)")

# daily distribution for p90 cutoffs (days where all three 10d-MA non-NaN)
daily = pd.DataFrame({c: smooth(c, 10) for c in DIALS}).dropna()
print(f"daily rows all-dials non-NaN: {len(daily)} (claimed 2443)")
p90 = daily.quantile(0.90)
print("p90 cutoffs (10d MA):", p90.round(2).to_dict(), " (claimed 5d 28.8 / 21d 38.9 / 63d 54.9)")

print("\n================ CLAIM 1: 63d>=50 headline ================")
flag = tj["ma10_63d"] >= 50
t, p, n, hi, lo = monthly_t(tj, flag)
lm, lM = loyo(tj, flag)
print(f"N_hi={n} avgR_hi={hi:+.3f} avgR_lo={lo:+.3f} t={t:.2f} p={p:.3f} LOYO=[{lm:.2f},{lM:.2f}]")
print("claimed: N=242 +0.19 vs +0.65 t=-2.73 p=.010 LOYO [-2.95,-2.05]")

print("\n================ CLAIM 2: combinations ================")
combos = {
    "63d>=50": tj["ma10_63d"] >= 50,
    "63d p90": tj["ma10_63d"] >= p90["63d"],
    "21d p90": tj["ma10_21d"] >= p90["21d"],
    "5d p90": tj["ma10_5d"] >= p90["5d"],
    "any p90": (tj["ma10_63d"] >= p90["63d"]) | (tj["ma10_21d"] >= p90["21d"]) | (tj["ma10_5d"] >= p90["5d"]),
    "63d>=50 OR 21d p90": (tj["ma10_63d"] >= 50) | (tj["ma10_21d"] >= p90["21d"]),
}
for name, fl in combos.items():
    t, p, n, hi, lo = monthly_t(tj, fl)
    row = f"{name:22s} N={n:4d} hi={hi:+.3f} lo={lo:+.3f} t={t:.2f} p={p:.3f}"
    if name in ("any p90", "63d>=50", "63d>=50 OR 21d p90"):
        lm, lM = loyo(tj, fl)
        row += f" LOYO=[{lm:.2f},{lM:.2f}]"
    print(row)

print("\n================ CLAIM 3: 5d dial dead ================")
best = (0.0, None)
# thresholds p60..p95 at w=10
for q in [0.60, 0.70, 0.80, 0.90, 0.95]:
    thr = daily["5d"].quantile(q)
    fl = tj["ma10_5d"] >= thr
    t, p, n, hi, lo = monthly_t(tj, fl)
    print(f"w=10 q={q:.2f} thr={thr:5.1f} N={n:4d} hi={hi:+.3f} t={t:+.2f} p={p:.3f}")
    if not np.isnan(t) and t < best[0]:
        best = (t, ("w10", q))
# windows 1..21 at own p90
for w in [1, 3, 5, 8, 10, 13, 15, 18, 21]:
    s = smooth("5d", w)
    tw = join_dial(base_trades(), "5d", w, "x").dropna(subset=["x"])
    thr = s.dropna().loc[daily.index.min():].quantile(0.90)
    fl = tw["x"] >= thr
    t, p, n, hi, lo = monthly_t(tw, fl)
    print(f"w={w:2d} p90 thr={thr:5.1f} N={n:4d} hi={hi:+.3f} t={t:+.2f} p={p:.3f}")
    if not np.isnan(t) and t < best[0]:
        best = (t, ("p90", w))
print(f"best (most negative) 5d t = {best[0]:.2f} at {best[1]}  (claimed best -0.87)")

print("\n================ CLAIM 4: 63d threshold sweep (10d MA) ================")
for thr in [35, 40, 42.5, 45, 47.5, 50, 52.5, 55, 60, 65]:
    fl = tj["ma10_63d"] >= thr
    t, p, n, hi, lo = monthly_t(tj, fl)
    print(f"thr={thr:5.1f} N={n:4d} hi={hi:+.3f} lo={lo:+.3f} t={t:+.2f} p={p:.3f}")

print("\n================ CLAIM 5: MA window sweep, 63d thr=50 ================")
for w in [1, 3, 5, 8, 10, 13, 15, 18, 21]:
    tw = join_dial(base_trades(), "63d", w, "x").dropna(subset=["x"])
    fl = tw["x"] >= 50
    t, p, n, hi, lo = monthly_t(tw, fl)
    print(f"w={w:2d} N_total={len(tw)} N_hi={n:4d} hi={hi:+.3f} t={t:+.2f} p={p:.3f}")

print("\n================ CLAIM 6: hysteresis / whipsaw (63d, 10d MA daily) ================")
s63 = smooth("63d", 10).dropna()
yrs = (s63.index[-1] - s63.index[0]).days / 365.25


def episodes(on: pd.Series):
    on = on.astype(int)
    starts = ((on.diff() == 1) | ((on == 1) & (on.index == on.index[0]))).sum()
    flips = int(on.diff().abs().sum())
    # episode lengths
    grp = (on.diff().fillna(0) != 0).cumsum()
    lens = on.groupby(grp).agg(["first", "size"])
    ep = lens[lens["first"] == 1]["size"]
    return flips, len(ep), ep


def hyst(series: pd.Series, on_thr: float, off_thr: float) -> pd.Series:
    st = np.zeros(len(series), dtype=bool)
    cur = False
    for i, v in enumerate(series.values):
        if not cur and v >= on_thr:
            cur = True
        elif cur and v < off_thr:
            cur = False
        st[i] = cur
    return pd.Series(st, index=series.index)


for name, on in [
    ("plain 50", s63 >= 50),
    ("hyst 50/45", hyst(s63, 50, 45)),
    ("hyst 50/40", hyst(s63, 50, 40)),
    ("plain 55", s63 >= 55),
]:
    flips, neps, ep = episodes(on)
    # map to trades
    reg = on.rename("on").to_frame().reset_index()
    reg.columns = ["Date", "on"]
    tt = pd.merge_asof(base_trades().sort_values("Signal Date"), reg,
                       left_on="Signal Date", right_on="Date",
                       direction="backward", tolerance=pd.Timedelta("7D")).dropna(subset=["on"])
    t, p, n, hi, lo = monthly_t(tt, tt["on"].astype(bool))
    print(f"{name:10s} flips/yr={flips/yrs:.1f} episodes={neps} eps<=5d={(ep<=5).sum()} "
          f"eps<=10d={(ep<=10).sum()} median_len={ep.median():.1f} %on={on.mean()*100:.1f} "
          f"N_tr={n} hi={hi:+.2f} t={t:+.2f} p={p:.3f}")
print(f"span years = {yrs:.1f}")

print("\n================ CLAIM 7: runner-up 21d dial / 21d MA / thr~36 ================")
t21 = join_dial(base_trades(), "21d", 21, "x").dropna(subset=["x"])
s21 = smooth("21d", 21).dropna()
thr21 = s21.loc[daily.index.min():].quantile(0.90)
print(f"21d/21MA p90 threshold = {thr21:.1f} (claimed ~36)")
fl21 = t21["x"] >= thr21
t, p, n, hi, lo = monthly_t(t21, fl21)
lm, lM = loyo(t21, fl21)
print(f"N={n} hi={hi:+.3f} lo={lo:+.3f} t={t:.2f} p={p:.3f} LOYO=[{lm:.2f},{lM:.2f}]")
print("claimed t=-3.93 LOYO [-4.67,-3.14] N=145")
# overlap with 63d>=50 flags at trade level (align by trade identity)
key = ["Strategy", "Ticker", "Signal Date"]
f63 = set(map(tuple, tj.loc[tj["ma10_63d"] >= 50, key].values))
f21 = set(map(tuple, t21.loc[fl21, key].values))
uniq = f21 - f63
uniq_mask = t21.apply(lambda r: (r["Strategy"], r["Ticker"], r["Signal Date"]) in uniq, axis=1) & fl21
print(f"21d flags={len(f21)} overlap with 63d>=50={len(f21 & f63)} unique={len(uniq)} "
      f"unique avgR={t21.loc[uniq_mask,'R_Multiple'].mean():+.2f}")
print("claimed 123/145 overlap, 22 unique avg +0.49")

print("\n================ CLAIM 8: 2026 YTD inversion ================")
t26 = tj[tj["Signal Date"].dt.year == 2026]
fl = t26["ma10_63d"] >= 50
print(f"2026 flagged avgR={t26.loc[fl,'R_Multiple'].mean():+.2f} N={fl.sum()} | "
      f"unflagged avgR={t26.loc[~fl,'R_Multiple'].mean():+.2f} N={(~fl).sum()}")
print("claimed flagged +0.49..+0.55 N~25 vs unflagged -0.09..-0.12 N~68")

print("\n================ extra: sensitivity checks ================")
# (a) live join convention: ffill limit 5 days instead of 7-day tolerance
s = smooth("63d", 10)
sd = s.reindex(pd.date_range(s.index.min(), s.index.max())).ffill(limit=5)
tb = base_trades()
tb["v"] = sd.reindex(tb["Signal Date"]).values
tb = tb.dropna(subset=["v"])
t, p, n, hi, lo = monthly_t(tb, tb["v"] >= 50)
print(f"(a) ffill-limit-5 join: N_hi={n} hi={hi:+.3f} t={t:+.2f} p={p:.3f}")
# (b) trade-level (unclustered) t for reference
hi_tr = tj.loc[tj['ma10_63d'] >= 50, 'R_Multiple']
lo_tr = tj.loc[tj['ma10_63d'] < 50, 'R_Multiple']
t, p = stats.ttest_ind(hi_tr, lo_tr, equal_var=False)
print(f"(b) raw per-trade t={t:+.2f} p={p:.3f} (must NOT be the basis of claims)")
# (c) quarterly clustering (coarser)
d = tj.copy(); d['flag'] = d['ma10_63d'] >= 50
d['q'] = d['Signal Date'].dt.to_period('Q')
hq = d[d['flag']].groupby('q')['R_Multiple'].mean(); lq = d[~d['flag']].groupby('q')['R_Multiple'].mean()
t, p = stats.ttest_ind(hq, lq, equal_var=False)
print(f"(c) quarterly-clustered t={t:+.2f} p={p:.3f} (nQhi={len(hq)}, nQlo={len(lq)})")
