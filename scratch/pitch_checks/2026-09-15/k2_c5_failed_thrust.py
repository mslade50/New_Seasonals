"""C5 round 1: failed thrust. Single name with 5d rank >= 80 on t-1 that falls
>= 1.0 Wilder ATR on day t (prior-day ATR) while SPY 1d > -1%; long the name
hedged by trailing-252 SPY beta, h=1..10, lag=1. Pooled over the tape's single
names. Date-clustered: average across names per date first, then across dates.
Gate attribution (no drop / no thrust / no SPY gate), dose, cluster size, eras,
earnings, and same-date cross-sectional excess."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitch_lab import *  # noqa
from k2_common import *  # noqa

import numpy as np
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

pd.set_option("display.width", 250)
SN = single_names()
px, cal, P = load_panels(sorted(SN) + ["SPY", "^VIX"])
NAMES = [t for t in sorted(SN) if t in px]
print("single names:", len(NAMES))
D = derive(px, cal)
C = P["Close"]
ret1, sh, r5 = D["ret1"], D["shock"], D["r5"]
bS = rolling_beta(ret1[NAMES + ["SPY"]], "SPY")
spy1 = ret1["SPY"]
spy_ok = (spy1 > -0.01).to_numpy()[:, None]
sma200 = C["SPY"].rolling(200).mean()
above = (C["SPY"] > sma200).to_numpy()
vix = C["^VIX"]
vq = vix.quantile([1 / 3, 2 / 3]).values
vixa = vix.to_numpy()

r5p = r5[NAMES].shift(1).to_numpy()
shk = sh[NAMES].to_numpy()
thr = r5p >= 80
drop = shk <= -1.0
CELL = thr & drop & spy_ok

# earnings: day of print or the session after
ern = pd.read_parquet(ROOT / "data" / "earnings_calendar.parquet", columns=["ticker", "date"])
ern["date"] = pd.to_datetime(ern["date"])
EARN = np.zeros_like(CELL)
for j, t in enumerate(NAMES):
    e = ern.loc[ern["ticker"] == t, "date"]
    pos = cal.searchsorted(pd.DatetimeIndex(e))
    for p in pos:
        if p < len(cal):
            EARN[p, j] = True
            if p + 1 < len(cal):
                EARN[p + 1, j] = True

cnt = CELL.sum(axis=1)
CNT = np.broadcast_to(cnt[:, None], CELL.shape)

PAIR, XS = {}, {}
for h in (1, 2, 3, 5, 10):
    F = fwd_panel(C, h)
    pr = F[NAMES].sub(bS[NAMES].mul(F["SPY"], axis=0)).to_numpy()
    PAIR[h] = pr
    XS[h] = pr - np.nanmean(pr, axis=1, keepdims=True)
RAW5 = fwd_panel(C, 5)[NAMES].to_numpy()


def nw_t(x: np.ndarray, L: int) -> float:
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 5:
        return np.nan
    e = x - x.mean()
    s = (e * e).sum() / n
    for lag in range(1, L + 1):
        s += 2 * (1 - lag / (L + 1)) * (e[lag:] * e[:-lag]).sum() / n
    return x.mean() / np.sqrt(s / n)


def stat(M: np.ndarray, h: int, label: str, arr=None) -> dict:
    A = PAIR[h] if arr is None else arr
    X = XS[h]
    Mv = M & ~np.isnan(A)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.where(Mv, A, 0).sum(1) / Mv.sum(1)
        sx = np.where(Mv, X, 0).sum(1) / Mv.sum(1)
    ok = Mv.sum(1) > 0
    dates = cal[ok]
    sv, sxv = s[ok], sx[ok]
    if len(sv) == 0:
        return {"label": label, "n_dates": 0}
    keep = declusters(dates, max(h, 5), cal)
    kp = pd.Series(sv, index=dates).loc[keep].values
    w = int((kp > 0).sum())
    yrs = pd.Series(sv, index=dates).groupby(dates.year).mean()
    return {"label": label, "name_days": int(Mv.sum()), "n_dates": len(sv),
            "date_mean_pct": 100 * sv.mean(), "nw_t": nw_t(sv, h),
            "xs_excess_pp": 100 * np.nanmean(sxv), "xs_nw_t": nw_t(sxv, h),
            "hit_dates": 100 * (sv > 0).mean(),
            "decl_n": len(kp), "decl_mean_pct": 100 * kp.mean(),
            "decl_rec": f"{w}-{len(kp) - w}", "decl_sign_p": round(sign_test(w, len(kp)), 4),
            "pos_yrs": f"{int((yrs > 0).sum())}/{len(yrs)}"}


def ctrl(h: int) -> float:
    return 100 * np.nanmean(np.nanmean(PAIR[h], axis=1))


live_i = len(cal) - 1
print("\nlive 09-14 cell names:", [NAMES[j] for j in np.where(CELL[live_i])[0]],
      " count", int(cnt[live_i]))
for t in ["GLW", "ADI", "AMD", "INTC", "HPQ"]:
    j = NAMES.index(t)
    print(f"  {t}: r5@t-1 {r5p[live_i, j]:.1f}  shock {shk[live_i, j]:+.3f}  beta {bS[t].iloc[-1]:.2f}")

print("\n##### 1. horizons: CELL vs controls #####")
rows = []
for h in (1, 2, 3, 5, 10):
    r = stat(CELL, h, f"h={h} CELL")
    r["ctrl_all_pct"] = round(ctrl(h), 3)
    rows.append(r)
show(rows, "CELL (pair = name - beta*SPY; xs_excess = vs same-date mean of all single names)")

# local +/-126 control on the pooled name-days (h=5)
for h in (1, 5):
    A = PAIR[h]
    vals, cv = [], []
    for j in range(len(NAMES)):
        e = CELL[:, j].astype(float)
        if e.sum() == 0:
            continue
        win = np.convolve(e, np.ones(253), mode="same") > 0
        keep = win & ~CELL[:, j] & ~np.isnan(A[:, j])
        vals.append(A[keep, j])
        c = A[CELL[:, j] & ~np.isnan(A[:, j]), j]
        cv.append(c)
    print(f"h={h}: pooled name-day cell mean {100*np.concatenate(cv).mean():+.3f}%  "
          f"local +/-126 ex-trigger {100*np.concatenate(vals).mean():+.3f}%")

print("\n##### 2. gate attribution (h=1,3,5) #####")
for h in (1, 3, 5):
    rows = [stat(CELL, h, f"h={h} CELL r5>=80 & drop>=1ATR & SPY>-1%"),
            stat(thr & spy_ok, h, "thrust only (r5>=80, SPY ok), any day-t"),
            stat(thr & ~drop & spy_ok, h, "thrust, NO drop"),
            stat(drop & spy_ok, h, "drop only (any r5)"),
            stat(drop & ~thr & spy_ok & (r5p < 50), h, "drop, r5@t-1 < 50"),
            stat(thr & drop, h, "CELL without SPY gate"),
            stat(thr & drop & ~spy_ok, h, "CELL on SPY <= -1% days")]
    show(rows, f"gate attribution h={h}")

print("\n##### 3. dose + neighbours (h=1,3,5) #####")
for h in (1, 3, 5):
    rows = []
    for lo, hi in [(-1.5, -1.0), (-2.0, -1.5), (-3.0, -2.0), (-99, -3.0)]:
        rows.append(stat(thr & spy_ok & (shk <= hi) & (shk > lo), h, f"drop in ({lo},{hi}]"))
    for rr in (70, 90, 95):
        rows.append(stat((r5p >= rr) & drop & spy_ok, h, f"r5>={rr} & drop>=1"))
    rows.append(stat(thr & spy_ok & (shk <= -2.0), h, "r5>=80 & drop>=2"))
    show(rows, f"dose/neighbours h={h}")

print("\n##### 4. clustering, eras, regime, earnings (h=1,3,5) #####")
for h in (1, 3, 5):
    rows = [stat(CELL & (CNT == 1), h, "isolated (1 name that date)"),
            stat(CELL & (CNT >= 2) & (CNT <= 3), h, "2-3 names"),
            stat(CELL & (CNT >= 4), h, "4+ names"),
            stat(CELL & (CNT >= 8), h, "8+ names")]
    pre = np.asarray(cal < pd.Timestamp("2018-01-01"))[:, None]
    rows += [stat(CELL & pre, h, "pre-2018"), stat(CELL & ~pre, h, "2018+"),
             stat(CELL & above[:, None], h, "SPY above 200d"),
             stat(CELL & ~above[:, None], h, "SPY below 200d"),
             stat(CELL & (vixa <= vq[0])[:, None], h, "VIX low tercile"),
             stat(CELL & ((vixa > vq[0]) & (vixa <= vq[1]))[:, None], h, "VIX mid"),
             stat(CELL & (vixa > vq[1])[:, None], h, "VIX high"),
             stat(CELL & EARN, h, "earnings day/after"),
             stat(CELL & ~EARN, h, "not earnings"),
             stat(CELL & ~EARN & (CNT >= 4), h, "not earnings & 4+ names")]
    ex = ~np.isin(cal.year, [2008, 2009, 2020])[:, None]
    rows.append(stat(CELL & ex, h, "ex 2008/09/2020"))
    show(rows, f"splits h={h}")

print("\n##### 5. raw long (unhedged) h=5, and cost #####")
show([stat(CELL, 5, "CELL raw long h=5", RAW5)], "raw")
r5c = stat(CELL, 5, "x")
r1c = stat(CELL, 1, "x")
for lbl, r in [("h=1", r1c), ("h=5", r5c)]:
    print(f"{lbl}: date-mean {r['date_mean_pct']:+.3f}% = {100*r['date_mean_pct']:.1f} bp; "
          f"vs 10-13 bp two-leg round trip -> {100*r['date_mean_pct']/11.5:.1f}x; "
          f"xs excess {r['xs_excess_pp']:+.3f}pp -> {100*r['xs_excess_pp']/11.5:.1f}x")

# by-year table h=5
A = PAIR[5]
Mv = CELL & ~np.isnan(A)
with np.errstate(invalid="ignore", divide="ignore"):
    s = np.where(Mv, A, 0).sum(1) / Mv.sum(1)
ok = Mv.sum(1) > 0
ser = pd.Series(s[ok], index=cal[ok])
print("\nby-year date-mean h=5 (pct) and n_dates:")
print(pd.DataFrame({"mean_pct": 100 * ser.groupby(ser.index.year).mean(),
                    "n": ser.groupby(ser.index.year).size()}).round(3).T.to_string())
