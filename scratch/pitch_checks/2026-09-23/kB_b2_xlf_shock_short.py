"""kB_b2 -- ADVERSARIAL round 1 on CANDIDATE B2.

B2: SHORT XLF against beta-SPY after a sector-idiosyncratic high-volume shock
day. Pre-specified sign: CONTINUATION (short the residual), h = 1..5.

Definitions (fixed before looking):
  beta_t   = OLS slope of sector vs SPY daily returns over the 126 sessions
             ending t-1 (ex-ante; the shock day does not enter its own beta)
  resid_t  = r_sector_t - beta_t * r_spy_t
  zres_t   = resid_t / sd(resid over the 126 sessions ending t-1)
  volx_t   = Volume_t / mean(Volume over the 63 sessions ending t-1)
  SHOCK    = resid <= -1.5% AND volx >= 2        (headline)
  SHOCK_Z  = zres <= -2   AND volx >= 2          (alternate, pre-declared)
  trade    = -(r_sector - beta_t * r_spy) forward, entry lag=1 MOC
Family: XLB XLE XLF XLI XLK XLP XLU XLV XLY XLC XLRE KRE (12). Order: FILTER then
DECLUSTER (gap = h) per member, then pool. Pre-specified, not searched: no
multiplicity charge beyond the 5 horizons, which are REPORTED, not picked.
Cost: ~2.5 bp RT per leg, pair = 2.5 x (1+beta).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change  # noqa

ROOT = Path(__file__).resolve().parents[3]
FAM = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLC", "XLRE", "KRE"]
LEG_BP = 2.5
HS = (1, 2, 3, 5)
P = load_prices(FAM + ["SPY"])
CAL = P["SPY"].index
px = pd.DataFrame({t: P[t]["Close"] for t in P}).reindex(CAL)
vol = pd.DataFrame({t: P[t]["Volume"] for t in P}).reindex(CAL)
DR = {t: _valid_pct_change(px[t], 1) for t in px.columns}

ST = {}
for t in FAM:
    j = pd.concat([DR[t], DR["SPY"]], axis=1, keys=["a", "b"]).dropna()
    b = (j["a"].rolling(126, min_periods=100).cov(j["b"]) /
         j["b"].rolling(126, min_periods=100).var()).shift(1)
    res = j["a"] - b * j["b"]
    sd = res.rolling(126, min_periods=100).std().shift(1)
    v = vol[t].dropna()
    v = v[v > 0]
    vx = (v / v.rolling(63, min_periods=50).mean().shift(1)).reindex(j.index)
    ST[t] = pd.DataFrame({"beta": b, "res": res, "z": res / sd, "volx": vx}).reindex(CAL)


def trade(t: str, h: int, lag: int = 1) -> pd.Series:
    return -(fwd_lag(px[t], h, lag) - ST[t]["beta"] * fwd_lag(px["SPY"], h, lag))


def shock(t, thr=-0.015, vx=2.0, zmode=False):
    s = ST[t]
    base = (s["z"] <= -2) if zmode else (s["res"] <= thr)
    if vx:
        base = base & (s["volx"] >= vx)
    return base.fillna(False)


print("live", CAL[-1].date())
for t in FAM:
    s = ST[t].iloc[-1]
    print(f"  {t:4s} beta {s.beta:.2f} resid {100*s.res:+.2f}% z {s.z:+.2f} volx {s.volx:.2f}")


def member_table(h, **kw):
    rows, V, D = [], [], []
    for t in FAM:
        ret = trade(t, h)
        m = shock(t, **kw)
        days = CAL[m.values & ret.notna().values]
        if len(days) == 0:
            rows.append({"ticker": t, "n": 0}); continue
        e = declusters(days, h, CAL)
        v = ret.loc[e].values
        span = (CAL >= e[0]) & (CAL <= e[-1]) & ret.notna().values
        base = ret[span].values
        se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
        rows.append({"ticker": t, "n": len(v), "mean_pct": 100 * v.mean(),
                     "drift_pct": 100 * base.mean(), "excess_pct": 100 * (v.mean() - base.mean()),
                     "hit": 100 * (v > 0).mean(), "se_pct": 100 * se})
        V.append(v); D.extend(list(e))
    return pd.DataFrame(rows), np.concatenate(V), pd.DatetimeIndex(D)


def fe(df):
    ok = (df["n"] > 1) & (df["se_pct"] > 0)
    ex, se = df.loc[ok, "excess_pct"].values, df.loc[ok, "se_pct"].values
    w = 1 / se**2
    m = (w * ex).sum() / w.sum(); s = np.sqrt(1 / w.sum())
    Q = (w * (ex - m) ** 2).sum(); k = len(ex) - 1
    I2 = max(0.0, 100 * (Q - k) / Q) if Q > 0 else 0.0
    return m, s, Q, k, I2, int((ex > 0).sum()), len(ex)


def dcl(V, D):
    s = pd.Series(V).groupby(pd.DatetimeIndex(D).values).mean()
    t = s.mean() / (s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 2 else np.nan
    return s, t


for zmode in (False, True):
    tag = "SHOCK_Z (z<=-2 & volx>=2)" if zmode else "SHOCK (resid<=-1.5% & volx>=2)"
    print(f"\n################ FAMILY {tag}")
    for h in HS:
        df, V, D = member_table(h, zmode=zmode)
        m, s, Q, k, I2, npos, nm = fe(df)
        ser, tt = dcl(V, D)
        w = int((ser > 0).sum())
        print(f" h={h}: epi {len(V)} dates {len(ser)} pooled {100*V.mean():+.3f}% | FE excess "
              f"{m:+.3f}% t {m/s:+.2f} Q {Q:.1f}/{k} p {1-st.chi2.cdf(Q,k):.3f} I2 {I2:.0f}% "
              f"+{npos}/{nm} | dcl {100*ser.mean():+.3f}% t {tt:+.2f} rec {w}-{len(ser)-w} "
              f"sign p {sign_test(w, len(ser)):.3f}")
        if h == 1 and not zmode:
            print(df.round(3).to_string(index=False))

# ------------------------------------------------------------ XLF alone
print("\n################ XLF ALONE (headline shock)")
xm = shock("XLF")
jpm = pd.to_datetime(pd.read_parquet(ROOT / "data" / "earnings_calendar.parquet")
                     .query("ticker == 'JPM'")["date"])
pos = pd.Series(range(len(CAL)), index=CAL)
jpos = [int(CAL.searchsorted(d)) for d in jpm if CAL[0] <= d <= CAL[-1]]
for h in HS:
    ret = trade("XLF", h)
    days = CAL[xm.values & ret.notna().values]
    e = declusters(days, h, CAL)
    v = ret.loc[e].values
    span = (CAL >= e[0]) & (CAL <= e[-1]) & ret.notna().values
    loc = local_control(CAL[ret.notna().values], days)
    w = int((v > 0).sum())
    show([summarize(v, f"XLF h={h} episodes"), summarize(ret[span].values, "own drift"),
          summarize(ret.loc[loc].values, "local +/-126"),
          summarize(trade("XLF", h, 0).loc[e].values, "lag=0 contrast")], f"XLF h={h}")
    print(f"  record {w}-{len(v)-w} sign p {sign_test(w, len(v)):.4f}")
    if h == 5:
        XE, XV = e, v
        print("  episodes:", ", ".join(f"{d.date()}:{100*x:+.2f}" for d, x in zip(e, v)))
        show(era_split(e, v), "XLF era split h=5")
        erng = np.array([any(0 <= pos[d] - jp <= 3 for jp in jpos) for d in e])
        crisis = np.array([d.year in (2008, 2009, 2020) for d in e])
        show([summarize(v[erng], "shock 0..3 td after a JPM print"),
              summarize(v[~erng], "not bank-earnings"),
              summarize(v[crisis], "2008/09/2020"), summarize(v[~crisis], "ex-crisis")],
             "XLF h=5 earnings / crisis splits")
        print(" ", cluster_note(e, v))

# ------------------------------------------------------------ family splits
print("\n################ FAMILY splits (headline, h=1 and h=5, date-clustered)")
for h in (1, 5):
    df, V, D = member_table(h)
    ser, tt = dcl(V, D)
    sd = pd.DatetimeIndex(ser.index)
    season = np.array(((sd.month.isin([1, 4, 7, 10])) & (sd.day >= 12)) |
                      ((sd.month.isin([2, 5, 8, 11])) & (sd.day <= 7)))
    crisis = np.array(sd.year.isin([2008, 2009, 2020]))
    mid = np.array(sd.year % 4 == 2)
    show([summarize(ser.values[season], "earnings season"),
          summarize(ser.values[~season], "off season"),
          summarize(ser.values[crisis], "2008/09/2020"),
          summarize(ser.values[~crisis], "ex-crisis"),
          summarize(ser.values[mid], "midterm yrs"),
          summarize(ser.values[~mid], "non-midterm")] + era_split(sd, ser.values),
         f"family h={h} splits")
    print(" ", cluster_note(sd, ser.values))
    worst = ser.idxmin()
    print(f"  worst date-clustered window {100*ser.min():+.2f}% on {pd.Timestamp(worst).date()}")

b = ST["XLF"]["beta"].iloc[-1]
print(f"\ncost: XLF pair RT ~{LEG_BP*(1+abs(b)):.1f} bp at beta {b:.2f}; 5x bar "
      f"{5*LEG_BP*(1+abs(b)):.1f} bp")
