"""kB_b1 -- ADVERSARIAL round 1 on CANDIDATE B1.

B1: beta-hedged SHORT SMH after a deep-laggard V-turn (r63 <= 10 AND r5 >= 95),
h=5, entry lag=1 MOC. Pre-specified sign: short vs beta.

This is an INVERSION of the registry kill a3_laggard_turn.py (2026-09-10): the
long of r63<=10 & r5>=75 lost -0.381% common excess, alpha -0.313% t -4.37.
CHARGE (stated before looking): sign (2) x horizon (1,2,3,5,10 = 5) = 10
comparisons -> Bonferroni two-sided 0.005 -> the hedged pooled alpha must clear
|t| >= 2.81 on the DATE-CLUSTERED series. The r5 threshold also moved 75 -> 95,
which sits inside a3's 16-cell grid; the grid is re-run on the hedged short.

Reference class: a3's declared universe minus SPY (the hedge leg) = 31 members.
Hedge: ex-ante 126d OLS beta vs SPY on daily returns through close t (known at
the signal close). Hedged short fwd return = -(r_etf - beta * r_spy).
Order: FILTER then DECLUSTER (gap = h) per member, then pool.
Cost: ~2.5 bp round trip per leg, pair cost = 2.5 x (1+|beta|).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change  # noqa

H = 5
LEG_BP = 2.5
UNIVERSE = [
    "QQQ", "IWM", "DIA",
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
    "SMH", "XBI", "IBB", "ITA", "IHI", "ITB", "XHB", "XRT", "XME", "XOP",
    "OIH", "KRE", "IYR", "GDX", "VNQ", "EFA", "EEM",
]
raw = close_panel(UNIVERSE + ["SPY"])
CAL = raw["SPY"].dropna().index
px = raw.reindex(CAL)
DR = {t: _valid_pct_change(px[t], 1) for t in px.columns}


def beta(t: str, hedger: str = "SPY", win: int = 126) -> pd.Series:
    j = pd.concat([DR[t], DR[hedger]], axis=1, keys=["a", "b"]).dropna()
    b = j["a"].rolling(win, min_periods=100).cov(j["b"]) / \
        j["b"].rolling(win, min_periods=100).var()
    return b.reindex(CAL).ffill(limit=3)


BETA = {t: beta(t) for t in UNIVERSE}
R63 = {t: pct_rank(px[t], 63) for t in UNIVERSE}
R5 = {t: pct_rank(px[t], 5) for t in UNIVERSE}


def hedged(t: str, h: int, hedger: str = "SPY", b: pd.Series | None = None,
           lag: int = 1) -> pd.Series:
    b = BETA[t] if b is None else b
    return -(fwd_lag(px[t], h, lag) - b * fwd_lag(px[hedger], h, lag))


def member_table(h=H, f63=10.0, t5=95.0, kind="hedged"):
    rows, V, D, T = [], [], [], []
    for t in UNIVERSE:
        if kind == "hedged":
            ret = hedged(t, h)
        elif kind == "raw_short":
            ret = -fwd_lag(px[t], h)
        else:  # dollar-neutral beta=1
            ret = -(fwd_lag(px[t], h) - fwd_lag(px["SPY"], h))
        m = ((R63[t] <= f63) & (R5[t] >= t5)).reindex(CAL, fill_value=False)
        days = CAL[m.values & ret.notna().values]
        if len(days) == 0:
            continue
        e = declusters(days, h, CAL)
        v = ret.loc[e].values
        span = (CAL >= e[0]) & (CAL <= e[-1]) & ret.notna().values
        base = ret[span].values
        se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
        rows.append({"ticker": t, "n": len(v), "mean_pct": 100 * v.mean(),
                     "drift_pct": 100 * base.mean(),
                     "excess_pct": 100 * (v.mean() - base.mean()),
                     "hit": 100 * (v > 0).mean(), "se_pct": 100 * se})
        V.append(v); D.extend(list(e)); T.extend([t] * len(v))
    return pd.DataFrame(rows), np.concatenate(V), pd.DatetimeIndex(D), np.asarray(T)


def fe(df):
    ok = (df["n"] > 1) & (df["se_pct"] > 0)
    ex, se = df.loc[ok, "excess_pct"].values, df.loc[ok, "se_pct"].values
    w = 1 / se**2
    m = (w * ex).sum() / w.sum()
    s = np.sqrt(1 / w.sum())
    Q = (w * (ex - m) ** 2).sum()
    k = len(ex) - 1
    I2 = max(0.0, 100 * (Q - k) / Q) if Q > 0 else 0.0
    return m, s, Q, k, I2, int((ex > 0).sum()), len(ex)


def dcl(V, D):
    s = pd.Series(V).groupby(pd.DatetimeIndex(D).values).mean()
    t = s.mean() / (s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 2 else np.nan
    return s, t


print("live (", CAL[-1].date(), "):")
for t in ("SMH", "QQQ", "XLK", "XLF"):
    print(f"  {t:4s} r63 {R63[t].iloc[-1]:5.1f} r5 {R5[t].iloc[-1]:5.1f} "
          f"beta126(SPY) {BETA[t].iloc[-1]:.2f}")
live = [t for t in UNIVERSE if R63[t].iloc[-1] <= 10 and R5[t].iloc[-1] >= 95]
print("  live members:", live)
bq = beta("SMH", "QQQ")
print(f"  SMH beta126 vs QQQ {bq.iloc[-1]:.2f}")

# ------------------------------------------------------------------ 1. family
for kind in ("hedged", "raw_short", "beta1_neutral"):
    df, V, D, T = member_table(kind=kind)
    m, s, Q, k, I2, npos, nm = fe(df)
    ser, tt = dcl(V, D)
    wins = int((ser > 0).sum())
    print(f"\n=== FAMILY {kind} (r63<=10 & r5>=95, h={H}) ===")
    print(f"  episodes {len(V)} on {len(ser)} dates; pooled mean {100*V.mean():+.3f}%")
    print(f"  FE common excess vs own drift {m:+.3f}% se {s:.3f} t {m/s:+.2f}; "
          f"Q {Q:.1f}/{k} df p {1-st.chi2.cdf(Q,k):.3f} I2 {I2:.1f}%; "
          f"members +excess {npos}/{nm}")
    print(f"  date-clustered mean {100*ser.mean():+.3f}% t {tt:+.2f} record "
          f"{wins}-{len(ser)-wins} sign p {sign_test(wins, len(ser)):.4f} "
          f"boot P<=0 {bootstrap_p_le0(ser.values):.3f}")
    if kind == "hedged":
        print(df.round(3).to_string(index=False))
        HV, HD, HT, HSER = V, D, T, ser

# ------------------------------------------------------------------ 2. era
sd = pd.DatetimeIndex(HSER.index)
show(era_split(sd, HSER.values), "hedged family, date-clustered, era split")
mid = sd.year % 4 == 2
show([summarize(HSER.values[mid], "midterm yrs"),
      summarize(HSER.values[~mid], "non-midterm")], "midterm split")
print(" ", cluster_note(sd, HSER.values))
by = pd.Series(HSER.values).groupby(sd.year).agg(["count", "sum"])
print("  year sums (pp):", {int(y): round(100 * r, 2) for y, r in by["sum"].items()})

# ------------------------------------------------------------------ 3. gate attribution on the SHORT
print(f"\n=== GATE ATTRIBUTION, hedged short, day-level pooled then date-clustered (h={H})")
rows = []
cells = {
    "JOIN r63<=10 & r5>=95": lambda t: (R63[t] <= 10) & (R5[t] >= 95),
    "r63<=10 ALONE": lambda t: R63[t] <= 10,
    "r63<=10 & r5<95 (discarded)": lambda t: (R63[t] <= 10) & (R5[t] < 95),
    "r5>=95 ALONE": lambda t: R5[t] >= 95,
    "r5>=95 & r63>10": lambda t: (R5[t] >= 95) & (R63[t] > 10),
    "ALL DAYS": lambda t: pd.Series(True, index=CAL),
}
for lbl, fn in cells.items():
    vs, ds = [], []
    for t in UNIVERSE:
        ret = hedged(t, H)
        m = fn(t).reindex(CAL, fill_value=False).values & ret.notna().values
        vs.append(ret.values[m]); ds.extend(list(CAL[m]))
    v = np.concatenate(vs)
    s, tt = dcl(v, ds)
    rows.append({"cell": lbl, "n_days": len(v), "mean_pct": round(100 * v.mean(), 3),
                 "n_dates": len(s), "dcl_mean_pct": round(100 * s.mean(), 3),
                 "dcl_t": round(tt, 2)})
print(pd.DataFrame(rows).to_string(index=False))

# ------------------------------------------------------------------ 4. grid (charged)
print("\n=== 4x4 GRID on the hedged short (episode-level, date-clustered)")
g = []
for f63 in (5, 10, 15, 20):
    for t5 in (75, 90, 95, 98):
        df, V, D, T = member_table(H, f63, t5)
        s, tt = dcl(V, D)
        m, se_, *_ = fe(df)
        g.append({"r63<=": f63, "r5>=": t5, "n": len(V), "dcl_mean": round(100 * s.mean(), 3),
                  "dcl_t": round(tt, 2), "FE_excess": round(m, 3), "FE_t": round(m / se_, 2)})
gdf = pd.DataFrame(g)
print(gdf.to_string(index=False))
print(f"  FE excess positive {int((gdf.FE_excess>0).sum())}/16, spread "
      f"{gdf.FE_excess.min():+.3f}..{gdf.FE_excess.max():+.3f}")

# ------------------------------------------------------------------ 5. horizons (charged)
print("\n=== HORIZONS, hedged family (charged in the 10-way budget)")
hr = []
for h in (1, 2, 3, 5, 10):
    df, V, D, T = member_table(h)
    s, tt = dcl(V, D)
    m, se_, *_ = fe(df)
    hr.append({"h": h, "n": len(V), "dcl_mean": round(100 * s.mean(), 3), "dcl_t": round(tt, 2),
               "FE_excess": round(m, 3), "FE_t": round(m / se_, 2)})
print(pd.DataFrame(hr).to_string(index=False))

# ------------------------------------------------------------------ 6. SMH own cell
print("\n=== SMH OWN CELL ===")
m_smh = ((R63["SMH"] <= 10) & (R5["SMH"] >= 95)).reindex(CAL, fill_value=False)
for lbl, ret in (("vs beta-SPY", hedged("SMH", H)),
                 ("vs beta-QQQ", -(fwd_lag(px["SMH"], H) - bq * fwd_lag(px["QQQ"], H))),
                 ("raw short", -fwd_lag(px["SMH"], H))):
    days = CAL[m_smh.values & ret.notna().values]
    e = declusters(days, H, CAL)
    v = ret.loc[e].values
    span = (CAL >= e[0]) & (CAL <= e[-1]) & ret.notna().values
    loc = local_control(CAL[ret.notna().values], days)
    w = int((v > 0).sum())
    show([summarize(v, f"SMH {lbl} episodes"), summarize(ret[span].values, "own drift"),
          summarize(ret.loc[loc].values, "local +/-126")], f"SMH {lbl}")
    print(f"  record {w}-{len(v)-w} sign p {sign_test(w, len(v)):.4f}; "
          f"worst {100*v.min():+.2f}%")
    if lbl == "vs beta-SPY":
        print("  episodes:", ", ".join(f"{d.date()}:{100*x:+.2f}" for d, x in zip(e, v)))
        show(era_split(e, v), "SMH era split")
        SMH_E, SMH_V = e, v

# MU print inside the hold?
ec = pd.read_parquet(REPO_ROOT / "data" / "earnings_calendar.parquet") if "REPO_ROOT" in dir() else \
    pd.read_parquet(Path(__file__).resolve().parents[3] / "data" / "earnings_calendar.parquet")
mu = pd.to_datetime(ec.loc[ec.ticker == "MU", "date"]).values.astype("datetime64[ns]")
pos = pd.Series(range(len(CAL)), index=CAL)
fl = []
for d in SMH_E:
    p = pos[d]
    lo, hi = CAL[p + 1], CAL[min(p + 1 + H, len(CAL) - 1)]
    # after-close print on day X moves day X+1: inside if lo <= X < hi
    fl.append(bool(((mu >= np.datetime64(lo)) & (mu < np.datetime64(hi))).any()))
fl = np.array(fl)
show([summarize(SMH_V[fl], "MU print moves inside hold"), summarize(SMH_V[~fl], "no MU")],
     "SMH episodes x MU print")
p = pos[CAL[-1]]
print(f"  live hold: entry {CAL[-1].date()}+1 close (09-23), h=5 exit = 09-30 close; "
      "MU reports 09-30 AFTER close -> reaction 10-01 is OUTSIDE h=5, INSIDE h>=6")

# cost
b_now = BETA["SMH"].iloc[-1]
print(f"\ncost: pair RT ~{LEG_BP*(1+abs(b_now)):.1f} bp at beta {b_now:.2f}; 5x bar "
      f"{5*LEG_BP*(1+abs(b_now)):.1f} bp = {5*LEG_BP*(1+abs(b_now))/100:.2f}%")
