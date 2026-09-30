"""c11 round 2 (2026-09-21): the pooled QE-7 cell passed round 1 (ladder rank
3 of 68, 9.7x cost) while the LIVE September row was below drift. Round 2 asks
whether the pooled edge is the mechanism or something else:
  1. per-quarter rows (Mar/Jun/Sep/Dec) for LQD and HYG spread proxies, with and
     without 2008-09 / 2020-03; Sept-vs-rest permutation
  2. where in the window the return sits: QE-7 -> QE vs QE -> QE+10
  3. quarter-end vs ordinary month-end (ME-7 -> ME+10 in non-quarter months)
  4. equity-neutral form: LQD - b1*IEF - b2*SPY with ex-ante betas (trailing
     252 multivariate OLS through the signal close)
  5. midterm split per quarter; SPY-above-200d split; concentration
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
BAR = pd.Timestamp("2026-09-18")

px = close_panel(["LQD", "IEF", "HYG", "SPY"]).loc[:BAR]
px = px[px["SPY"].notna()]
idx = px.index
ser = pd.Series(range(len(idx)), index=idx)
dr = (px / px.shift(1) - 1.0)
sma200 = px["SPY"].rolling(200).mean()


def ols_betas(y: str, xs: list[str], p_sig: int, n: int = 252) -> np.ndarray | None:
    lo = p_sig - n + 1
    if lo < 1:
        return None
    Y = dr[y].iloc[lo:p_sig + 1]
    X = dr[xs].iloc[lo:p_sig + 1]
    m = Y.notna() & X.notna().all(axis=1)
    if m.sum() < 200:
        return None
    Xm = np.column_stack([np.ones(m.sum()), X[m].values])
    b = np.linalg.lstsq(Xm, Y[m].values, rcond=None)[0]
    return b[1:]


def pr(t: str, p0: int, p1: int) -> float:
    if p1 >= len(idx) or p0 < 0:
        return np.nan
    return px[t].iloc[p1] / px[t].iloc[p0] - 1.0


def spread(leg: str, p0: int, p1: int, hedge=("IEF",)) -> float:
    b = ols_betas(leg, list(hedge), p0 - 1)
    if b is None:
        return np.nan
    v = pr(leg, p0, p1)
    for bi, x in zip(b, hedge):
        v -= bi * pr(x, p0, p1)
    return v


# month-end sessions
me = []
for y in range(2002, 2027):
    for m in range(1, 13):
        s = idx[(idx.year == y) & (idx.month == m)]
        if len(s) and s[-1] < BAR:
            me.append(s[-1])
ME = pd.DataFrame({"d": me})
ME["p"] = [int(ser[d]) for d in ME.d]
ME["m"] = ME.d.dt.month
ME["qe"] = ME.m.isin([3, 6, 9, 12])

rows = []
for r in ME.itertuples():
    p0 = r.p - 7
    rec = {"qe_date": r.d, "month": r.m, "is_qe": r.qe, "year": r.d.year,
           "above200": bool(px["SPY"].iloc[p0 - 1] > sma200.iloc[p0 - 1]) if p0 > 200 else np.nan}
    for leg in ("LQD", "HYG"):
        rec[f"{leg}_h17"] = spread(leg, p0, p0 + 17)
        rec[f"{leg}_pre"] = spread(leg, p0, r.p)          # QE-7 -> QE
        rec[f"{leg}_post"] = spread(leg, r.p, r.p + 10)   # QE -> QE+10 (beta from QE-1)
        rec[f"{leg}_h17_eqn"] = spread(leg, p0, p0 + 17, hedge=("IEF", "SPY"))
    rec["SPY_h17"] = pr("SPY", p0, p0 + 17)
    rows.append(rec)
D = pd.DataFrame(rows)
D.to_pickle(Path(__file__).with_name("_kD_c11_panel.pkl"))


def s(v, lbl):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    o = summarize(v, lbl)
    if o["n"]:
        w = int((v > 0).sum())
        o["rec"] = f"{w}-{len(v)-w}"
        o["sign_p"] = round(sign_test(w, len(v)), 4)
    return o


Q = D[D.is_qe]
for leg in ("LQD", "HYG"):
    col = f"{leg}_h17"
    out = []
    for m, lbl in [(3, "Mar QE"), (6, "Jun QE"), (9, "SEP QE (live)"), (12, "Dec QE")]:
        out.append(s(Q.loc[Q.month == m, col], lbl))
    out.append(s(Q[col], "all QE"))
    ex = ~Q.qe_date.isin([pd.Timestamp("2008-09-30"), pd.Timestamp("2020-03-31")])
    out.append(s(Q.loc[ex, col], "all QE ex 2008-09 & 2020-03"))
    for m, lbl in [(3, "Mar ex-2020"), (9, "Sep ex-2008")]:
        mm = (Q.month == m) & ex
        out.append(s(Q.loc[mm, col], lbl))
    out.append(s(D.loc[~D.is_qe, col], "ordinary month-ends (ME-7 -> ME+10)"))
    out.append(s(D.loc[~D.is_qe & ~D.qe_date.isin([pd.Timestamp('2008-10-31'), pd.Timestamp('2020-04-30'), pd.Timestamp('2008-11-28')]), col],
                 "ordinary ME ex 2008-10/11, 2020-04"))
    show(out, f"1+3. {leg} - b*IEF, h=17 from QE-7 (and ME-7), per quarter")
    # permutation: 23 random QE rows vs the September mean
    rng = np.random.default_rng(42)
    allv = Q[col].dropna().values
    sep = Q.loc[Q.month == 9, col].dropna().values
    perm = np.array([rng.choice(allv, len(sep), replace=False).mean() for _ in range(20000)])
    print(f"  P(random {len(sep)} quarters <= September mean {1e4*sep.mean():+.1f} bps) = {(perm <= sep.mean()).mean():.3f}")
    # 2. where in the window
    out = []
    for m, lbl in [(3, "Mar"), (6, "Jun"), (9, "SEP"), (12, "Dec")]:
        qq = Q[Q.month == m]
        out.append({"q": lbl, "pre_QE-7..QE_bps": 1e4 * qq[f"{leg}_pre"].mean(),
                    "post_QE..QE+10_bps": 1e4 * qq[f"{leg}_post"].mean(),
                    "pre_med": 1e4 * qq[f"{leg}_pre"].median(), "post_med": 1e4 * qq[f"{leg}_post"].median()})
    out.append({"q": "all QE", "pre_QE-7..QE_bps": 1e4 * Q[f"{leg}_pre"].mean(),
                "post_QE..QE+10_bps": 1e4 * Q[f"{leg}_post"].mean(),
                "pre_med": 1e4 * Q[f"{leg}_pre"].median(), "post_med": 1e4 * Q[f"{leg}_post"].median()})
    out.append({"q": "ordinary ME", "pre_QE-7..QE_bps": 1e4 * D.loc[~D.is_qe, f"{leg}_pre"].mean(),
                "post_QE..QE+10_bps": 1e4 * D.loc[~D.is_qe, f"{leg}_post"].mean(),
                "pre_med": 1e4 * D.loc[~D.is_qe, f"{leg}_pre"].median(),
                "post_med": 1e4 * D.loc[~D.is_qe, f"{leg}_post"].median()})
    print(f"\n=== 2. {leg}: where the window return sits (bps, beta-IEF spread) ===")
    print(pd.DataFrame(out).round(1).to_string(index=False))
    # 4. equity-neutral
    col2 = f"{leg}_h17_eqn"
    out = [s(Q.loc[Q.month == m, col2], f"{lbl} eq-neutral") for m, lbl in
           [(3, "Mar"), (6, "Jun"), (9, "SEP"), (12, "Dec")]]
    out.append(s(Q[col2], "all QE eq-neutral"))
    out.append(s(Q.loc[ex, col2], "all QE eq-neutral ex 2008-09/2020-03"))
    out.append(s(D.loc[~D.is_qe, col2], "ordinary ME eq-neutral"))
    show(out, f"4. {leg} - b1*IEF - b2*SPY (ex-ante betas), h=17")
    # 5. midterm + regime
    out = [s(Q.loc[(Q.month == 9) & (Q.year % 4 == 2), col], "Sep QE midterm"),
           s(Q.loc[(Q.year % 4 == 2), col], "all QE midterm"),
           s(Q.loc[(Q.year % 4 != 2), col], "all QE non-midterm"),
           s(Q.loc[Q.above200 == True, col], "all QE, SPY above 200d"),
           s(Q.loc[Q.above200 == False, col], "all QE, SPY below 200d"),
           s(Q.loc[(Q.month == 9) & (Q.above200 == True), col], "Sep QE, SPY above 200d")]
    show(out, f"5. {leg} midterm / 200d split (h=17)")
    qq = Q[[ "qe_date", col]].dropna()
    print(f"  concentration all QE: {cluster_note(pd.DatetimeIndex(qq.qe_date), qq[col].values)}")
    yrs = qq.assign(y=qq.qe_date.dt.year).groupby("y")[col].sum().sort_values()
    print(f"  per-year sums (bps), worst 3 {dict((k, round(1e4*v)) for k, v in yrs.head(3).items())} "
          f"best 3 {dict((k, round(1e4*v)) for k, v in yrs.tail(3).items())}")

# live row detail
print("\n=== live: September rows, eq-neutral LQD and SPY loading ===")
sep = Q[Q.month == 9][["qe_date", "LQD_h17", "LQD_h17_eqn", "HYG_h17", "HYG_h17_eqn", "SPY_h17", "above200"]].copy()
for c in ["LQD_h17", "LQD_h17_eqn", "HYG_h17", "HYG_h17_eqn"]:
    sep[c] = (1e4 * sep[c]).round(1)
sep["SPY_h17"] = (100 * sep["SPY_h17"]).round(2)
print(sep.to_string(index=False))
