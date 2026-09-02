"""Risk-architect lens, part 3: the stack/clone risk budget as a TAIL formula.

Reconstructs open ATR risk and open gross notional per day (from the ledger,
Entry..Exit inclusive) by strategy and theme, then calibrates the one number a
tail budget needs: on the book's worst days, how many units of open ATR risk
were actually lost (realized daily PnL / open risk). That ratio (the 'stress
multiple' k) turns an open-risk budget into a same-day loss bound:

    loss_bound(theme) = k_q * open_risk(theme)        q = 1% / 0.1% tail of k

and lets a stack be capped by loss bound rather than by leg count or rho.

Outputs risk_arch_stack_stress.json.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from risk_arch_common import NAV, THEMES, dump, load_ledger, load_spy, load_strategy_daily, sessions, ROOT
import sys
sys.path.insert(0, str(ROOT))
import strategy_config as sc

INDEX_ETFS = {"SPY", "QQQ", "DIA", "IWM", "VOO", "^GSPC", "^NDX"}
LEV3X = set(sc.LEV3X_ALL)


def open_matrix(df: pd.DataFrame, idx: pd.DatetimeIndex, value_col: str, key: str) -> pd.DataFrame:
    """Sum of value_col over trades open (Entry..Exit inclusive) per day, by key."""
    keys = sorted(df[key].dropna().unique())
    pos = {d: i for i, d in enumerate(idx)}
    M = np.zeros((len(idx), len(keys)))
    kpos = {k: i for i, k in enumerate(keys)}
    ent = np.searchsorted(idx.values, df["Entry Date"].values, side="left")
    ex = np.searchsorted(idx.values, df["Exit Date"].values, side="right")
    for (e, x, k, v) in zip(ent, ex, df[key].values, df[value_col].values):
        if k is None or (isinstance(k, float) and np.isnan(k)):
            continue
        if e >= len(idx):
            continue
        M[e:max(e + 1, x), kpos[k]] += v
    return pd.DataFrame(M, index=idx, columns=keys)


def main() -> None:
    df = load_ledger()
    df = df[df["Entry Date"].notna() & df["Exit Date"].notna()].copy()
    df["notional"] = (df["Shares_flat"].abs() * df["Entry Price"]).astype(float)
    df["cls"] = np.where(df["Ticker"].isin(LEV3X), "lev3x", np.where(df["Ticker"].isin(INDEX_ETFS), "index", "stock"))
    strat, total = load_strategy_daily()
    spy = load_spy()
    idx = sessions(strat, spy)
    pnl = (total.reindex(idx)).astype(float)
    sd = strat.reindex(idx)

    open_risk_strat = open_matrix(df, idx, "Risk_flat_750k", "Strategy")
    open_risk_theme = open_matrix(df, idx, "Risk_flat_750k", "theme")
    open_not_theme = open_matrix(df, idx, "notional", "theme")
    open_not_cls = open_matrix(df, idx, "notional", "cls")
    open_risk_book = open_risk_strat.sum(1)
    open_not_book = open_not_theme.sum(1)

    out: dict = {}
    # 1. open risk distribution by theme (bps of NAV), 2010+ and 2016+
    for label, start in (("2010+", "2010-01-01"), ("2016+", "2016-01-01")):
        m = idx >= start
        R = open_risk_theme[m] / NAV * 1e4
        N = open_not_theme[m] / NAV
        out[f"open_risk_bps_{label}"] = {t: dict(mean=float(R[t].mean()), p90=float(R[t].quantile(.9)), p99=float(R[t].quantile(.99)), max=float(R[t].max()),
                                                 max_date=str(R[t].idxmax().date())) for t in R.columns}
        out[f"open_risk_bps_{label}"]["book"] = dict(mean=float(R.sum(1).mean()), p90=float(R.sum(1).quantile(.9)), p99=float(R.sum(1).quantile(.99)), max=float(R.sum(1).max()), max_date=str(R.sum(1).idxmax().date()))
        out[f"open_notional_nav_{label}"] = {t: dict(mean=float(N[t].mean()), p99=float(N[t].quantile(.99)), max=float(N[t].max()), max_date=str(N[t].idxmax().date())) for t in N.columns}
        out[f"open_notional_nav_{label}"]["book_gross"] = dict(mean=float(N.sum(1).mean()), p95=float(N.sum(1).quantile(.95)), p99=float(N.sum(1).quantile(.99)), max=float(N.sum(1).max()), max_date=str(N.sum(1).idxmax().date()))
        C = open_not_cls[m] / NAV
        out[f"open_notional_by_class_{label}"] = {c: dict(mean=float(C[c].mean()), p99=float(C[c].quantile(.99)), max=float(C[c].max())) for c in C.columns}

    # 2. stress multiple k = -pnl / open_risk on days with open risk >= 100 bps
    m = (idx >= "2005-01-01")
    orb = open_risk_book[m]; p = pnl[m]
    ok = orb >= 100 / 1e4 * NAV
    k = (-p[ok] / orb[ok])
    out["stress_multiple_book"] = dict(days=int(ok.sum()), mean=float(k.mean()), p90=float(k.quantile(.9)), p95=float(k.quantile(.95)), p99=float(k.quantile(.99)), p999=float(k.quantile(.999)), max=float(k.max()),
                                       max_date=str(k.idxmax().date()), note="k = realized day loss / open ATR risk; 1.0 = every open leg lost one full stop in one day")
    # worst 20 days: loss, open risk, k, theme composition of the loss
    worst = p.nsmallest(20)
    rows = []
    for d, v in worst.items():
        comp = sd.loc[d]
        th = {t: float(comp[[s for s in ss if s in comp.index]].sum()) for t, ss in THEMES.items()}
        big = min(th, key=lambda t: th[t])
        rows.append(dict(date=str(d.date()), pnl_pct=float(v / NAV * 100), open_risk_bps=float(open_risk_book[d] / NAV * 1e4), k=float(-v / open_risk_book[d]) if open_risk_book[d] > 0 else None,
                         open_gross_nav=float(open_not_book[d] / NAV), worst_theme=big, worst_theme_share=float(th[big] / v) if v != 0 else None,
                         theme_open_risk_bps={t: float(open_risk_theme.loc[d, t] / NAV * 1e4) for t in open_risk_theme.columns}))
    out["worst_20_days"] = rows
    # 3. per-theme stress multiple on that theme's own worst days (theme pnl / theme open risk)
    per = {}
    for t, ss in THEMES.items():
        tp = sd[[s for s in ss if s in sd.columns]].sum(1)[m]
        orr = open_risk_theme[t][m]
        okt = orr >= 50 / 1e4 * NAV
        kk = (-tp[okt] / orr[okt])
        if len(kk) > 50:
            per[t] = dict(days=int(okt.sum()), p95=float(kk.quantile(.95)), p99=float(kk.quantile(.99)), p999=float(kk.quantile(.999)), max=float(kk.max()), max_date=str(kk.idxmax().date()),
                          worst_day_pct=float(tp.min() / NAV * 100), worst_day_date=str(tp.idxmin().date()),
                          worst21_pct=float((tp.rolling(21).sum()).min() / NAV * 100))
    out["stress_multiple_by_theme"] = per
    # 4. does realized daily loss scale with open risk? regress |pnl| on open risk, and tail-day open risk vs typical
    x = orb / NAV * 1e4; y = p / NAV * 1e4
    tail = y <= y.quantile(0.01)
    out["open_risk_vs_loss"] = dict(corr_absloss_openrisk=float(np.corrcoef(x, y.abs())[0, 1]),
                                    mean_open_risk_bps_all=float(x.mean()), mean_open_risk_bps_worst1pct=float(x[tail].mean()),
                                    mean_open_risk_bps_best1pct=float(x[y >= y.quantile(0.99)].mean()),
                                    elasticity_loglog=float(np.polyfit(np.log(x[x > 0]), np.log(y.abs()[x > 0] + 1), 1)[0]))
    # 5. OLV stack tail: single-ticker open risk/notional and stack depth on OLV's worst days
    olv = df[df["Strategy"] == "Oversold Low Volume"]
    olv_open_risk = open_matrix(olv, idx, "Risk_flat_750k", "Strategy")["Oversold Low Volume"]
    olv_open_not = open_matrix(olv, idx, "notional", "Strategy")["Oversold Low Volume"]
    olv_legs = open_matrix(olv.assign(one=1.0), idx, "one", "Strategy")["Oversold Low Volume"]
    olv_p = sd["Oversold Low Volume"]
    w = olv_p.nsmallest(10)
    out["olv_worst_days"] = [dict(date=str(d.date()), pnl_pct=float(v / NAV * 100), legs=int(olv_legs[d]), open_risk_bps=float(olv_open_risk[d] / NAV * 1e4),
                                  open_notional_nav=float(olv_open_not[d] / NAV), k=float(-v / olv_open_risk[d]) if olv_open_risk[d] > 0 else None) for d, v in w.items()]
    ok2 = olv_open_risk >= 50 / 1e4 * NAV
    kk = -olv_p[ok2] / olv_open_risk[ok2]
    out["olv_stress_by_depth"] = {}
    for lo, hi in ((1, 2), (2, 4), (4, 7), (7, 99)):
        mm = ok2 & (olv_legs >= lo) & (olv_legs < hi)
        if mm.sum() > 30:
            kd = -olv_p[mm] / olv_open_risk[mm]
            out["olv_stress_by_depth"][f"{lo}-{hi-1}"] = dict(days=int(mm.sum()), mean_pnl_per_openrisk=float(-kd.mean()), sd=float(kd.std()), p99=float(kd.quantile(.99)), max=float(kd.max()))
    # 6. same for OVS (short book) and breakout
    for s in ("Overbot Vol Spike", "52wh Breakout"):
        sub = df[df["Strategy"] == s]
        orr = open_matrix(sub, idx, "Risk_flat_750k", "Strategy")[s]
        legs = open_matrix(sub.assign(one=1.0), idx, "one", "Strategy")[s]
        sp = sd[s]
        okk = orr >= 50 / 1e4 * NAV
        kd = -sp[okk] / orr[okk]
        out[f"stress_{s}"] = dict(days=int(okk.sum()), p99=float(kd.quantile(.99)), max=float(kd.max()), max_date=str(kd.idxmax().date()), legs_on_max=int(legs[kd.idxmax()]),
                                  worst_day_pct=float(sp.min() / NAV * 100))
    # 7. margin feasibility by class rates (plain PM 15/8/45, and index legs moved to futures 6%)
    m16 = idx >= "2016-01-01"
    C = open_not_cls[m16]
    for name, rates in (("pm_plain", dict(stock=.15, index=.08, lev3x=.45)), ("pm_rules_3x", dict(stock=.15, index=.08, lev3x=.90)),
                        ("pm_index_to_futures", dict(stock=.15, index=.06, lev3x=.45)), ("pm_conc30", dict(stock=.30, index=.08, lev3x=.45))):
        req = sum(C.get(c, 0) * r for c, r in rates.items()) / NAV
        out[f"margin_req_nav_{name}"] = dict(p95=float(req.quantile(.95)), p99=float(req.quantile(.99)), max=float(req.max()), max_date=str(req.idxmax().date()),
                                            feasible_m_max=float(1 / req.max()), feasible_m_p99=float(1 / req.quantile(.99)),
                                            feasible_m_p99_live632k=float((632 / 750) / req.quantile(.99)))
    out["class_share_of_gross_top1pct_days_2016+"] = (C[C.sum(1) >= C.sum(1).quantile(.99)].sum() / C[C.sum(1) >= C.sum(1).quantile(.99)].sum().sum()).round(3).to_dict()
    dump(out, "risk_arch_stack_stress.json")

    print("open risk bps 2016+:", {t: (round(v["p99"]), round(v["max"])) for t, v in out["open_risk_bps_2016+"].items()})
    print("gross notional/NAV 2016+:", out["open_notional_nav_2016+"]["book_gross"])
    print("stress multiple book:", out["stress_multiple_book"])
    print("worst 20 days:")
    for r in rows[:12]:
        print(f"  {r['date']} {r['pnl_pct']:.2f}% open {r['open_risk_bps']:.0f}bps k={r['k']:.2f} gross {r['open_gross_nav']:.2f} theme {r['worst_theme']} ({r['worst_theme_share']:.0%})")
    print("theme stress:", {t: (round(v["p99"], 2), round(v["max"], 2), v["max_date"]) for t, v in per.items()})
    print("open risk vs loss:", out["open_risk_vs_loss"])
    print("OLV worst:", out["olv_worst_days"][:5])
    print("OLV stress by depth:", out["olv_stress_by_depth"])
    for k in out:
        if k.startswith("margin_req"):
            print(k, out[k])
    print("class share top1% gross days:", out["class_share_of_gross_top1pct_days_2016+"])


if __name__ == "__main__":
    main()
