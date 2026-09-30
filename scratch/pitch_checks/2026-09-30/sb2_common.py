"""Shared helpers for the 2026-09-30 seasonal-board kill checks (WMT long, TXN short).

Anchor T = last trading day on/before Sep 30 each year (today is Wed 2026-09-30).
Entry at close T+lag, exit close T+lag+h. Gate measured at T-1 (the board's asof
bar, 2026-09-29), PIT expanding full-history percentile like trailing_return_pctile.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

MIDTERMS = {2002, 2006, 2010, 2014, 2018, 2022}


def anchors(idx: pd.DatetimeIndex, month: int = 9, day: int = 30,
            shift: int = 0, last_year: int = 2025) -> dict[int, int]:
    out = {}
    for y in range(idx[0].year, last_year + 1):
        loc = int(idx.searchsorted(pd.Timestamp(y, month, day), side="right")) - 1
        if loc < 0 or idx[loc].year != y:
            continue
        p = loc + shift
        if 0 <= p < len(idx):
            out[y] = p
    return out


def win_ret(c: np.ndarray, p: int, lag: int, h: int) -> float:
    a, b = p + lag, p + lag + h
    if b >= len(c):
        return np.nan
    return c[b] / c[a] - 1.0


def yearly(c: pd.Series, anc: dict[int, int], lag: int, h: int, side: float) -> pd.Series:
    v = c.values
    return pd.Series({y: side * win_ret(v, p, lag, h) for y, p in anc.items()}).dropna()


def pit_pctile(c: pd.Series, p: int, w: int) -> float:
    r = c.iloc[: p + 1].pct_change(w).dropna()
    if len(r) < 60:
        return np.nan
    return float(r.rank(pct=True).iloc[-1] * 100)


def all_day_drift(c: pd.Series, lag: int, h: int, side: float) -> float:
    return 100 * side * float(fwd_lag(c, h, lag).mean())


def local_ctrl(c: pd.Series, anc: dict[int, int], lag: int, h: int, side: float) -> float:
    r = fwd_lag(c, h, lag)
    trig = pd.DatetimeIndex([c.index[p] for p in anc.values()])
    loc = local_control(c.index, trig)
    return 100 * side * float(r.loc[loc].mean())


def cohort_line(s: pd.Series, label: str) -> str:
    n = len(s)
    if n == 0:
        return f"{label}: n=0"
    w = int((s > 0).sum())
    return (f"{label}: n={n} mean {100*s.mean():+.2f}% med {100*s.median():+.2f}% "
            f"hit {w}/{n} signp {sign_test(w, n):.3f} worst {100*s.min():+.2f}% "
            f"({s.idxmin()})")


def drop_best(s: pd.Series) -> str:
    o = s.sort_values(ascending=False)
    parts = []
    for k in (0, 1, 2):
        t = o.iloc[k:]
        parts.append(f"drop{k}: mean {100*t.mean():+.2f}% hit {int((t>0).sum())}/{len(t)}")
    return " | ".join(parts) + f"  best yrs {[(y, round(100*v, 2)) for y, v in o.head(2).items()]}"


def concentration(s: pd.Series) -> str:
    tot = s.sum()
    top2 = s.sort_values(ascending=False).head(2).sum()
    return (f"total {100*tot:+.2f}pp, top-2 yrs {100*top2:+.2f}pp "
            f"({100*top2/tot if tot else np.nan:.0f}% of total)")


def era(s: pd.Series, cut: int) -> str:
    a, b = s[s.index < cut], s[s.index >= cut]
    return (f"pre-{cut}: n={len(a)} mean {100*a.mean():+.2f}% hit {int((a>0).sum())}/{len(a)} | "
            f"{cut}+: n={len(b)} mean {100*b.mean():+.2f}% hit {int((b>0).sum())}/{len(b)}")


def beta_at(y: pd.Series, x: pd.Series, p: int, n: int = 252) -> float:
    ry = y.pct_change().iloc[max(1, p - n + 1): p + 1]
    rx = x.pct_change().iloc[max(1, p - n + 1): p + 1]
    d = pd.concat([ry, rx], axis=1).dropna()
    if len(d) < 60:
        return np.nan
    return float(np.cov(d.iloc[:, 0], d.iloc[:, 1])[0, 1] / d.iloc[:, 1].var())


def full_report(c: pd.Series, anc: dict[int, int], lag: int, h: int, side: float,
                label: str) -> pd.Series:
    s = yearly(c, anc, lag, h, side)
    print(f"\n--- {label}: lag={lag} h={h} side={'+' if side > 0 else '-'} ---")
    print(cohort_line(s, "ALL yrs"))
    print(f"  own all-day drift {all_day_drift(c, lag, h, side):+.2f}% | "
          f"local +/-126td ctrl {local_ctrl(c, anc, lag, h, side):+.2f}%")
    m = s[s.index.isin(MIDTERMS)]
    print(cohort_line(m, "MIDTERM"))
    print("  midterm " + drop_best(m))
    print("  midterm yrs: " + ", ".join(f"{y}:{100*v:+.2f}" for y, v in m.items()))
    print("  all-yrs " + drop_best(s))
    print("  concentration all-yrs: " + concentration(s))
    print("  era " + era(s, 2018))
    print("  era " + era(s, 2010))
    nm = s[~s.index.isin(MIDTERMS)]
    print(cohort_line(nm, "NON-midterm"))
    return s
