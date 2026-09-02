"""Practitioner lens, check 1 (2026-09-02): does the book's own P&L path carry
information a drawdown- or streak-triggered size cut could use?

Kaminski-Lo (2014): a stop-loss / drawdown cut on a P&L series has a positive
stopping premium only if the series is positively autocorrelated. This script
measures (a) the ACF of the flat-basis daily book P&L at lags 1-21 with
Ljung-Box, (b) variance ratios at 5/10/21 days, (c) forward 5/21-day book
return conditioned on trailing drawdown from the 252d high, on losing streaks,
and on worst-5%/best-5% days. Windows: 2003+ and 2016-07+.

Inputs: data/backtest_daily_pnl.parquet (pnl_flat, ends 2026-08-07).
Output: practitioner_01_acf_drawdown.json beside this script.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"C:/Users/McKinley Slade/dev/New_Seasonals")
OUT = ROOT / "scratch/ultracode_sizing_2026-09-02/practitioner_01_acf_drawdown.json"
NAV = 750_000.0

dp = pd.read_parquet(ROOT / "data/backtest_daily_pnl.parquet")
dp["date"] = pd.to_datetime(dp["date"])
s_all = dp.set_index("date")["pnl_flat"] / NAV
s_all = s_all[s_all.index >= "2003-06-01"]


def acf(x: np.ndarray, maxlag: int) -> np.ndarray:
    x = x - x.mean()
    denom = (x * x).sum()
    return np.array([(x[:-k] * x[k:]).sum() / denom for k in range(1, maxlag + 1)])


def ljung_box(r: np.ndarray, n: int) -> tuple[float, float]:
    from scipy import stats
    q = n * (n + 2) * np.sum(r ** 2 / (n - np.arange(1, len(r) + 1)))
    return float(q), float(1 - stats.chi2.cdf(q, len(r)))


def variance_ratio(x: np.ndarray, q: int) -> float:
    x = x - x.mean()
    v1 = np.var(x, ddof=1)
    xs = np.convolve(x, np.ones(q), "valid")
    return float(np.var(xs, ddof=1) / (q * v1))


def fwd(s: pd.Series, h: int) -> pd.Series:
    return s[::-1].rolling(h).sum()[::-1].shift(-1)


def cond_table(s: pd.Series, mask: pd.Series, label: str, h_list=(5, 21)) -> dict:
    out = {"label": label, "n_days": int(mask.sum())}
    for h in h_list:
        f = fwd(s, h)
        a, b = f[mask].dropna(), f[~mask].dropna()
        # block-ish SE: cluster forward windows by non-overlapping blocks of h days
        se = a.std() / np.sqrt(max(len(a) / h, 1))
        out[f"fwd{h}_mean_bps"] = float(a.mean() * 1e4)
        out[f"fwd{h}_rest_bps"] = float(b.mean() * 1e4)
        out[f"fwd{h}_t_vs_rest_blocked"] = float((a.mean() - b.mean()) / se) if se > 0 else np.nan
        out[f"fwd{h}_hit"] = float((a > 0).mean())
        out[f"fwd{h}_sharpe_ann"] = float(a.mean() / a.std() * np.sqrt(252 / h)) if a.std() > 0 else np.nan
    return out


res = {}
for win, s in {"2003+": s_all, "2016-07+": s_all[s_all.index >= "2016-07-01"]}.items():
    x = s.values
    r = acf(x, 21)
    q10, p10 = ljung_box(r[:10], len(x))
    q21, p21 = ljung_box(r[:21], len(x))
    # ACF of the SIGNED loss series (down days) and of |r| for clustering
    ra = acf(np.abs(x), 21)
    eq = s.cumsum()
    dd = eq - eq.cummax()                       # flat-basis drawdown in NAV fraction
    hi252 = eq.rolling(252, min_periods=60).max()
    dd252 = eq - hi252
    streak = (s < 0).astype(int)
    streak = streak.groupby((streak != streak.shift()).cumsum()).cumsum() * (s < 0)
    worst = s <= s.quantile(0.05)
    best = s >= s.quantile(0.95)
    r21 = s.rolling(21).sum()
    tables = [
        cond_table(s, dd252 <= -0.05, "dd from 252d high <= -5%"),
        cond_table(s, (dd252 <= -0.05) & (dd252 > -0.10), "dd 5 to 10%"),
        cond_table(s, dd252 <= -0.075, "dd <= -7.5%"),
        cond_table(s, dd252 <= -0.10, "dd <= -10%"),
        cond_table(s, dd252 == 0, "at 252d high"),
        cond_table(s, streak >= 3, "3+ consecutive down days"),
        cond_table(s, streak >= 5, "5+ consecutive down days"),
        cond_table(s, worst, "worst-5% day"),
        cond_table(s, best, "best-5% day"),
        cond_table(s, r21 <= r21.quantile(0.10), "trailing 21d P&L bottom decile"),
        cond_table(s, r21 >= r21.quantile(0.90), "trailing 21d P&L top decile"),
    ]
    res[win] = {
        "n": int(len(x)), "mean_bps": float(x.mean() * 1e4), "sd_bps": float(x.std() * 1e4),
        "sharpe": float(x.mean() / x.std() * np.sqrt(252)),
        "acf_1_21": [round(float(v), 4) for v in r],
        "acf_abs_1_21": [round(float(v), 4) for v in ra],
        "ljung_box_10": {"Q": q10, "p": p10}, "ljung_box_21": {"Q": q21, "p": p21},
        "acf_2se": float(2 / np.sqrt(len(x))),
        "variance_ratio": {str(q): variance_ratio(x, q) for q in (2, 5, 10, 21)},
        "conditional": tables,
        "worst_flat_dd_pct": float(dd.min() * 100),
    }
    print(f"== {win}: n={len(x)} Sharpe {res[win]['sharpe']:.2f}  ACF1..5 {np.round(r[:5],3)}  LB10 p={p10:.3f} LB21 p={p21:.3f}  VR(5/10/21) "
          f"{variance_ratio(x,5):.2f}/{variance_ratio(x,10):.2f}/{variance_ratio(x,21):.2f}  2se={2/np.sqrt(len(x)):.3f}")
    print(f"   |r| ACF1..5 {np.round(ra[:5],3)}")
    for t in tables:
        print(f"   {t['label']:<34} n={t['n_days']:5d}  fwd21 {t['fwd21_mean_bps']:7.1f} vs rest {t['fwd21_rest_bps']:6.1f} bps (t_blk {t['fwd21_t_vs_rest_blocked']:+.2f}, hit {t['fwd21_hit']:.2f})"
              f"  fwd5 {t['fwd5_mean_bps']:6.1f} vs {t['fwd5_rest_bps']:5.1f} (t {t['fwd5_t_vs_rest_blocked']:+.2f})")

res["reading"] = ("Kaminski-Lo gate: a drawdown/streak cut only earns a positive stopping premium when the P&L series is "
                  "positively autocorrelated at the cut horizon. See acf_1_21 vs acf_2se and the conditional tables.")
json.dump(res, open(OUT, "w"), indent=1)
print("wrote", OUT)
