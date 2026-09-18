"""CHECK B (2026-09-09) — energy extension: XLE closing AT a 252-day high while
stretched (z10 >= 1.5), and USO very stretched (z10 >= 2.5 with a 21d return
>= 15%). Fade, continuation, or nothing?

State tonight (2026-09-09 close): XLE, XOP, VLO, CVX and DBC all closed exactly
at their 52-week highs (dist_52w_high 0.0) with z10 1.74 / 2.05 / 2.93 / 2.05 /
2.50; USO is z10 2.78 on a +19.1% 21d return.

Definitions matched to scripts/build_pitch_state.py::_metrics_for:
  z10           = close.pct_change(10) / (close.pct_change().rolling(21).std()
                                          * sqrt(10))
  dist_52w_high = close / close.rolling(252).max() - 1
"at a 252-day high" = dist_52w_high >= 0 (the close IS the rolling max).

Forward returns are LAG-1 MOC: entry at close D+1, exit at close D+1+h, i.e.
the order an idea seen at tonight's close can actually place tomorrow.
Anchors declustered at 5 sessions. Every cell is printed against (i) the same
cell without the extension condition and (ii) the unconditional instrument.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa: F401,F403

import numpy as np
import pandas as pd

ASOF = pd.Timestamp("2026-09-09")
ERA = "2018-01-01"
GAP_TD = 5
HS = (1, 3, 5, 10, 21)
TICKERS = ["XLE", "USO", "XOP", "VLO", "CVX", "DBC"]

px = load_prices(TICKERS)
for t in list(px):
    px[t] = px[t][px[t].index <= ASOF]


def z10(df):
    c = df["Close"].astype(float)
    return c.pct_change(10) / (c.pct_change().rolling(21).std() * np.sqrt(10))


def dist_hi(df, lb=252):
    c = df["Close"].astype(float)
    return c / c.rolling(lb).max() - 1.0


def ret_n(df, n):
    return df["Close"].astype(float).pct_change(n)


def moc(df, h):
    """lag-1 MOC: close[D+1] -> close[D+1+h], indexed by the anchor D."""
    c = df["Close"].astype(float)
    return (c.shift(-(1 + h)) / c.shift(-1) - 1.0).replace([np.inf, -np.inf], np.nan)


def _signp(k, n):
    """Exact binomial sign p, but skipped on huge control cells.

    pitch_lab.sign_test's p=0.5 branch is exact rational arithmetic; at
    n ~ 6700 (an all-days control) one call costs ~8 seconds of big-int work.
    The sign test exists for the SMALL conditional cell, so it is reported
    there and suppressed (None) on cells above 1500 observations, where the
    hit-rate column already says everything a coin test would.
    """
    if n > 1500:
        return None
    return round(sign_test(k, n), 4)


def stat(vals, label, dates=None):
    v = np.asarray(vals, dtype=float)
    keep = ~np.isnan(v)
    v = v[keep]
    d = pd.DatetimeIndex(dates)[keep] if dates is not None else None
    s = summarize(v, label)
    if s["n"] == 0:
        return s
    up, dn = int((v > 0).sum()), int((v < 0).sum())
    s["record"] = f"{up}-{dn}"
    s["sign_p_up"] = _signp(up, len(v))
    s["sign_p_dn"] = _signp(dn, len(v))
    if d is not None and len(d):
        s["worst_on"] = str(d[int(np.argmin(v))].date())
        s["best_on"] = str(d[int(np.argmax(v))].date())
    s.pop("sd_pct", None)
    return s


def ex_top2_years(dates, vals):
    v = np.asarray(vals, float)
    d = pd.DatetimeIndex(dates)
    if len(v) < 3:
        return "n/a"
    by_yr = pd.Series(v, index=d.year).groupby(level=0).sum()
    drop = set(by_yr.sort_values(ascending=False).head(2).index)
    keep = ~np.isin(d.year, list(drop))
    w = v[keep]
    if len(w) == 0:
        return f"drop years {sorted(drop)} -> nothing left"
    up, dn = int((w > 0).sum()), int((w < 0).sum())
    return (f"drop 2 best years {sorted(drop)} -> n={len(w)} mean "
            f"{100*w.mean():+.3f}% median {100*np.median(w):+.3f}% record "
            f"{up}-{dn} sign p(up) {sign_test(up, len(w)):.4f} "
            f"p(dn) {sign_test(dn, len(w)):.4f}")


print("=" * 96)
print("CHECK B — energy extension  (asof %s)" % ASOF.date())
print("=" * 96)
print("tonight's readings, recomputed here from the cache (not copied from the brief):")
for t in TICKERS:
    df = px[t]
    print(f"  {t:<4} bars {len(df):>5} {df.index[0].date()}..{df.index[-1].date()}"
          f"  close {float(df['Close'].iloc[-1]):.4f}"
          f"  z10 {float(z10(df).iloc[-1]):+.2f}"
          f"  dist52wh {100*float(dist_hi(df).iloc[-1]):+.2f}%"
          f"  ret21d {100*float(ret_n(df, 21).iloc[-1]):+.2f}%")


def run(tkr, cond, label, horizons=HS, extra_ctrl=None):
    df = px[tkr]
    spine = df.index
    trig = spine[cond.reindex(spine).fillna(False).values]
    trig = trig[trig < ASOF]
    epi = declusters(trig, GAP_TD, spine)
    print("\n" + "=" * 96)
    print(f"CELL {tkr}: {label}")
    print("=" * 96)
    print(f"  trigger sessions {len(trig)} of {len(spine)}   episodes ({GAP_TD}td) {len(epi)}")
    if len(trig) == 0:
        print("  NO TRIGGERS. Dead.")
        return
    print(f"  span {trig[0].date()} .. {trig[-1].date()}")
    yr = pd.Series(pd.DatetimeIndex(epi).year).value_counts().sort_index()
    print("  episodes by year: " + ", ".join(f"{y}:{n}" for y, n in yr.items()))
    if len(epi) <= 40:
        print("  episode dates: " + ", ".join(str(d.date()) for d in epi))
    rows = []
    for h in horizons:
        s = moc(df, h)
        valid = s.dropna().index
        e = pd.DatetimeIndex(epi).intersection(valid)
        ta = pd.DatetimeIndex(trig).intersection(valid)
        rows.append(stat(s.loc[e].values, f"CELL episodes h={h}", e))
        rows.append(stat(s.loc[ta].values, f"  cell day-level h={h}", ta))
        if extra_ctrl is not None:
            ctd = spine[extra_ctrl.reindex(spine).fillna(False).values]
            ctd = pd.DatetimeIndex(ctd[ctd < ASOF]).intersection(valid)
            cep = declusters(ctd, GAP_TD, spine)
            rows.append(stat(s.loc[pd.DatetimeIndex(cep).intersection(valid)].values,
                             f"  CTRL no-extension episodes h={h}",
                             pd.DatetimeIndex(cep).intersection(valid)))
        in_span = valid[(valid >= trig[0]) & (valid <= trig[-1])]
        rows.append(stat(s.loc[in_span].values, f"  CTRL-a all days, span h={h}", in_span))
        rows.append(stat(s.loc[valid].values, f"  CTRL-b all days, full hist h={h}", valid))
        loc = local_control(valid, ta, 126)
        rows.append(stat(s.loc[loc].values, f"  CTRL-c local +/-126td h={h}", loc))
    show(rows, f"{tkr} lag-1 MOC forward, cell vs controls")

    for h in horizons:
        s = moc(df, h)
        e = pd.DatetimeIndex(epi).intersection(s.dropna().index)
        v = s.loc[e].values.astype(float)
        if len(v) == 0:
            continue
        print(f"\n-- {tkr} h={h}  n={len(v)}")
        er = era_split(e, v, ERA)
        for x in er:
            x.pop("sd_pct", None)
        show(er, f"   era split (cut {ERA})")
        print("   " + ex_top2_years(e, v))
        print(f"   bootstrap P(mean <= 0) = {bootstrap_p_le0(v):.3f}")
        print("   " + cluster_note(e, v))


# --- XLE ------------------------------------------------------------------
xle = px["XLE"]
xle_hi = dist_hi(xle) >= 0.0
xle_z = z10(xle) >= 1.5
run("XLE", xle_hi & xle_z, "at a 252d high AND z10 >= 1.5",
    extra_ctrl=xle_hi & ~xle_z)
run("XLE", xle_hi, "at a 252d high, NO z10 condition (control cell)")

# sensitivity on the z10 threshold
print("\n" + "=" * 96)
print("XLE z10 THRESHOLD SENSITIVITY (at a 252d high), episodes, lag-1 MOC")
print("=" * 96)
rows = []
for thr in (1.0, 1.25, 1.5, 1.74, 2.0):
    cond = xle_hi & (z10(xle) >= thr)
    trig = xle.index[cond.reindex(xle.index).fillna(False).values]
    trig = trig[trig < ASOF]
    epi = declusters(trig, GAP_TD, xle.index)
    for h in (5, 21):
        s = moc(xle, h)
        e = pd.DatetimeIndex(epi).intersection(s.dropna().index)
        rows.append(stat(s.loc[e].values, f"z10>={thr} h={h}", e))
show(rows, "threshold sweep")

# --- USO ------------------------------------------------------------------
uso = px["USO"]
uso_cond = (z10(uso) >= 2.5) & (ret_n(uso, 21) >= 0.15)
run("USO", uso_cond, "z10 >= 2.5 AND 21d return >= 15%",
    extra_ctrl=(z10(uso) >= 2.5) & (ret_n(uso, 21) < 0.15))
run("USO", z10(uso) >= 2.5, "z10 >= 2.5 only (control cell)")

# --- the other four names, headline cell only -----------------------------
print("\n" + "=" * 96)
print("COMPANION NAMES — at a 252d high AND z10 >= 1.5, episodes, lag-1 MOC")
print("=" * 96)
for t in ("XOP", "VLO", "CVX", "DBC"):
    df = px[t]
    cond = (dist_hi(df) >= 0.0) & (z10(df) >= 1.5)
    trig = df.index[cond.reindex(df.index).fillna(False).values]
    trig = trig[trig < ASOF]
    epi = declusters(trig, GAP_TD, df.index)
    rows = []
    for h in HS:
        s = moc(df, h)
        e = pd.DatetimeIndex(epi).intersection(s.dropna().index)
        valid = s.dropna().index
        rows.append(stat(s.loc[e].values, f"CELL h={h}", e))
        rows.append(stat(s.loc[valid].values, f"  CTRL all days h={h}", valid))
    show(rows, f"{t}  (episodes {len(epi)})")

# --- frozen parameters ----------------------------------------------------
print("\n" + "=" * 96)
print("FROZEN PARAMETERS as of %s" % ASOF.date())
print("=" * 96)
for t in TICKERS:
    df = px[t]
    atr = wilder_atr(df["High"].values, df["Low"].values, df["Close"].values)
    c, a = float(df["Close"].iloc[-1]), float(atr[-1])
    print(f"  {t:<4} bar {df.index[-1].date()}  O {float(df['Open'].iloc[-1]):.4f} "
          f"H {float(df['High'].iloc[-1]):.4f} L {float(df['Low'].iloc[-1]):.4f} "
          f"C {c:.4f}   Wilder-14 ATR {a:.4f} ({100*a/c:.2f}% of price)")
print("\nDONE.")
