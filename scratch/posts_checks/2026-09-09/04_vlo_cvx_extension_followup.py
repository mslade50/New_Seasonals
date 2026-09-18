"""CHECK B follow-up (2026-09-09) — the single names that carried check B.

02_energy_extension.py ran the "at a 252-day high AND z10 >= 1.5" cell across
XLE / XOP / VLO / CVX / DBC. The ETF cell (XLE) had a hit-rate tilt and no mean
edge; VLO and CVX were the two names where the cell beat its own control on the
MEAN as well. Both are live tonight (VLO z10 2.93, CVX z10 2.05, both closing
exactly at a 252d high), so they get the full battery here rather than the
companion-table summary: local control, era split, midterm split, drop-2-best-
years haircut, bootstrap, the sign test, and the earnings-proximity check a
single-stock idea needs before it can be written as an order.

Same conventions as 02: _metrics_for definitions, lag-1 MOC (entry close D+1,
exit close D+1+h), 5-session declustering.
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
NAMES = ["VLO", "CVX", "XOP"]

px = load_prices(NAMES)
for t in list(px):
    px[t] = px[t][px[t].index <= ASOF]


def z10(df):
    c = df["Close"].astype(float)
    return c.pct_change(10) / (c.pct_change().rolling(21).std() * np.sqrt(10))


def dist_hi(df, lb=252):
    c = df["Close"].astype(float)
    return c / c.rolling(lb).max() - 1.0


def moc(df, h):
    c = df["Close"].astype(float)
    return (c.shift(-(1 + h)) / c.shift(-1) - 1.0).replace([np.inf, -np.inf], np.nan)


def _signp(k, n):
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
    by_yr = pd.Series(v, index=d.year).groupby(level=0).sum()
    drop = set(by_yr.sort_values(ascending=False).head(2).index)
    keep = ~np.isin(d.year, list(drop))
    w = v[keep]
    up, dn = int((w > 0).sum()), int((w < 0).sum())
    return (f"drop 2 best years {sorted(drop)} -> n={len(w)} mean "
            f"{100*w.mean():+.3f}% median {100*np.median(w):+.3f}% record "
            f"{up}-{dn} sign p(up) {sign_test(up, len(w)):.4f}")


for tkr in NAMES:
    df = px[tkr]
    spine = df.index
    cond = (dist_hi(df) >= 0.0) & (z10(df) >= 1.5)
    trig = spine[cond.reindex(spine).fillna(False).values]
    trig = trig[trig < ASOF]
    epi = declusters(trig, GAP_TD, spine)
    print("\n" + "=" * 96)
    print(f"{tkr}: at a 252d high AND z10 >= 1.5   (tonight z10 "
          f"{float(z10(df).iloc[-1]):+.2f}, dist52wh "
          f"{100*float(dist_hi(df).iloc[-1]):+.2f}%)")
    print("=" * 96)
    print(f"  trigger sessions {len(trig)}   episodes {len(epi)}   "
          f"span {trig[0].date()} .. {trig[-1].date()}")
    yr = pd.Series(pd.DatetimeIndex(epi).year).value_counts().sort_index()
    print("  episodes by year: " + ", ".join(f"{y}:{n}" for y, n in yr.items()))

    rows = []
    for h in HS:
        s = moc(df, h)
        valid = s.dropna().index
        e = pd.DatetimeIndex(epi).intersection(valid)
        ta = pd.DatetimeIndex(trig).intersection(valid)
        in_span = valid[(valid >= trig[0]) & (valid <= trig[-1])]
        loc = local_control(valid, ta, 126)
        rows.append(stat(s.loc[e].values, f"CELL episodes h={h}", e))
        rows.append(stat(s.loc[in_span].values, f"  CTRL-a all days, span h={h}", in_span))
        rows.append(stat(s.loc[valid].values, f"  CTRL-b all days, full h={h}", valid))
        rows.append(stat(s.loc[loc].values, f"  CTRL-c local +/-126td h={h}", loc))
        rows[-4]["edge_vs_local_pp"] = round(rows[-4]["mean_pct"] - rows[-1]["mean_pct"], 3)
    show(rows, f"{tkr} lag-1 MOC, cell vs three controls")

    for h in HS:
        s = moc(df, h)
        e = pd.DatetimeIndex(epi).intersection(s.dropna().index)
        v = s.loc[e].values.astype(float)
        print(f"\n-- {tkr} h={h}  n={len(v)}")
        er = era_split(e, v, ERA)
        for x in er:
            x.pop("sd_pct", None)
        show(er, f"   era split (cut {ERA})")
        mid = np.array([(d.year % 4) == 2 for d in e])
        show([stat(v[mid], f"midterm (n={int(mid.sum())})", e[mid]),
              stat(v[~mid], f"non-midterm (n={int((~mid).sum())})", e[~mid])],
             "   midterm split")
        print("   " + ex_top2_years(e, v))
        print(f"   bootstrap P(mean <= 0) = {bootstrap_p_le0(v):.3f}")
        print("   " + cluster_note(e, v))

# ---------------------------------------------------------------------------
# earnings proximity: a single-stock 5-day hold has to know where the print is
# ---------------------------------------------------------------------------
print("\n" + "=" * 96)
print("EARNINGS PROXIMITY (data/earnings_calendar.parquet)")
print("=" * 96)
ep = Path(__file__).resolve().parents[3] / "data" / "earnings_calendar.parquet"
if not ep.exists():
    print("  earnings_calendar.parquet MISSING — cannot check. Say so in the report.")
else:
    ec = pd.read_parquet(ep)
    col = "date" if "date" in ec.columns else ec.columns[1]
    tcol = "ticker" if "ticker" in ec.columns else ec.columns[0]
    ec[col] = pd.to_datetime(ec[col])
    for tkr in NAMES:
        g = ec[ec[tcol] == tkr].sort_values(col)
        if g.empty:
            print(f"  {tkr}: no rows in the earnings calendar")
            continue
        past = g[g[col] <= ASOF][col]
        fut = g[g[col] > ASOF][col]
        print(f"  {tkr}: last print "
              f"{past.iloc[-1].date() if len(past) else 'n/a'}   next "
              f"{fut.iloc[0].date() if len(fut) else 'n/a'}"
              + (f"   ({(fut.iloc[0] - ASOF).days} calendar days away)"
                 if len(fut) else ""))

print("\n" + "=" * 96)
print("FROZEN PARAMETERS as of %s" % ASOF.date())
print("=" * 96)
for tkr in NAMES:
    df = px[tkr]
    atr = wilder_atr(df["High"].values, df["Low"].values, df["Close"].values)
    c, a = float(df["Close"].iloc[-1]), float(atr[-1])
    print(f"  {tkr:<4} bar {df.index[-1].date()}  O {float(df['Open'].iloc[-1]):.4f} "
          f"H {float(df['High'].iloc[-1]):.4f} L {float(df['Low'].iloc[-1]):.4f} "
          f"C {c:.4f}   Wilder-14 ATR {a:.4f} ({100*a/c:.2f}% of price)")
print("\nDONE.")
