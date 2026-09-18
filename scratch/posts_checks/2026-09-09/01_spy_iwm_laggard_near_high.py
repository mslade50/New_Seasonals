"""CHECK A (2026-09-09) — SPY long while the small-cap leg lags badly and the
index is still pinned near its own 52-week high.

State tonight (2026-09-09 close, the last bar in master_prices):
  IWM rank_21d = 11.5   SPY dist_52w_high = -1.99%
so the cell is LIVE and the order would be placed for Thursday 2026-09-10
(PPI), with CPI on Friday 2026-09-11 and FOMC on 2026-09-16.

Definitions, matched to scripts/build_pitch_state.py::_metrics_for so the
tape numbers the brief quotes and the numbers here are the same statistic:
  rank_21d = close.pct_change(21).rolling(252).rank(pct=True) * 100
  dist_52w_high = close / close.rolling(252).max() - 1

Trigger:  IWM rank_21d <= 15  AND  SPY dist_52w_high >= -3%.
Anchors are declustered at 5 sessions (this is a state cell, not a calendar
cell). Both vehicles are US-equity-session ETFs, so the calendars agree; the
anchor spine is SPY's own index and IWM anchors are the intersection.

Forms — the doctrine is lag-1, an idea seen at TONIGHT's close is entered
TOMORROW, so both tradeable forms are scored explicitly:

  MOC h   entry = Close[D+1], exit = Close[D+1+h]     time_td = h
  MOO h   entry = Open[D+1],  exit = Close[D+h]       time_td = h
          (h=1 is the entry session's own open->close)

Controls: the same form over the trigger span, over all history, and over the
local +/-126td neighbourhood ex-trigger. Splits: era pre/post 2018, midterm
years (year % 4 == 2), and whether a CPI print lands within 2 sessions of the
anchor (tonight's case: CPI is D+2).
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
IWM_RANK_MAX = 15.0
SPY_NEAR_HIGH = -0.03
HS = (1, 2, 3, 5)

px = load_prices(["SPY", "IWM"])
for t in px:
    px[t] = px[t][px[t].index <= ASOF]
SPY, IWM = px["SPY"], px["IWM"]


# ---------------------------------------------------------------------------
# state, exactly as _metrics_for computes it (single-ticker frames, so the
# raw pct_change form and the valid-session form coincide)
# ---------------------------------------------------------------------------
def rank_n(df, n=21, lb=252):
    c = df["Close"].astype(float)
    return c.pct_change(n).rolling(lb).rank(pct=True) * 100.0


def dist_hi(df, lb=252):
    c = df["Close"].astype(float)
    return c / c.rolling(lb).max() - 1.0


IWM_R21 = rank_n(IWM)
SPY_DHI = dist_hi(SPY)
IWM_DHI = dist_hi(IWM)


def form_series(df, form, h):
    """Return indexed by the ANCHOR (signal) session D."""
    o = df["Open"].astype(float)
    c = df["Close"].astype(float)
    if form == "moc":            # close D+1 -> close D+1+h
        r = c.shift(-(1 + h)) / c.shift(-1) - 1.0
    elif form == "moo":          # open D+1 -> close D+h
        r = c.shift(-h) / o.shift(-1) - 1.0
    else:
        raise ValueError(form)
    return r.replace([np.inf, -np.inf], np.nan)


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
    """Drop the two best CALENDAR YEARS entirely and restate."""
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
            f"{up}-{dn} sign p(up) {sign_test(up, len(w)):.4f}")


# ---------------------------------------------------------------------------
print("=" * 92)
print("CHECK A — SPY near its 52w high while IWM's 21d rank is <= 15 (asof %s)"
      % ASOF.date())
print("=" * 92)
print(f"  SPY bars {len(SPY)}  {SPY.index[0].date()} .. {SPY.index[-1].date()}"
      f"   last close {SPY['Close'].iloc[-1]:.4f}")
print(f"  IWM bars {len(IWM)}  {IWM.index[0].date()} .. {IWM.index[-1].date()}"
      f"   last close {IWM['Close'].iloc[-1]:.4f}")
print(f"\n  TONIGHT: IWM rank_21d = {IWM_R21.loc[ASOF]:.2f}   "
      f"SPY dist_52w_high = {100*SPY_DHI.loc[ASOF]:+.2f}%   "
      f"IWM dist_52w_high = {100*IWM_DHI.loc[ASOF]:+.2f}%")
print(f"  trigger definition: IWM rank_21d <= {IWM_RANK_MAX} AND "
      f"SPY dist_52w_high >= {100*SPY_NEAR_HIGH:.0f}%   -> live tonight: "
      f"{bool(IWM_R21.loc[ASOF] <= IWM_RANK_MAX and SPY_DHI.loc[ASOF] >= SPY_NEAR_HIGH)}")

spine = SPY.index
mask = (IWM_R21.reindex(spine) <= IWM_RANK_MAX) & (SPY_DHI.reindex(spine) >= SPY_NEAR_HIGH)
trig = spine[mask.fillna(False).values]
trig = trig[trig < ASOF]          # tonight's own bar has no forward return
epi = declusters(trig, GAP_TD, spine)
print(f"\n  trigger sessions (excl. tonight): {len(trig)}   "
      f"episodes after {GAP_TD}td declustering: {len(epi)}")
if len(trig):
    print(f"  span {trig[0].date()} .. {trig[-1].date()}")
    yr = pd.Series(pd.DatetimeIndex(epi).year).value_counts().sort_index()
    print("  episodes by year: " + ", ".join(f"{y}:{n}" for y, n in yr.items()))
    print("  episode dates: " + ", ".join(str(d.date()) for d in epi))

# ---------------------------------------------------------------------------
print("\n" + "=" * 92)
print("1. SPY FORWARD, BOTH TRADEABLE FORMS, WITH CONTROLS")
print("=" * 92)
for form, flab in (("moc", "MOC  close[D+1]->close[D+1+h]"),
                   ("moo", "MOO  open[D+1]->close[D+h]")):
    rows = []
    for h in HS:
        s = form_series(SPY, form, h)
        valid = s.dropna().index
        e = pd.DatetimeIndex(epi).intersection(valid)
        t_all = pd.DatetimeIndex(trig).intersection(valid)
        rows.append(stat(s.loc[e].values, f"CELL episodes h={h}", e))
        rows.append(stat(s.loc[t_all].values, f"  cell day-level h={h}", t_all))
        in_span = valid[(valid >= trig[0]) & (valid <= trig[-1])]
        rows.append(stat(s.loc[in_span].values, f"  CTRL-a all days, span h={h}", in_span))
        rows.append(stat(s.loc[valid].values, f"  CTRL-b all days, full hist h={h}", valid))
        loc = local_control(valid, t_all, 126)
        rows.append(stat(s.loc[loc].values, f"  CTRL-c local +/-126td h={h}", loc))
        if rows[-5]["n"] and rows[-2]["n"]:
            rows[-5]["edge_vs_allhist_pp"] = round(
                rows[-5]["mean_pct"] - rows[-2]["mean_pct"], 3)
    show(rows, f"SPY {flab}")

print("\n" + "=" * 92)
print("2. IWM (THE LAGGARD ITSELF), SAME CELL, SAME FORMS")
print("=" * 92)
iwm_spine = IWM.index
for form, flab in (("moc", "MOC  close[D+1]->close[D+1+h]"),
                   ("moo", "MOO  open[D+1]->close[D+h]")):
    rows = []
    for h in HS:
        s = form_series(IWM, form, h)
        valid = s.dropna().index
        e = pd.DatetimeIndex(epi).intersection(valid)
        rows.append(stat(s.loc[e].values, f"CELL episodes h={h}", e))
        rows.append(stat(s.loc[valid].values, f"  CTRL all days full hist h={h}", valid))
        if rows[-2]["n"] and rows[-1]["n"]:
            rows[-2]["edge_vs_allhist_pp"] = round(
                rows[-2]["mean_pct"] - rows[-1]["mean_pct"], 3)
    show(rows, f"IWM {flab}")

# SPY-minus-IWM spread, since the cell is a divergence
print("\n  SPY minus IWM (the relative leg), MOC episodes:")
rows = []
for h in HS:
    a = form_series(SPY, "moc", h)
    b = form_series(IWM, "moc", h)
    sp = (a - b).replace([np.inf, -np.inf], np.nan)
    valid = sp.dropna().index
    e = pd.DatetimeIndex(epi).intersection(valid)
    rows.append(stat(sp.loc[e].values, f"CELL SPY-IWM h={h}", e))
    rows.append(stat(sp.loc[valid].values, f"  CTRL all days h={h}", valid))
show(rows, "SPY-IWM spread, MOC form")

# ---------------------------------------------------------------------------
print("\n" + "=" * 92)
print("3. ERA SPLIT / MIDTERM SPLIT / HAIRCUT  (SPY, episodes)")
print("=" * 92)
for form in ("moc", "moo"):
    for h in HS:
        s = form_series(SPY, form, h)
        valid = s.dropna().index
        e = pd.DatetimeIndex(epi).intersection(valid)
        v = s.loc[e].values.astype(float)
        if len(v) == 0:
            continue
        print(f"\n-- SPY {form.upper()} h={h}  n={len(v)}")
        er = era_split(e, v, ERA)
        for x in er:
            x.pop("sd_pct", None)
        show(er, f"   era split (cut {ERA})")
        mid = np.array([(d.year % 4) == 2 for d in e])
        show([stat(v[mid], f"midterm years (n={int(mid.sum())})", e[mid]),
              stat(v[~mid], f"non-midterm (n={int((~mid).sum())})", e[~mid])],
             "   midterm split")
        print("   " + ex_top2_years(e, v))
        print(f"   bootstrap P(mean <= 0) = {bootstrap_p_le0(v):.3f}")
        print("   " + cluster_note(e, v))

# ---------------------------------------------------------------------------
print("\n" + "=" * 92)
print("4. CPI-WITHIN-2-SESSIONS SPLIT OF THE ANCHORS")
print("   (tonight PPI is D+1 and CPI is D+2, so this is the live configuration)")
print("=" * 92)
ev = load_events(["cpi"])["date"]
ppi_ev = load_events(["ppi"])["date"]
pos = pd.Series(range(len(spine)), index=spine)


def event_within(anchors, evdates, k=2):
    out = []
    for d in pd.DatetimeIndex(anchors):
        p = pos.get(d)
        if p is None or p + k >= len(spine):
            out.append(False)
            continue
        lo, hi = spine[p + 1], spine[p + k]
        out.append(bool(((evdates > lo - pd.Timedelta(days=1)) &
                         (evdates <= hi)).any()))
    return np.asarray(out, dtype=bool)


for form in ("moc", "moo"):
    for h in (1, 2, 3, 5):
        s = form_series(SPY, form, h)
        valid = s.dropna().index
        e = pd.DatetimeIndex(epi).intersection(valid)
        v = s.loc[e].values.astype(float)
        if len(v) == 0:
            continue
        cpi_flag = event_within(e, ev, 2)
        rows = [stat(v[cpi_flag], f"CPI within 2 sessions (n={int(cpi_flag.sum())})",
                     e[cpi_flag]),
                stat(v[~cpi_flag], f"no CPI within 2 (n={int((~cpi_flag).sum())})",
                     e[~cpi_flag])]
        show(rows, f"SPY {form.upper()} h={h}")
        if form == "moc" and h == 5:
            print("   anchors WITH a CPI in D+1..D+2: "
                  + ", ".join(str(d.date()) for d in e[cpi_flag]))

ppi_flag_live = True
print(f"\n  live calendar check: next session {spine[-1].date()} + 1 = Thu 2026-09-10 "
      f"(PPI), Fri 2026-09-11 (CPI), FOMC 2026-09-16 — from macro_events.csv:")
for k in ("ppi", "cpi", "fomc_decision"):
    nxt = load_events([k])
    nxt = nxt[nxt["date"] > ASOF].head(2)
    print(f"    {k:<14} " + ", ".join(str(pd.Timestamp(d).date()) for d in nxt["date"]))

# ---------------------------------------------------------------------------
print("\n" + "=" * 92)
print("5. FROZEN PARAMETERS as of %s" % ASOF.date())
print("=" * 92)
for name, df in (("SPY", SPY), ("IWM", IWM)):
    atr = wilder_atr(df["High"].values, df["Low"].values, df["Close"].values)
    c, a = float(df["Close"].iloc[-1]), float(atr[-1])
    print(f"  {name}  bar {df.index[-1].date()}  O {float(df['Open'].iloc[-1]):.4f} "
          f"H {float(df['High'].iloc[-1]):.4f} L {float(df['Low'].iloc[-1]):.4f} "
          f"C {c:.4f}   Wilder-14 ATR {a:.4f}  ({100*a/c:.2f}% of price)")
print("\nDONE.")
