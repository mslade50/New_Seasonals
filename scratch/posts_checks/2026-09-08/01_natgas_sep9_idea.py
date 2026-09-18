"""NG=F / UNG "Sep 09 +/-2 trading-day-of-year" cell, in TRADEABLE form.

The evening context sweep found the cell as a lag-0 close-to-close number:
anchor on the analogue of TODAY (trading-day-of-year 175, tol +/-2, one pick
per year), h1 = the NEXT session's close-to-close return.  n=25, mean +1.351%,
record 18-7, sign p 0.0216, vs all-September +0.436% / 271-258.

That is NOT the order you can place tonight.  Tomorrow (Wed 2026-09-09) IS the
anchor session, so the only entries available are:

  (a) MOO_C0   long at the ANCHOR OPEN, out at the ANCHOR CLOSE   (time_td=1)
  (b) MOO_C1   long at the ANCHOR OPEN, out at the NEXT CLOSE     (time_td=2)
  (c) MOC_C1   long at the ANCHOR CLOSE, out at the NEXT CLOSE    (= the cell)

(c) is the cell itself and is the plain lag-1 close form of the pitch doctrine.
(a) and (b) are the forms an MOO order actually buys, and they are DIFFERENT
statistics -- (c) never touches an open print.

Both vehicles are scored: NG=F (the continuous future the cell was found on,
2000+) and UNG (the ETF anyone would actually trade, 2007+).  Anchors are the
hand-built picks from scratch/context_checks/2026-09-08/07_natgas_september.py
(2001-09-14 .. 2025-09-11), reused verbatim so the tables line up.

Controls for every form: the same form on all September sessions and on all
sessions.  Splits: era pre/post 2018, 52-week-drawdown state at the anchor
(NG=F is currently 61% below its 52w high), top-2-year concentration.

Anchor convention note: the anchors are defined on NG=F's calendar.  UNG has a
different calendar (2026-09-07 is a US equity holiday NG=F traded through), so
UNG anchors are the subset of those dates present in UNG's own index; any drop
is printed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa: F401,F403  (load_prices, summarize, sign_test, ...)

import numpy as np
import pandas as pd

ASOF = pd.Timestamp("2026-09-08")
ERA = "2018-01-01"
TOL = 2
SEED = 42

FORMS = [
    ("a MOO_C0  open->same close ", "moo_c0"),
    ("b MOO_C1  open->next close ", "moo_c1"),
    ("c MOC_C1  close->next close", "moc_c1"),
]

px = load_prices(["NG=F", "UNG"])
NG = px["NG=F"]
NG = NG[NG.index <= ASOF]
UNG = px["UNG"]
UNG = UNG[UNG.index <= ASOF]
VEH = {"NG=F": NG, "UNG": UNG}


# ---------------------------------------------------------------------------
# anchors -- rebuilt exactly as the context drill builds them
# ---------------------------------------------------------------------------
ng_close = NG["Close"].astype(float).dropna()
doy = pd.Series(ng_close.index.year, index=ng_close.index)
doy = pd.Series(doy.groupby(doy.values).cumcount().values + 1, index=ng_close.index)
TARGET_DOY = int(doy.iloc[-1])

picks = []
for y in sorted(set(ng_close.index.year)):
    if y >= ASOF.year:
        continue
    cand = doy[(ng_close.index.year == y) & (doy - TARGET_DOY).abs().le(TOL)]
    if cand.empty:
        continue
    picks.append(cand.index[(cand - TARGET_DOY).abs().values.argmin()])
PICKS = pd.DatetimeIndex(picks)

# 52-week drawdown state at each anchor (house rule: rolling on valid sessions)
HI252 = rolling_on_valid(ng_close, lambda x: x.rolling(252).max())
DD52 = ng_close / HI252 - 1.0


# ---------------------------------------------------------------------------
# form machinery
# ---------------------------------------------------------------------------
def form_series(df: pd.DataFrame, form: str) -> pd.Series:
    """Return the form's return, indexed by the ANCHOR session date."""
    o = df["Open"].astype(float)
    c = df["Close"].astype(float)
    if form == "moo_c0":
        r = c / o - 1.0
    elif form == "moo_c1":
        r = c.shift(-1) / o - 1.0
    elif form == "moc_c1":
        r = c.shift(-1) / c - 1.0
    else:
        raise ValueError(form)
    return r.replace([np.inf, -np.inf], np.nan)


SER = {v: {f: form_series(VEH[v], f) for _, f in FORMS} for v in VEH}


def stat(vals, label, dates=None):
    v = np.asarray(vals, dtype=float)
    keep = ~np.isnan(v)
    v = v[keep]
    d = pd.DatetimeIndex(dates)[keep] if dates is not None else None
    s = summarize(v, label)
    if s["n"] == 0:
        return s
    up, down = int((v > 0).sum()), int((v < 0).sum())
    s["record"] = f"{up}-{down}"
    s["sign_p"] = round(sign_test(max(up, down), len(v)), 4)
    s["sign_dir"] = "up" if up >= down else "down"
    if d is not None and len(d):
        s["worst_on"] = str(d[int(np.argmin(v))].date())
        s["best_on"] = str(d[int(np.argmax(v))].date())
    for k in ("sd_pct",):
        s.pop(k, None)
    return s


def pull(veh: str, form: str, dates) -> tuple[pd.DatetimeIndex, np.ndarray]:
    s = SER[veh][form]
    d = pd.DatetimeIndex(dates).intersection(s.dropna().index)
    return d, s.loc[d].values.astype(float)


def anchors_for(veh: str) -> pd.DatetimeIndex:
    return PICKS.intersection(VEH[veh].index)


def top2_year_share(dates, vals) -> str:
    v = np.asarray(vals, float)
    if len(v) == 0:
        return "n/a"
    by_yr = pd.Series(v, index=pd.DatetimeIndex(dates).year).groupby(level=0).sum()
    tot = by_yr.sum()
    top = by_yr.sort_values(ascending=False).head(2)
    share = (top.sum() / tot * 100) if tot != 0 else np.nan
    names = ", ".join(f"{y} {100*r:+.2f}%" for y, r in top.items())
    return (f"top2 years [{names}] = {100*top.sum():+.2f}pp of {100*tot:+.2f}pp "
            f"total ({share:.0f}%)")


def ex_top2(dates, vals) -> str:
    v = np.asarray(vals, float)
    if len(v) < 3:
        return "n/a"
    order = np.argsort(-v)[:2]
    keep = np.ones(len(v), bool)
    keep[order] = False
    w = v[keep]
    up = int((w > 0).sum())
    return (f"drop 2 best years -> mean {100*w.mean():+.3f}%  median "
            f"{100*np.median(w):+.3f}%  n={len(w)}  record {up}-{len(w)-up}  "
            f"sign p {sign_test(max(up, len(w)-up), len(w)):.4f}")


print("=" * 78)
print("0. STATE, ANCHORS, VEHICLES")
print("=" * 78)
for v, df in VEH.items():
    c = df["Close"].astype(float).dropna()
    print(f"  {v:<5} bars {len(c):>5}   {c.index[0].date()} .. {c.index[-1].date()}"
          f"   last close {c.iloc[-1]:.4f}")
print(f"  today's trading-day-of-year = {TARGET_DOY} (tol +/-{TOL})")
print(f"  NG=F anchors: {len(PICKS)}  ({PICKS[0].date()} .. {PICKS[-1].date()})")
ung_anch = anchors_for("UNG")
missed = pd.DatetimeIndex([d for d in PICKS if d.year >= 2008 and d not in UNG.index])
print(f"  UNG anchors : {len(ung_anch)} ({ung_anch[0].date()} .. {ung_anch[-1].date()})"
      f"   post-2007 anchors absent from UNG's calendar: "
      f"{[str(d.date()) for d in missed] or 'none'}")
print(f"  NG=F dd52 today {100*DD52.iloc[-1]:+.1f}%  (52w high {HI252.iloc[-1]:.3f})")
print("  NOTE: the anchor session is TOMORROW (Wed 2026-09-09), so a MOO order")
print("        buys the anchor's OWN open -- forms (a)/(b), not the cell (c).")


# ---------------------------------------------------------------------------
# 1. per-year table, both vehicles, all three forms
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("1. PER-YEAR VALUES (percent), both vehicles, all three forms")
print("=" * 78)
tab = []
for d in PICKS:
    row = {"year": d.year, "anchor": str(d.date()),
           "dd52_pct": round(100 * DD52.loc[d], 1) if pd.notna(DD52.get(d)) else None}
    for veh in ("NG=F", "UNG"):
        for lbl, f in FORMS:
            key = f"{veh.replace('=F','')}_{lbl.split()[1]}"
            s = SER[veh][f]
            val = s.get(d, np.nan) if d in s.index else np.nan
            row[key] = round(100 * val, 2) if pd.notna(val) else None
    tab.append(row)
print(pd.DataFrame(tab).to_string(index=False))


# ---------------------------------------------------------------------------
# 2. headline table: every form x vehicle, with controls
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("2. FORMS AND CONTROLS")
print("=" * 78)
for veh in ("NG=F", "UNG"):
    rows = []
    anch = anchors_for(veh)
    idx = VEH[veh].index
    sept = idx[idx.month == 9]
    for lbl, f in FORMS:
        d, v = pull(veh, f, anch)
        rows.append(stat(v, f"CELL {lbl}", d))
        d2, v2 = pull(veh, f, sept)
        rows.append(stat(v2, f"  CTRL all Sept  {lbl}", d2))
        d3, v3 = pull(veh, f, idx)
        rows.append(stat(v3, f"  CTRL all days  {lbl}", d3))
        # edge vs the month, which is the control that matters for natgas
        if rows[-3]["n"] and rows[-2]["n"]:
            rows[-3]["edge_vs_sept_pp"] = round(
                rows[-3]["mean_pct"] - rows[-2]["mean_pct"], 3)
    show(rows, f"{veh}: cell vs September vs all days")


# ---------------------------------------------------------------------------
# 3. era split, concentration, bootstrap
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("3. ERA SPLIT / CONCENTRATION / BOOTSTRAP (seed %d)" % SEED)
print("=" * 78)
for veh in ("NG=F", "UNG"):
    anch = anchors_for(veh)
    for lbl, f in FORMS:
        d, v = pull(veh, f, anch)
        if len(v) == 0:
            continue
        print(f"\n-- {veh} {lbl}  n={len(v)}")
        er = era_split(d, v, ERA)
        for e in er:
            e.pop("sd_pct", None)
        show(er, f"   era split (cut {ERA})")
        print("   " + top2_year_share(d, v))
        print("   " + ex_top2(d, v))
        print(f"   bootstrap P(mean <= 0) = {bootstrap_p_le0(v, seed=SEED):.3f}"
              f"   (iid over years; anchors are 1/yr so no overlap)")


# ---------------------------------------------------------------------------
# 4. drawdown split -- today's state is -61%
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("4. 52-WEEK DRAWDOWN SPLIT AT THE ANCHOR (today: %.0f%%)"
      % (100 * DD52.iloc[-1]))
print("=" * 78)
for thr, tlab in ((-0.50, "-50%"), (-0.60, "-60% = TODAY")):
    deep = pd.DatetimeIndex([d for d in PICKS
                             if pd.notna(DD52.get(d)) and DD52.loc[d] <= thr])
    shal = pd.DatetimeIndex([d for d in PICKS
                             if pd.notna(DD52.get(d)) and DD52.loc[d] > thr])
    print(f"\n  threshold dd52 <= {tlab}")
    print(f"    deep years  ({len(deep)}): "
          + ", ".join(f"{d.year}({100*DD52.loc[d]:.0f}%)" for d in deep))
    print(f"    other years ({len(shal)})")
    for veh in ("NG=F", "UNG"):
        rows = []
        for lbl, f in FORMS:
            d, v = pull(veh, f, deep.intersection(VEH[veh].index))
            rows.append(stat(v, f"deep  {lbl}", d))
            d, v = pull(veh, f, shal.intersection(VEH[veh].index))
            rows.append(stat(v, f"other {lbl}", d))
        show(rows, f"   {veh} split at {tlab}")
        for r in rows:
            if r.get("n", 0) and r["n"] < 15:
                print(f"     ** {r['label']}: n={r['n']} -- anecdote tier, "
                      f"sign test only **")

print("\n  Unconditional deep-drawdown control on NG=F (any session, not just "
      "the cell):")
rows = []
for thr, tlab in ((-0.50, "-50%"), (-0.60, "-60%")):
    mask = (DD52 <= thr).reindex(NG.index, fill_value=False).values
    dd_days = NG.index[mask]
    for lbl, f in FORMS:
        d, v = pull("NG=F", f, dd_days)
        rows.append(stat(v, f"any day dd52<={tlab} {lbl}", d))
show(rows, "   NG=F deep-drawdown sessions, same forms")


# ---------------------------------------------------------------------------
# 5. frozen idea parameters
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("5. FROZEN IDEA PARAMETERS as of %s" % ASOF.date())
print("=" * 78)
for veh in ("NG=F", "UNG"):
    df = VEH[veh]
    atr = wilder_atr(df["High"].values, df["Low"].values, df["Close"].values)
    c = float(df["Close"].iloc[-1])
    a = float(atr[-1])
    print(f"  {veh:<5} bar {df.index[-1].date()}   close {c:.4f}   "
          f"Wilder-14 ATR {a:.4f}   ATR/price {100*a/c:.2f}%")
    print(f"        open {float(df['Open'].iloc[-1]):.4f}  "
          f"high {float(df['High'].iloc[-1]):.4f}  "
          f"low {float(df['Low'].iloc[-1]):.4f}")
    for k in (0.5, 1.0, 1.5, 2.0):
        print(f"        {k:>3.1f} ATR = {k*a:.4f}  "
              f"(-> {c - k*a:.4f} / {c + k*a:.4f})")

print("\n  Idea as it would be written:")
print("    LONG UNG (or NG=F), entry MOO on 2026-09-09, time_td = 1 or 2,")
print("    no stop implied by the cell; ATR figures above are the frozen")
print("    reference for any stop/target a card chooses to attach.")

print("\nDONE.")
