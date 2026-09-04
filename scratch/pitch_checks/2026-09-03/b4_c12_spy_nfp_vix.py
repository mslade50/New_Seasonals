"""C12 KILL CHECK -- the equity DIRECTION leg of the pre-print cross.

Long or short SPY MOC on the last session before the payrolls print (k=-1 entry
off the k=-2 anchor, lag=1), exit MOC at the print close (h=1), gated on the VIX
21-day RANGE percentile being in its bottom 15% (today 3.57 on the rel-range
form, 1.98 on the abs-range form).

The registry (2026-08-07) swept POST-NFP equity direction and found it empty.
This is the PRE-print session out of a dead range, which was not swept. What
this check owes back, whatever the verdict:
  - the SIGN
  - the excess over SPY's own unconditional drift at the same horizon
  - the GATE ATTRIBUTION (run it without the gate)
  - corr(SPY h=1, SVXY h=1) on the gated anchor set, which tells us whether the
    sibling volatility idea (C1) is a vol trade or a disguised equity beta trade
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import (load_prices, load_events, anchor_positions, battery,
                       summarize, show, sign_test, declusters, rolling_on_valid,
                       bootstrap_p_le0)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 230)

TK = ["SPY", "^VIX", "SVXY", "QQQ"]
raw = load_prices(TK)
px = pd.DataFrame({t: raw[t]["Close"] for t in TK}).dropna(subset=["SPY"])
cal = px.index

vix = raw["^VIX"]["Close"].dropna()
rng21 = vix.rolling(21).max() - vix.rolling(21).min()
rel = rng21 / vix.rolling(21).mean()
REL = (rel.rolling(252).rank(pct=True) * 100).reindex(cal).ffill(limit=3)
ABS = (rng21.rolling(252).rank(pct=True) * 100).reindex(cal).ffill(limit=3)
print("cal %s .. %s" % (cal[0].date(), cal[-1].date()))
print("live 2026-09-02: VIX rel-range pctile %.2f  abs-range pctile %.2f"
      % (REL.iloc[-1], ABS.iloc[-1]))

nfp = load_events(["nfp"])["date"]
pos, _ = anchor_positions(cal, nfp, -2)
anchor = pd.DatetimeIndex([cal[i] for i in pos])
entry_all = pd.DatetimeIndex([cal[i + 1] for i in pos if i + 1 < len(cal)])
_one = pd.Series(1, index=cal)
tdom = pd.Series(_one.groupby([cal.year, cal.month]).cumcount().values + 1, index=cal)


def ret_from_entry(s, h):
    return s.shift(-h) / s - 1.0


ANCH = pd.Series(cal.isin(anchor), index=cal)
GATED = ANCH & (REL <= 15.0)
print("NFP k=-2 anchors: %d   gated (rel-range pctile <= 15): %d"
      % (int(ANCH.sum()), int(GATED.fillna(False).sum())))
print("gated anchor dates:",
      ", ".join(str(d.date()) for d in cal[GATED.fillna(False).values]))

# ==========================================================================
print("\n" + "=" * 78)
print("ATTACK 1 -- battery, long SPY through the print, gated and ungated")
print("=" * 78)
variants = {"gate rel<=15 (LIVE)": GATED,
            "gate rel<=5": ANCH & (REL <= 5.0),
            "gate rel<=25": ANCH & (REL <= 25.0),
            "gate abs<=15": ANCH & (ABS <= 15.0),
            "gate OFF (all NFP)": ANCH,
            "COMPLEMENT rel>15": ANCH & (REL > 15.0)}
for h in (1, 2, 3):
    battery(px, GATED, [("SPY", 1.0)], h=h,
            title="C12 SPY long, last session before the print, dead 21d VIX range",
            cost_bps=2.0, variants=variants if h == 1 else None,
            lag=1, min_gap=5, event_kinds=("nfp",))

# ==========================================================================
print("\n" + "=" * 78)
print("ATTACK 2 -- sign, excess over SPY's own drift, tdom control")
print("=" * 78)
tdom_set = sorted(set(tdom.loc[entry_all].values))
for h in (1, 2, 3, 5):
    r = ret_from_entry(px["SPY"], h)
    gent = pd.DatetimeIndex([cal[i + 1] for i in pos
                             if i + 1 < len(cal) and (REL.iloc[i] <= 15.0)])
    rows = []
    for lbl, sel in (("GATED entries (LIVE cell)", gent),
                     ("ALL NFP entries", entry_all)):
        v = r.reindex(sel).dropna()
        s = summarize(v.values, f"SPY h={h} {lbl}")
        if s["n"]:
            s["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
            s["edge_vs_drift_pp"] = round(100 * (v.mean() - r.dropna().mean()), 3)
            b = r[tdom.isin(tdom_set).values].dropna()
            s["edge_vs_tdom_pp"] = round(100 * (v.mean() - b.mean()), 3)
        rows.append(s)
    rows.append(summarize(r.dropna().values, f"SPY h={h} all-days drift"))
    rows.append(summarize(r[tdom.isin(tdom_set).values].dropna().values,
                          f"SPY h={h} tdom-matched"))
    # the gate on NON-anchor days: is a dead range bullish or bearish generally?
    nong = cal[(~ANCH & (REL <= 15.0)).fillna(False).values]
    rows.append(summarize(r.reindex(nong).dropna().values,
                          f"SPY h={h} dead range, NO event (n days)"))
    show(rows, f"SPY direction, h={h}")

# ==========================================================================
print("\n" + "=" * 78)
print("ATTACK 3 -- gate attribution ladder (does the range gate move anything?)")
print("=" * 78)
for h in (1, 3):
    rows = []
    for thr in (5, 10, 15, 25, 50, 100):
        sel = pd.DatetimeIndex([cal[i + 1] for i in pos
                                if i + 1 < len(cal) and REL.iloc[i] <= thr])
        disc = pd.DatetimeIndex([cal[i + 1] for i in pos
                                 if i + 1 < len(cal) and REL.iloc[i] > thr])
        r = ret_from_entry(px["SPY"], h)
        c, d = r.reindex(sel).dropna(), r.reindex(disc).dropna()
        base = r.reindex(entry_all).dropna().mean()
        rows.append({"h": h, "rel<=": thr, "n_kept": len(c),
                     "kept_pct": round(100 * c.mean(), 3),
                     "hit": round(100 * (c > 0).mean(), 1),
                     "n_disc": len(d),
                     "disc_pct": round(100 * d.mean(), 3) if len(d) else np.nan,
                     "gate_value_pp": round(100 * (c.mean() - base), 3)})
    show(rows, f"gate ladder h={h}  (gate_value = gated minus all-NFP)")

# ==========================================================================
print("\n" + "=" * 78)
print("ATTACK 4 -- era / midterm split on the gated cell")
print("=" * 78)
for h in (1, 3):
    r = ret_from_entry(px["SPY"], h)
    gent = pd.DatetimeIndex([cal[i + 1] for i in pos
                             if i + 1 < len(cal) and (REL.iloc[i] <= 15.0)])
    v = r.reindex(gent).dropna()
    rows = []
    for lbl, sel in (("pre-2018", [d.year < 2018 for d in v.index]),
                     ("2018+", [d.year >= 2018 for d in v.index]),
                     ("MIDTERM", [d.year % 4 == 2 for d in v.index]),
                     ("non-midterm", [d.year % 4 != 2 for d in v.index]),
                     ("SEPTEMBER", [d.month == 9 for d in v.index])):
        x = v[sel]
        s = summarize(x.values, f"h={h} {lbl}")
        if s["n"]:
            s["sign_p"] = round(sign_test(int((x > 0).sum()), len(x)), 4)
        rows.append(s)
    show(rows, f"gated-cell splits h={h}")
    print("  gated entries and their h=%d returns (%%):" % h)
    print("   ", {str(d.date()): round(100 * x, 2) for d, x in v.items()})

# ==========================================================================
print("\n" + "=" * 78)
print("ATTACK 5 -- corr(SPY h=1, SVXY h=1) on the gated anchor set")
print("  is the sibling vol idea a vol trade or a disguised equity beta trade?")
print("=" * 78)
sv = px["SVXY"].dropna()
for h in (1, 2, 3):
    rs = ret_from_entry(px["SPY"], h)
    rv = ret_from_entry(px["SVXY"], h)
    both = pd.concat([rs, rv], axis=1).dropna()
    both.columns = ["SPY", "SVXY"]
    gent = pd.DatetimeIndex([cal[i + 1] for i in pos
                             if i + 1 < len(cal) and (REL.iloc[i] <= 15.0)])
    aent = entry_all
    for lbl, sel in (("GATED anchors", gent), ("ALL NFP anchors", aent),
                     ("all days (SVXY era)", both.index)):
        b = both.reindex(pd.DatetimeIndex(sel)).dropna()
        if len(b) < 3:
            print("  h=%d %-22s n=%d  (too few)" % (h, lbl, len(b)))
            continue
        c = float(np.corrcoef(b["SPY"], b["SVXY"])[0, 1])
        beta = np.polyfit(b["SPY"], b["SVXY"], 1)[0]
        r2 = c ** 2
        print("  h=%d %-22s n=%3d  corr %+.3f  R2 %.3f  beta(SVXY on SPY) %+.2f  "
              "SPY %+.3f%%  SVXY %+.3f%%"
              % (h, lbl, len(b), c, r2, beta, 100 * b["SPY"].mean(),
                 100 * b["SVXY"].mean()))
print("\nDONE.")
