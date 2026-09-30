"""kA c5 round 1+2 - long XLF on a bank-breadth flush under a bear steepener.

Pre-specified cell (surface map c5): >= 80% of the 11-bank complex
(JPM BAC C WFC GS MS USB PNC SCHW BNY STT, as in 02_watch_extra.py) at a 5d pct_rank <= 20
AND the 10y-3m spread (^TNX - ^IRX) widened >= 25 bp over 21 sessions with the ten-year
rising (bear steepener). Long XLF, rows KRE and the equal-weight 11 banks; h=3/5.

The curve gate is the only new element, so the decisive tests are:
  gate attribution (breadth alone vs joint vs the discarded complement, and the
  breadth floor with the curve flat/flattening), filter_vs_reanchor, definition
  neighbours (10y-5y via ^FVX; 10/21/63 sessions; 15/25/40 bp; bear clause on/off),
  midterm and SPY 200d splits, and SPY-relative residual.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change
import numpy as np
import pandas as pd

BANKS = ["JPM", "BAC", "C", "WFC", "GS", "MS", "USB", "PNC", "SCHW", "BNY", "STT"]
TK = BANKS + ["XLF", "KRE", "SPY", "^TNX", "^IRX", "^FVX"]
px = close_panel(TK)
px = px[px.index >= "2000-01-01"]
idx = px.index

r5 = pd.DataFrame({b: pct_rank(px[b], 5) for b in BANKS})
valid_n = r5.notna().sum(axis=1)
share = (r5 <= 20).sum(axis=1) / valid_n.replace(0, np.nan)
breadth = (share >= 0.80) & (valid_n >= 9)

y10, y3m, y5 = px["^TNX"], px["^IRX"], px["^FVX"]


def dchg(s, n):
    v = s.dropna()
    return (v - v.shift(n)).reindex(idx)


spread = y10 - y3m
spread5 = y10 - y5
d_sp = {n: dchg(spread, n) for n in (10, 21, 63)}
d_sp5 = {n: dchg(spread5, n) for n in (10, 21, 63)}
d_10 = {n: dchg(y10, n) for n in (10, 21, 63)}
curve = (d_sp[21] >= 0.25) & (d_10[21] > 0)
joint = (breadth & curve).reindex(idx, fill_value=False)

# equal-weight bank basket (daily rebalanced index of available names)
dr = px[BANKS].pct_change(fill_method=None)
px["EWB"] = (1 + dr.mean(axis=1).fillna(0)).cumprod()
sma200 = rolling_on_valid(px["SPY"], lambda x: x.rolling(200).mean())
above200 = (px["SPY"] > sma200).reindex(idx, fill_value=False)
midterm = pd.Series(idx.year % 4 == 2, index=idx)

print("=== live (", idx[-1].date(), ") ===")
print(f"  bank r5 share <= 20: {share.iloc[-1]:.2f}  breadth {bool(breadth.iloc[-1])}")
print(f"  10y-3m {100*spread.iloc[-1]:.1f} bp; 21d change {100*d_sp[21].iloc[-1]:+.1f} bp "
      f"(10d {100*d_sp[10].iloc[-1]:+.1f}, 63d {100*d_sp[63].iloc[-1]:+.1f}); TNX 21d {100*d_10[21].iloc[-1]:+.1f} bp")
print(f"  10y-5y {100*spread5.iloc[-1]:.1f} bp; 21d change {100*d_sp5[21].iloc[-1]:+.1f} bp")
print(f"  curve gate {bool(curve.iloc[-1])}; joint {bool(joint.iloc[-1])}; SPY>200d {bool(above200.iloc[-1])}")
print(f"  breadth days last 15 sessions: {[str(d.date()) for d in idx[-15:][breadth.reindex(idx, fill_value=False).iloc[-15:].values]]}")
print(f"  joint days last 15 sessions: {[str(d.date()) for d in idx[-15:][joint.iloc[-15:].values]]}")

GAP = 10


def ep(mask, h, veh="XLF", gap=GAP, lbl="", legs=None):
    legs = legs or [(veh, 1.0)]
    ret = vehicle_ret(px, legs, h)
    m = mask.reindex(idx, fill_value=False)
    d = idx[m.values].intersection(ret.dropna().index)
    if len(d) == 0:
        return {"label": lbl, "n": 0}, np.array([]), pd.DatetimeIndex([])
    e = declusters(d, gap, idx)
    v = ret.loc[e].values
    r = summarize(v, lbl)
    r["n_days"] = len(d)
    r["base"] = 100 * ret.dropna().mean()
    r["edge_pp"] = r["mean_pct"] - r["base"]
    r["sign_p"] = sign_test(int((v > 0).sum()), len(v))
    return r, v, e


# ------------------------------------------------------------ battery (round 1)
for h in (3, 5):
    battery(px, joint, [("XLF", 1.0)], h, "c5 long XLF, breadth & bear steepener", cost_bps=3.0,
            min_gap=GAP, event_kinds=("nfp",),
            variants={"breadth alone": breadth,
                      "breadth & NOT curve (discarded)": breadth & ~curve,
                      "breadth & spread21 <= 0 (flat/flattening)": breadth & (d_sp[21] <= 0),
                      "breadth & spread21 >= 25 (no bear clause)": breadth & (d_sp[21] >= 0.25)})

# ------------------------------------------------------------ vehicles
print("\n" + "=" * 78, "\nvehicle rows (episodes, gap 10)\n" + "=" * 78)
for h in (3, 5):
    rows = []
    for veh in ("XLF", "KRE", "EWB"):
        rows.append(ep(joint, h, veh, lbl=f"joint {veh}")[0])
        rows.append(ep(breadth, h, veh, lbl=f"breadth alone {veh}")[0])
    r, v, e = ep(joint, h, lbl="joint XLF-SPY", legs=[("XLF", 1.0), ("SPY", -1.0)])
    rows.append(r)
    rows.append(ep(breadth, h, lbl="breadth XLF-SPY", legs=[("XLF", 1.0), ("SPY", -1.0)])[0])
    show(rows, f"h={h}")

# ------------------------------------------------------------ gate attribution
print("\n" + "=" * 78, "\nGATE ATTRIBUTION\n" + "=" * 78)
for h in (3, 5):
    ret = vehicle_ret(px, [("XLF", 1.0)], h)
    p_ep = declusters(idx[breadth.reindex(idx, fill_value=False).values].intersection(ret.dropna().index), GAP, idx)
    c_ep = declusters(idx[joint.values].intersection(ret.dropna().index), GAP, idx)
    out = filter_vs_reanchor(ret, pd.Series(idx.isin(p_ep), index=idx), pd.Series(idx.isin(c_ep), index=idx),
                             idx, window_td=21, label=f"h={h} XLF episode anchors")
    dj = idx[joint.values].intersection(ret.dropna().index)
    dc = idx[(breadth & ~curve).reindex(idx, fill_value=False).values].intersection(ret.dropna().index)
    show([summarize(ret.loc[dj].values, "day-level joint"),
          summarize(ret.loc[dc].values, "day-level breadth & not curve")])

# ------------------------------------------------------------ neighbours
print("\n" + "=" * 78, "\nDEFINITION NEIGHBOURS (XLF, episodes gap 10)\n" + "=" * 78)
for h in (3, 5):
    rows = []
    for nm, dd in (("10y-3m", d_sp), ("10y-5y", d_sp5)):
        for n in (10, 21, 63):
            for bp in (0.15, 0.25, 0.40):
                for bear in (True, False):
                    g = (dd[n] >= bp) & ((d_10[n] > 0) if bear else True)
                    r = ep(breadth & g, h, lbl=f"{nm} {n}d >= {int(bp*100)}bp{' bear' if bear else ''}")[0]
                    rows.append(r)
    show(rows, f"h={h}")
    rows = []
    for thr in (0.6, 0.7, 0.8, 0.9, 1.0):
        b2 = (share >= thr) & (valid_n >= 9)
        rows.append(ep(b2 & curve, h, lbl=f"share>={thr} & curve")[0])
        rows.append(ep(b2, h, lbl=f"share>={thr} alone")[0])
    show(rows, f"h={h} breadth-threshold neighbours")

# ------------------------------------------------------------ regime / era
print("\n" + "=" * 78, "\nREGIME / ERA (XLF, episodes gap 10)\n" + "=" * 78)
for h in (3, 5):
    rows = []
    for lbl, m in [("joint", joint), ("joint midterm", joint & midterm), ("joint non-mid", joint & ~midterm),
                   ("joint SPY>200d", joint & above200), ("joint SPY<200d", joint & ~above200),
                   ("joint midterm & >200d <-- TODAY", joint & midterm & above200),
                   ("joint pre-2018", joint & pd.Series(idx < "2018-01-01", index=idx)),
                   ("joint 2018+", joint & pd.Series(idx >= "2018-01-01", index=idx)),
                   ("joint ex-2008/09", joint & pd.Series(~idx.year.isin([2008, 2009]), index=idx)),
                   ("breadth SPY>200d", breadth & above200), ("breadth midterm & >200d", breadth & midterm & above200)]:
        rows.append(ep(m, h, lbl=lbl)[0])
    show(rows, f"h={h}")
r, v, e = ep(joint, 5, lbl="x")
print("  joint episodes h=5 XLF:", [(str(a.date()), round(100 * b, 2)) for a, b in zip(e, v)])
print("  concentration:", cluster_note(e, v))

# ------------------------------------------------------------ book overlap
print("\n=== book overlap (ledger rows signalled since 2026-09-01 on XLF/KRE/banks) ===")
t = pd.read_parquet(Path(__file__).resolve().parents[3] / "data" / "backtest_trades_full.parquet")
rec = t[t["Ticker"].isin(["XLF", "KRE"] + BANKS) & (pd.to_datetime(t["Signal Date"]) >= "2026-09-01")]
print(rec[["Strategy", "Ticker", "Direction", "Signal Date", "Entry Date", "Exit Date", "Exit Type"]].to_string(index=False)
      if len(rec) else "  (none)")
