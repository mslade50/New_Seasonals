"""C4 — the semis complex into the NVDA print, exiting before it.

Today's geometry: anchor 2026-08-13, NVDA prints 2026-08-26 (9 td ahead).
Entry MOC 2026-08-14 (= anchor + 1), exit MOC 2026-08-25 = pE-1, so h=7.
Same k/h parameterisation as c3: k = td from anchor to print, h = k-2.

Order of operations the brief demands: the ANCHOR cell first (no price
gate), the laggard state second as a GATE-ATTRIBUTION question, and the
placebo anchor ladder because "the run into NVDA" and "the last week of
August/February/May/November" are the same days.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pitch_lab import close_panel, pct_rank, show, sign_test, summarize, cluster_note  # noqa: E402

ASOF = pd.Timestamp("2026-08-13")
K, H = 9, 7

VEH = ["SMH", "QQQ", "SPY", "NVDA", "MU", "INTC", "ADI"]
px = close_panel(VEH)
px = px[px.index <= ASOF]
idx = px.index
pos = pd.Series(range(len(idx)), index=idx)

earn = pd.read_parquet("data/earnings_calendar.parquet", columns=["ticker", "date"])
earn["date"] = pd.to_datetime(earn["date"])
nv = earn[earn["ticker"] == "NVDA"]["date"].sort_values()
nv = nv[(nv >= idx[0]) & (nv <= idx[-1] + pd.Timedelta(days=20))]
pE = np.searchsorted(idx.values, nv.values, side="left")
ok = (pE > 0) & (pE < len(idx))
pE, nvd = pE[ok], nv.values[ok]
print(f"NVDA prints usable: {len(pE)}  ({pd.Timestamp(nvd[0]).date()} .. "
      f"{pd.Timestamp(nvd[-1]).date()})")
print("print months:", pd.Series(pd.DatetimeIndex(nvd).month).value_counts().sort_index().to_dict())


def leg_ret(tkr: str, pe_arr: np.ndarray, k: int, h: int):
    """entry MOC at pE-k+1, exit MOC at pE-k+1+h. Returns (ret, anchor_dates)."""
    c = px[tkr]
    a = pe_arr - k
    e_, x_ = a + 1, a + 1 + h
    m = (a >= 260) & (x_ < len(idx))
    v = c.values[x_[m]] / c.values[e_[m]] - 1.0
    return v, idx[a[m]], m


print("\n" + "=" * 74)
print(f"1. THE ANCHOR CELL, NO PRICE GATE. long, k={K} h={H} (exit = pE-1)")
print("=" * 74)
rows = []
for tk in ["SMH", "QQQ", "SPY", "NVDA", "MU", "INTC", "ADI"]:
    v, d, _ = leg_ret(tk, pE, K, H)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        rows.append({"label": f"{tk} (no data)", "n": 0})
        continue
    r = summarize(v, f"{tk} into the NVDA print")
    # unconditional control, same h
    base = (px[tk].shift(-(1 + H)) / px[tk].shift(-1) - 1.0).dropna()
    r["ctl_all_days_pct"] = round(100 * base.mean(), 3)
    r["edge_pct"] = round(r["mean_pct"] - 100 * base.mean(), 3)
    rows.append(r)
show(rows, "does the complex drift into an NVDA print at all?")

v_smh, d_smh, msk = leg_ret("SMH", pE, K, H)
good = ~np.isnan(v_smh)
v_smh, d_smh = v_smh[good], d_smh[good]
w = int((v_smh > 0).sum())
print(f"  SMH record {w}-{len(v_smh)-w}, sign p = {sign_test(w, len(v_smh)):.4f}")
print(f"  concentration: {cluster_note(d_smh, v_smh)}")

print("\n" + "=" * 74)
print("2. PLACEBO ANCHOR LADDER — shift the print date, keep k and h")
print("=" * 74)
print("  (m<0 = window sits EARLIER, still pre-print. m>0 straddles the real")
print("   print, so those rows carry the event's own variance - noted, not used.)")
lad = []
for m in range(-12, 5):
    v, d, _ = leg_ret("SMH", pE + m, K, H)
    v = v[~np.isnan(v)]
    if len(v) < 5:
        continue
    lad.append(summarize(v, f"shift {m:+d}" + ("  <-- TRUE" if m == 0 else "")))
show(lad, "anchor ladder on SMH")
real = [r for r in lad if r["label"].startswith("shift +0")][0]
pre = [r for r in lad if r["label"].startswith("shift -") or r["label"].startswith("shift +0")]
rank = 1 + sum(1 for r in pre if r["mean_pct"] > real["mean_pct"])
print(f"  TRUE anchor mean {real['mean_pct']:+.3f}% ranks {rank} of {len(pre)} "
      f"PRE-print offsets. SPIKE = event; PLATEAU = calendar position.")

print("\n" + "=" * 74)
print("3. GATE ATTRIBUTION — does the 63d-laggard state condition anything?")
print("=" * 74)
r63 = pct_rank(px["SMH"], 63)
lag_state = r63.reindex(idx).values[(pE - K)[msk]][good]
rows = [summarize(v_smh, f"gate OFF, all prints (N={len(v_smh)})")]
for lo, hi in [(0, 10), (0, 25), (25, 75), (75, 101)]:
    sel = (lag_state >= lo) & (lag_state < hi)
    rows.append(summarize(v_smh[sel], f"SMH rank63 in [{lo},{hi}) (N={int(sel.sum())})"))
show(rows, "SMH rank63 at the anchor (today = 2.4)")
sel_lo = lag_state < 25
if sel_lo.sum() >= 3:
    ww = int((v_smh[sel_lo] > 0).sum())
    print(f"  laggard cell record {ww}-{int(sel_lo.sum())-ww}, sign p = "
          f"{sign_test(ww, int(sel_lo.sum())):.4f}")
print(f"  today's SMH rank63 = {r63.iloc[-1]:.1f}")

print("\n" + "=" * 74)
print("4. CALENDAR CONFOUND — August prints ARE the last week of August")
print("=" * 74)
mth = pd.DatetimeIndex(d_smh).month
rows = [summarize(v_smh[mth == m_], f"print month {m_:02d} (N={int((mth==m_).sum())})")
        for m_ in sorted(set(mth))]
show(rows, "by print month")
# month-matched calendar control: SMH h=7 from the SAME calendar position,
# in years when there is no NVDA print anchor at that spot is impossible
# (NVDA prints every quarter), so use the tdom-of-month control instead.
base = (px["SMH"].shift(-(1 + H)) / px["SMH"].shift(-1) - 1.0)
bidx = base.dropna().index
tdom = pd.Series(pd.Series(bidx, index=bidx).groupby([bidx.year, bidx.month]).cumcount().values + 1,
                 index=bidx)
ent_dates = pd.DatetimeIndex([idx[pos[d] + 1] for d in d_smh])
ent_td = [int(tdom.get(x, -1)) for x in ent_dates]
print("  entry tdom distribution:", pd.Series(ent_td).value_counts().sort_index().to_dict())
rows = []
for lo, hi in [(1, 5), (6, 10), (11, 15), (16, 23)]:
    m_ = (tdom >= lo) & (tdom <= hi)
    rows.append(summarize(base.loc[bidx[m_.values]].values, f"tdom {lo}-{hi}"))
show(rows, "SMH h=7 unconditional by entry tdom")
aug = mth == 8
augm = (pd.DatetimeIndex(d_smh).month == 8)
show([summarize(v_smh[aug], f"August prints (N={int(aug.sum())})"),
      summarize(base.loc[bidx[(bidx.month == 8)]].values, "SMH h=7 ALL August days")],
     "the August cell against unconditional August")

print("\n" + "=" * 74)
print("5. ERA + CYCLE")
print("=" * 74)
dd = pd.DatetimeIndex(d_smh)
show([summarize(v_smh[dd < pd.Timestamp("2013-01-01")], "pre-2013"),
      summarize(v_smh[(dd >= pd.Timestamp("2013-01-01")) & (dd < pd.Timestamp("2020-01-01"))], "2013-2019"),
      summarize(v_smh[dd >= pd.Timestamp("2020-01-01")], "2020+ (NVDA drives the complex)"),
      summarize(v_smh[(dd.year % 4) == 2], "midterm years")], "era / cycle")
print("  episode returns 2020+:",
      {str(x.date()): round(100 * y, 2) for x, y in zip(dd[dd >= "2020-01-01"],
                                                        v_smh[dd >= pd.Timestamp("2020-01-01")])})
print("\n6. cost: SMH 1 leg ~4 bps round trip; cell mean "
      f"{100*v_smh.mean():.3f}% = {10000*v_smh.mean():.1f} bps -> "
      f"{10000*v_smh.mean()/4.0:.1f}x cost (need >=5x)")
