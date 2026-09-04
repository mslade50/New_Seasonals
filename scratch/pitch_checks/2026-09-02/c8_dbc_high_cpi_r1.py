"""C8 round 1 -- broad commodities (DBC) at a fresh 252-day high with a CPI
print inside the hold.

Two questions, in order:
  A. is the PARENT alive?  long DBC at a fresh 252d high, h=6, lag=1, against
     DBC's own drift (which is where the K-1 / roll drag gets charged).
  B. does the CPI-in-hold conditioner do anything in EITHER direction, and is
     it a gate that gates (attribution) or just a sample splitter?
Plus: vehicle alternatives in master_prices, placebo anchor ladder on the CPI
leg, reference class across commodity vehicles, concentration and era.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa: F401,F403,E402
from pitch_lab import (close_panel, load_prices, fwd_lag, declusters, summarize,
                       sign_test, cluster_note, battery, rolling_on_valid,
                       event_in_window, load_events, show, horizon_scan,
                       episode_paths, anchor_positions)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)

ROOT = Path(__file__).resolve().parents[3]
TK = ["DBC", "USO", "GLD", "SLV", "UNG", "SPY", "XLE"]
px = close_panel(TK)

hi252 = rolling_on_valid(px["DBC"], lambda x: x.rolling(252).max())
off_high = px["DBC"] / hi252 - 1.0
at_high = off_high >= -0.0005      # within 5 bp == "at" a fresh high
print("=" * 110)
print("LIVE: DBC close %.4f  252d max %.4f  off-high %+.4f%%  -> at_high=%s"
      % (px["DBC"].iloc[-1], hi252.iloc[-1], 100 * off_high.iloc[-1], bool(at_high.iloc[-1])))
print("  DBC bar span %s .. %s" % (px["DBC"].dropna().index[0].date(),
                                   px["DBC"].dropna().index[-1].date()))
print("=" * 110)

H = 6

# ---------------------------------------------------------------------------
# A. IS THE PARENT ALIVE?
# ---------------------------------------------------------------------------
variants = {
    "at high (<=5bp)": at_high,
    "within 0.5%": off_high >= -0.005,
    "within 1%": off_high >= -0.01,
    "within 2%": off_high >= -0.02,
    "fresh high AND not at one yesterday": at_high & ~at_high.shift(1).fillna(False),
}
battery(px, at_high, [("DBC", 1.0)], h=H,
        title="C8 PARENT: long DBC at a fresh 252d high, h=6",
        cost_bps=6.0, variants=variants, min_gap=6, event_kinds=("cpi",))

# ---------------------------------------------------------------------------
# B. THE CPI CONDITIONER -- gate attribution
# ---------------------------------------------------------------------------
print("\n" + "=" * 110)
print("B. CPI-IN-HOLD GATE ATTRIBUTION (h=6, lag=1)")
print("=" * 110)
s = px["DBC"].dropna()
f = fwd_lag(s, H, lag=1)
own = f.dropna().mean()
mm = at_high.reindex(s.index).fillna(False)
epi = declusters(s.index[mm.values], 6, s.index)
v = f.reindex(epi).dropna()
epi = v.index
cpi_in = event_in_window(epi, s.index, H, 1, ("cpi",))
print(f"  DBC own h=6 drift (all days) {100*own:+.4f}%   N={f.dropna().shape[0]}")
rows = []
for tag, sel in (("parent (all at-high episodes)", np.ones(len(epi), bool)),
                 ("CPI IN hold", cpi_in), ("CPI OUT", ~cpi_in)):
    st = summarize(v.values[sel], tag)
    st["excess_pp"] = round(st["mean_pct"] - 100 * own, 3)
    st["signp"] = round(sign_test(int((v.values[sel] > 0).sum()), int(sel.sum())), 4)
    rows.append(st)
show(rows)
a, b = v.values[cpi_in], v.values[~cpi_in]
se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
print(f"  welch t (IN - OUT) = {(a.mean()-b.mean())/se:+.2f}   "
      f"gate moves the cell {100*(a.mean()-v.values.mean()):+.3f}pp for "
      f"{len(epi)-int(cpi_in.sum())} discarded episodes of {len(epi)}")

# CPI main effect on DBC (is any split just the CPI main effect?)
ev = load_events(["cpi"])["date"]
pos_all, kept = anchor_positions(s.index, ev, 0)
allcpi = s.index[pos_all]
mask_cpi_any = pd.Series(False, index=s.index)
cpi_in_all = event_in_window(s.index, s.index, H, 1, ("cpi",))
rows = [summarize(f.reindex(s.index[cpi_in_all]).dropna().values, "ALL days, CPI in the h=6 hold"),
        summarize(f.reindex(s.index[~cpi_in_all]).dropna().values, "ALL days, no CPI in hold")]
show(rows, "CPI main effect on DBC (unconditional)")

# ---------------------------------------------------------------------------
# C. PLACEBO ANCHOR LADDER -- slide the "at a high" anchor around the CPI
# ---------------------------------------------------------------------------
print("\n" + "=" * 110)
print("C. PLACEBO ANCHOR LADDER -- at-high episodes k sessions from the nearest CPI")
print("=" * 110)
cpi_dates = load_events(["cpi"])["date"]
pos = pd.Series(range(len(s.index)), index=s.index)
cpi_pos = np.array(sorted({int(pos.get(d)) for d in cpi_dates
                           if d in pos.index} |
                          {int(s.index.searchsorted(d)) for d in cpi_dates
                           if s.index[0] <= d <= s.index[-1]}))
epi_pos = np.array([int(pos.get(d)) for d in epi])
nearest = np.array([cpi_pos[np.argmin(np.abs(cpi_pos - p))] - p for p in epi_pos])
rows = []
for k in range(-8, 9):
    sel = nearest == k
    if sel.sum() < 3:
        continue
    st = summarize(v.values[sel], f"k={k:+d} (CPI is {k:+d} td from anchor)")
    st["excess_pp"] = round(st["mean_pct"] - 100 * own, 3)
    rows.append(st)
show(rows)
print("  (the live configuration is a CPI +6 td from the anchor, i.e. the last")
print("   session of a 6-session hold entered at lag=1 -> k=+6)")

# ---------------------------------------------------------------------------
# D. VEHICLE ALTERNATIVES + REFERENCE CLASS
# ---------------------------------------------------------------------------
print("\n" + "=" * 110)
print("D. REFERENCE CLASS -- the identical rule (own fresh 252d high) per vehicle")
print("=" * 110)
rows = []
for t in ("DBC", "USO", "GLD", "SLV", "UNG", "XLE", "SPY"):
    ss = px[t].dropna()
    h_ = rolling_on_valid(px[t], lambda x: x.rolling(252).max())
    ah = (px[t] / h_ - 1.0) >= -0.0005
    ff = fwd_lag(ss, H, lag=1)
    m2 = ah.reindex(ss.index).fillna(False)
    e = declusters(ss.index[m2.values], 6, ss.index)
    vv = ff.reindex(e).dropna()
    if len(vv) < 5:
        continue
    ci = event_in_window(vv.index, ss.index, H, 1, ("cpi",))
    st = summarize(vv.values, t)
    rows.append({"vehicle": t, "n": st["n"], "mean_pct": round(st["mean_pct"], 3),
                 "own_drift": round(100 * ff.dropna().mean(), 3),
                 "excess_pp": round(st["mean_pct"] - 100 * ff.dropna().mean(), 3),
                 "hit": round(st["hit"], 1),
                 "cpi_in_mean": round(100 * vv.values[ci].mean(), 3) if ci.sum() else np.nan,
                 "cpi_out_mean": round(100 * vv.values[~ci].mean(), 3) if (~ci).sum() else np.nan,
                 "n_cpi_in": int(ci.sum())})
print(pd.DataFrame(rows).to_string(index=False))

# ---------------------------------------------------------------------------
# E. CONCENTRATION / ERA / HORIZON
# ---------------------------------------------------------------------------
print("\n" + "=" * 110)
print("E. CONCENTRATION, ERA, HORIZON")
print("=" * 110)
print("  parent: " + cluster_note(epi, v.values, k=2))
print("  CPI-in: " + cluster_note(epi[cpi_in], v.values[cpi_in], k=2))
yr = pd.Series(v.values, index=epi).groupby(epi.year).agg(["count", "mean"])
yr["mean_pct"] = (100 * yr["mean"]).round(3)
print(yr[["count", "mean_pct"]].to_string())
show(horizon_scan(px, epi, [("DBC", 1.0)], hs=(1, 2, 3, 5, 6, 10), lag=1, min_gap=6),
     "parent horizon scan")
show(horizon_scan(px, epi[cpi_in], [("DBC", 1.0)], hs=(1, 2, 3, 5, 6, 10), lag=1, min_gap=6),
     "CPI-in-hold horizon scan")

# fragility dial coverage
frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
ma10 = frag["63d"].rolling(10).mean()
rd = ma10.reindex(epi[cpi_in])
print(f"\n  fragility dial on CPI-in episodes: {int(rd.notna().sum())} of {int(cpi_in.sum())} covered, "
      f"max {rd.max():.1f} against today's {ma10.iloc[-1]:.1f}")
