"""C2 round 1 -- long duration (IEF, TLT) across the back-to-back PPI->CPI pair.

Same anchor as C1: signal D = PPI-3, entry MOC D+1 = PPI-2, PPI at entry+2,
CPI at entry+3.

The registry's standing correction is the whole point of this script:
"CPI/PPI/FOMC work on duration" is a TRADING-DAY-OF-MONTH profile, not an
event effect.  Long TLT into CPI reads +0.178% raw and +6.7 bps against a
tdom-matched control (2026-08-10, d5b_tdom_control.py).  So the tdom control
is built FIRST and every number is quoted against it.

Second standing kill to clear: the 2026-08-27 containment work already
measured TLT/IEF over a 10 td hold containing both prints and found them
BELOW their own drift (-0.019 / -0.044pp).  This is the same object at a
nearer anchor and shorter horizons.

Kills:
  0. TLT's own tdom profile -- rebuild it, do not cite it.
  1. tdom-matched excess at every horizon, both vehicles.
  2. gate attribution: pair vs isolated-PPI vs all-PPI vs all-CPI.
  3. placebo anchor ladder.
  4. gap-share: an 08:30 print must be earned in the overnight gap.
  5. era / midterm splits, and where the LIVE state (10y at a 52w high,
     TLT 1.4% off its low) sits inside the cell.
  6. cost: IEF and LQD are thin; the registry's own vehicle work says TLT
     strictly dominates (net 24.4 bps at 10.8x vs IEF 10.0 at 6.0x).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)

TK = ["TLT", "IEF"]
pxd = load_prices(TK)
px = pd.DataFrame({t: pxd[t]["Close"] for t in TK}).dropna()
idx = px.index
pos = pd.Series(range(len(idx)), index=idx)
print(f"common calendar {idx[0].date()} .. {idx[-1].date()}  N={len(idx)}")

ppi = load_events(["ppi"])["date"]
cpi = load_events(["cpi"])["date"]
K_SIG = -3

tdom = pd.Series(pd.Series(idx, index=idx).groupby([idx.year, idx.month]).cumcount().values + 1,
                 index=idx)


def sig_dates(print_dates, k=K_SIG, require_next=None, forbid_next=False):
    nxt = set(pd.DatetimeIndex(require_next)) if require_next is not None else None
    p, _ = anchor_positions(idx, print_dates, 0)
    out = []
    for pp in p:
        if pp + 1 >= len(idx):
            continue
        if nxt is not None:
            has = idx[pp + 1] in nxt
            if forbid_next and has:
                continue
            if not forbid_next and not has:
                continue
        s = pp + k
        if 0 <= s < len(idx):
            out.append(idx[s])
    return pd.DatetimeIndex(sorted(set(out)))


PAIR = sig_dates(ppi, require_next=cpi)
PPI_ISO = sig_dates(ppi, require_next=cpi, forbid_next=True)
PPI_ALL = sig_dates(ppi)
CPI_ALL = sig_dates(cpi)
print(f"PAIR anchors N={len(PAIR)}  {PAIR[0].date()} .. {PAIR[-1].date()}")

# ------------------------------------------------------------ 0. tdom profile
print("\n" + "=" * 100)
print("0. REBUILD TLT's OWN tdom PROFILE (h=3, no events anywhere).")
print("   The registry says this profile IS the 'prints work on duration'")
print("   pattern. Rebuilt here so nothing is borrowed.")
print("=" * 100)
for t in TK:
    r = fwd_lag(px[t], 3, lag=1)
    prof = pd.DataFrame({"d": tdom.values, "r": r.values}, index=idx).dropna()
    g = prof.groupby("d")["r"].agg(["count", "mean"])
    g["mean_pct"] = (100 * g["mean"]).round(3)
    print(f"\n  {t} h=3 by trading-day-of-month:")
    print(g[["count", "mean_pct"]].head(16).to_string())
print("\n  pair-anchor tdom histogram:",
      dict(tdom.reindex(PAIR).value_counts().sort_index()))


def tdom_excess(t, h, dates, by_month=False):
    r = fwd_lag(px[t], h, lag=1)
    ok = r.notna()
    cells = pd.DataFrame({"m": idx.month, "d": tdom.values, "r": r.values},
                         index=idx)[ok.values]
    keys = ["m", "d"] if by_month else ["d"]
    cellmean = cells.groupby(keys)["r"].mean()
    sig = cells.reindex(pd.DatetimeIndex(dates).intersection(cells.index))
    matched = sig.set_index(keys).index.map(cellmean)
    exc = sig["r"].values - np.asarray(matched, float)
    exc = exc[~np.isnan(exc)]
    return sig["r"].values, exc


print("\n" + "=" * 100)
print("1. HORIZON SCAN, pair anchor, RAW vs tdom-matched vs month-x-tdom.")
print("=" * 100)
for t in TK:
    rows = []
    for h in (1, 2, 3, 4, 5, 6, 8, 10):
        raw, exc = tdom_excess(t, h, PAIR)
        _, exc2 = tdom_excess(t, h, PAIR, by_month=True)
        allr = fwd_lag(px[t], h, lag=1).dropna()
        rows.append({
            "h": h, "n": len(raw), "raw_pct": round(100 * raw.mean(), 3),
            "drift_pct": round(100 * allr.mean(), 3),
            "vs_drift_pp": round(100 * (raw.mean() - allr.mean()), 3),
            "tdom_exc_pp": round(100 * exc.mean(), 4),
            "tdom_t": round(exc.mean() / (exc.std(ddof=1) / np.sqrt(len(exc))), 2),
            "tdom_hit": round(100 * (exc > 0).mean(), 1),
            "mxt_exc_pp": round(100 * exc2.mean(), 4),
        })
    show(rows, f"{t} long, pair anchor (entry = PPI-2)")

H = 3
mask = pd.Series(False, index=idx)
mask.loc[PAIR] = True
for t in TK:
    battery(px, mask, [(t, 1.0)], h=H,
            title=f"C2 {t}: long, entry PPI-2, exit CPI close (h=3)",
            cost_bps=3.0, min_gap=5, event_kinds=("fomc_decision",))

# --------------------------------------------------------- gate attribution
print("\n" + "=" * 100)
print("2. GATE ATTRIBUTION, h=3, tdom-matched excess (the only valid basis).")
print("=" * 100)
for t in TK:
    rows = []
    for lbl, dts in (("PAIR (PPI, CPI next)", PAIR), ("PPI, NO CPI next", PPI_ISO),
                     ("ALL PPI anchors", PPI_ALL), ("ALL CPI anchors", CPI_ALL)):
        raw, exc = tdom_excess(t, H, dts)
        rows.append({"label": lbl, "n": len(raw),
                     "raw_pct": round(100 * raw.mean(), 3),
                     "tdom_exc_pp": round(100 * exc.mean(), 4),
                     "hit": round(100 * (exc > 0).mean(), 1),
                     "t": round(exc.mean() / (exc.std(ddof=1) / np.sqrt(len(exc))), 2),
                     "sign_p": round(sign_test(int((exc > 0).sum()), len(exc)), 4)})
    show(rows, f"{t} h={H} gate attribution (tdom-matched)")

# ------------------------------------------------------------ placebo ladder
print("\n" + "=" * 100)
print(f"3. PLACEBO ANCHOR LADDER k=PPI-8..PPI+2, h={H}, tdom-matched excess.")
print("=" * 100)
pair_pos = [pos[d] + 3 for d in PAIR]
for t in TK:
    rows = []
    for k in range(-8, 3):
        d = idx[[p + k for p in pair_pos if 0 <= p + k < len(idx)]]
        raw, exc = tdom_excess(t, H, d)
        if not len(exc):
            continue
        rows.append({"k": k, "n": len(exc), "raw_pct": round(100 * raw.mean(), 3),
                     "tdom_exc_pp": round(100 * exc.mean(), 4),
                     "hit": round(100 * (exc > 0).mean(), 1)})
    d = pd.DataFrame(rows).sort_values("tdom_exc_pp", ascending=False).reset_index(drop=True)
    rank = int(d.index[d["k"] == K_SIG][0]) + 1
    print(f"\n  {t}: TRUE ANCHOR k={K_SIG} RANKS {rank} of {len(d)} on tdom-matched excess")
    print(d.to_string(index=False))

# --------------------------------------------------------------- gap share
print("\n" + "=" * 100)
print("4. GAP-SHARE TEST. Both 08:30 prints sit inside the overnight gap of")
print("   entry+2 (PPI) and entry+3 (CPI). If the hold does not earn there,")
print("   no release mechanism is operating.")
print("=" * 100)
for t in TK:
    s = pxd[t].dropna(subset=["Open", "Close"])
    si = s.index
    sp = pd.Series(range(len(si)), index=si)
    c, o = s["Close"].values, s["Open"].values
    tot_l, gap_l = [], []
    for d in PAIR:
        p = sp.get(d)
        if p is None or p + 1 + H >= len(si):
            continue
        e = p + 1
        tot_l.append(c[e + H] / c[e] - 1.0)
        gap_l.append((o[e + 2] / c[e + 1] - 1.0) + (o[e + 3] / c[e + 2] - 1.0))
    tot_a, gap_a = np.array(tot_l), np.array(gap_l)
    allgap = (s["Open"] / s["Close"].shift(1) - 1.0).dropna()
    print(f"  {t}: N={len(tot_a)} hold {100*tot_a.mean():+.3f}% = "
          f"2 release gaps {100*gap_a.mean():+.4f}% + rest "
          f"{100*(tot_a-gap_a).mean():+.4f}%  | unconditional 2 gaps "
          f"{200*allgap.mean():+.4f}%  | gap excess "
          f"{100*(gap_a.mean()-2*allgap.mean()):+.4f}pp")

# --------------------------------------------------------------- splits
print("\n" + "=" * 100)
print("5. ERA / MIDTERM / SEPTEMBER + THE LIVE STATE BAND.")
print("   Today: ^TNX within 0.25% of a 52w HIGH, TLT 1.44% off its 52w LOW.")
print("   Locate today INSIDE the cell (registry house rule, 2026-09-04).")
print("=" * 100)
for t in TK:
    raw_all, exc_all = tdom_excess(t, H, PAIR)
    d = pd.DatetimeIndex(PAIR).intersection(fwd_lag(px[t], H, 1).dropna().index)
    yrs = d.year
    # distance from the 52w low, own series
    lo = rolling_on_valid(px[t], lambda x: x.rolling(252).min())
    dist_lo = (px[t] / lo - 1.0).reindex(d).values
    rows = []
    for lbl, m in (("pre-2013", yrs < 2013), ("2013+", yrs >= 2013),
                   ("pre-2018", yrs < 2018), ("2018+", yrs >= 2018),
                   ("MIDTERM", (yrs % 4) == 2), ("non-midterm", (yrs % 4) != 2),
                   ("SEPTEMBER", d.month == 9),
                   ("LIVE BAND: <=3% off 52w low", dist_lo <= 0.03),
                   ("  >3% off 52w low", dist_lo > 0.03)):
        sub = exc_all[m]
        if not len(sub):
            continue
        rows.append({"label": lbl, "n": len(sub),
                     "tdom_exc_pp": round(100 * sub.mean(), 4),
                     "hit": round(100 * (sub > 0).mean(), 1),
                     "raw_pct": round(100 * raw_all[m].mean(), 3),
                     "sign_p": round(sign_test(int((sub > 0).sum()), len(sub)), 4)})
    show(rows, f"{t} h={H} splits (tdom-matched excess)")
