"""E2 round 1 -- the OVERNIGHT component of a scheduled macro release.

PRE-SPECIFIED BEFORE MEASURING:

  H1  US macro releases print at 08:30 ET, BEFORE the cash open.  The entire
      market reaction to a scheduled print is resolved in the OPENING AUCTION,
      not during the cash session.  A close-to-close return around a print
      therefore blends two structurally different objects: an OVERNIGHT
      segment (prior close -> open) that CONTAINS the event, and an INTRADAY
      segment (open -> close) that contains the fade or the follow-through.
  H2  Two close-to-close cells died this morning as nulls (SPY on PPI day
      -0.009% / 317 episodes vs +0.039% drift; SPY NFP-close-into-CPI
      +0.129% vs a +0.221% all-days control).  A NULL TOTAL DOES NOT IMPLY
      TWO NULL COMPONENTS.  That is the hypothesis and it is pre-specified.
  H3  Tradeable form: MOC on the session BEFORE the print, MOO on the print
      session.  A pure overnight hold, time_td=1 with a MOO exit.  Measured
      exactly and only that.
  H4  THE PLACEBO IS THE BEST TEST AVAILABLE.  FOMC decisions print at
      14:00 ET, INSIDE the cash session.  If the 08:30 releases and the 14:00
      release show the SAME overnight pattern, the mechanism is not what it
      claims to be.  Run and reported in section 3.
  H5  THE CONTROL IS THE INSTRUMENT'S OWN UNCONDITIONAL OVERNIGHT DRIFT, NOT
      ZERO.  US equity overnight drift is historically positive and intraday
      drift near zero.  An "overnight premium into CPI" that merely
      reproduces SPY's unconditional overnight drift is a filter that does
      not filter.  This is the single most likely way this dies and it is
      tested FIRST, in section 1.
  H6  BASIS CAVEAT.  master_prices stores ADJUSTED OHLCV.  On an ex-dividend
      session the prior close has been scaled by the dividend factor and the
      open has not, so the OVERNIGHT segment mechanically absorbs the entire
      dividend.  Section 0 quantifies where those sessions sit so it is
      visible whether a print cell is sitting on one.
  H7  Instruments: SPY, TLT, GLD, GDX.  Events: cpi, ppi, nfp,
      fomc_decision.  Fixed in advance; no instrument is added after looking.
  H8  A one-session overnight hold cannot overlap another, so day-level IS
      episode-level for this object and no declustering is required.  Stated
      so the absence of a decluster step is not mistaken for an omission.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

TKRS = ["SPY", "TLT", "GLD", "GDX"]
COST = {"SPY": 1.0, "TLT": 2.5, "GLD": 2.0, "GDX": 5.0}   # bps round trip
KINDS = ["cpi", "ppi", "nfp", "fomc_decision"]

raw = load_prices(TKRS)
ev = load_events()

D = {}
for t in TKRS:
    g = raw[t]
    o, c = g["Open"].values.astype(float), g["Close"].values.astype(float)
    n = len(c)
    on = np.full(n, np.nan)      # prior close -> open   (CONTAINS the 08:30 print)
    on[1:] = o[1:] / c[:-1] - 1.0
    idr = c / o - 1.0            # open -> close
    c2c = np.full(n, np.nan)
    c2c[1:] = c[1:] / c[:-1] - 1.0
    idx = g.index
    ym = pd.Series(idx.year * 100 + idx.month, index=idx)
    D[t] = {"idx": idx, "on": on, "id": idr, "c2c": c2c,
            "tdom": ym.groupby(ym.values).cumcount().values + 1,
            "yr": idx.year.values}

print("=" * 100)
print("0. DATA AUDIT")
print("=" * 100)
print(f"  columns: {list(raw['SPY'].columns)}")
for t in TKRS:
    print(f"  {t}: {D[t]['idx'][0].date()} .. {D[t]['idx'][-1].date()}  "
          f"N={len(D[t]['idx'])}")

print("\n  H6 ex-dividend diagnostic -- mean OVERNIGHT (bps) by trading-day-of-month.")
print("  A dividend absorbed by the overnight segment shows as a spike at the "
      "ticker's ex-div tdom (TLT ~tdom 1, SPY quarterly opex).")
rows = []
for j in range(1, 22):
    r = {"tdom": j}
    for t in TKRS:
        m = (D[t]["tdom"] == j) & ~np.isnan(D[t]["on"])
        r[t] = round(1e4 * D[t]["on"][m].mean(), 2) if m.sum() > 20 else np.nan
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))


def pos_of(t, kind):
    """Session positions of an event, mapped forward when the date is not a
    session.  Requires a prior session (for the overnight leg)."""
    idx = D[t]["idx"]
    out, exact, fwd_map = [], 0, 0
    for x in ev[ev.event == kind]["date"]:
        p = int(idx.searchsorted(x, "left"))
        if p <= 0 or p >= len(idx):
            continue
        if idx[p].normalize() == pd.Timestamp(x).normalize():
            exact += 1
        else:
            fwd_map += 1
        out.append(p)
    return np.array(sorted(set(out))), exact, fwd_map


def rep(v, lbl, extra=None):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return {"cell": lbl, "N": len(v)}
    w = int((v > 0).sum())
    d = {"cell": lbl, "N": len(v), "bps": round(1e4 * v.mean(), 2),
         "hit": round(100 * w / len(v), 1),
         "t": round(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), 2),
         "sign_p": round(sign_test(w, len(v)), 4),
         "sd_bps": round(1e4 * v.std(ddof=1), 1)}
    if extra:
        d.update(extra)
    return d


print("\n" + "=" * 100)
print("1. H5 FIRST -- THE UNCONDITIONAL DECOMPOSITION.  This is the control.")
print("   If the print-day overnight just reproduces this, the filter does "
      "not filter.")
print("=" * 100)
uncond = {}
for t in TKRS:
    rows = []
    for seg in ("on", "id", "c2c"):
        v = D[t][seg]
        rows.append(rep(v, f"{t} {seg} all days"))
        uncond[(t, seg)] = np.nanmean(v)
    # era
    for lbl, m in [("pre-2008", D[t]["yr"] < 2008),
                   ("2008-2019", (D[t]["yr"] >= 2008) & (D[t]["yr"] < 2020)),
                   ("2020+", D[t]["yr"] >= 2020)]:
        for seg in ("on", "id"):
            rows.append(rep(D[t][seg][m], f"{t} {seg} {lbl}"))
    print(pd.DataFrame(rows).to_string(index=False))
    print()

print("=" * 100)
print("2. THE PRE-SPECIFIED CELL -- print-session OVERNIGHT "
      "(MOC prior close -> MOO print open)")
print("   EXCESS is over the instrument's OWN unconditional overnight drift, "
      "full history.")
print("=" * 100)
best = []
for kind in KINDS:
    rows = []
    for t in TKRS:
        p, exact, fmap = pos_of(t, kind)
        p = p[~np.isnan(D[t]["on"][p])]
        if len(p) < 5:
            continue
        v = D[t]["on"][p]
        base = uncond[(t, "on")]
        w = int((v > 0).sum())
        # excess series (per-event) for a sign test on the EXCESS
        exc = v - base
        rows.append({"tkr": t, "N": len(v),
                     "on_bps": round(1e4 * v.mean(), 2),
                     "uncond_on_bps": round(1e4 * base, 2),
                     "EXCESS_bps": round(1e4 * (v.mean() - base), 2),
                     "hit": round(100 * w / len(v), 1),
                     "t_vs_0": round(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), 2),
                     "t_vs_ctrl": round((v.mean() - base) /
                                        (v.std(ddof=1) / np.sqrt(len(v))), 2),
                     "sign_p": round(sign_test(w, len(v)), 4),
                     "sd_bps": round(1e4 * v.std(ddof=1), 1),
                     "worst_pct": round(100 * v.min(), 2),
                     "xcost": round(1e4 * (v.mean() - base) / COST[t], 1),
                     "exact_map": exact, "fwd_map": fmap})
        best.append((kind, t, 1e4 * (v.mean() - base), len(v)))
    print(f"\n  --- {kind} --- overnight leg")
    print(pd.DataFrame(rows).to_string(index=False))

print("\n" + "=" * 100)
print("2b. THE OTHER HALF -- print-session INTRADAY (open -> close), same "
      "events.  EXCESS over own unconditional intraday.")
print("=" * 100)
for kind in KINDS:
    rows = []
    for t in TKRS:
        p, _, _ = pos_of(t, kind)
        v = D[t]["id"][p]
        v = v[~np.isnan(v)]
        if len(v) < 5:
            continue
        base = uncond[(t, "id")]
        w = int((v > 0).sum())
        rows.append({"tkr": t, "N": len(v), "id_bps": round(1e4 * v.mean(), 2),
                     "uncond_id_bps": round(1e4 * base, 2),
                     "EXCESS_bps": round(1e4 * (v.mean() - base), 2),
                     "hit": round(100 * w / len(v), 1),
                     "t_vs_ctrl": round((v.mean() - base) /
                                        (v.std(ddof=1) / np.sqrt(len(v))), 2),
                     "sd_bps": round(1e4 * v.std(ddof=1), 1)})
    print(f"\n  --- {kind} --- intraday leg")
    print(pd.DataFrame(rows).to_string(index=False))

print("\n" + "=" * 100)
print("3. H4 THE PLACEBO -- 08:30 releases vs the 14:00 FOMC decision.")
print("   Mechanism check on DISPERSION: the event's content must show up in "
      "the segment that contains it.")
print("   sd_ratio = segment sd on event days / segment sd on all days.")
print("=" * 100)
rows = []
for t in TKRS:
    sd_on_all = np.nanstd(D[t]["on"], ddof=1)
    sd_id_all = np.nanstd(D[t]["id"], ddof=1)
    for kind in KINDS:
        p, _, _ = pos_of(t, kind)
        von = D[t]["on"][p]
        vid = D[t]["id"][p]
        von, vid = von[~np.isnan(von)], vid[~np.isnan(vid)]
        if len(von) < 5:
            continue
        rows.append({"tkr": t, "event": kind,
                     "release": "14:00 ET" if kind == "fomc_decision" else "08:30 ET",
                     "N": len(von),
                     "sd_ON_ratio": round(von.std(ddof=1) / sd_on_all, 2),
                     "sd_ID_ratio": round(vid.std(ddof=1) / sd_id_all, 2),
                     "ON_bps": round(1e4 * von.mean(), 1),
                     "ID_bps": round(1e4 * vid.mean(), 1)})
print(pd.DataFrame(rows).to_string(index=False))
print("\n  MECHANISM CONFIRMED only if sd_ON_ratio > 1 for the 08:30 events "
      "and ~1 for FOMC,")
print("  while sd_ID_ratio is elevated for FOMC.  Elevated dispersion with a "
      "null mean is NOT a trade;")
print("  it is confirmation the decomposition is measuring what it claims.")

print("\n" + "=" * 100)
print("4. TDOM-MATCHED CONTROL on the overnight cells "
      "(the control that killed TLT-into-CPI this morning)")
print("=" * 100)
rows = []
for kind in KINDS:
    for t in TKRS:
        p, _, _ = pos_of(t, kind)
        p = p[~np.isnan(D[t]["on"][p])]
        if len(p) < 5:
            continue
        v = D[t]["on"][p]
        tset = set(int(x) for x in D[t]["tdom"][p])
        evpos = set(p.tolist())
        m = (np.isin(D[t]["tdom"], list(tset)) & ~np.isnan(D[t]["on"])
             & ~np.isin(np.arange(len(D[t]["on"])), list(evpos)))
        ctl = D[t]["on"][m]
        bucket = {j: D[t]["on"][(D[t]["tdom"] == j) & ~np.isnan(D[t]["on"])
                                & ~np.isin(np.arange(len(D[t]["on"])), list(evpos))].mean()
                  for j in tset}
        exc = np.array([D[t]["on"][q] - bucket[int(D[t]["tdom"][q])] for q in p])
        w = int((exc > 0).sum())
        rows.append({"event": kind, "tkr": t, "N": len(v),
                     "cell_bps": round(1e4 * v.mean(), 2),
                     "tdom_ctrl_bps": round(1e4 * ctl.mean(), 2),
                     "EXCESS_bps": round(1e4 * exc.mean(), 2),
                     "hit": round(100 * w / len(exc), 1),
                     "sign_p": round(sign_test(w, len(exc)), 4),
                     "xcost": round(1e4 * exc.mean() / COST[t], 1)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n" + "=" * 100)
print("5. DECOMPOSE THE TWO CELLS THAT DIED THIS MORNING AS NULL TOTALS")
print("=" * 100)
# 5a. SPY on PPI day, c2c = on x id
t = "SPY"
p, _, _ = pos_of(t, "ppi")
p = p[~np.isnan(D[t]["c2c"][p])]
print(f"\n  5a. SPY PPI-day close-to-close, decomposed  (N={len(p)})")
print(pd.DataFrame([
    rep(D[t]["c2c"][p], "PPI day c2c"),
    rep(D[t]["on"][p], "  of which OVERNIGHT"),
    rep(D[t]["id"][p], "  of which INTRADAY"),
    rep(D[t]["c2c"], "CTRL c2c all days"),
    rep(D[t]["on"], "CTRL overnight all days"),
    rep(D[t]["id"], "CTRL intraday all days"),
]).to_string(index=False))

# 5b. SPY from the NFP close into the CPI close -- cumulative on vs id
print("\n  5b. SPY, NFP close -> CPI close window, cumulative OVERNIGHT vs "
      "cumulative INTRADAY")
idx = D[t]["idx"]
nfp = pd.DatetimeIndex(ev[ev.event == "nfp"]["date"])
cpi = pd.DatetimeIndex(ev[ev.event == "cpi"]["date"])
rows_on, rows_id, rows_tot, nsess = [], [], [], []
for d0 in nfp:
    nxt = cpi[cpi > d0]
    if len(nxt) == 0:
        continue
    d1 = nxt[0]
    a = int(idx.searchsorted(d0, "left"))
    b = int(idx.searchsorted(d1, "left"))
    if a <= 0 or b >= len(idx) or b <= a or (b - a) > 12:
        continue
    seg = slice(a + 1, b + 1)
    von, vid = D[t]["on"][seg], D[t]["id"][seg]
    if np.isnan(von).any() or np.isnan(vid).any():
        continue
    rows_on.append(np.prod(1 + von) - 1)
    rows_id.append(np.prod(1 + vid) - 1)
    rows_tot.append(D[t]["c2c"][seg])
    nsess.append(b - a)
tot = np.array([np.prod(1 + x) - 1 for x in rows_tot])
print(f"      windows N={len(tot)}, median {int(np.median(nsess))} sessions")
print(pd.DataFrame([
    rep(tot, "NFP->CPI total c2c"),
    rep(np.array(rows_on), "  cumulative OVERNIGHT"),
    rep(np.array(rows_id), "  cumulative INTRADAY"),
]).to_string(index=False))
k = int(np.median(nsess))
print(f"      control: {k} sessions of unconditional overnight = "
      f"{1e4*k*uncond[('SPY','on')]:.1f} bps; intraday = "
      f"{1e4*k*uncond[('SPY','id')]:.1f} bps")

print("\n" + "=" * 100)
print("6. ERA / MIDTERM / CONCENTRATION on the overnight cells")
print("=" * 100)
for kind in KINDS:
    for t in TKRS:
        p, _, _ = pos_of(t, kind)
        p = p[~np.isnan(D[t]["on"][p])]
        if len(p) < 20:
            continue
        v = D[t]["on"][p]
        y = D[t]["yr"][p]
        base = uncond[(t, "on")]
        segs = []
        for lbl, m in [("pre-2008", y < 2008), ("2008-19", (y >= 2008) & (y < 2020)),
                       ("2020+", y >= 2020), ("pre-2018", y < 2018),
                       ("2018+", y >= 2018), ("midterm", (y % 4) == 2),
                       ("non-mid", (y % 4) != 2)]:
            segs.append(f"{lbl} {1e4*(v[m].mean()-base):+.1f}({int(m.sum())})"
                        if m.sum() > 3 else f"{lbl} n/a")
        print(f"  {kind:14s} {t:4s} EXCESS bps by era: " + "  ".join(segs))
    print()

print("=" * 100)
print("7. TODAY'S EXECUTABLE VERSION")
print("=" * 100)
print("  A pitch published the morning of 2026-08-10 can place MOC tonight "
      "(2026-08-10)")
print("  and exit MOO 2026-08-11.  The CPI print is 2026-08-12 and PPI is "
      "2026-08-13, so the")
print("  overnight session this pitch can actually buy is 08-10 -> 08-11, "
      "which contains NO print.")
print("  The CPI overnight is MOC 2026-08-11 -> MOO 2026-08-12: TOMORROW's "
      "trade, not today's.")
for t in TKRS:
    p, _, _ = pos_of(t, "cpi")
    pm1 = p - 1
    pm1 = pm1[pm1 > 0]
    v = D[t]["on"][pm1]
    v = v[~np.isnan(v)]
    base = uncond[(t, "on")]
    w = int((v > 0).sum())
    print(f"  {t}: the PRINT-MINUS-1 overnight (what today CAN buy) N={len(v)} "
          f"{1e4*v.mean():+.2f} bps, excess {1e4*(v.mean()-base):+.2f} bps, "
          f"hit {100*w/len(v):.1f}%, sign p {sign_test(w, len(v)):.4f}")
