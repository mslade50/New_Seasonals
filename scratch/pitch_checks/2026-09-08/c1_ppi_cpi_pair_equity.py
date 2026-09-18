"""C1 round 1 -- long SPY / IWM MOC into a BACK-TO-BACK PPI->CPI pair.

Today's exact geometry: anchor close 2026-09-04, entry MOC 2026-09-08,
PPI 2026-09-10 = entry+2, CPI 2026-09-11 = entry+3.  So the signal date D
sits 3 sessions before the PPI print and the entry (D+1) sits 2 before it.

Pair definition (the object): a PPI print with a CPI on the very NEXT
trading session.  That is the reverse of the registry's 2026-08-10
"CPI-then-PPI" line; the surface map claims the reverse ordering is
unmeasured, and step 0 below checks that claim against the registry number
(-0.071% on N=133) before anything else runs.

Kills it must clear:
  0. registry collision -- is this the 2026-08-10 ordering null, or the
     2026-08-27 containment corpse, wearing a nearer anchor?
  1. beat SPY's own unconditional h drift, the local +/-126td control, and
     an all-days control.  Never zero.
  2. GATE ATTRIBUTION -- does "CPI on the next session" do anything that
     "a PPI anchor" does not?  Run pair / isolated-PPI / all-PPI / all-CPI.
  3. placebo anchor ladder k=-5..+5 on the signal offset.
  4. month x trading-day-of-month control (the registry's standing
     correction; PPI/CPI cluster at a fixed tdom by construction).
  5. era, midterm, September splits.  Today is midterm September.
  6. cost.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)

TK = ["SPY", "IWM"]
pxd = load_prices(TK)
px = pd.DataFrame({t: pxd[t]["Close"] for t in TK}).dropna()
idx = px.index
pos = pd.Series(range(len(idx)), index=idx)
print(f"common calendar {idx[0].date()} .. {idx[-1].date()}  N={len(idx)}")

ppi = load_events(["ppi"])["date"]
cpi = load_events(["cpi"])["date"]
nfp = load_events(["nfp"])["date"]
fomc = load_events(["fomc_decision"])["date"]

# --------------------------------------------------------------- anchors
# signal date = PPI position - 3  (entry = PPI - 2)
K_SIG = -3


def sig_dates(print_dates, k=K_SIG, require_next=None, forbid_next=False):
    """Signal dates k sessions before each print; optionally require the
    NEXT session to carry a print of `require_next` kind."""
    nxt = set(pd.DatetimeIndex(require_next)) if require_next is not None else None
    p, kept = anchor_positions(idx, print_dates, 0)
    out = []
    for i, pp in enumerate(p):
        if pp + 1 >= len(idx):
            continue
        nxt_sess_has = idx[pp + 1] in nxt if nxt is not None else None
        if nxt is not None:
            if forbid_next and nxt_sess_has:
                continue
            if not forbid_next and not nxt_sess_has:
                continue
        s = pp + k
        if 0 <= s < len(idx):
            out.append(idx[s])
    return pd.DatetimeIndex(sorted(set(out)))


PAIR = sig_dates(ppi, require_next=cpi)                       # today's object
PPI_ISO = sig_dates(ppi, require_next=cpi, forbid_next=True)  # PPI, no CPI next
PPI_ALL = sig_dates(ppi)
CPI_ALL = sig_dates(cpi)

print(f"\nPAIR (PPI with CPI the very next session) signal days: N={len(PAIR)}"
      f"  {PAIR[0].date()} .. {PAIR[-1].date()}")
print(f"  all-PPI anchors {len(PPI_ALL)} | PPI-without-CPI-next {len(PPI_ISO)}"
      f" | all-CPI anchors {len(CPI_ALL)}")

print("\n" + "=" * 100)
print("0. REGISTRY COLLISION CHECK -- reproduce the 2026-08-10 ordering line.")
print("   'CPI-then-PPI pair +0.002% N=55; the reverse -0.071% N=133'.")
print("   That line anchors 2 td before the PPI print, h=1 (the print session).")
print("=" * 100)
ppi_after_cpi, ppi_before_cpi = [], []
cpi_pos = {pos.get(d) for d in pd.DatetimeIndex(cpi) if pos.get(d) is not None}
p_all, kept = anchor_positions(idx, ppi, 0)
for pp in p_all:
    if pp - 1 in cpi_pos:
        ppi_after_cpi.append(pp)
    if pp + 1 in cpi_pos:
        ppi_before_cpi.append(pp)
r1 = fwd_lag(px["SPY"], 1, lag=1)
for lbl, plist in (("CPI-then-PPI (gap +1)", ppi_after_cpi),
                   ("PPI-then-CPI (gap -1, TODAY)", ppi_before_cpi)):
    d = idx[[p - 2 for p in plist if p - 2 >= 0]]
    v = r1.reindex(d).dropna().values
    print(f"  {lbl:32s} anchor=print-2, h=1: N={len(v)}  "
          f"mean {100*v.mean():+.4f}%  hit {100*(v>0).mean():.1f}%")
print("  -> if the second line reproduces -0.071% on ~133, the surface map's"
      "\n     novelty claim is FALSE and today's ordering is registry-measured.")

# --------------------------------------------------------------- horizons
print("\n" + "=" * 100)
print("1. HORIZON SCAN on the pair anchor, episode level, both vehicles.")
print("   h=2 exits on the PPI close, h=3 on the CPI close.")
print("=" * 100)
for t in TK:
    rows = horizon_scan(px, PAIR, [(t, 1.0)], hs=(1, 2, 3, 4, 5, 6, 8, 10),
                        lag=1, min_gap=5)
    show(rows, f"{t} long, pair anchor (entry = PPI-2)")

# ------------------------------------------------------------- the battery
H = 3   # pre-specified: hold across BOTH prints, exit the CPI close
mask = pd.Series(False, index=idx)
mask.loc[PAIR] = True
for t in TK:
    battery(px, mask, [(t, 1.0)], h=H,
            title=f"C1 {t}: long, entry PPI-2, exit CPI close (h=3)",
            cost_bps=2.0, min_gap=5, event_kinds=("fomc_decision",))

# ------------------------------------------------------- gate attribution
print("\n" + "=" * 100)
print("2. GATE ATTRIBUTION -- does 'CPI on the next session' filter anything?")
print("=" * 100)
for t in TK:
    ret = fwd_lag(px[t], H, lag=1)
    base = ret.dropna()
    rows = []
    for lbl, dts in (("PAIR (PPI, CPI next)", PAIR),
                     ("PPI, NO CPI next", PPI_ISO),
                     ("ALL PPI anchors", PPI_ALL),
                     ("ALL CPI anchors", CPI_ALL)):
        d = declusters(pd.DatetimeIndex(dts).intersection(base.index), 5, base.index)
        v = ret.loc[d].values
        r = summarize(v, lbl)
        r["excess_pp"] = round(r["mean_pct"] - 100 * base.mean(), 3)
        r["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
        rows.append(r)
    rows.append(summarize(base.values, "ALL DAYS (drift)"))
    show(rows, f"{t} h={H}: gate attribution")

# --------------------------------------------------------- placebo ladder
print("\n" + "=" * 100)
print(f"3. PLACEBO ANCHOR LADDER, k = PPI-8 .. PPI+2, h={H}, pair anchors only.")
print("   The true anchor is k=-3 (entry PPI-2). A plateau kills.")
print("=" * 100)
pair_pos = [pos[d] + 3 for d in PAIR]   # back to the PPI position
for t in TK:
    ret = fwd_lag(px[t], H, lag=1)
    rows = []
    for k in range(-8, 3):
        d = idx[[p + k for p in pair_pos if 0 <= p + k < len(idx)]]
        v = ret.reindex(d).dropna().values
        if not len(v):
            continue
        rows.append({"k": k, "n": len(v), "mean_pct": round(100 * v.mean(), 3),
                     "hit": round(100 * (v > 0).mean(), 1),
                     "t": round(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), 2)})
    d = pd.DataFrame(rows).sort_values("mean_pct", ascending=False).reset_index(drop=True)
    rank = int(d.index[d["k"] == K_SIG][0]) + 1
    print(f"\n  {t}: TRUE ANCHOR k={K_SIG} RANKS {rank} of {len(d)}")
    print(d.to_string(index=False))

# -------------------------------------------------------- tdom control
print("\n" + "=" * 100)
print("4. MONTH x TRADING-DAY-OF-MONTH matched control.")
print("   Prints land at a fixed tdom by construction; the all-days control")
print("   flatters them. Registry standing correction, 2026-08-10.")
print("=" * 100)
tdom = pd.Series(idx, index=idx).groupby([idx.year, idx.month]).cumcount() + 1
tdom = pd.Series(tdom.values, index=idx)
print("  pair-anchor tdom distribution:",
      dict(tdom.reindex(PAIR).value_counts().sort_index().head(20)))
for t in TK:
    ret = fwd_lag(px[t], H, lag=1)
    ok = ret.notna()
    cells = pd.DataFrame({"m": idx.month, "d": tdom.values, "r": ret.values},
                         index=idx)[ok.values]
    cellmean = cells.groupby(["m", "d"])["r"].mean()
    sig = cells.reindex(pd.DatetimeIndex(PAIR).intersection(cells.index))
    matched = sig.set_index(["m", "d"]).index.map(cellmean)
    exc = sig["r"].values - np.asarray(matched, float)
    exc = exc[~np.isnan(exc)]
    print(f"  {t}: raw {100*sig['r'].mean():+.3f}%   month-x-tdom matched excess "
          f"{100*exc.mean():+.4f}pp  (N={len(exc)}, hit {100*(exc>0).mean():.1f}%, "
          f"t {exc.mean()/(exc.std(ddof=1)/np.sqrt(len(exc))):+.2f})")

# ----------------------------------------------------------------- splits
print("\n" + "=" * 100)
print("5. ERA / MIDTERM / SEPTEMBER splits, pair anchors, h=3")
print("=" * 100)
for t in TK:
    ret = fwd_lag(px[t], H, lag=1)
    base = ret.dropna()
    d = pd.DatetimeIndex(PAIR).intersection(base.index)
    v = ret.loc[d].values
    yrs = d.year
    rows = []
    for lbl, m in (("pre-2013", yrs < 2013), ("2013+", yrs >= 2013),
                   ("pre-2018", yrs < 2018), ("2018+", yrs >= 2018),
                   ("MIDTERM (y%4==2)", (yrs % 4) == 2),
                   ("non-midterm", (yrs % 4) != 2),
                   ("SEPTEMBER", d.month == 9),
                   ("MIDTERM x SEPT", ((yrs % 4) == 2) & (d.month == 9))):
        sub = v[m]
        if not len(sub):
            continue
        r = summarize(sub, lbl)
        r["excess_pp"] = round(r["mean_pct"] - 100 * base.mean(), 3)
        r["sign_p"] = round(sign_test(int((sub > 0).sum()), len(sub)), 4)
        rows.append(r)
    show(rows, f"{t} h={H} splits (excess vs all-day drift)")

print("\n" + "=" * 100)
print("6. FOMC-inside-the-window contamination: the registry's Sept-quad kill")
print("   says a September run-in with an FOMC in it is the FOMC cell.")
print("=" * 100)
for t in TK:
    ret = fwd_lag(px[t], H, lag=1)
    d = pd.DatetimeIndex(PAIR).intersection(ret.dropna().index)
    fl = event_in_window(d, idx, H, 1, ("fomc_decision",))
    show([summarize(ret.loc[d].values[fl], f"FOMC in hold (N={int(fl.sum())})"),
          summarize(ret.loc[d].values[~fl], f"no FOMC (N={int((~fl).sum())})")],
         f"{t} h={H}")
