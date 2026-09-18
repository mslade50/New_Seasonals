"""A8 round 1 - the industrial-complex washout COUNT (energy thrust-count
object mirrored onto the downside).

Pre-specified cell: 11-name industrial complex MMM ITW GD PH DOV NSC EMR HON
CAT UNP ROK; count of members at pitch_lab.zscore(close,10) <= -2.0; long XLI,
lag=1 MOC entry.

CONVENTION STATED: z10 is `pitch_lab.zscore` (10-day return standardised by its
own trailing-252 mean and sd), NOT build_pitch_state._metrics_for. The two are
different objects (CLAUDE.md records it; watchlist 19's parked numbers were
produced under pitch_lab, so pitch_lab binds).

The checks the registry says this family owes, all in round 1 because each can
kill on its own:
  (a) today's count under the binding convention
  (b) full dose response 1..8+ and monotonicity
  (c) PC1 share / participation ratio / effective N
  (d) P(XLI itself triggering | count >= k) -- set membership BEFORE effect
  (e) the count against plain XLI z10 <= -2
  (f) reference class: the identical rule on 8 other sectors, charged max-of-K
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

ASOF = pd.Timestamp("2026-09-10")
Z_THR = -2.0
H = 5
COST_BPS = 4.0

COMPLEX = {
    "XLI": ["MMM", "ITW", "GD", "PH", "DOV", "NSC", "EMR", "HON", "CAT",
            "UNP", "ROK"],
    "XLE": ["XOM", "CVX", "COP", "SLB", "EOG", "PSX", "MPC", "VLO", "OXY",
            "HAL", "WMB"],
    "XLK": ["AAPL", "MSFT", "NVDA", "AVGO", "CSCO", "ORCL", "CRM", "ACN",
            "ADBE", "TXN", "INTC"],
    "XLF": ["JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC", "AXP",
            "SCHW", "BLK"],
    "XLV": ["JNJ", "PFE", "MRK", "CI", "LLY", "UNH", "TMO", "ABT", "BMY",
            "AMGN", "CVS"],
    "XLP": ["PG", "KO", "PEP", "WMT", "COST", "MO", "PM", "MDLZ", "CL",
            "KMB", "GIS"],
    "XLY": ["AMZN", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "YUM", "TGT",
            "F", "GM"],
    "XLB": ["LIN", "APD", "SHW", "ECL", "NEM", "FCX", "IP", "DD", "PPG",
            "NUE", "VMC"],
    "XLU": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "ED",
            "PEG", "WEC"],
}
DEFENDED = "XLI"

need = sorted({t for v in COMPLEX.values() for t in v} | set(COMPLEX))
px = load_prices(need)
missing = [t for t in need if t not in px]
if missing:
    print("DROPPED (not cached):", missing)

Z = {t: zscore(px[t]["Close"], 10) for t in px}


def count_series(sector):
    members = [t for t in COMPLEX[sector] if t in px]
    idx = px[sector].index
    df = pd.DataFrame({t: (Z[t] <= Z_THR).reindex(idx).fillna(False)
                       for t in members})
    # only count on days every member has a defined z (avoids inception drift)
    defined = pd.DataFrame({t: Z[t].reindex(idx).notna() for t in members})
    c = df.sum(axis=1).where(defined.all(axis=1))
    return c, members


print("=" * 78)
print("(a) TODAY'S COUNT under pitch_lab.zscore, 2026-09-10 close")
for sector in COMPLEX:
    c, members = count_series(sector)
    vals = {t: float(Z[t].loc[:ASOF].iloc[-1]) for t in members}
    cnt = c.loc[:ASOF].iloc[-1]
    own = float(Z[sector].loc[:ASOF].iloc[-1])
    mark = "  <== DEFENDED" if sector == DEFENDED else ""
    print(f"  {sector}: count={cnt:.0f}/{len(members)}  own z10={own:+.2f}{mark}")
    if sector == DEFENDED:
        print("    " + "  ".join(f"{t}={vals[t]:+.2f}" for t in members))

c_xli, members_xli = count_series(DEFENDED)
print(f"\n  NOTE handover quoted tape-convention z10s (MMM -3.55 etc). "
      f"Under pitch_lab the count is {c_xli.loc[:ASOF].iloc[-1]:.0f}.")

# ---------------------------------------------------------------------------
# (b) dose response
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print(f"(b) DOSE RESPONSE, long {DEFENDED}, h={H}, lag=1")
xli = px[DEFENDED]["Close"]
fw = fwd_lag(xli, H, 1)
valid = fw.dropna().index
base = fw.loc[valid]
print(f"  all days: N={len(base)} mean={100*base.mean():+.3f}% "
      f"hit={100*(base>0).mean():.1f}%")
rows = []
for k in range(0, 12):
    m = (c_xli.reindex(valid) == k)
    v = base[m.fillna(False).values]
    if len(v) == 0:
        continue
    rows.append(summarize(v.values, f"count=={k}"))
show(rows, "exact count")
rows = []
for k in range(1, 10):
    m = (c_xli.reindex(valid) >= k)
    v = base[m.fillna(False).values]
    if len(v) < 3:
        continue
    r = summarize(v.values, f"count>={k}")
    ep = declusters(v.index, H, valid)
    r["epi_n"] = len(ep)
    r["epi_mean_pct"] = round(100 * base.loc[ep].mean(), 3)
    rows.append(r)
show(rows, "cumulative count (with episode-level column)")

# ---------------------------------------------------------------------------
# (c) PC1 / participation ratio / effective N
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("(c) FACTOR STRUCTURE of each complex (daily returns, full overlap)")
for sector in COMPLEX:
    members = [t for t in COMPLEX[sector] if t in px]
    R = pd.DataFrame({t: px[t]["Close"].pct_change() for t in members}).dropna()
    C = R.corr().values
    ev = np.linalg.eigvalsh(C)[::-1]
    pc1 = ev[0] / ev.sum()
    part = (ev.sum() ** 2) / (ev ** 2).sum()
    mpc = (C.sum() - len(C)) / (len(C) * (len(C) - 1))
    tag = "  <== DEFENDED" if sector == DEFENDED else ""
    print(f"  {sector}: mean pairwise corr {mpc:.3f}  PC1 {100*pc1:.1f}%  "
          f"participation ratio {part:.2f} effective of {len(members)}{tag}")

# ---------------------------------------------------------------------------
# (d) set membership: does the count ever fire without XLI itself?
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("(d) SET MEMBERSHIP - P(XLI's own z10 <= -2 | count >= k)")
zx = Z[DEFENDED].reindex(valid)
own_trig = (zx <= Z_THR)
print(f"  XLI own z10 <= -2 base rate: {100*own_trig.mean():.1f}% "
      f"({int(own_trig.sum())} of {len(valid)} days)")
for k in range(1, 10):
    m = (c_xli.reindex(valid) >= k).fillna(False)
    n = int(m.sum())
    if n == 0:
        continue
    p = float(own_trig[m.values].mean())
    print(f"  count>={k}: N={n:>5}  P(XLI own trigger) = {p:.3f}   "
          f"count-ON but XLI NOT triggering: {int((m & ~own_trig).sum())} days")

# ---------------------------------------------------------------------------
# (e) count vs plain XLI z10 <= -2, and the residual cell
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("(e) COUNT vs the single-instrument cell it may be relabelling")
rows = []
for lbl, m in [
    ("XLI z10<=-2 ALONE", own_trig),
    ("count>=4 ALONE", (c_xli.reindex(valid) >= 4).fillna(False)),
    ("count>=6 ALONE", (c_xli.reindex(valid) >= 6).fillna(False)),
    ("XLI z<=-2 AND count>=4", own_trig & (c_xli.reindex(valid) >= 4).fillna(False)),
    ("XLI z<=-2 AND count<4 (complement)",
     own_trig & ~(c_xli.reindex(valid) >= 4).fillna(False)),
    ("count>=4 AND XLI z>-2 (breadth w/o headline)",
     (c_xli.reindex(valid) >= 4).fillna(False) & ~own_trig),
]:
    v = base[np.asarray(m.values, bool)]
    if len(v) == 0:
        rows.append({"label": lbl, "n": 0})
        continue
    r = summarize(v.values, lbl)
    ep = declusters(v.index, H, valid)
    r["epi_n"] = len(ep)
    r["epi_mean_pct"] = round(100 * base.loc[ep].mean(), 3)
    rows.append(r)
show(rows, "count vs single instrument")

# ---------------------------------------------------------------------------
# (f) reference class: identical rule on every sector, charged max-of-K
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print(f"(f) REFERENCE CLASS - same count>=K rule, long the sector ETF, h={H}")
for K in (3, 4, 5, 6):
    print(f"\n  --- threshold count >= {K} ---")
    res = []
    for sector in COMPLEX:
        c, members = count_series(sector)
        s = px[sector]["Close"]
        f = fwd_lag(s, H, 1)
        val = f.dropna().index
        m = (c.reindex(val) >= K).fillna(False)
        v = f.loc[val][m.values]
        ep = declusters(v.index, H, val)
        if len(ep) < 2:
            res.append((sector, len(v), len(ep), np.nan, np.nan, np.nan))
            continue
        e = f.loc[ep].values
        drift = f.loc[val].mean()
        res.append((sector, len(v), len(ep), 100 * e.mean(),
                    100 * (e.mean() - drift),
                    e.mean() / (e.std(ddof=1) / np.sqrt(len(e)))))
    rd = pd.DataFrame(res, columns=["sector", "n_days", "n_epi", "mean_pct",
                                    "excess_pct", "t"])
    print(rd.round(3).to_string(index=False))
    ok = rd.dropna(subset=["excess_pct"])
    if DEFENDED in set(ok["sector"]) and len(ok) > 1:
        d = float(ok.set_index("sector").loc[DEFENDED, "excess_pct"])
        rank = int((ok["excess_pct"] > d).sum()) + 1
        print(f"   {DEFENDED} excess {d:+.3f}% ranks {rank} of {len(ok)}")

# ---------------------------------------------------------------------------
# battery on the defended cell at the live count (reported after (a))
# ---------------------------------------------------------------------------
live_cnt = int(c_xli.loc[:ASOF].iloc[-1])
print("\n" + "=" * 78)
print(f"BATTERY at the LIVE count (>= {live_cnt})")
frame = pd.DataFrame({DEFENDED: xli})
mask = (c_xli >= live_cnt).reindex(frame.index, fill_value=False).fillna(False)
variants = {f"count>={k}": (c_xli >= k).reindex(frame.index).fillna(False)
            for k in (2, 3, 4, 5, 6, 7)}
variants["XLI z10<=-2 alone"] = own_trig.reindex(frame.index).fillna(False)
battery(frame, mask, [(DEFENDED, 1.0)], H,
        f"A8 {DEFENDED} count>={live_cnt}", COST_BPS, variants=variants,
        event_kinds=("cpi", "fomc_decision"))

print("\n  horizon scan at the live count:")
sig = frame.index[mask.values]
show(horizon_scan(frame, sig, [(DEFENDED, 1.0)], hs=(1, 2, 3, 5, 7, 10)),
     "XLI count horizon scan")
