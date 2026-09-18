"""S3 -- VIX range compression at the extreme WHILE tail hedging is bid.

Live 2026-09-04: ^VIX 14.53 close (53% below its 52w high); its 21-session RANGE
sits at the ~1st percentile of its own trailing history; ^SKEW 21d return rank
98.0 (+12.5% over 21d); SVXY within 0.3% of its 52w high, z10 +1.18.

GATE ATTRIBUTION IS THE POINT. Three masks measured on identical vehicles and
identical controls:
  (a) SKEW 21d rank >= 95 alone
  (b) VIX 21d range in its bottom decile (trailing-252 rank) alone
  (c) the CONJUNCTION
plus (a ex-c) and (b ex-c) so the conjunction's increment over each PARENT is
readable rather than asserted.

Vehicles: SPY long, SVXY long (short-vol expression), SPY short.
CAVEAT carried in the output: SVXY was a -1.0x product until 2018-02-28 and
-0.5x after; its pre-2018 leg is a different instrument, so the era split is
load-bearing, not decoration.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _survey_lib import (  # noqa: E402
    align, cell, declusters, era_split, fwd_lag, hscan, load_prices, np,
    pct_rank, pd, roll_max, roll_min, rolling_on_valid, show, sign_test, sma,
    summarize,
)

TICKERS = ["SPY", "SVXY", "^VIX", "^SKEW", "^VIX3M", "QQQ"]
PX = load_prices(TICKERS)
IDX = PX["SPY"].index
C = {t: PX[t]["Close"] for t in PX}

print("=" * 78)
print("S3  VIX RANGE COMPRESSION + SKEW BID   (asof 2026-09-04, entry lag=1)")
print("=" * 78)

# ------------------------------------------------- VIX 21d range + percentile
vix = C["^VIX"]
vix_hi21 = rolling_on_valid(vix, lambda x: x.rolling(21).max())
vix_lo21 = rolling_on_valid(vix, lambda x: x.rolling(21).min())
vix_rng = vix_hi21 - vix_lo21
# percentile of the range statistic against its own trailing 252 sessions,
# and against full history to date (expanding), reported both ways.
rng_rank252 = rolling_on_valid(vix_rng, lambda x: x.rolling(252).rank(pct=True) * 100)
rng_rank_exp = rolling_on_valid(vix_rng, lambda x: x.expanding(252).rank(pct=True) * 100)
skew_rank = pct_rank(C["^SKEW"], 21)

print("\nLIVE STATE VERIFICATION:")
print(f"  ^VIX close {vix.iloc[-1]:.2f}   21d range {vix_rng.iloc[-1]:.2f} "
      f"(hi {vix_hi21.iloc[-1]:.2f} lo {vix_lo21.iloc[-1]:.2f})")
print(f"  21d-range percentile: trailing-252 {rng_rank252.iloc[-1]:.1f}   "
      f"expanding-full {rng_rank_exp.iloc[-1]:.1f}   [brief claimed ~1st pctile]")
print(f"  ^VIX vs 52w high: {100 * (vix.iloc[-1] / roll_max(vix, 252).iloc[-1] - 1):+.1f}%")
print(f"  ^SKEW {C['^SKEW'].iloc[-1]:.2f}  21d ret {100 * (C['^SKEW'].iloc[-1] / C['^SKEW'].iloc[-22] - 1):+.2f}%  "
      f"rank21 {skew_rank.iloc[-1]:.1f}")
s = C["SVXY"]
print(f"  SVXY {s.iloc[-1]:.2f}  off 52w high {100 * (s.iloc[-1] / roll_max(s, 252).iloc[-1] - 1):+.2f}%")
print(f"  SPY  off 52w high {100 * (C['SPY'].iloc[-1] / roll_max(C['SPY'], 252).iloc[-1] - 1):+.2f}%")
print(f"  freshest bar: {IDX[-1].date()}")

# ------------------------------------------------------------------ the masks
m_skew = align(skew_rank >= 95, IDX).fillna(False).astype(bool)
m_comp = align(rng_rank252 <= 10, IDX).fillna(False).astype(bool)
m_conj = m_skew & m_comp
m_skew_only = m_skew & ~m_comp
m_comp_only = m_comp & ~m_skew

MASKS = {
    "(a) SKEW rank21>=95 ALONE": m_skew,
    "(b) VIX 21d-range bottom decile ALONE": m_comp,
    "(c) CONJUNCTION a&b": m_conj,
    "(a ex-c) SKEW bid, range NOT compressed": m_skew_only,
    "(b ex-c) range compressed, SKEW NOT bid": m_comp_only,
}
print("\nMASK COUNTS (trigger days over the shared SPY calendar):")
for k, m in MASKS.items():
    d = IDX[m.values]
    print(f"  {k:<44s} {len(d):5d} days"
          + (f"   {d[0].date()} .. {d[-1].date()}   yrs {len(set(d.year))}" if len(d) else ""))
print(f"  live day in conjunction: {bool(m_conj.iloc[-1])}   "
      f"(skew {bool(m_skew.iloc[-1])}, compression {bool(m_comp.iloc[-1])})")


def spy(h):
    return fwd_lag(C["SPY"], h, 1)


def spy_short(h):
    return -fwd_lag(C["SPY"], h, 1)


def svxy(h):
    return align(fwd_lag(C["SVXY"], h, 1), IDX)


for vname, vf in (("SPY long", spy), ("SVXY long", svxy)):
    for mname, m in MASKS.items():
        t = IDX[m.values]
        if len(t) == 0:
            print(f"\n{vname} x {mname}: NO TRIGGERS")
            continue
        hscan(vf, t, f"{vname}  |  {mname}")

# --------------------------------------------- attribution table at h=5 & h=10
print("\n" + "=" * 78)
print("GATE ATTRIBUTION: what the conjunction adds over each parent")
print("=" * 78)
rows = []
for h in (1, 2, 3, 5, 10):
    for vname, vf in (("SPY", spy), ("SVXY", svxy)):
        r = vf(h)
        valid = r.dropna().index
        base = float(r.loc[valid].mean())
        cellmeans = {}
        for mname, m in MASKS.items():
            t = pd.DatetimeIndex(IDX[m.values]).intersection(valid)
            if len(t) == 0:
                cellmeans[mname] = np.nan
                continue
            epi = declusters(t, h, valid)
            ep = r.loc[epi].values
            w = int((ep > 0).sum())
            cellmeans[mname] = 100 * ep.mean()
            rows.append({"h": h, "veh": vname, "mask": mname, "n_days": len(t),
                         "n_epi": len(epi), "mean_pct": round(100 * ep.mean(), 3),
                         "edge_all_pct": round(100 * (ep.mean() - base), 3),
                         "hit": round(100 * (ep > 0).mean(), 1),
                         "worst_pct": round(100 * ep.min(), 2),
                         "rec": f"{w}-{len(epi) - w}",
                         "sign_p": round(sign_test(w, len(epi)), 4)})
        c = cellmeans.get("(c) CONJUNCTION a&b", np.nan)
        rows.append({"h": h, "veh": vname, "mask": ">>> conj minus (a ex-c)",
                     "mean_pct": round(c - cellmeans.get("(a ex-c) SKEW bid, range NOT compressed", np.nan), 3)})
        rows.append({"h": h, "veh": vname, "mask": ">>> conj minus (b ex-c)",
                     "mean_pct": round(c - cellmeans.get("(b ex-c) range compressed, SKEW NOT bid", np.nan), 3)})
df = pd.DataFrame(rows)
print(df.to_string(index=False))

# ------------------------------------------------------- full cells + eras
for h in (5, 10):
    for vname, vf in (("SPY long", spy), ("SVXY long", svxy)):
        t = IDX[m_conj.values]
        if len(t):
            cell(vf(h), t, h, f"CONJUNCTION -> {vname}")

print("\n=== S3 ERA SPLIT on the conjunction (h=5 episodes) ===")
print("    SVXY was -1.0x before 2018-02-28 and -0.5x after; read its pre-2018")
print("    leg as a different instrument.")
for vname, vf in (("SPY", spy), ("SVXY", svxy)):
    r = vf(5)
    t = pd.DatetimeIndex(IDX[m_conj.values]).intersection(r.dropna().index)
    if len(t) == 0:
        continue
    epi = declusters(t, 5, r.dropna().index)
    show(era_split(epi, r.loc[epi].values), vname)

print("\n=== S3 SENSITIVITY: SKEW rank and compression-decile thresholds (SPY, h=5) ===")
rows = []
for sk in (90, 95, 98):
    for dec in (5, 10, 20):
        m = (align(skew_rank >= sk, IDX).fillna(False).astype(bool)
             & align(rng_rank252 <= dec, IDX).fillna(False).astype(bool))
        for vname, vf in (("SPY", spy), ("SVXY", svxy)):
            r = vf(5)
            t = pd.DatetimeIndex(IDX[m.values]).intersection(r.dropna().index)
            if len(t) == 0:
                rows.append({"skew>=": sk, "rng<=": dec, "veh": vname, "n_days": 0, "n": 0})
                continue
            epi = declusters(t, 5, r.dropna().index)
            ep = r.loc[epi].values
            w = int((ep > 0).sum())
            rows.append({"skew>=": sk, "rng<=": dec, "veh": vname, "n_days": len(t),
                         "n": len(epi), "mean_pct": round(100 * ep.mean(), 3),
                         "hit": round(100 * (ep > 0).mean(), 1),
                         "worst_pct": round(100 * ep.min(), 2),
                         "rec": f"{w}-{len(epi) - w}",
                         "sign_p": round(sign_test(w, len(epi)), 4)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== S3 COST NOTE ===")
print("  SPY round trip ~2-4 bps, SVXY wider (~8-12 bps, thinner book).")
print("  3x cost bar: SPY ~+0.09%, SVXY ~+0.30% per episode.")
