"""C4 round 1: SHORT the semis complex into the August NVDA print with SMH at
a 63-day rank floor.

This is a RESCUE of a sign that was killed on 2026-08-14 as a long. It is
priced as such: the family-wise cost of the search that produced it is
computed explicitly, and the placebo offset ladder is run in b1b.

Anchor convention. NVDA prints after the close on date P. Today is
2026-08-19 with P = 2026-08-26 at +5 sessions from today's close. So the
tradeable analogue is: signal on close D, entry MOC on close D+1, where
D+1 sits exactly 5 sessions before P. That is D = pos(P) - 6.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

ASOF = pd.Timestamp("2026-08-18")
SEMI = ["SMH", "NVDA", "AMD", "AVGO", "MU", "INTC", "TXN", "ADI", "AMAT",
        "LRCX", "KLAC", "QCOM", "ASML", "TSM", "MRVL", "ON", "MCHP", "SWKS",
        "TER", "MPWR", "NXPI"]
px = close_panel(SEMI + ["SPY", "QQQ"])
idx = px.index
pos = pd.Series(range(len(idx)), index=idx)

# ---------------------------------------------------------------- prints
ec = pd.read_parquet("data/earnings_calendar.parquet")
nv = ec[ec["ticker"] == "NVDA"]["date"]
nv = pd.DatetimeIndex(pd.to_datetime(nv).sort_values().unique())
aug = nv[nv.month == 8]
print("NVDA prints in cache:", len(nv), nv[0].date(), "..", nv[-1].date())
print("August prints (n=%d):" % len(aug), ", ".join(str(d.date()) for d in aug))

ENTRY_LEAD = 5   # entry close sits 5 sessions before the print date


def anchor_pairs(prints, k=0):
    """(print, D) where entry (D+1) is ENTRY_LEAD-k sessions before the print.
    Returns PAIRS so the print and its anchor can never drift apart in the
    display (they did in the first draft: prints out of range at either end
    were dropped from the anchor list only, and a naive zip mislabelled the
    whole table by one year)."""
    out = []
    for p in prints:
        loc = idx.searchsorted(p)          # nearest trading date on/after p
        if loc >= len(idx):
            continue
        d = loc - (ENTRY_LEAD + 1) + k
        if 0 <= d < len(idx):
            out.append((p, idx[d]))
    return out


def anchors_for(prints, k=0):
    return pd.DatetimeIndex([d for _, d in anchor_pairs(prints, k)])


pairs_aug = anchor_pairs(aug)
anc_aug = anchors_for(aug)
anc_all = anchors_for(nv)
print("\nanchor days (August, k=0):")
for p, d in pairs_aug:
    print(f"   print {p.date()}  anchor close {d.date()}  entry close {idx[pos[d]+1].date()}")

# ---------------------------------------------------------------- gate
smh = px["SMH"].dropna()
r63 = pct_rank(smh, 63).reindex(idx)
print(f"\nSMH 63d rank today ({ASOF.date()}) = {r63.loc[ASOF]:.1f}")
print("SMH 63d rank at each August anchor:")
for p, d in pairs_aug:
    print(f"   print {p.date()} anchor {d.date()}  r63rank {r63.get(d, np.nan):6.1f}")


def mk(anchor_idx, gate=None):
    m = pd.Series(False, index=idx)
    m.loc[anchor_idx] = True
    if gate is not None:
        m &= (r63 <= gate).fillna(False)
    return m


H = 7
LEGS = [("SMH", -1.0)]

variants = {}
for g in (10, 15, 20, 25, 33, 50, 100):
    variants[f"aug print, r63rank<={g}"] = mk(anc_aug, g)
variants["aug print, NO gate"] = mk(anc_aug, None)
variants["ALL prints, r63rank<=25"] = mk(anc_all, 25)
variants["ALL prints, NO gate"] = mk(anc_all, None)

battery(px, mk(anc_aug, 25), LEGS, H,
        "C4  SHORT SMH, entry 5 sessions before an AUGUST NVDA print, SMH r63rank<=25",
        cost_bps=2.0, variants=variants, min_gap=200)

# ------------------------------------------------- explicit small-N record
ret = vehicle_ret(px, LEGS, H, 1)
for lbl, m in [("gated r63<=25", mk(anc_aug, 25)), ("gate OFF", mk(anc_aug, None))]:
    d = idx[m.values & ret.notna().values]
    v = ret.loc[d].values
    w = int((v > 0).sum())
    print(f"\n{lbl}: N={len(v)} record {w}-{len(v)-w} sign p={sign_test(w, len(v)):.4f} "
          f"mean {100*v.mean():+.3f}%")
    for dd, vv in zip(d, v):
        print(f"    anchor {dd.date()}  short-SMH h={H} {100*vv:+6.2f}%   "
              f"(SMH r63rank {r63.get(dd, np.nan):5.1f})")

# ------------------------------------------------- gate-off vs gate-on delta
print("\n== does the 63d-rank floor ADD anything? (August prints only) ==")
d_off = idx[mk(anc_aug, None).values & ret.notna().values]
rows = []
for g in (10, 15, 20, 25, 33, 50, 100):
    dg = idx[mk(anc_aug, g).values & ret.notna().values]
    if len(dg) == 0:
        rows.append({"gate": g, "n": 0})
        continue
    comp = d_off.difference(dg)
    rows.append({"gate": g, "n_in": len(dg), "mean_in_pct": 100*ret.loc[dg].mean(),
                 "n_out": len(comp),
                 "mean_out_pct": 100*ret.loc[comp].mean() if len(comp) else np.nan,
                 "gate_edge_pp": 100*(ret.loc[dg].mean() - ret.loc[d_off].mean())})
show(rows, "gate attribution: in-gate minus ALL August prints")

# ------------------------------------------------- SEARCH COST
print("""
== SEARCH COST of the 2026-08-14 work that produced this sign ==
The 08-14 study grouped NVDA prints BY PRINT MONTH. NVDA prints 4x a year
(Feb / May / Aug / Nov), so the month grouping is a 4-cell split. The kill
report then quoted a 2020+ sub-era for the August cell (2 eras) and the
candidate today is the opposite SIGN of what was tested (2 signs).
Minimum family = 4 months x 2 eras x 2 signs = 16 comparisons.
Bonferroni-corrected alpha at 0.05 -> p must be < 0.05/16 = 0.003125.
A 4-0 record is sign p = 0.0625; 5-0 is 0.03125; 6-0 is 0.015625.
NO achievable record at N<=7 clears 0.003125 (7-0 = 0.0078).
""")
