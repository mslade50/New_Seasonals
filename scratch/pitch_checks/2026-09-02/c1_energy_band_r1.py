"""C1 round 1 -- kill attempt on watchlist 4, "long XLE on a crude one-day
thrust in the [5,6)% band at >= 1.50 ATR, no CPI/PPI inside a 3-session hold".

Mandatory items from the brief:
 1. reproduce the [5,6)% band cell at h=3 with the full battery (lag=1)
 2. the FULL bucket ladder -- is [5,6) an interior spike in a non-monotone
    ladder? (definition fragility)
 3. does the >=1.50 ATR conjunction DO anything, or just remove observations?
    (gate attribution, with the number)
 4. NFP inside the hold -- the stated arm names only CPI/PPI, but a 3-session
    hold from 2026-09-02 CONTAINS the 2026-09-04 payrolls print.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa: F401,F403,E402
from pitch_lab import (close_panel, load_prices, fwd_lag, declusters, summarize,
                       sign_test, bootstrap_p_le0, cluster_note, battery,
                       local_control, wilder_atr, event_in_window, load_events,
                       show, horizon_scan)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)

TK = ["USO", "XLE", "XOP", "OIH", "SPY", "DBC"]
px = close_panel(TK)
uso = load_prices(["USO"])["USO"]

# USO 1d return + Wilder-14 ATR on USO's OWN calendar, then reindexed.
uso_1d_own = uso["Close"] / uso["Close"].shift(1) - 1.0
atr_own = pd.Series(wilder_atr(uso["High"], uso["Low"], uso["Close"]),
                    index=uso.index)
atrpct_own = atr_own / uso["Close"].shift(1)
uso_1d = uso_1d_own.reindex(px.index)
atr_pct = atrpct_own.reindex(px.index)
thrust_atr = uso_1d / atr_pct

print("=" * 110)
print("LIVE READING 2026-09-01")
print(f"  USO close {uso['Close'].iloc[-1]:.4f}  1d {100*uso_1d_own.iloc[-1]:+.3f}%  "
      f"ATR14 {atr_own.iloc[-1]:.4f} = {100*atrpct_own.iloc[-1]:.3f}% of prior close  "
      f"-> thrust {uso_1d_own.iloc[-1]/atrpct_own.iloc[-1]:.3f} ATR")
print(f"  last USO bar {uso.index[-1].date()}   last panel bar {px.index[-1].date()}")
print("=" * 110)

# ---------------------------------------------------------------------------
# 1. the ARMED cell, full battery.  band + ATR gate, event gate applied after.
# ---------------------------------------------------------------------------
band = (uso_1d >= 0.05) & (uso_1d < 0.06)
band_atr = band & (thrust_atr >= 1.50)

variants = {
    "[5,6)% band alone": band,
    "[5,6)% + atr>=1.50 (ARMED)": band_atr,
    "atr>=1.50 alone": thrust_atr >= 1.50,
    ">=5% alone (the dead parent)": uso_1d >= 0.05,
    "[5,6)% + atr>=1.25": band & (thrust_atr >= 1.25),
    "[5,6)% + atr>=1.75": band & (thrust_atr >= 1.75),
    "[4.5,6.5)% + atr>=1.50": (uso_1d >= 0.045) & (uso_1d < 0.065) & (thrust_atr >= 1.50),
    "[4,6)% + atr>=1.50": (uso_1d >= 0.04) & (uso_1d < 0.06) & (thrust_atr >= 1.50),
    "[5,7)% + atr>=1.50": (uso_1d >= 0.05) & (uso_1d < 0.07) & (thrust_atr >= 1.50),
}

battery(px, band_atr, [("XLE", 1.0)], h=3,
        title="C1 ARMED CELL: long XLE, USO 1d in [5,6)% AND >=1.50 ATR, h=3",
        cost_bps=4.0, variants=variants, min_gap=5, event_kinds=("cpi", "ppi"))

# the parked headline was on the band ALONE -- reproduce that too
battery(px, band, [("XLE", 1.0)], h=3,
        title="C1 PARKED HEADLINE: long XLE, USO 1d in [5,6)% (band only), h=3",
        cost_bps=4.0, variants=None, min_gap=5, event_kinds=("cpi", "ppi"))

# ---------------------------------------------------------------------------
# 2. FULL BUCKET LADDER -- monotone or interior spike?
# ---------------------------------------------------------------------------
print("\n" + "=" * 110)
print("2. BUCKET LADDER -- XLE h=3 episode mean and EXCESS over own drift, lag=1")
print("=" * 110)
s = px["XLE"].dropna()
f3 = fwd_lag(s, 3, lag=1)
own3 = f3.dropna().mean()
print(f"XLE own h=3 drift (all days, lag=1) = {100*own3:+.4f}%   N={f3.dropna().shape[0]}")

buckets = {
    "[2,3)%": (uso_1d >= 0.02) & (uso_1d < 0.03),
    "[3,4)%": (uso_1d >= 0.03) & (uso_1d < 0.04),
    "[4,5)%": (uso_1d >= 0.04) & (uso_1d < 0.05),
    "[5,6)%": (uso_1d >= 0.05) & (uso_1d < 0.06),
    "[6,7)%": (uso_1d >= 0.06) & (uso_1d < 0.07),
    "[7,inf)%": uso_1d >= 0.07,
}
rows = []
for lbl, m in buckets.items():
    mm = m.reindex(s.index).fillna(False)
    trig = s.index[mm.values]
    epi = declusters(trig, 5, s.index)
    v = f3.reindex(epi).dropna()
    if len(v) == 0:
        rows.append({"bucket": lbl, "n_epi": 0})
        continue
    w = int((v > 0).sum())
    rows.append({"bucket": lbl, "n_days": len(trig), "n_epi": len(v),
                 "mean_pct": round(100 * v.mean(), 3),
                 "excess_pp": round(100 * (v.mean() - own3), 3),
                 "hit": round(100 * (v > 0).mean(), 1),
                 "t": round(summarize(v.values)["t"], 2),
                 "signp": round(sign_test(w, len(v)), 4)})
print(pd.DataFrame(rows).to_string(index=False))

# same ladder with the ATR>=1.50 gate stapled on
print("\n  ... same ladder WITH the >=1.50 ATR gate:")
rows = []
for lbl, m in buckets.items():
    mm = (m & (thrust_atr >= 1.50)).reindex(s.index).fillna(False)
    trig = s.index[mm.values]
    epi = declusters(trig, 5, s.index)
    v = f3.reindex(epi).dropna()
    if len(v) == 0:
        rows.append({"bucket": lbl, "n_epi": 0})
        continue
    w = int((v > 0).sum())
    rows.append({"bucket": lbl, "n_days": len(trig), "n_epi": len(v),
                 "mean_pct": round(100 * v.mean(), 3),
                 "excess_pp": round(100 * (v.mean() - own3), 3),
                 "hit": round(100 * (v > 0).mean(), 1),
                 "signp": round(sign_test(w, len(v)), 4)})
print(pd.DataFrame(rows).to_string(index=False))

# ---------------------------------------------------------------------------
# 3. GATE ATTRIBUTION on the ATR conjunction
# ---------------------------------------------------------------------------
print("\n" + "=" * 110)
print("3. ATR GATE ATTRIBUTION -- what does '>=1.50 ATR' buy inside the band?")
print("=" * 110)


def cell(mask, h=3, ticker="XLE", lag=1, gap=5):
    ss = px[ticker].dropna()
    f = fwd_lag(ss, h, lag=lag)
    own = f.dropna().mean()
    mm = mask.reindex(ss.index).fillna(False)
    trig = ss.index[mm.values]
    epi = declusters(trig, gap, ss.index)
    v = f.reindex(epi).dropna()
    if len(v) == 0:
        return None
    w = int((v > 0).sum())
    st = summarize(v.values)
    return {"n_days": len(trig), "n_epi": len(v), "mean_pct": round(st["mean_pct"], 3),
            "excess_pp": round(st["mean_pct"] - 100 * own, 3), "hit": round(st["hit"], 1),
            "t": round(st["t"], 2) if st["t"] == st["t"] else np.nan,
            "signp": round(sign_test(w, len(v)), 4),
            "worst_pct": round(st["worst_pct"], 2),
            "dates": epi}


rows = []
for lbl, m in [("band only", band),
               ("band AND atr>=1.50", band_atr),
               ("band AND atr<1.50 (the DISCARDS)", band & (thrust_atr < 1.50))]:
    r = cell(m)
    if r:
        d = r.pop("dates")
        rows.append({"gate": lbl, **r})
print(pd.DataFrame(rows).to_string(index=False))
r_band, r_gated = cell(band), cell(band_atr)
print(f"\n  gate moves the band from {r_band['excess_pp']:+.3f}pp (n_epi {r_band['n_epi']}) "
      f"to {r_gated['excess_pp']:+.3f}pp (n_epi {r_gated['n_epi']}) "
      f"-- discards {r_band['n_epi'] - r_gated['n_epi']} episodes")
print("  ARMED-cell episode dates:", ", ".join(str(d.date()) for d in r_gated["dates"]))

# ---------------------------------------------------------------------------
# 4. NFP INSIDE THE HOLD -- the brief's decisive item
# ---------------------------------------------------------------------------
print("\n" + "=" * 110)
print("4. WHAT LANDS INSIDE THE 3-SESSION HOLD (entry lag=1, exit +3)")
print("=" * 110)
for label, mask in (("[5,6)% band alone", band), ("ARMED band+atr>=1.50", band_atr)):
    mm = mask.reindex(s.index).fillna(False)
    epi = declusters(s.index[mm.values], 5, s.index)
    v = f3.reindex(epi).dropna()
    epi = v.index
    print(f"\n--- {label}  (N episodes = {len(epi)}) ---")
    for kinds in (("nfp",), ("cpi", "ppi"), ("cpi",), ("ppi",),
                  ("fomc_decision",), ("cpi", "ppi", "nfp")):
        fl = event_in_window(epi, s.index, 3, 1, kinds)
        lab = "+".join(kinds)
        rr = []
        for tag, sel in ((f"{lab} IN hold", fl), (f"{lab} OUT", ~fl)):
            if sel.sum() == 0:
                rr.append({"label": tag, "n": 0})
                continue
            st = summarize(v.values[sel], tag)
            st["excess_pp"] = round(st["mean_pct"] - 100 * own3, 3)
            w = int((v.values[sel] > 0).sum())
            st["signp"] = round(sign_test(w, int(sel.sum())), 4)
            rr.append(st)
        df = pd.DataFrame(rr)
        for c in df.columns:
            if df[c].dtype.kind == "f":
                df[c] = df[c].round(3)
        print(df.to_string(index=False))
        if fl.sum() > 1 and (~fl).sum() > 1:
            a, b = v.values[fl], v.values[~fl]
            se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
            print(f"    welch t (IN - OUT) = {(a.mean()-b.mean())/se:+.2f}")

# the ARMED cell's live configuration: NFP in, CPI/PPI out
print("\n--- the LIVE configuration: NFP inside the hold, CPI and PPI outside ---")
mm = band_atr.reindex(s.index).fillna(False)
epi = declusters(s.index[mm.values], 5, s.index)
v = f3.reindex(epi).dropna()
epi = v.index
nfp_in = event_in_window(epi, s.index, 3, 1, ("nfp",))
cpp_out = ~event_in_window(epi, s.index, 3, 1, ("cpi", "ppi"))
live_cfg = nfp_in & cpp_out
print(f"  episodes matching TODAY's exact event configuration: {int(live_cfg.sum())} of {len(epi)}")
if live_cfg.sum():
    st = summarize(v.values[live_cfg], "LIVE CONFIG (nfp in, cpi/ppi out)")
    st["excess_pp"] = round(st["mean_pct"] - 100 * own3, 3)
    st["signp"] = round(sign_test(int((v.values[live_cfg] > 0).sum()), int(live_cfg.sum())), 4)
    show([st])
    print("  dates:", ", ".join(str(d.date()) for d in epi[live_cfg]))
    print("  returns %:", np.round(100 * v.values[live_cfg], 2).tolist())
st = summarize(v.values[~live_cfg], "everything else")
st["excess_pp"] = round(st["mean_pct"] - 100 * own3, 3)
show([st])

# and on the band alone, where N is larger
print("\n--- LIVE configuration on the BAND ALONE (larger N) ---")
mm = band.reindex(s.index).fillna(False)
epi_b = declusters(s.index[mm.values], 5, s.index)
vb = f3.reindex(epi_b).dropna()
epi_b = vb.index
nfp_in_b = event_in_window(epi_b, s.index, 3, 1, ("nfp",))
cpp_out_b = ~event_in_window(epi_b, s.index, 3, 1, ("cpi", "ppi"))
lc = nfp_in_b & cpp_out_b
rows = []
for tag, sel in (("LIVE CONFIG nfp-in/cpi-ppi-out", lc), ("all other", ~lc)):
    st = summarize(vb.values[sel], tag)
    st["excess_pp"] = round(st["mean_pct"] - 100 * own3, 3)
    st["signp"] = round(sign_test(int((vb.values[sel] > 0).sum()), int(sel.sum())), 4)
    rows.append(st)
show(rows)
print("  LIVE-CONFIG dates:", ", ".join(str(d.date()) for d in epi_b[lc]))

# ---------------------------------------------------------------------------
# 5. horizon scan on the armed cell (feeds round 3 if it survives)
# ---------------------------------------------------------------------------
print("\n" + "=" * 110)
print("5. HORIZON SCAN, armed cell, long XLE")
print("=" * 110)
mm = band_atr.reindex(px.index).fillna(False)
show(horizon_scan(px, px.index[mm.values], [("XLE", 1.0)],
                  hs=(1, 2, 3, 4, 5, 7, 10), lag=1, min_gap=5), "armed cell")
