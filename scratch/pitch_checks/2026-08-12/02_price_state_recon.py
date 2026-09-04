"""B1 recon: every tape extreme by class, one excess-vs-own-drift reading.

Not a check. Round-1 lives in the c*/a* scripts. This exists so the surface
map's dismissals rest on a number instead of an impression.

Registry traps this script is written against:
- pct_rank takes a PRICE series (it does pct_change internally). Feeding it a
  return series silently ranks a second difference. Every trigger below prints
  TODAY's value so it can be eyeballed against pitch_tape.json.
- entry is lag=1 (MOC tomorrow), never lag=0.
- the control is the instrument's own drift over the same span, never zero.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

TKRS = ["SPY", "QQQ", "IWM", "DIA", "TLT", "IEF", "LQD", "HYG", "GLD", "GDX",
        "SLV", "USO", "DBC", "XLE", "UUP", "EFA", "EEM", "FXI", "EWZ", "SVXY",
        "XLU", "XLP", "XLRE", "VNQ", "XLV", "XLF", "XLK", "SMH", "XLC", "XME",
        "IBB", "^VIX", "^SKEW", "^MOVE", "^TNX", "UVXY", "XLY", "XLB"]
px = close_panel(TKRS)
idx = px.index


def dist_52w_high(s):
    return s / s.rolling(252).max() - 1.0


def dist_52w_low(s):
    return s / s.rolling(252).min() - 1.0


CELLS = {}


def cell(name, mask, legs, note=""):
    CELLS[name] = (mask, legs, note)


# --- rate-sensitive defensive washout (watchlist owes this a real check) ----
cell("XLU 21d rank <= 5", pct_rank(px["XLU"], 21) <= 5, [("XLU", 1.0)])
cell("XLRE 5d rank <= 8", pct_rank(px["XLRE"], 5) <= 8, [("XLRE", 1.0)])
cell("VNQ z10 <= -1.5", zscore(px["VNQ"], 10) <= -1.5, [("VNQ", 1.0)])

# --- semis laggard (watchlist owes this a real check) ----------------------
cell("SMH 63d rank <= 2", pct_rank(px["SMH"], 63) <= 2, [("SMH", 1.0)])
cell("SMH 63d rank <= 2 vs QQQ", pct_rank(px["SMH"], 63) <= 2,
     [("SMH", 1.0), ("QQQ", -1.0)])

# --- sector leadership at the top of its own range -------------------------
cell("XLV 63d rank >= 99", pct_rank(px["XLV"], 63) >= 99, [("XLV", 1.0)])
cell("XLF 63d rank >= 99", pct_rank(px["XLF"], 63) >= 99, [("XLF", 1.0)])
cell("XLV>=99 vs SPY", pct_rank(px["XLV"], 63) >= 99,
     [("XLV", 1.0), ("SPY", -1.0)])

# --- the whole IG duration complex pinned at a 52w low ---------------------
ig_low = (dist_52w_low(px["TLT"]) <= 0.01) & (dist_52w_low(px["IEF"]) <= 0.015) \
    & (dist_52w_low(px["LQD"]) <= 0.015)
cell("TLT+IEF+LQD all within ~1% of 52w low", ig_low, [("TLT", 1.0)])
cell("  same, IEF leg", ig_low, [("IEF", 1.0)])
cell("  same, LQD leg", ig_low, [("LQD", 1.0)])

# --- credit divergence, HYG high / LQD low (watchlist, arm at N>=8) --------
cell("HYG<0.5% of high & LQD<2% of low",
     (dist_52w_high(px["HYG"]) >= -0.005) & (dist_52w_low(px["LQD"]) <= 0.02),
     [("LQD", 1.0), ("HYG", -1.0)])

# --- vol complex at an extreme --------------------------------------------
cell("SVXY at a 52w high", dist_52w_high(px["SVXY"]) >= -0.001, [("SVXY", 1.0)])
cell("SKEW 5d rank>=95 & VIX 5d rank<=35",
     (pct_rank(px["^SKEW"], 5) >= 95) & (pct_rank(px["^VIX"], 5) <= 35),
     [("SPY", 1.0)])
cell("  same, short SVXY leg",
     (pct_rank(px["^SKEW"], 5) >= 95) & (pct_rank(px["^VIX"], 5) <= 35),
     [("SVXY", -1.0)])

# --- metals / energy thrust ------------------------------------------------
cell("GDX 5d rank>=99 & GLD 5d rank>=95",
     (pct_rank(px["GDX"], 5) >= 99) & (pct_rank(px["GLD"], 5) >= 95),
     [("GDX", 1.0)])
cell("XME 5d rank >= 96", pct_rank(px["XME"], 5) >= 96, [("XME", 1.0)])
cell("DBC 5d rank >= 97", pct_rank(px["DBC"], 5) >= 97, [("DBC", 1.0)])
cell("USO 5d rank>=90 & 63d rank<=20",
     (pct_rank(px["USO"], 5) >= 90) & (pct_rank(px["USO"], 63) <= 20),
     [("USO", 1.0)])

# --- intl washout ----------------------------------------------------------
cell("EWZ 5d rank <= 3", pct_rank(px["EWZ"], 5) <= 3, [("EWZ", 1.0)])
cell("EWZ 5d rank<=3 vs EEM", pct_rank(px["EWZ"], 5) <= 3,
     [("EWZ", 1.0), ("EEM", -1.0)])

# --- dispersion inside US large -------------------------------------------
cell("DIA 63d rank>=84 & QQQ 63d rank<=21",
     (pct_rank(px["DIA"], 63) >= 84) & (pct_rank(px["QQQ"], 63) <= 21),
     [("QQQ", 1.0), ("DIA", -1.0)])

# --- MOVE thrust while the long end sits at a low -------------------------
cell("MOVE 1d>=3% & TLT within 1% of 52w low",
     (px["^MOVE"].pct_change() >= 0.03) & (dist_52w_low(px["TLT"]) <= 0.01),
     [("TLT", 1.0)])

print("cell                                     h  N   Nepi  cond%   drift%   "
      "excess  hit%   signp   today?")
for name, (mask, legs, note) in CELLS.items():
    m = mask.reindex(idx, fill_value=False).fillna(False)
    today = bool(m.iloc[-1])
    for h in (1, 3, 5):
        r = vehicle_ret(px, legs, h, 1)
        sig = idx[m.values & r.notna().values]
        if len(sig) < 5:
            print(f"{name[:40]:40s} {h}  N={len(sig)}  (too few, dead on count)")
            break
        epi = declusters(sig, max(h, 5), idx)
        v = r.loc[epi].values
        span = (idx >= sig[0]) & (idx <= sig[-1])
        ctrl = r[span].dropna()
        wins = int((v > 0).sum())
        base = float((ctrl > 0).mean())
        print(f"{name[:40]:40s} {h} {len(sig):4d} {len(epi):5d} "
              f"{100*v.mean():+7.3f} {100*ctrl.mean():+7.3f} "
              f"{100*(v.mean()-ctrl.mean()):+7.3f} {100*wins/len(v):5.1f} "
              f"{sign_test(wins, len(v), base):7.4f}  {'LIVE' if today else ''}")
