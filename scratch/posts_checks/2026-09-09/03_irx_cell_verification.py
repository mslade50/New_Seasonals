"""CHECK C (2026-09-09) — VERIFICATION ONLY of the context brief's cell:

  "^IRX 63-day return rank >= 95 with ^GSPC within 3% of its 252-day high,
   ZIRP readings removed -> VIX h5 mean -3.246%, hit 5.6%, n=18, t -3.56"

The producing drill is scratch/context_checks/2026-09-09/07_front_end_repricing.py.
Nothing is re-derived here: that module is imported and executed AS WRITTEN,
with the single change of extending its HORIZONS tuple from (1, 5, 21) to
(1, 5, 10, 21) so the h=10 question can be answered on the same methodology.
Everything else — the rank definition, the 3%-of-252d-high definition, the
lag=0 close-to-close context convention, the 10td declustering, the ^IRX >= 0.50
ZIRP exclusion, the controls — is the drill's own code.

The drill's own caveat, quoted from its docstring: ^IRX is a 13-week BILL YIELD
LEVEL, so a 63-day PERCENT change of a near-zero yield explodes (0.02 -> 0.04 is
+100%) and the ZIRP era manufactures rank>=95 readings out of noise. The ZIRP
cut is therefore a level filter, ^IRX >= 0.50, applied to the TRIGGER sessions.

Also printed below, computed here: whether the condition is live on tonight's
bar, and an explicit wins-losses record plus the unconditional VIX control at
each horizon (the drill prints controls for the ^GSPC leg only).
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from pitch_lab import *  # noqa: F401,F403

import numpy as np
import pandas as pd

DRILL = ROOT / "scratch" / "context_checks" / "2026-09-09" / "07_front_end_repricing.py"
ASOF = pd.Timestamp("2026-09-09")

print("=" * 96)
print("CHECK C — verification of the ^IRX x ^GSPC-near-high cell")
print("drill of record:", DRILL)
print("=" * 96)

spec = importlib.util.spec_from_file_location("drill07", DRILL)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print("\ndrill constants as written: RANK_MIN=%s  SPX_NEAR_HIGH=%s  GAP_TD=%s  "
      "HORIZONS=%s" % (mod.RANK_MIN, mod.SPX_NEAR_HIGH, mod.GAP_TD, mod.HORIZONS))
mod.HORIZONS = (1, 5, 10, 21)
print("HORIZONS extended to %s for this verification (only change made).\n"
      % (mod.HORIZONS,))

print("#" * 96)
print("# DRILL OUTPUT BEGINS (unmodified code)")
print("#" * 96)
mod.main()
print("#" * 96)
print("# DRILL OUTPUT ENDS")
print("#" * 96)

# ---------------------------------------------------------------------------
# supplementary: explicit W-L record and an unconditional VIX control, which
# the drill prints only for the ^GSPC leg.
# ---------------------------------------------------------------------------
px = load_prices(["^IRX", "^GSPC", "^VIX"])
irx = px["^IRX"]["Close"].dropna()
gspc = px["^GSPC"]["Close"].dropna()
vix = px["^VIX"]["Close"].dropna()

r63 = pct_rank(px["^IRX"]["Close"], 63, 252)
g_hi = gspc.rolling(252).max()
g_near = gspc >= mod.SPX_NEAR_HIGH * g_hi

print("\n" + "=" * 96)
print("SUPPLEMENT 1 — is the cell live on tonight's bar?")
print("=" * 96)
print(f"  ^IRX close {irx.loc[ASOF]:.4f}   63d-return rank (252 lookback) "
      f"{r63.loc[ASOF]:.2f}   >= {mod.RANK_MIN}? {bool(r63.loc[ASOF] >= mod.RANK_MIN)}")
print(f"  ^IRX raw 63d change in the yield level "
      f"{100*(irx.loc[ASOF]/irx.shift(63).loc[ASOF] - 1):+.2f}%")
print(f"  ^IRX >= 0.50 (passes the ZIRP filter)? {bool(irx.loc[ASOF] >= 0.5)}")
print(f"  ^GSPC close {gspc.loc[ASOF]:.2f}  252d max {g_hi.loc[ASOF]:.2f}  "
      f"{100*(gspc.loc[ASOF]/g_hi.loc[ASOF]-1):+.2f}% vs high   within 3%? "
      f"{bool(g_near.loc[ASOF])}")

trig = pd.DatetimeIndex(r63.index[(r63 >= mod.RANK_MIN).fillna(False).values])
trig = trig.intersection(pd.DatetimeIndex(g_near.index[g_near.fillna(False).values]))
trig_nz = trig.intersection(pd.DatetimeIndex(irx.index[(irx >= 0.5).values]))
spine = gspc.index
epi = declusters(trig.intersection(spine), mod.GAP_TD, spine)
epi_nz = declusters(trig_nz.intersection(spine), mod.GAP_TD, spine)
print(f"\n  headline cell: {len(trig)} trigger sessions -> {len(epi)} episodes")
print(f"  ZIRP-removed : {len(trig_nz)} trigger sessions -> {len(epi_nz)} episodes")
print("  ZIRP-removed episode dates: " + ", ".join(str(d.date()) for d in epi_nz))
lv = irx.reindex(epi).dropna()
print(f"  ^IRX level on headline episodes: min {lv.min():.3f} med {lv.median():.3f} "
      f"max {lv.max():.3f};  below 0.50: {int((lv < 0.5).sum())} of {len(lv)}")


def _signp(k, n):
    """Exact binomial sign p, but skipped on huge control cells.

    pitch_lab.sign_test's p=0.5 branch is exact rational arithmetic; at
    n ~ 6700 (an all-days control) one call costs ~8 seconds of big-int work.
    The sign test exists for the SMALL conditional cell, so it is reported
    there and suppressed (None) on cells above 1500 observations, where the
    hit-rate column already says everything a coin test would.
    """
    if n > 1500:
        return None
    return round(sign_test(k, n), 4)


def rec(vals, label, dates=None):
    v = np.asarray(vals, float)
    v = v[~np.isnan(v)]
    s = summarize(v, label)
    if s["n"] == 0:
        return s
    up, dn = int((v > 0).sum()), int((v < 0).sum())
    s["record"] = f"{up}-{dn}"
    s["sign_p_dn"] = _signp(dn, len(v))
    s["sign_p_up"] = _signp(up, len(v))
    s.pop("sd_pct", None)
    return s


print("\n" + "=" * 96)
print("SUPPLEMENT 2 — ^VIX leg, ZIRP-removed episodes, wins-losses + control")
print("   (lag=0 close-to-close, the drill's CONTEXT convention, not a trade)")
print("=" * 96)
rows = []
for h in (1, 5, 10, 21):
    f = fwd_ret(vix, h)
    valid = f.dropna().index
    e = pd.DatetimeIndex(epi_nz).intersection(valid)
    e_all = pd.DatetimeIndex(epi).intersection(valid)
    rows.append(rec(f.loc[e].values, f"ZIRP-removed episodes h={h}", e))
    rows.append(rec(f.loc[e_all].values, f"  headline (ZIRP kept) h={h}", e_all))
    rows.append(rec(f.loc[valid].values, f"  CTRL all days h={h}", valid))
    ctd = pd.DatetimeIndex(trig_nz).intersection(valid)
    rows.append(rec(f.loc[ctd].values, f"  ZIRP-removed day-level h={h}", ctd))
show(rows, "^VIX forward, percent change in the index level")

print("\n" + "=" * 96)
print("SUPPLEMENT 3 — same on ^GSPC, ZIRP-removed episodes")
print("=" * 96)
rows = []
for h in (1, 5, 10, 21):
    f = fwd_ret(gspc, h)
    valid = f.dropna().index
    e = pd.DatetimeIndex(epi_nz).intersection(valid)
    rows.append(rec(f.loc[e].values, f"ZIRP-removed episodes h={h}", e))
    rows.append(rec(f.loc[valid].values, f"  CTRL all days h={h}", valid))
show(rows, "^GSPC forward")

print("\n" + "=" * 96)
print("SUPPLEMENT 4 — era split + haircut on the ZIRP-removed ^VIX h=5 cell")
print("=" * 96)
f5 = fwd_ret(vix, 5)
e = pd.DatetimeIndex(epi_nz).intersection(f5.dropna().index)
v = f5.loc[e].values.astype(float)
er = era_split(e, v)
for x in er:
    x.pop("sd_pct", None)
show(er, "era split (cut 2018-01-01)")
print("  episode years: " + ", ".join(str(d.year) for d in e))
by_yr = pd.Series(v, index=pd.DatetimeIndex(e).year).groupby(level=0).sum()
drop = set(by_yr.sort_values().head(2).index)   # 2 most NEGATIVE years = best for a short
keep = ~np.isin(pd.DatetimeIndex(e).year, list(drop))
w = v[keep]
up, dn = int((w > 0).sum()), int((w < 0).sum())
print(f"  drop the 2 best years for the short {sorted(drop)} -> n={len(w)} "
      f"mean {100*w.mean():+.3f}% median {100*np.median(w):+.3f}% "
      f"record {up}-{dn} sign p(dn) {sign_test(dn, len(w)):.4f}")
print("  " + cluster_note(e, v))
print("\nDONE.")
