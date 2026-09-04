"""C1 DEBT 1 -- the fragility dial reads 87.9 and the gated sample tops out at 68.0.

Three attacks, in the order the brief demands:

 A. The dial is NOT independent of the entry. Production's VIX Range Compression
    signal is one of the 8 fragility contributors, so a compressed range
    mechanically pushes the dial up. Rebuild the PRODUCTION signal exactly
    (risk_dashboard_v2.compute_vix_range_compression: ABSOLUTE 21d range,
    504d percentile, < 15, VIX > 13, VIX > 20d SMA) and measure its overlap
    with the pitch's gate and with the dial.

 B. n=15 cannot answer "does a dial of 88 break it". Ask the big-N question
    instead: over ALL dial-covered days, what does the 10d-MA 63d dial predict
    for a ONE-SESSION long-SVXY / short-^VIX return? Decile buckets, N in the
    thousands. No one-session signal -> the debt is discharged. Wrong-signed at
    the top decile -> C1 is dead.

 C. Long-history state proxies with the SAME intent that DO cover all 21
    gated anchors: VIX level percentile, VIX3M/VIX term structure, SPY
    drawdown from its 252d high, 21d realised-vol percentile. Today's values
    are benign; does the cell survive in the benign subsample?

VINTAGE (stated per CLAUDE.md): data/rd2_fragility.parquet rows before
2026-07-02 are a RECOMPUTE vintage (drifted up to ~7 dial points); 2026-07-02+
are point-in-time appends. Every historical number below is therefore on the
recompute vintage, and only the last ~43 rows are PIT. That cuts BOTH ways and
is reported, not hidden.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa: F401,F403,E402
from pitch_lab import (close_panel, fwd_lag, summarize, sign_test, load_events,
                       rolling_on_valid, show, anchor_positions, bootstrap_p_le0)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
ROOT = Path(__file__).resolve().parents[3]

px = close_panel(["^VIX", "^VIX3M", "SVXY", "UVXY", "SPY"])
cal = px["SPY"].dropna().index
vix = px["^VIX"]

# --- the pitch's gate (rel range / 21d mean, 252 lookback, <= 15) -----------
rng21 = (rolling_on_valid(vix, lambda x: x.rolling(21).max())
         - rolling_on_valid(vix, lambda x: x.rolling(21).min()))
REL = rolling_on_valid(rng21 / rolling_on_valid(vix, lambda x: x.rolling(21).mean()),
                       lambda x: x.rolling(252).rank(pct=True) * 100)
G15 = REL <= 15.0

# --- PRODUCTION VIX Range Compression, copied from risk_dashboard_v2 -------
# _rolling_percentile there is rank(pct) over `lookback`; reproduce on valid.
comp_pctile_504 = rolling_on_valid(rng21, lambda x: x.rolling(504).rank(pct=True) * 100)
vix_sma20 = rolling_on_valid(vix, lambda x: x.rolling(20, min_periods=16).mean())
PROD_VRC = (comp_pctile_504 < 15.0) & (vix > 13.0) & (vix > vix_sma20)

# --- fragility dial --------------------------------------------------------
frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
frag.index = pd.to_datetime(frag.index)
DIAL = frag["63d"].rolling(10).mean()
PIT_CUT = pd.Timestamp("2026-07-02")

nfp = load_events(["nfp"])["date"]


def anchors(dates, k=-2, gate=None):
    p, _ = anchor_positions(cal, dates, k)
    a = pd.DatetimeIndex([cal[i] for i in p])
    if gate is not None:
        a = a[gate.reindex(a).fillna(False).values]
    return a


def stat(dates, tkr, h, lag=1, label="", sign=1.0):
    ss = px[tkr].dropna()
    f = sign * fwd_lag(ss, h, lag=lag)
    v = f.reindex(pd.DatetimeIndex(dates)).dropna()
    if len(v) == 0:
        return {"label": label, "n": 0}, v
    st = summarize(v.values, label)
    st["excess_pp"] = round(st["mean_pct"] - 100 * f.dropna().mean(), 3)
    st["signp"] = round(sign_test(int((v.values > 0).sum()), len(v)), 4)
    return st, v


A = anchors(nfp, -2, G15)
print("=" * 118)
print(f"LIVE 2026-09-02 close: rel-range pctile {REL.iloc[-1]:.2f}  "
      f"prod-VRC pctile(504) {comp_pctile_504.iloc[-1]:.2f}  "
      f"prod signal ON = {bool(PROD_VRC.iloc[-1])}  VIX {vix.iloc[-1]:.2f}")
print(f"  dial ma10(63d) = {DIAL.iloc[-1]:.2f}   raw 63d = {frag['63d'].iloc[-1]:.2f}"
      f"   raw 21d = {frag['21d'].iloc[-1]:.2f}")
print(f"  VINTAGE: dial rows {(frag.index < PIT_CUT).sum()} recompute / "
      f"{(frag.index >= PIT_CUT).sum()} point-in-time (cut {PIT_CUT.date()})")
print(f"  gated NFP anchors (k=-2, rel<=15): n={len(A)}  span "
      f"{A[0].date()}..{A[-1].date()}")
print("=" * 118)

# ===========================================================================
# A. IS THE DIAL ENDOGENOUS TO THE GATE?
# ===========================================================================
print("\n" + "=" * 118)
print("A. THE DIAL IS NOT INDEPENDENT OF THE ENTRY")
print("=" * 118)
d_all = DIAL.dropna()
common = d_all.index.intersection(cal)
g_on = G15.reindex(common).fillna(False)
p_on = PROD_VRC.reindex(common).fillna(False)
dv = d_all.reindex(common)

print(f"\nA1. day-level overlap of the two range definitions, {common[0].date()}.."
      f"{common[-1].date()} (N={len(common)})")
inter = int((g_on & p_on).sum())
union = int((g_on | p_on).sum())
print(f"    pitch gate ON days       : {int(g_on.sum())}")
print(f"    production VRC ON days   : {int(p_on.sum())}")
print(f"    both                     : {inter}   Jaccard = {inter/union:.3f}"
      if union else "")
print(f"    point-biserial corr(pitch gate, dial) = "
      f"{np.corrcoef(g_on.astype(float), dv)[0,1]:+.3f}")
print(f"    point-biserial corr(prod VRC,  dial) = "
      f"{np.corrcoef(p_on.astype(float), dv)[0,1]:+.3f}")
print(f"    mean dial | pitch gate ON  = {dv[g_on.values].mean():.1f}  "
      f"(OFF {dv[~g_on.values].mean():.1f})")
print(f"    mean dial | prod VRC ON    = {dv[p_on.values].mean():.1f}  "
      f"(OFF {dv[~p_on.values].mean():.1f})")

print("\nA2. dial on NFP anchors, gated vs discarded (dial-covered only)")
a_all = anchors(nfp, -2)
for lbl, s in (("gated (rel<=15)", A), ("discarded by the gate", a_all.difference(A))):
    dd = DIAL.reindex(s).dropna()
    if len(dd):
        print(f"    {lbl:24s} n={len(dd):3d}  mean dial {dd.mean():5.1f}  "
              f"median {dd.median():5.1f}  max {dd.max():5.1f}  "
              f">=60: {int((dd>=60).sum())}  >=80: {int((dd>=80).sum())}")

print("\nA3. today's dial is outside the gated population -- by how much?")
dd = DIAL.reindex(A).dropna()
print(f"    gated-anchor dial: n={len(dd)} max {dd.max():.1f} "
      f"p90 {dd.quantile(.9):.1f}; today {DIAL.iloc[-1]:.1f}")
pctile_today = 100 * (dv < DIAL.iloc[-1]).mean()
print(f"    but across ALL dial-covered days today's {DIAL.iloc[-1]:.1f} sits at the "
      f"{pctile_today:.1f}th percentile (N={len(dv)}) -- so the 15-anchor sample's")
print(f"    ceiling of {dd.max():.1f} reflects 15 draws, not a structural bar.")
n_days_ge = int((dv >= DIAL.iloc[-1]).sum())
print(f"    days at dial >= {DIAL.iloc[-1]:.1f} in the whole series: {n_days_ge} "
      f"({100*n_days_ge/len(dv):.1f}%)")

# ===========================================================================
# B. BIG-N: DOES THE DIAL PREDICT A ONE-SESSION SHORT-VOL RETURN AT ALL?
# ===========================================================================
print("\n" + "=" * 118)
print("B. BIG-N DIAL TEST -- all dial-covered days, 1-session forward, lag=1")
print("   long SVXY and short ^VIX. If the dial carries no one-session short-vol")
print("   signal at N in the thousands, debt 1 is discharged.")
print("=" * 118)

for tkr, sgn, nm in (("SVXY", 1.0, "long SVXY"), ("^VIX", -1.0, "short ^VIX"),
                     ("UVXY", -1.0, "short UVXY")):
    ss = px[tkr].dropna()
    f = sgn * fwd_lag(ss, 1, lag=1)
    j = pd.DataFrame({"dial": dv, "r": f.reindex(dv.index)}).dropna()
    q = pd.qcut(j["dial"], 10, labels=False, duplicates="drop")
    rows = []
    for b in sorted(q.unique()):
        m = q == b
        st = summarize(j.loc[m, "r"].values,
                       f"D{b+1} dial [{j.loc[m,'dial'].min():.0f},"
                       f"{j.loc[m,'dial'].max():.0f}]")
        st["signp"] = round(sign_test(int((j.loc[m, 'r'] > 0).sum()), int(m.sum())), 4)
        rows.append(st)
    rows.append(summarize(j["r"].values, "ALL dial-covered days"))
    show(rows, f"{nm} h=1 by dial decile (N={len(j)})")
    top = j[j["dial"] >= 80]
    tt = j[j["dial"] >= DIAL.iloc[-1] - 3]
    print(f"    dial >= 80 : n={len(top)} mean {100*top['r'].mean():+.3f}% "
          f"hit {100*(top['r']>0).mean():.1f}%   "
          f"| dial within +-3 of TODAY ({DIAL.iloc[-1]:.1f}): n={len(tt)} "
          f"mean {100*tt['r'].mean():+.3f}% hit {100*(tt['r']>0).mean():.1f}%")
    if len(j) > 2:
        c = np.corrcoef(j["dial"], j["r"])[0, 1]
        # OLS slope in pp of return per 10 dial points
        b1 = np.polyfit(j["dial"], 100 * j["r"], 1)[0]
        sl_t = c * np.sqrt((len(j) - 2) / max(1e-12, 1 - c * c))
        print(f"    corr(dial, next-session {nm} ret) = {c:+.4f}  t = {sl_t:+.2f}  "
              f"slope = {10*b1:+.4f} pp per 10 dial pts")

print("\nB2. the SAME question restricted to the gate being ON (any day, not just NFP)")
for tkr, sgn, nm in (("SVXY", 1.0, "long SVXY"), ("^VIX", -1.0, "short ^VIX")):
    ss = px[tkr].dropna()
    f = sgn * fwd_lag(ss, 1, lag=1)
    j = pd.DataFrame({"dial": dv, "r": f.reindex(dv.index),
                      "g": g_on.values}).dropna()
    j = j[j["g"].astype(bool)]
    if len(j) < 20:
        print(f"    {nm}: n={len(j)} too thin"); continue
    rows = []
    for lo, hi in ((0, 30), (30, 50), (50, 70), (70, 200)):
        m = (j["dial"] >= lo) & (j["dial"] < hi)
        rows.append(summarize(j.loc[m, "r"].values, f"gate ON, dial [{lo},{hi})"))
    rows.append(summarize(j["r"].values, "gate ON, all dials"))
    show(rows, f"{nm} h=1, gate ON days only (N={len(j)})")

# ===========================================================================
# C. LONG-HISTORY STATE PROXIES THAT COVER ALL 21 ANCHORS
# ===========================================================================
print("\n" + "=" * 118)
print("C. LONG-HISTORY STATE PROXIES -- same intent as the dial, full coverage")
print("=" * 118)

vix_lvl_pct = rolling_on_valid(vix, lambda x: x.rolling(252).rank(pct=True) * 100)
ts = px["^VIX3M"] / px["^VIX"] - 1.0                       # contango, +ve = calm
ts_pct = rolling_on_valid(ts, lambda x: x.rolling(252).rank(pct=True) * 100)
hi252 = rolling_on_valid(px["SPY"], lambda x: x.rolling(252).max())
dd_spy = px["SPY"] / hi252 - 1.0                           # 0 = at the high
rv21 = rolling_on_valid(px["SPY"].pct_change(),
                        lambda x: x.rolling(21).std() * np.sqrt(252) * 100)
rv_pct = rolling_on_valid(rv21, lambda x: x.rolling(252).rank(pct=True) * 100)

live = {"VIX level pctile": vix_lvl_pct.iloc[-1],
        "VIX3M/VIX contango": 100 * ts.iloc[-1],
        "contango pctile": ts_pct.iloc[-1],
        "SPY dd from 252d high": 100 * dd_spy.iloc[-1],
        "realised-vol pctile": rv_pct.iloc[-1]}
print("  LIVE: " + "  |  ".join(f"{k} {v:.2f}" for k, v in live.items()))

_, v_s = stat(A, "SVXY", 1)
_, v_v = stat(A, "^VIX", 1, sign=-1.0)
tbl = pd.DataFrame({
    "svxy_h1": (100 * v_s).round(2),
    "vix_lvl_pct": vix_lvl_pct.reindex(v_s.index).round(1),
    "contango_pct": (100 * ts.reindex(v_s.index)).round(1),
    "contango_pctile": ts_pct.reindex(v_s.index).round(1),
    "spy_dd_pct": (100 * dd_spy.reindex(v_s.index)).round(2),
    "rv_pctile": rv_pct.reindex(v_s.index).round(1),
    "dial": DIAL.reindex(v_s.index).round(1)})
print("\nC1. per-anchor state table (gated NFP set, SVXY h=1)")
print(tbl.to_string())

print("\nC2. does the cell survive in the BENIGN subsample (today's side of each split)?")
splits = [
    ("contango > 0 (today +16.6%)", ts.reindex(v_s.index) > 0),
    ("contango >= +10%", ts.reindex(v_s.index) >= 0.10),
    ("contango pctile >= 50", ts_pct.reindex(v_s.index) >= 50),
    ("SPY within 3% of 252d high (today -1.64%)",
     dd_spy.reindex(v_s.index) >= -0.03),
    ("SPY within 2% of high", dd_spy.reindex(v_s.index) >= -0.02),
    ("VIX level pctile <= 40 (today see live)",
     vix_lvl_pct.reindex(v_s.index) <= 40),
    ("realised-vol pctile <= 30 (today 17.9-ish)",
     rv_pct.reindex(v_s.index) <= 30),
]
rows = []
for lbl, m in splits:
    m = m.fillna(False).values
    st = summarize(v_s.values[m], f"SVXY | {lbl}")
    if st["n"]:
        st["signp"] = round(sign_test(int((v_s.values[m] > 0).sum()), int(m.sum())), 4)
    rows.append(st)
    st2 = summarize(v_s.values[~m], f"     ...complement")
    rows.append(st2)
show(rows, "SVXY h=1 conditioned on benign-state proxies")

print("\nC3. the JOINT benign cell -- contango>0 AND SPY within 3% of its high")
m = ((ts.reindex(v_s.index) > 0) & (dd_spy.reindex(v_s.index) >= -0.03)).fillna(False).values
st = summarize(v_s.values[m], "SVXY | joint benign")
st["signp"] = round(sign_test(int((v_s.values[m] > 0).sum()), int(m.sum())), 4)
show([st, summarize(v_s.values[~m], "SVXY | not benign")], "joint")
print("    benign-cell dates: " + ", ".join(
    f"{d.date()}:{100*r:+.2f}" for d, r in v_s[m].items()))
if m.sum() >= 3:
    print(f"    bootstrap P(mean<=0) on the benign cell = "
          f"{bootstrap_p_le0(v_s.values[m]):.3f}")

print("\nC4. the same proxies on the SHORT-^VIX leg (n=45, full coverage)")
rows = []
for lbl, mm in (("contango > 0", ts.reindex(v_v.index) > 0),
                ("SPY within 3% of high", dd_spy.reindex(v_v.index) >= -0.03),
                ("realised-vol pctile <= 30", rv_pct.reindex(v_v.index) <= 30)):
    mm = mm.fillna(False).values
    st = summarize(v_v.values[mm], f"-^VIX | {lbl}")
    st["signp"] = round(sign_test(int((v_v.values[mm] > 0).sum()), int(mm.sum())), 4)
    rows.append(st)
    rows.append(summarize(v_v.values[~mm], "     ...complement"))
rows.append(summarize(v_v.values, "-^VIX | all gated anchors"))
show(rows, "short ^VIX h=1")
