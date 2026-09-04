"""B4 round 1 -- the decision that IS the settle.

42 of 212 FOMC decisions land ON a VIX expiry and 2026-09-16 is one. The
parents are closed (post-FOMC and post-VIX-expiry vol cells swept and empty;
VIX-expiry-week drift dead as mid-month position plus noise; SVXY as a
pre-FOMC leg closed on a 0.78 correlation), so this only lives as the
COINCIDENCE. Charged four ways:

  1. count + WHAT MAKES A DECISION COINCIDENT (the confound: both events are
     mid-month Wednesdays, so the split may be trading-day-of-month in
     disguise -- the exact kill the registry already applied to
     VIX-expiry-week drift)
  2. the run-in split: does the coincidence change the dec-10 -> decision
     window on SPY and the vol complex?
  3. tdom-matched control for the coincident set
  4. the 14:00 ET placebo. VIX settlement (VRO) is struck off SPX OPENING
     prints on the expiry morning, so settle flow resolves in the OPENING
     auction; the FOMC prints at 14:00. If the coincidence is a flow object
     it has to show up OVERNIGHT / at the open, not in the afternoon.
     Registry: "the largest overnight premium in the whole study is SPY on
     FOMC day (+13.5 bps tdom-matched, hit 64.2%, sign p 0.0000), and FOMC
     prints at 14:00 ... the overnight premium is a session-of-day effect,
     not an event effect."
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

ASOF = pd.Timestamp("2026-08-31")
H, K_ENTRY = 10, -10

TICK = ["SPY", "^VIX", "SVXY", "QQQ", "IWM"]
px = load_prices(TICK)
S = {t: px[t]["Close"].dropna()[lambda s: s.index <= ASOF] for t in px}
OPN = {t: px[t]["Open"].dropna()[lambda s: s.index <= ASOF] for t in px}

ev = load_events(["fomc_decision", "vix_expiry", "opex"])
FOM = pd.DatetimeIndex(sorted(ev.loc[ev.event == "fomc_decision", "date"].unique()))
FOM = FOM[FOM <= ASOF]
VX = pd.DatetimeIndex(sorted(ev.loc[ev.event == "vix_expiry", "date"].unique()))
VXset = set(VX)
COIN = pd.DatetimeIndex([d for d in FOM if d in VXset])
NONC = FOM.difference(COIN)
print("FOMC %d | vix_expiry %d | COINCIDENT %d | non-coincident %d"
      % (len(FOM), len(VX[VX <= ASOF]), len(COIN), len(NONC)))
print("coincident dates:", ", ".join(str(d.date()) for d in COIN[-8:]), "(last 8)")


spy = S["SPY"]
tdom = pd.Series(range(len(spy.index)), index=spy.index)
# trading day of month for every session
tdm = spy.index.to_series().groupby([spy.index.year, spy.index.month]).cumcount() + 1
tdm = pd.Series(tdm.values, index=spy.index)

print("\n" + "=" * 78)
print("1. WHAT MAKES A DECISION COINCIDENT?  (the confound test)")
print("=" * 78)
for lbl, dd in (("coincident", COIN), ("non-coincident", NONC)):
    t = tdm.reindex(dd).dropna()
    print("  %-15s n=%d | trading-day-of-month: min %d p25 %.0f median %.0f p75 %.0f max %d"
          % (lbl, len(t), t.min(), np.percentile(t, 25), np.median(t),
             np.percentile(t, 75), t.max()))
    print("                  month histogram: %s"
          % dict(pd.Series(pd.DatetimeIndex(dd).month).value_counts().sort_index()))
print("  -> if the two sets sit at different month positions, the split is")
print("     trading-day-of-month wearing an event label (registry: the")
print("     VIX-expiry-week cell died exactly this way).")


def window(s, anchors, k=K_ENTRY, h=H):
    pos, kept = anchor_positions(s.index, anchors, offset=k)
    d, e, v = [], [], []
    a = s.values
    for p, dd in zip(pos, kept):
        if p < 0 or p + h >= len(a):
            continue
        d.append(dd)
        e.append(s.index[p])
        v.append(a[p + h] / a[p] - 1.0)
    return pd.DatetimeIndex(d), pd.DatetimeIndex(e), np.asarray(v, float)


def drift(s, h=H, span=None):
    r = (s.shift(-h) / s - 1.0).dropna()
    if span:
        r = r[(r.index >= span[0]) & (r.index <= span[1])]
    return r.values


print("\n" + "=" * 78)
print("2. DOES THE COINCIDENCE CHANGE THE RUN-IN?  dec-10 close -> decision close")
print("=" * 78)
rows = []
for t in TICK:
    s = S[t]
    for lbl, dd in (("coincident", COIN), ("non-coinc", NONC)):
        d, e, v = window(s, dd)
        if len(v) < 5:
            rows.append({"tic": t, "set": lbl, "n": len(v)})
            continue
        b = drift(s, H, (d[0], d[-1]))
        w = int((v > 0).sum())
        rows.append({"tic": t, "set": lbl, "n": len(v), "mean_pct": 100 * v.mean(),
                     "drift_pct": 100 * b.mean(), "edge_pp": 100 * (v.mean() - b.mean()),
                     "hit": 100 * w / len(v), "sign_p": sign_test(w, len(v))})
show(rows, "run-in by coincidence")
for t in TICK:
    a = [r for r in rows if r["tic"] == t and r["set"] == "coincident"][0]
    b_ = [r for r in rows if r["tic"] == t and r["set"] == "non-coinc"][0]
    if a.get("n", 0) < 5 or b_.get("n", 0) < 5:
        continue
    print("  %-5s coincident minus non-coincident edge = %+.3fpp"
          % (t, a["edge_pp"] - b_["edge_pp"]))

print("\n" + "=" * 78)
print("3. TDOM-MATCHED CONTROL for the coincident set")
print("=" * 78)
d, e, v = window(spy, COIN)
tt = tdm.reindex(e).values
lo, hi = int(np.percentile(tt, 5)), int(np.percentile(tt, 95))
r10 = (spy.shift(-H) / spy - 1.0).dropna()
mm = tdm.reindex(r10.index)
matched = r10[((mm >= lo) & (mm <= hi)).values]
matched = matched[(matched.index >= e[0]) & (matched.index <= e[-1])]
print("  coincident entry sessions sit at tdom %d..%d" % (lo, hi))
print("  SPY coincident run-in %+.3f%% (n=%d) vs tdom-matched all-days %+.3f%% (n=%d)"
      % (100 * v.mean(), len(v), 100 * matched.mean(), len(matched)))
print("  -> tdom-matched edge %+.3fpp (against the naive all-days edge %+.3fpp)"
      % (100 * (v.mean() - matched.mean()),
         100 * (v.mean() - drift(spy, H, (e[0], e[-1])).mean())))

print("\n" + "=" * 78)
print("4. THE 14:00 ET PLACEBO -- decompose the DECISION SESSION itself.")
print("   VRO settles off the OPENING prints; the decision prints at 14:00.")
print("=" * 78)
cv = spy.values
ov = OPN["SPY"].reindex(spy.index).values
pos_all = pd.Series(range(len(spy.index)), index=spy.index)


def decompose(dates, label):
    on, intr, full = [], [], []
    for dd in pd.DatetimeIndex(dates):
        p = pos_all.get(dd)
        if p is None or p < 1:
            continue
        if np.isnan(ov[p]) or np.isnan(cv[p]) or np.isnan(cv[p - 1]):
            continue
        on.append(ov[p] / cv[p - 1] - 1.0)
        intr.append(cv[p] / ov[p] - 1.0)
        full.append(cv[p] / cv[p - 1] - 1.0)
    on, intr, full = map(lambda x: np.array(x, float), (on, intr, full))
    w = int((on > 0).sum())
    return {"set": label, "n": len(on),
            "overnight_bps": 1e4 * on.mean(), "on_hit": 100 * (on > 0).mean(),
            "on_sign_p": sign_test(w, len(on)),
            "intraday_bps": 1e4 * intr.mean(), "intra_hit": 100 * (intr > 0).mean(),
            "session_bps": 1e4 * full.mean()}


VXonly = pd.DatetimeIndex([d for d in VX if d <= ASOF and d not in set(FOM)])
rows = [decompose(COIN, "FOMC & VIX expiry"),
        decompose(NONC, "FOMC only"),
        decompose(VXonly, "VIX expiry only"),
        decompose(spy.index[1:], "ALL sessions")]
show(rows, "decision-session decomposition, SPY (bps)")
print("  The settle story predicts the coincident set's EXTRA return lands")
print("  OVERNIGHT / in the opening auction. Read the overnight column:")
c, f = rows[0], rows[1]
print("    coincident overnight %.1f bps vs FOMC-only %.1f bps -> difference %.1f bps"
      % (c["overnight_bps"], f["overnight_bps"], c["overnight_bps"] - f["overnight_bps"]))
print("    coincident intraday  %.1f bps vs FOMC-only %.1f bps -> difference %.1f bps"
      % (c["intraday_bps"], f["intraday_bps"], c["intraday_bps"] - f["intraday_bps"]))
print("    VIX-expiry-only sessions overnight %.1f bps vs ALL sessions %.1f bps"
      % (rows[2]["overnight_bps"], rows[3]["overnight_bps"]))

print("\n" + "=" * 78)
print("5. VOL COMPLEX on the decision session and the day after")
print("=" * 78)
vix = S["^VIX"]
pv = pd.Series(range(len(vix.index)), index=vix.index)
rows = []
for lbl, dd in (("FOMC & VIX exp", COIN), ("FOMC only", NONC), ("VIX exp only", VXonly)):
    d1, d2 = [], []
    for x in pd.DatetimeIndex(dd):
        p = pv.get(x)
        if p is None or p < 1 or p + 1 >= len(vix):
            continue
        d1.append(vix.values[p] / vix.values[p - 1] - 1.0)
        d2.append(vix.values[p + 1] / vix.values[p] - 1.0)
    rows.append({"set": lbl, "n": len(d1), "vix_day_pct": 100 * np.mean(d1),
                 "vix_next_pct": 100 * np.mean(d2),
                 "day_hit": 100 * np.mean(np.array(d1) > 0)})
show(rows, "^VIX change on the event session and the next")
print("\n  (registry: post-FOMC and post-VIX-expiry vol cells are already swept")
print("   and empty; this table exists only to check the COINCIDENCE adds")
print("   something the parents did not have.)")
