"""C8 round 1 — long SPY against short QQQ on a tech-absent-from-the-highs trigger.

Live: SPY -1.56% off its 52w high, QQQ -4.28% (gap +2.72pp); QQQ r63 rank 17.5
vs SPY 30.2; XLK r63 rank 28.2 vs XLV 99.6; zero tech names in the 20-name
new-high list.

Round-1 obligations discharged here:
  0. fingerprint collision against the 2026-08-11 QQQ/SPY pitch, computed
  1. gate-OFF FIRST: each half of the trigger alone on the pair
  2. battery() vs three controls, three trigger forms
  3. LEG ATTRIBUTION (2026-08-07 + -08-19 registry): what each leg earns
     against its OWN drift, equal-dollar vs beta-neutral residual, and
     whether the naked single leg BEATS the pair
  4. cost at 4 bps two-leg, era, concentration
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BAR = pd.Timestamp("2026-08-21")

SECT9 = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
NAMES = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLV"] + SECT9
px_all = load_prices(sorted(set(NAMES)))
spy = px_all["SPY"]["Close"].dropna()
CAL = spy.index[spy.index <= BAR]
px = pd.DataFrame({t: px_all[t]["Close"] for t in set(NAMES)}).reindex(CAL)


def dist_52wh(c, look=252):
    return c / c.rolling(look).max() - 1.0


spy_d = dist_52wh(px_all["SPY"]["Close"].dropna()).reindex(CAL)
qqq_d = dist_52wh(px_all["QQQ"]["Close"].dropna()).reindex(CAL)
xlk_d = dist_52wh(px_all["XLK"]["Close"].dropna()).reindex(CAL)
gap = spy_d - qqq_d

r63_spy = pct_rank(px_all["SPY"]["Close"].dropna(), 63).reindex(CAL)
r63_qqq = pct_rank(px_all["QQQ"]["Close"].dropna(), 63).reindex(CAL)

print("=" * 100)
print("0. FINGERPRINT COLLISION with the 2026-08-11 pitch, computed not asserted")
print("=" * 100)


def fp(legs, entry, h):
    ls = sorted(f"{t.upper()}:{s.upper()}" for t, s in legs)
    bucket = "short" if h <= 10 else "long"
    raw = "|".join(ls) + f"|{entry}|{bucket}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


prior = "e9e5534ea788239f"   # 2026-08-11 "Own the Nasdaq against the index"
for entry in ("MOC", "MOO", "LIMIT@CLOSE"):
    a = fp([("QQQ", "LONG"), ("SPY", "SHORT")], entry, 5)
    b = fp([("SPY", "LONG"), ("QQQ", "SHORT")], entry, 5)
    print(f"  entry={entry:12s} 08-11 side (QQQ long) -> {a}   "
          f"C8 side (SPY long) -> {b}   {'MATCHES PRIOR' if a == prior else ''}")
print(f"  recent_fingerprints blocked_since 2026-08-10 contains {prior} (2026-08-11).")
print("  NOTE: fingerprint() keys on TICKER:SIDE, so the opposite side is a "
      "DIFFERENT hash -> the code-level repeat block does NOT fire. The "
      "structural pair is the same object; changed_since is owed on the merits.")

print(f"\n  live state: SPY {100*spy_d.iloc[-1]:+.2f}%  QQQ {100*qqq_d.iloc[-1]:+.2f}%  "
      f"gap {100*gap.iloc[-1]:+.2f}pp  XLK {100*xlk_d.iloc[-1]:+.2f}%  "
      f"r63 SPY {r63_spy.iloc[-1]:.1f} QQQ {r63_qqq.iloc[-1]:.1f}")
gap_pit = rolling_on_valid(gap, lambda x: x.rolling(252).rank(pct=True) * 100).iloc[-1]
print(f"  gap PIT trailing-252 pctile = {gap_pit:.1f}   "
      f"FULL-SAMPLE pctile = {100*(gap <= gap.iloc[-1]).mean():.1f}")

# ---------------------------------------------------------------- triggers
GAP_LIVE = float(gap.iloc[-1])
TRIG = {
    "A gap>=2.72pp & SPY>-3%": (gap >= GAP_LIVE - 1e-9) & (spy_d > -0.03),
    "B r63 QQQ<=20 & SPY>=25": (r63_qqq <= 20) & (r63_spy >= 25),
    "C XLK<=-3% off high, >=2 of the other 8 sectors AT a high, SPY>-3%": None,
}
other8 = [s for s in SECT9 if s != "XLK"]
cnt = pd.Series(0.0, index=CAL)
den = pd.Series(0.0, index=CAL)
for s in other8:
    c = px_all[s]["Close"].dropna()
    c = c[c.index <= BAR]
    dd = dist_52wh(c)
    f = (dd >= -0.0025).astype(float)
    f[dd.isna()] = np.nan
    f = f.reindex(CAL)
    ok = f.notna()
    cnt[ok] += f[ok].values
    den[ok] += 1.0
TRIG["C XLK<=-3% off high, >=2 of the other 8 sectors AT a high, SPY>-3%"] = (
    (xlk_d <= -0.03) & (cnt >= 2) & (spy_d > -0.03))
print(f"  today: other-8 sectors at a high = {cnt.iloc[-1]:.0f} of {den.iloc[-1]:.0f} "
      f"(XLB, XLE);  XLK dist {100*xlk_d.iloc[-1]:+.2f}% "
      f"-> trigger C fires today: {bool(TRIG[list(TRIG)[2]].iloc[-1])}")
for k, v in TRIG.items():
    print(f"  {k:62s} fires today = {bool(v.iloc[-1])}   days = {int(v.sum())}")

PAIR = [("SPY", 1.0), ("QQQ", -1.0)]

# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("1. GATE-OFF FIRST: each half alone on the equal-dollar pair")
print("=" * 100)
halves = {
    "gap>=2.72pp ALONE": (gap >= GAP_LIVE - 1e-9),
    "SPY>-3% off high ALONE": (spy_d > -0.03),
    "r63 QQQ<=20 ALONE": (r63_qqq <= 20),
    "r63 SPY>=25 ALONE": (r63_spy >= 25),
    "XLK<=-3% off high ALONE": (xlk_d <= -0.03),
    ">=2 of other-8 sectors at a high ALONE": (cnt >= 2),
}
for h in (1, 3, 5, 10):
    ret = vehicle_ret(px, PAIR, h, 1)
    valid = ret.notna()
    rows = [summarize(ret[valid].values, f"CTRL-b all days (N={int(valid.sum())})")]
    for lbl, m in halves.items():
        mm = valid & m.reindex(CAL, fill_value=False)
        rows.append(summarize(ret[mm].values, f"{lbl} (N={int(mm.sum())})"))
    for lbl, m in TRIG.items():
        mm = valid & m.reindex(CAL, fill_value=False)
        rows.append(summarize(ret[mm].values, f"FULL {lbl[:24]} (N={int(mm.sum())})"))
    show(rows, f"h={h} td, long SPY / short QQQ equal dollar, day level")

# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("2. BATTERY on each trigger form, h=5, pair")
print("=" * 100)
for lbl, m in TRIG.items():
    battery(px, m, PAIR, 5, f"C8 {lbl}", 2.0, min_gap=10)

# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("3. LEG ATTRIBUTION — the 2026-08-07 / -08-19 registry test")
print("=" * 100)
# beta of SPY on QQQ, daily, full sample
d_spy = px["SPY"].pct_change()
d_qqq = px["QQQ"].pct_change()
ok = d_spy.notna() & d_qqq.notna()
BETA = float(np.polyfit(d_qqq[ok], d_spy[ok], 1)[0])
print(f"  measured beta of SPY on QQQ (daily, full sample) = {BETA:.3f}")
print(f"  => equal-dollar SPY-QQQ carries {BETA-1:+.3f} units of QQQ beta "
      f"(a SHORT-beta bet), the 2026-08-19 decomposition")

for lbl, m in TRIG.items():
    for h in (3, 5, 10):
        ret_pair = vehicle_ret(px, PAIR, h, 1)
        valid = ret_pair.dropna().index
        t = CAL[m.reindex(CAL, fill_value=False).values].intersection(valid)
        if len(t) == 0:
            continue
        epi = declusters(t, 10, valid)
        rows = []
        for name, legs in (("LONG leg: SPY alone", [("SPY", 1.0)]),
                           ("SHORT leg: -QQQ alone", [("QQQ", -1.0)]),
                           ("equal-dollar pair", PAIR),
                           (f"beta-neutral resid SPY-{BETA:.2f}*QQQ",
                            [("SPY", 1.0), ("QQQ", -BETA)])):
            r = vehicle_ret(px, legs, h, 1)
            s = summarize(r.loc[epi].values, name)
            base = r.loc[r.dropna().index].mean()
            s["own_drift_pct"] = round(100 * base, 3)
            s["excess_pp"] = round(s["mean_pct"] - 100 * base, 3)
            rows.append(s)
        show(rows, f"{lbl[:40]} | h={h}, N_epi={len(epi)}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("4. DOES THE NAKED LEG BEAT THE PAIR? (2026-08-19: 'if anything ever "
      "survives here it is not a pair trade')")
print("=" * 100)
for lbl, m in TRIG.items():
    rows = []
    for h in (1, 2, 3, 5, 7, 10):
        ret_pair = vehicle_ret(px, PAIR, h, 1)
        valid = ret_pair.dropna().index
        t = CAL[m.reindex(CAL, fill_value=False).values].intersection(valid)
        epi = declusters(t, 10, valid)
        if len(epi) == 0:
            continue
        row = {"h": h, "n_epi": len(epi)}
        for name, legs in (("pair", PAIR), ("SPY_only", [("SPY", 1.0)]),
                           ("QQQ_short_only", [("QQQ", -1.0)]),
                           ("QQQ_LONG_only", [("QQQ", 1.0)])):
            r = vehicle_ret(px, legs, h, 1)
            row[name] = round(100 * r.loc[epi].mean(), 3)
        rows.append(row)
    print(f"\n  --- {lbl}")
    print(pd.DataFrame(rows).to_string(index=False))
