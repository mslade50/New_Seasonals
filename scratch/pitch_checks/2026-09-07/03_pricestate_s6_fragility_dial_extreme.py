"""S6 -- the fragility dial at an extreme, and rising fast.

Sizing statistic per CLAUDE.md: the 10-day MA of the 63d column of
data/rd2_fragility.parquet. Live 2026-09-04 it reads 87.96, against 57.18
twenty-one sessions earlier -- a +30.8 point rise.

VINTAGE CAVEAT, carried in the output and load-bearing for anything built on
this: the parquet is APPEND-ONLY point-in-time only since 2026-07-02. Rows
before that date are a RECOMPUTE vintage that drifted up to ~7 points on the
63d dial. So every episode here except the live 2026 one is measured on a
series that had partial lookahead in its construction. The rd2_fragility_ts
sibling is a raw-basis full recompute and is explicitly never a sizing
fallback, so it is not substituted here either.

Cells (SPY, lag=1, declustered at h):
  (a) 10d-MA-63d in its top decile of the series to date (expanding, PIT-ish)
  (b) 10d-MA-63d in its top decile of the FULL series (in-sample, for contrast)
  (c) the dial having RISEN more than 25 points over 21 sessions -- the live
      condition
  (d) the conjunction of (a) and (c)
Also: the 5d and 21d columns as alternative extremes, and a short-SPY read of
the same masks, because the direction is not assumed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _survey_lib import (  # noqa: E402
    align, cell, declusters, era_split, fwd_lag, hscan, load_prices, np, pd,
    show, sign_test, summarize, bootstrap_p_le0, local_control, cluster_note,
)

ROOT = Path(__file__).resolve().parents[3]
FRAG = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
PX = load_prices(["SPY", "QQQ", "IWM"])
IDX = PX["SPY"].index
C = {t: PX[t]["Close"] for t in PX}

print("=" * 78)
print("S6  FRAGILITY DIAL AT AN EXTREME  (asof 2026-09-04, entry lag=1)")
print("=" * 78)
print("\n*** VINTAGE CAVEAT ***")
print("  data/rd2_fragility.parquet is APPEND-ONLY POINT-IN-TIME only since")
print("  2026-07-02. Rows before that are a RECOMPUTE vintage that drifted up to")
print("  ~7 points on the 63d dial. Every episode below except the live 2026 one")
print("  is therefore measured on a partially-lookahead series. Treat every")
print("  number in this file as an upper bound on what was knowable at the time.")

ma = FRAG["63d"].rolling(10).mean()
ma.name = "ma10_63d"
print(f"\nDIAL SERIES: {FRAG.index[0].date()} .. {FRAG.index[-1].date()}, "
      f"{len(FRAG)} rows; 10d-MA defined on {int(ma.notna().sum())}")
print(f"  live 10d-MA-63d = {ma.iloc[-1]:.2f}   21 sessions ago = {ma.iloc[-22]:.2f}"
      f"   rise = {ma.iloc[-1] - ma.iloc[-22]:+.2f} pts   [brief claimed 88.0 vs 57.2]")
print(f"  percentile of the live value in the FULL series: "
      f"{100 * (ma.dropna() <= ma.iloc[-1]).mean():.2f}")
print(f"  raw 63d = {FRAG['63d'].iloc[-1]:.2f}  21d = {FRAG['21d'].iloc[-1]:.2f}  "
      f"5d = {FRAG['5d'].iloc[-1]:.2f}")

# ---------------------------------------------------------------- the masks
q90_exp = ma.expanding(252).quantile(0.90)
q90_full = ma.dropna().quantile(0.90)
rise21 = ma - ma.shift(21)

print(f"\n  top-decile cut, EXPANDING (live): {q90_exp.iloc[-1]:.2f}"
      f"    top-decile cut, FULL SAMPLE: {q90_full:.2f}")
print(f"  live rise21 = {rise21.iloc[-1]:+.2f} pts  -> in cell (c) rise>25: "
      f"{bool(rise21.iloc[-1] > 25)}")

m_top_exp = align((ma >= q90_exp).fillna(False), IDX).fillna(0).astype(bool)
m_top_full = align((ma >= q90_full).fillna(False), IDX).fillna(0).astype(bool)
m_rise = align((rise21 > 25).fillna(False), IDX).fillna(0).astype(bool)
m_conj = m_top_exp & m_rise
m_dial_defined = align(ma.notna(), IDX).fillna(0).astype(bool)

MASKS = {
    "(a) top decile, EXPANDING cut (PIT-ish)": m_top_exp,
    "(b) top decile, FULL-SAMPLE cut (in-sample)": m_top_full,
    "(c) dial ROSE >25 pts over 21 sessions": m_rise,
    "(d) conj: top decile AND rose >25": m_conj,
    "(ALL) every day the dial exists": m_dial_defined,
}
print("\nMASK COUNTS (over the SPY calendar, dial history 2016-07 onward):")
for k, m in MASKS.items():
    d = IDX[m.values]
    print(f"  {k:<46s} {len(d):5d} days"
          + (f"   {d[0].date()} .. {d[-1].date()}   yrs {sorted(set(d.year))}"
             if len(d) else ""))
print(f"  live day in (d): {bool(m_conj.iloc[-1])}")


def spy(h):
    return fwd_lag(C["SPY"], h, 1)


def spy_short(h):
    return -fwd_lag(C["SPY"], h, 1)


def qqq(h):
    return align(fwd_lag(C["QQQ"], h, 1), IDX)


def iwm(h):
    return align(fwd_lag(C["IWM"], h, 1), IDX)


# The dial-era baseline is the honest control: SPY 2016+ only, not 2000+.
DIAL_DAYS = IDX[m_dial_defined.values]
print(f"\n  dial-era SPY baseline window: {DIAL_DAYS[0].date()} .. {DIAL_DAYS[-1].date()}")

for mname, m in MASKS.items():
    if mname.startswith("(ALL)"):
        continue
    t = IDX[m.values]
    if len(t) == 0:
        print(f"\n{mname}: NO TRIGGERS")
        continue
    hscan(spy, t, f"SPY long  |  {mname}")

print("\n" + "=" * 78)
print("DIAL-ERA BASELINE (the control that matters: SPY over the dial's own span)")
print("=" * 78)
rows = []
for h in (1, 2, 3, 5, 10):
    r = spy(h)
    base_all = r.loc[r.dropna().index].mean()
    base_dial = r.loc[pd.DatetimeIndex(DIAL_DAYS).intersection(r.dropna().index)].mean()
    rows.append({"h": h, "SPY_all_days_2000+_pct": round(100 * base_all, 3),
                 "SPY_dial_era_2016+_pct": round(100 * base_dial, 3)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n" + "=" * 78)
print("EPISODE CELLS vs the DIAL-ERA baseline (not the 2000+ one)")
print("=" * 78)
rows = []
for h in (1, 2, 3, 5, 10):
    for vname, vf in (("SPY", spy), ("QQQ", qqq), ("IWM", iwm)):
        r = vf(h)
        valid = r.dropna().index
        dial_valid = pd.DatetimeIndex(DIAL_DAYS).intersection(valid)
        base_dial = float(r.loc[dial_valid].mean())
        for mname, m in MASKS.items():
            t = pd.DatetimeIndex(IDX[m.values]).intersection(valid)
            if len(t) == 0:
                continue
            epi = declusters(t, h, valid)
            ep = r.loc[epi].values
            w = int((ep > 0).sum())
            rows.append({"h": h, "veh": vname, "mask": mname, "n_days": len(t),
                         "n": len(epi), "mean_pct": round(100 * ep.mean(), 3),
                         "dial_era_base_pct": round(100 * base_dial, 3),
                         "edge_vs_dialera_pct": round(100 * (ep.mean() - base_dial), 3),
                         "hit": round(100 * (ep > 0).mean(), 1),
                         "worst_pct": round(100 * ep.min(), 2),
                         "rec": f"{w}-{len(epi) - w}",
                         "sign_p": round(sign_test(w, len(epi)), 4)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n" + "=" * 78)
print("FULL CELLS on the LIVE condition (d), SPY long and SPY short")
print("=" * 78)
for h in (5, 10):
    t = IDX[m_conj.values]
    if len(t) == 0:
        print(f"  (d) has no triggers at h={h}")
        continue
    cell(spy(h), t, h, "(d) top decile AND rose>25 -> SPY LONG")
    cell(spy_short(h), t, h, "(d) top decile AND rose>25 -> SPY SHORT")

print("\n" + "=" * 78)
print("ALTERNATIVE DIAL COLUMNS at their own top decile (5d and 21d), SPY h=5/h=10")
print("=" * 78)
rows = []
for col in ("5d", "21d", "63d"):
    s = FRAG[col]
    m = align((s >= s.dropna().quantile(0.90)).fillna(False), IDX).fillna(0).astype(bool)
    for h in (5, 10):
        r = spy(h)
        valid = r.dropna().index
        t = pd.DatetimeIndex(IDX[m.values]).intersection(valid)
        if len(t) == 0:
            continue
        epi = declusters(t, h, valid)
        ep = r.loc[epi].values
        base = float(r.loc[pd.DatetimeIndex(DIAL_DAYS).intersection(valid)].mean())
        w = int((ep > 0).sum())
        rows.append({"col": col + " raw (no 10d MA)", "h": h, "n_days": len(t),
                     "n": len(epi), "mean_pct": round(100 * ep.mean(), 3),
                     "edge_vs_dialera_pct": round(100 * (ep.mean() - base), 3),
                     "hit": round(100 * (ep > 0).mean(), 1),
                     "rec": f"{w}-{len(epi) - w}",
                     "sign_p": round(sign_test(w, len(epi)), 4)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n" + "=" * 78)
print("VINTAGE SPLIT: episodes BEFORE 2026-07-02 (recompute vintage) vs AFTER (PIT)")
print("=" * 78)
for h in (5, 10):
    r = spy(h)
    valid = r.dropna().index
    t = pd.DatetimeIndex(IDX[m_top_exp.values]).intersection(valid)
    epi = declusters(t, h, valid)
    pre = epi[epi < pd.Timestamp("2026-07-02")]
    post = epi[epi >= pd.Timestamp("2026-07-02")]
    show([summarize(r.loc[pre].values, f"h={h} recompute vintage (pre 2026-07-02)"),
          summarize(r.loc[post].values, f"h={h} true PIT (2026-07-02+)")],
         "(a) top-decile expanding, split by vintage")

print("\n=== S6 CELL COUNT AND COST ===")
print("  grid here: 3 vehicles x 4 masks x 5 horizons = 60 cells, plus 3 dial")
print("  columns x 2 horizons. Any pulse is UNCHARGED for the grid.")
print("  SPY round trip ~2-4 bps -> 3x bar ~+0.09% per episode.")
