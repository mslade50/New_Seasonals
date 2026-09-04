"""E3 - Macro-print density into an index high (interaction_cell).

Cell: SPY within 1% of its trailing-252d closing high AND >= 3 tier-1 macro
prints (nfp/cpi/ppi) falling in the 5 sessions AFTER the entry session.
Today: entry 2026-08-06, prints NFP d+1 (08-07), CPI d+4 (08-12), PPI d+5
(08-13) -> density 3.

Measure: SPY forward 3/5/6 sessions, MOO basis LEADS, vs
  (i) unconditional drift
  (ii) the control that matters: near-52w-high AND density < 3.

Adversarial brief:
 - multiplicity: another checker today swept 132 NFP-family cells and found
   the grid max |t| was beaten by a coin-flip grid ~25% of the time. This is
   one more look in that family. A bootstrap null is rerun here.
 - confound: print density is a mechanical function of the calendar month
   (NFP = 1st Friday, CPI/PPI = mid-month), so the cell may just be "the
   second week of the month". Tested directly.
 - episodes / LOYO / eras / worst window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _common as C

pd.set_option("display.width", 230)
pd.set_option("display.max_columns", 60)

EVENTS = C.ROOT / "data" / "macro_events.csv"
TIER1 = {"nfp", "cpi", "ppi"}
HORIZONS = [3, 5, 6]
RNG = np.random.default_rng(20260806)


def hdr(s: str) -> None:
    print("\n" + "=" * 100)
    print(s)
    print("=" * 100)


def episodes(dates, vals, gap_td: int):
    d = pd.DatetimeIndex(dates)
    v = np.asarray(vals, dtype=float)
    if len(d) == 0:
        return d, v
    m = C.declusterize(d, gap_td=gap_td).astype(bool)
    return d[m], v[m]


def loyo_floor(dates, vals):
    d = pd.DatetimeIndex(dates)
    v = np.asarray(vals, dtype=float)
    ok = np.isfinite(v)
    d, v = d[ok], v[ok]
    rows = []
    for y in sorted(set(d.year)):
        m = d.year != y
        if m.sum() < 3:
            continue
        rows.append((y, int(m.sum()), round(float(v[m].mean()), 3), round(C.tstat(v[m]), 2)))
    if not rows:
        return float("nan"), rows
    return min(r[3] for r in rows), rows


def per_year(dates, vals):
    s = pd.Series(np.asarray(vals, float), index=pd.DatetimeIndex(dates)).dropna()
    if s.empty:
        return pd.DataFrame()
    g = s.groupby(s.index.year)
    return pd.DataFrame({"n": g.size(), "avg": g.mean().round(3),
                         "sum": g.sum().round(2), "worst": g.min().round(2)})


# ----------------------------------------------------------------- data
hdr("E3.0  DATA")
ev = pd.read_csv(EVENTS)
ev["date"] = pd.to_datetime(ev["date"])
t1 = ev[ev["event"].isin(TIER1)].copy()
print(f"  macro_events rows {len(ev)}; tier-1 (nfp/cpi/ppi) {len(t1)} "
      f"{t1['date'].min().date()} .. {t1['date'].max().date()}")
print(t1["event"].value_counts().to_string())
print("\n  NOTE: the calendar file legitimately contains FUTURE scheduled dates "
      "(they are published in advance);\n  only the PRICE data is truncated at "
      f"{C.ASOF_EXCL.date()}.")

px = C.load(["SPY"])
spy = px["SPY"]
bars = spy.index
n = len(bars)
print(f"  SPY {bars.min().date()} .. {bars.max().date()} n={n}")

t1_dates = set(t1["date"].dt.normalize())
print(f"  tier-1 print dates upcoming: "
      f"{sorted(str(d.date()) for d in t1_dates if d >= pd.Timestamp('2026-08-01'))[:6]}")

# ----------------------------------------------------------------- density
# entry session E = D+1 (trading). window = sessions E+1 .. E+w (trading).
bar_arr = bars.to_numpy()


def density(w: int) -> pd.Series:
    """# tier-1 prints in calendar span [date(E+1), date(E+w)] where E=D+1."""
    out = np.full(n, np.nan)
    dts = np.array(sorted(t1_dates), dtype="datetime64[ns]")
    for i in range(n):
        lo_i, hi_i = i + 2, i + 1 + w
        if hi_i >= n:
            continue
        lo, hi = bar_arr[lo_i], bar_arr[hi_i]
        out[i] = int(((dts >= lo) & (dts <= hi)).sum())
    return pd.Series(out, index=bars)


def density_incl_entry(w: int) -> pd.Series:
    """Alternative: prints in [date(E), date(E+w-1)] - the w sessions the
    position is open counting the entry session itself."""
    out = np.full(n, np.nan)
    dts = np.array(sorted(t1_dates), dtype="datetime64[ns]")
    for i in range(n):
        lo_i, hi_i = i + 1, i + w
        if hi_i >= n:
            continue
        lo, hi = bar_arr[lo_i], bar_arr[hi_i]
        out[i] = int(((dts >= lo) & (dts <= hi)).sum())
    return pd.Series(out, index=bars)


den5 = density(5)
den5_incl = density_incl_entry(5)

hi252 = spy["Close"].rolling(252, min_periods=200).max()
near = (spy["Close"] / hi252 - 1.0) * 100.0

print(f"\n  TODAY (signal bar {C.LAST_BAR.date()}, entry {C.ASOF_EXCL.date()}):")
print(f"    SPY vs 252d closing high: {near.iloc[-1]:+.2f}%  (near-1% = "
      f"{bool(near.iloc[-1] >= -1.0)})")
# today's density has to be computed against the calendar directly (no future bars)
future_prints = sorted(d for d in t1_dates
                       if pd.Timestamp("2026-08-06") < d <= pd.Timestamp("2026-08-14"))
print(f"    tier-1 prints after the entry session through 08-14: "
      f"{[str(d.date()) for d in future_prints]}")
print(f"    density (5 sessions after entry, matching the brief) = 3")
print(f"    density (5 sessions INCLUDING entry)                 = 2")
print(f"  distribution of density(w=5) over history: "
      f"{den5.value_counts().sort_index().to_dict()}")
print(f"  distribution of density_incl(w=5):         "
      f"{den5_incl.value_counts().sort_index().to_dict()}")

FWD = {}
for k in HORIZONS + [10]:
    FWD[("moo", k)] = C.fwd_from_next_open(spy, k)
    FWD[("close", k)] = C.fwd(spy["Close"], k)

# ----------------------------------------------------------------- cells
ok = near.notna() & den5.notna()
cells = {
    "TRIGGER near1% & den>=3": ok & (near >= -1.0) & (den5 >= 3),
    "CONTROL near1% & den<3": ok & (near >= -1.0) & (den5 < 3),
    "near1% (any density)": ok & (near >= -1.0),
    "den>=3 (any level)": ok & (den5 >= 3),
    "ALL bars": ok,
}

hdr("E3.1  CELL SIZES")
for nm, m in cells.items():
    print(f"  {nm:26s} n={int(m.sum()):5d}  rate={m.mean()*100:5.2f}%")
dtrig = bars[cells["TRIGGER near1% & den>=3"].values]
print(f"\n  trigger dates span {dtrig.min().date()} .. {dtrig.max().date()}")
print(f"  by year: {pd.Series(1, index=dtrig).groupby(dtrig.year).sum().to_dict()}")

hdr("E3.2  MAIN GRID - SPY forward, MOO basis LEADS")
rows = []
for nm, m in cells.items():
    if nm == "ALL bars":
        continue
    dts = bars[m.values]
    for basis in ("moo", "close"):
        for k in HORIZONS:
            f = FWD[(basis, k)]
            v = f.reindex(dts).to_numpy()
            base = f.reindex(bars[cells["ALL bars"].values]).dropna().to_numpy()
            d = C.describe(f"{nm} | {basis} k={k}", v, base)
            for g in (10, 21):
                ed, evv = episodes(dts, v, g)
                d[f"ep{g}_n"] = int(np.isfinite(evv).sum())
                d[f"ep{g}_t"] = round(C.tstat(evv), 2)
            rows.append(d)
C.show(rows)

hdr("E3.3  THE CONTROL THAT MATTERS - trigger vs near-high-but-fewer-prints")
for basis in ("moo", "close"):
    for k in HORIZONS:
        f = FWD[(basis, k)]
        x = f.reindex(bars[cells["TRIGGER near1% & den>=3"].values]).dropna().to_numpy()
        y = f.reindex(bars[cells["CONTROL near1% & den<3"].values]).dropna().to_numpy()
        se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
        print(f"  {basis} k={k}: TRIG n={len(x):4d} avg={x.mean():+.3f} | "
              f"CTRL n={len(y):4d} avg={y.mean():+.3f} | diff {x.mean()-y.mean():+.3f}pp "
              f"Welch t={(x.mean()-y.mean())/se:+.2f}")

hdr("E3.4  DEEP DIVE - trigger cell, MOO basis")
dts = bars[cells["TRIGGER near1% & den>=3"].values]
for k in HORIZONS:
    f = FWD[("moo", k)]
    v = f.reindex(dts).to_numpy()
    print(f"\n  --- k={k} MOO ---")
    rr = [C.describe(f"TRIG k={k} signal-days", v,
                     f.dropna().to_numpy())]
    for g in (10, 21):
        ed, evv = episodes(dts, v, g)
        rr.append(C.describe(f"TRIG k={k} episodes gap{g}", evv))
    C.show(rr)
    ed, evv = episodes(dts, v, 10)
    floor, tbl = loyo_floor(ed, evv)
    print(f"  LOYO (episodes gap10) floor t = {floor}")
    print("   " + " ".join(f"{y}:{t}" for y, _, _, t in tbl))
    print("  era split (signal-days, cut 2018):")
    C.show(C.era_split(dts, v))
    print("  per-year:")
    print(per_year(dts, v).to_string())

hdr("E3.5  CONFOUND - is this just 'the second week of the month'?")
tdom = pd.Series(0, index=bars, dtype=int)
for _, grp in pd.Series(bars, index=bars).groupby([bars.year, bars.month]):
    tdom.loc[grp.index] = np.arange(1, len(grp) + 1)
print("  trading-day-of-month distribution INSIDE the trigger cell:")
tt = tdom.reindex(dts)
print("   " + tt.value_counts().sort_index().to_string().replace("\n", "  "))
print(f"  mean tdom in trigger = {tt.mean():.1f}  vs near-high control "
      f"{tdom.reindex(bars[cells['CONTROL near1% & den<3'].values]).mean():.1f}")

tdom_set = set(tt.unique())
matched = cells["CONTROL near1% & den<3"] & tdom.isin(tdom_set)
print(f"\n  TDOM-MATCHED control (near-high, density<3, tdom in "
      f"{sorted(tdom_set)}): n={int(matched.sum())}")
for k in HORIZONS:
    f = FWD[("moo", k)]
    x = f.reindex(dts).dropna().to_numpy()
    y = f.reindex(bars[matched.values]).dropna().to_numpy()
    se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
    print(f"   moo k={k}: TRIG n={len(x)} avg={x.mean():+.3f} | MATCHED CTRL "
          f"n={len(y)} avg={y.mean():+.3f} | diff {x.mean()-y.mean():+.3f}pp "
          f"Welch t={(x.mean()-y.mean())/se:+.2f}")

print("\n  Pure calendar cell (ignore density AND the high): tdom in the trigger's range")
for k in HORIZONS:
    f = FWD[("moo", k)]
    m = ok & tdom.isin(tdom_set)
    v = f.reindex(bars[m.values]).dropna().to_numpy()
    print(f"   moo k={k}: n={len(v)} avg={v.mean():+.3f} t={C.tstat(v):+.2f} "
          f"| unconditional avg={f.dropna().mean():+.3f}")

hdr("E3.6  IS IT JUST 'TOMORROW IS NFP'?")
nfp_arr = np.array(sorted(set(pd.to_datetime(ev[ev["event"] == "nfp"]["date"])
                              .dt.normalize())), dtype="datetime64[ns]")
nfp_flag = np.zeros(n, dtype=bool)
nfp_flag[:n - 2] = np.isin(bar_arr[2:], nfp_arr)
next_is_nfp = pd.Series(nfp_flag, index=bars)
print(f"  sanity: bars whose E+1 session is an NFP print: {int(next_is_nfp.sum())}")
sub = {
    "TRIG & next-sess-after-entry is NFP": cells["TRIGGER near1% & den>=3"] & next_is_nfp,
    "TRIG & not": cells["TRIGGER near1% & den>=3"] & ~next_is_nfp,
    "near1% & next-sess-after-entry NFP": cells["near1% (any density)"] & next_is_nfp,
}
print(f"  today: session after entry is 2026-08-07 = NFP -> True")
rows = []
for nm, m in sub.items():
    dd = bars[m.values]
    for k in HORIZONS:
        f = FWD[("moo", k)]
        v = f.reindex(dd).to_numpy()
        d = C.describe(f"{nm} k={k}", v)
        ed, evv = episodes(dd, v, 10)
        d["ep10_n"] = int(np.isfinite(evv).sum())
        d["ep10_t"] = round(C.tstat(evv), 2)
        rows.append(d)
C.show(rows)

hdr("E3.7  DENSITY-DEFINITION ROBUSTNESS (incl-entry-session variant, and den>=2)")
alt = {
    "near1% & den_incl>=3": ok & (near >= -1.0) & (den5_incl >= 3),
    "near1% & den_incl>=2": ok & (near >= -1.0) & (den5_incl >= 2),
    "near1% & den>=2": ok & (near >= -1.0) & (den5 >= 2),
    "near1% & den==0": ok & (near >= -1.0) & (den5 == 0),
}
rows = []
for nm, m in alt.items():
    dd = bars[m.values]
    for k in HORIZONS:
        f = FWD[("moo", k)]
        v = f.reindex(dd).to_numpy()
        d = C.describe(f"{nm} k={k} moo", v, f.dropna().to_numpy())
        ed, evv = episodes(dd, v, 10)
        d["ep10_n"] = int(np.isfinite(evv).sum())
        d["ep10_t"] = round(C.tstat(evv), 2)
        rows.append(d)
C.show(rows)

hdr("E3.8  NEAR-HIGH THRESHOLD ROBUSTNESS")
rows = []
for thr in (-0.5, -1.0, -2.0, -3.0):
    for k in HORIZONS:
        m = ok & (near >= thr) & (den5 >= 3)
        dd = bars[m.values]
        f = FWD[("moo", k)]
        v = f.reindex(dd).to_numpy()
        d = C.describe(f"near>={thr}% & den>=3 k={k}", v, f.dropna().to_numpy())
        ed, evv = episodes(dd, v, 10)
        d["ep10_n"] = int(np.isfinite(evv).sum())
        d["ep10_t"] = round(C.tstat(evv), 2)
        rows.append(d)
C.show(rows)

hdr("E3.9  MULTIPLICITY - grid max|t| vs a circular-shift null")
grid_masks = {}
for thr in (-0.5, -1.0, -2.0, -3.0):
    for dmin in (2, 3):
        grid_masks[f"near>={thr} den>={dmin}"] = (ok & (near >= thr) & (den5 >= dmin)).to_numpy()
grid_ts = []
for gname, gm in grid_masks.items():
    for basis in ("moo", "close"):
        for k in HORIZONS:
            f = FWD[(basis, k)].to_numpy()
            v = f[gm]
            grid_ts.append((gname, basis, k, int(np.isfinite(v).sum()),
                            round(float(np.nanmean(v)), 3), round(C.tstat(v), 2)))
gt = pd.DataFrame(grid_ts, columns=["cell", "basis", "k", "n", "avg", "t"])
gt["abs_t"] = gt["t"].abs()
print(gt.sort_values("abs_t", ascending=False).head(10).to_string(index=False))
obs_max = float(gt["abs_t"].max())
print(f"\n  cells in grid: {len(gt)}   observed max |t| = {obs_max:.2f}")

NDRAW = 2000
null_max = np.empty(NDRAW)
fwd_arrays = {(b, k): FWD[(b, k)].to_numpy() for b in ("moo", "close") for k in HORIZONS}
mask_list = list(grid_masks.values())
for i in range(NDRAW):
    sh = int(RNG.integers(252, n - 252))
    best = 0.0
    for gm in mask_list:
        gms = np.roll(gm, sh)
        for key, f in fwd_arrays.items():
            v = f[gms]
            t = abs(C.tstat(v))
            if np.isfinite(t) and t > best:
                best = t
    null_max[i] = best
print(f"  circular-shift null max|t| (masks shifted, return series fixed, "
      f"{NDRAW} draws):")
print(f"    median {np.median(null_max):.2f}  90th {np.quantile(null_max, .90):.2f}  "
      f"95th {np.quantile(null_max, .95):.2f}")
print(f"    P(null max|t| >= observed {obs_max:.2f}) = "
      f"{(null_max >= obs_max).mean():.4f}")

hdr("E3.10  EVERY TRIGGER DAY (audit trail)")
tab = pd.DataFrame({
    "near_52h_%": near.reindex(dts).round(2),
    "den5": den5.reindex(dts).astype(int),
    "tdom": tdom.reindex(dts),
    "fwd3_moo": FWD[("moo", 3)].reindex(dts).round(2),
    "fwd5_moo": FWD[("moo", 5)].reindex(dts).round(2),
    "fwd6_moo": FWD[("moo", 6)].reindex(dts).round(2),
})
print(tab.to_string())

hdr("E3.11  EVEN-HANDED SUPPLEMENT - the ADJACENT den>=2 cell (today also qualifies)")
print("  The pitched den>=3 cell is empty, but the grid's max |t| came from")
print("  near-high & den>=2 at k=6. Today has den>=3 so it qualifies for den>=2")
print("  too. Diligence: is THAT a survivor, or the turn-of-the-month effect?")
m2 = ok & (near >= -1.0) & (den5 >= 2)
d2 = bars[m2.values]
print(f"\n  n={len(d2)}  {d2.min().date()} .. {d2.max().date()}")
print("  tdom distribution inside den>=2 cell:")
print("   " + tdom.reindex(d2).value_counts().sort_index().to_string().replace("\n", "  "))

for k in HORIZONS:
    f = FWD[("moo", k)]
    v = f.reindex(d2).to_numpy()
    print(f"\n  --- den>=2 k={k} MOO ---")
    rr = [C.describe(f"den>=2 k={k} signal-days", v, f.dropna().to_numpy())]
    for g in (10, 21):
        ed, evv = episodes(d2, v, g)
        rr.append(C.describe(f"den>=2 k={k} episodes gap{g}", evv))
    C.show(rr)
    ed, evv = episodes(d2, v, 10)
    floor, tbl = loyo_floor(ed, evv)
    print(f"  LOYO (episodes gap10) floor t = {floor}")
    print("   " + " ".join(f"{y}:{t}" for y, _, _, t in tbl))
    C.show(C.era_split(d2, v))

print("\n  TURN-OF-MONTH DECOMPOSITION (den>=2 cell split by trading day of month)")
tdom2 = tdom.reindex(d2)
for k in HORIZONS:
    f = FWD[("moo", k)]
    rr = []
    for nm, sel in (("tdom<=3 (turn of month)", tdom2 <= 3),
                    ("tdom 4-7", (tdom2 >= 4) & (tdom2 <= 7)),
                    ("tdom>=8", tdom2 >= 8)):
        dd = d2[sel.values]
        v = f.reindex(dd).to_numpy()
        d = C.describe(f"den>=2 k={k} {nm}", v, f.dropna().to_numpy())
        ed, evv = episodes(dd, v, 10)
        d["ep10_n"] = int(np.isfinite(evv).sum())
        d["ep10_t"] = round(C.tstat(evv), 2)
        rr.append(d)
    C.show(rr)

print("\n  TDOM-MATCHED: near-high days with the SAME tdom mix but den<2")
tdom_vals2 = set(tdom2.unique())
matched2 = ok & (near >= -1.0) & (den5 < 2) & tdom.isin(tdom_vals2)
for k in HORIZONS:
    f = FWD[("moo", k)]
    x = f.reindex(d2).dropna().to_numpy()
    y = f.reindex(bars[matched2.values]).dropna().to_numpy()
    se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
    print(f"   moo k={k}: den>=2 n={len(x)} avg={x.mean():+.3f} | matched den<2 "
          f"n={len(y)} avg={y.mean():+.3f} | diff {x.mean()-y.mean():+.3f}pp "
          f"Welch t={(x.mean()-y.mean())/se:+.2f}")

print("\n  PLAIN turn-of-month (no density, no near-high): tdom<=3 vs rest")
for k in HORIZONS:
    f = FWD[("moo", k)]
    a = f.reindex(bars[(ok & (tdom <= 3)).values]).dropna().to_numpy()
    b = f.reindex(bars[(ok & (tdom > 3)).values]).dropna().to_numpy()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    print(f"   moo k={k}: tdom<=3 n={len(a)} avg={a.mean():+.3f} t={C.tstat(a):+.2f} | "
          f"tdom>3 n={len(b)} avg={b.mean():+.3f} | diff Welch t={(a.mean()-b.mean())/se:+.2f}")
print("\n  ... and PLAIN turn-of-month by era (the registry says these cells died post-2013):")
for k in HORIZONS:
    f = FWD[("moo", k)]
    s = f.reindex(bars[(ok & (tdom <= 3)).values]).dropna()
    for lo, hi in ((2000, 2012), (2013, 2026)):
        v = s[(s.index.year >= lo) & (s.index.year <= hi)].to_numpy()
        print(f"   moo k={k} tdom<=3 {lo}-{hi}: n={len(v)} avg={v.mean():+.3f} "
              f"t={C.tstat(v):+.2f}")

hdr("E3.12  EXACT-TDOM STRATIFIED TEST of the den>=2 cell + print ladder")
print("  The tdom-SET match above is loose (the den>=2 cell piles up at tdom 4-7).")
print("  Stratify on the EXACT trading day of month and pool the within-stratum")
print("  differences - if density is a calendar proxy this collapses.\n")
for k in HORIZONS:
    f = FWD[("moo", k)]
    num = den = 0.0
    var = 0.0
    rows = []
    for td in sorted(set(tdom.reindex(d2).unique())):
        sel_hi = ok & (near >= -1.0) & (den5 >= 2) & (tdom == td)
        sel_lo = ok & (near >= -1.0) & (den5 < 2) & (tdom == td)
        x = f.reindex(bars[sel_hi.values]).dropna().to_numpy()
        y = f.reindex(bars[sel_lo.values]).dropna().to_numpy()
        if len(x) < 5 or len(y) < 5:
            continue
        w = len(x)
        diff = x.mean() - y.mean()
        se2 = x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y)
        num += w * diff
        den += w
        var += (w ** 2) * se2
        rows.append((int(td), len(x), len(y), round(x.mean(), 3), round(y.mean(), 3),
                     round(diff, 3)))
    pooled = num / den if den else float("nan")
    pooled_t = pooled / (np.sqrt(var) / den) if den else float("nan")
    print(f"  --- k={k} MOO, stratified by exact tdom ---")
    print(pd.DataFrame(rows, columns=["tdom", "n_den2+", "n_den<2", "avg_den2+",
                                      "avg_den<2", "diff"]).to_string(index=False))
    print(f"   POOLED within-tdom difference = {pooled:+.3f}pp  t = {pooled_t:+.2f}\n")

print("  PRINT LADDER within near-52w-high days (MOO):")
rows = []
for dv, sel in (("den==0", den5 == 0), ("den==1", den5 == 1),
                ("den==2", den5 == 2), ("den>=3", den5 >= 3)):
    m = ok & (near >= -1.0) & sel
    dd = bars[m.values]
    for k in HORIZONS:
        f = FWD[("moo", k)]
        v = f.reindex(dd).to_numpy()
        d = C.describe(f"near1% {dv} k={k}", v, f.dropna().to_numpy())
        ed, evv = episodes(dd, v, 10)
        d["ep10_n"] = int(np.isfinite(evv).sum())
        d["ep10_t"] = round(C.tstat(evv), 2)
        rows.append(d)
C.show(rows)

print("\n  TODAY'S OWN SUB-BUCKET: signal-bar tdom for 2026-08-05")
aug = bars[(bars.year == 2026) & (bars.month == 8)]
print(f"   August 2026 sessions so far: {[str(d.date()) for d in aug]}")
print(f"   signal bar {C.LAST_BAR.date()} tdom = {int(tdom.loc[C.LAST_BAR])}"
      f"  -> falls in the den>=2 cell's WEAKEST tdom bucket (<=3)")

hdr("E3.END")
