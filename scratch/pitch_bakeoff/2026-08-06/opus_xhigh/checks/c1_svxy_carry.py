"""C1 — adversarial kill check: LONG SVXY on steep VIX term-structure carry
into a new equity high.

Trigger under test: VIX/VIX3M <= 0.85 AND SPY within 1% of its trailing 252d
CLOSING high. Today (2026-08-05 close): ratio 0.834, dist -0.20%.

Kill vectors run here:
  (a) lift over SVXY's unconditional drift, era-matched
  (b) leverage-regime contamination (-1x pre 2018-02-15 vs -0.5x after)
  (c) clustering -> declustered episodes at gap 10 and 21, episodes named
  (d) conditional left tail: worst windows + intra-hold trough distribution
  (e) registry collision: in-cell corr(SVXY fwd, SPY fwd) and Event-Sleeve V4
      window overlap
  (f) calendar: CPI inside the hold
  (g) threshold grid 0.80/0.825/0.85/0.875/0.90 x near-high 0.5/1/3%

Executable basis is LEAD: entry MOO the session after the signal, exit MOC k
sessions later (C.fwd_from_next_open). Close-basis reported alongside.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _common as C

pd.set_option("display.width", 220)

HORIZONS = [3, 5, 10]
LEV_CUT = pd.Timestamp("2018-02-15")   # SVXY -1x -> -0.5x
RATIO_MAX = 0.85
NEARHIGH_PCT = 1.0


def hdr(s: str) -> None:
    print("\n" + "=" * 100)
    print(s)
    print("=" * 100)


px = C.load(["SVXY", "^VIX", "^VIX3M", "SPY"])
sv = px["SVXY"]
spy = px["SPY"]
vix = px["^VIX"]["Close"]
vix3m = px["^VIX3M"]["Close"]

ratio = (vix / vix3m.reindex(vix.index)).dropna()
spy_c = spy["Close"]
dist = (spy_c / spy_c.rolling(252).max() - 1.0) * 100.0   # <= 0, in percent

sig = pd.DataFrame(index=sv.index)
sig["ratio"] = ratio.reindex(sv.index)
sig["dist"] = dist.reindex(sv.index)
sig["svxy"] = sv["Close"]

# forward returns
for k in HORIZONS:
    sig[f"o{k}"] = C.fwd_from_next_open(sv, k)          # executable (MOO->MOC)
    sig[f"c{k}"] = C.fwd(sv["Close"], k)                # close-to-close
    sig[f"spy_o{k}"] = C.fwd_from_next_open(spy, k).reindex(sv.index)

# intra-hold trough vs the MOO entry (executable drawdown)
op = sv["Open"].to_numpy()
lo = sv["Low"].to_numpy()
n = len(sv)
for k in HORIZONS:
    dd = np.full(n, np.nan)
    for i in range(n - k):
        entry = op[i + 1]
        trough = np.nanmin(lo[i + 1: i + 1 + k])
        dd[i] = (trough / entry - 1.0) * 100.0
    sig[f"dd{k}"] = dd

sig["era"] = np.where(sig.index < LEV_CUT, "-1x (pre-2018-02)", "-0.5x (2018-02+)")

hdr("0. SANITY — today's reading and sample bounds")
print(f"SVXY bars {sv.index[0].date()} .. {sv.index[-1].date()}  N={len(sv)}")
last = sig.dropna(subset=["ratio", "dist"]).iloc[-1]
print(f"last usable bar {sig.dropna(subset=['ratio','dist']).index[-1].date()}: "
      f"VIX/VIX3M={last['ratio']:.4f}  SPY dist-to-252d-high={last['dist']:+.2f}%  "
      f"SVXY close={last['svxy']:.2f}")
print(f"trigger today: ratio<={RATIO_MAX} -> {last['ratio']<=RATIO_MAX} ; "
      f"dist>=-{NEARHIGH_PCT}% -> {last['dist']>=-NEARHIGH_PCT}")

base_mask = sig["ratio"].notna() & sig["dist"].notna()
cell = base_mask & (sig["ratio"] <= RATIO_MAX) & (sig["dist"] >= -NEARHIGH_PCT)
print(f"\nraw cell days (2011-10+): {int(cell.sum())} of {int(base_mask.sum())} "
      f"usable days = {cell.sum()/base_mask.sum()*100:.1f}% of the tape")

# ---------------------------------------------------------------- (a) lift
hdr("A. CELL vs UNCONDITIONAL DRIFT — executable MOO basis (lead), full sample")
rows = []
for k in HORIZONS:
    x = sig.loc[cell, f"o{k}"]
    b = sig.loc[base_mask, f"o{k}"]
    rows.append(C.describe(f"cell o{k}", x, baseline=b))
    rows.append(C.describe(f"ALL   o{k}", b))
C.show(rows)

print("\nWelch two-sample t (cell vs all-other-days), executable basis:")
wrows = []
for k in HORIZONS:
    x = sig.loc[cell, f"o{k}"].dropna().to_numpy()
    y = sig.loc[base_mask & ~cell, f"o{k}"].dropna().to_numpy()
    se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
    wrows.append({"h": f"o{k}", "n_cell": len(x), "n_other": len(y),
                  "cell_avg": round(x.mean(), 3), "other_avg": round(y.mean(), 3),
                  "diff": round(x.mean() - y.mean(), 3),
                  "welch_t": round((x.mean() - y.mean()) / se, 2)})
C.show(wrows)

hdr("A2. CLOSE-BASIS mirror (non-executable, for reference only)")
rows = []
for k in HORIZONS:
    rows.append(C.describe(f"cell c{k}", sig.loc[cell, f"c{k}"],
                           baseline=sig.loc[base_mask, f"c{k}"]))
C.show(rows)

# --------------------------------------------------------- (b) leverage era
hdr("B. LEVERAGE-REGIME SPLIT — pre-2018-02-15 is a DIFFERENT INSTRUMENT (-1x)")
for k in HORIZONS:
    rows = []
    for era in ["-1x (pre-2018-02)", "-0.5x (2018-02+)"]:
        m = cell & (sig["era"] == era)
        bm = base_mask & (sig["era"] == era)
        rows.append(C.describe(f"o{k} cell {era}", sig.loc[m, f"o{k}"],
                               baseline=sig.loc[bm, f"o{k}"]))
        rows.append(C.describe(f"o{k} ALL  {era}", sig.loc[bm, f"o{k}"]))
    C.show(rows)
    print()

# ------------------------------------------------------------ (c) episodes
hdr("C. DECLUSTERED EPISODES (the cell persists for weeks)")
cell_dates = sig.index[cell]
for gap in [10, 21]:
    keep = C.declusterize(cell_dates, gap_td=gap)
    ep = cell_dates[keep]
    print(f"\n--- gap={gap} td : {len(ep)} episodes from {int(cell.sum())} raw days")
    rows = []
    for k in HORIZONS:
        v = sig.loc[ep, f"o{k}"]
        rows.append(C.describe(f"episodes o{k} (gap{gap})", v,
                               baseline=sig.loc[base_mask, f"o{k}"]))
        v18 = sig.loc[[d for d in ep if d >= LEV_CUT], f"o{k}"]
        rows.append(C.describe(f"episodes o{k} 2018+ only", v18,
                               baseline=sig.loc[base_mask & (sig["era"] ==
                                                             "-0.5x (2018-02+)"),
                                                f"o{k}"]))
    C.show(rows)
    if gap == 21:
        print("\nEPISODE ROSTER (gap 21):")
        er = pd.DataFrame({
            "date": [d.date() for d in ep],
            "era": sig.loc[ep, "era"].to_numpy(),
            "ratio": sig.loc[ep, "ratio"].round(3).to_numpy(),
            "dist%": sig.loc[ep, "dist"].round(2).to_numpy(),
            "o3": sig.loc[ep, "o3"].round(2).to_numpy(),
            "o5": sig.loc[ep, "o5"].round(2).to_numpy(),
            "o10": sig.loc[ep, "o10"].round(2).to_numpy(),
            "dd5": sig.loc[ep, "dd5"].round(2).to_numpy(),
        })
        print(er.to_string(index=False))

# ------------------------------------------------------------- (d) the tail
hdr("D. CONDITIONAL LEFT TAIL — where a short-vol trade actually lives")
for k in HORIZONS:
    x = sig.loc[cell, f"o{k}"].dropna().sort_values()
    print(f"\no{k}: worst 5 windows in the cell")
    print(pd.DataFrame({"signal_date": [d.date() for d in x.index[:5]],
                        "ret%": x.iloc[:5].round(2).to_numpy(),
                        "era": sig.loc[x.index[:5], "era"].to_numpy()}
                       ).to_string(index=False))
    print(f"o{k} percentiles (cell) : "
          + "  ".join(f"p{p}={np.percentile(x, p):.2f}" for p in [1, 5, 10, 25, 50]))
    b = sig.loc[base_mask, f"o{k}"].dropna()
    print(f"o{k} percentiles (all)  : "
          + "  ".join(f"p{p}={np.percentile(b, p):.2f}" for p in [1, 5, 10, 25, 50]))

print("\nIntra-hold trough vs MOO entry (executable drawdown), cell vs all:")
rows = []
for k in HORIZONS:
    rows.append(C.describe(f"dd{k} cell", sig.loc[cell, f"dd{k}"],
                           baseline=sig.loc[base_mask, f"dd{k}"]))
    rows.append(C.describe(f"dd{k} ALL ", sig.loc[base_mask, f"dd{k}"]))
C.show(rows)
print("\nWorst intra-hold trough in the cell, 2018+ era only:")
for k in HORIZONS:
    m = cell & (sig["era"] == "-0.5x (2018-02+)")
    s = sig.loc[m, f"dd{k}"].dropna()
    if len(s):
        print(f"  dd{k}: worst {s.min():.2f}% on {s.idxmin().date()}  "
              f"| p5 {np.percentile(s,5):.2f}%  | median {s.median():.2f}%")

# ------------------------------------------------- (e) registry collision
hdr("E. REGISTRY COLLISION — is this a levered SPY long? (killed the pre-FOMC leg at 0.78)")
rows = []
for k in HORIZONS:
    sub = sig.loc[cell, [f"o{k}", f"spy_o{k}"]].dropna()
    r = float(np.corrcoef(sub[f"o{k}"], sub[f"spy_o{k}"])[0, 1])
    beta = float(np.polyfit(sub[f"spy_o{k}"], sub[f"o{k}"], 1)[0])
    allsub = sig.loc[base_mask, [f"o{k}", f"spy_o{k}"]].dropna()
    r_all = float(np.corrcoef(allsub[f"o{k}"], allsub[f"spy_o{k}"])[0, 1])
    # residual after hedging out SPY: does the cell add anything beyond beta*SPY?
    resid = sub[f"o{k}"] - beta * sub[f"spy_o{k}"]
    rows.append({"h": f"o{k}", "n": len(sub), "corr_in_cell": round(r, 3),
                 "corr_all_days": round(r_all, 3), "beta_vs_SPY": round(beta, 2),
                 "resid_avg": round(float(resid.mean()), 3),
                 "resid_t": round(C.tstat(resid.to_numpy()), 2)})
C.show(rows)

print("\nEvent-Sleeve V4 overlap (V4 = long SVXY, opex MOC -> +3 sessions MOC, ex-Sep):")
try:
    import sys
    sys.path.insert(0, str(C.ROOT))
    from macro_calendar import event_dates
    opex = pd.DatetimeIndex(event_dates("opex"))
    idx = sv.index
    v4days = set()
    for d in opex:
        if d.month == 9 or d not in idx:
            continue
        p = idx.get_loc(d)
        for j in range(p, min(p + 4, len(idx))):
            v4days.add(idx[j])
    ov = sum(1 for d in cell_dates if d in v4days)
    print(f"  cell days that sit inside a live V4 window: {ov} / {len(cell_dates)} "
          f"({ov/len(cell_dates)*100:.1f}%)")
    nonv4 = pd.DatetimeIndex([d for d in cell_dates if d not in v4days])
    rows = []
    for k in HORIZONS:
        rows.append(C.describe(f"cell o{k} EX-V4-window", sig.loc[nonv4, f"o{k}"],
                               baseline=sig.loc[base_mask, f"o{k}"]))
    C.show(rows)
    nxt = [d for d in opex if d > C.LAST_BAR and d.month != 9]
    print(f"  next V4 entry: {nxt[0].date()} (opex)")
except Exception as e:                                    # pragma: no cover
    print(f"  !! opex calendar unavailable: {e}")

# ---------------------------------------------------------------- (f) CPI
hdr("F. CALENDAR — does a CPI print inside the hold change the cell?")
try:
    ev = pd.read_csv(C.ROOT / "data" / "macro_events.csv", parse_dates=["date"])
    print("events available:", sorted(ev["event"].unique()))
    idx = sv.index
    pos = {d: i for i, d in enumerate(idx)}
    for evname in ["cpi", "nfp"]:
        dts = set(pd.DatetimeIndex(ev.loc[ev["event"] == evname, "date"]).normalize())
        if not dts:
            print(f"  (no {evname} rows)")
            continue
        rows = []
        for k in HORIZONS:
            insidef = []
            for d in idx:
                i = pos[d]
                if i + k >= len(idx):
                    insidef.append(np.nan)
                    continue
                win = idx[i + 1: i + 1 + k]
                insidef.append(any(w in dts for w in win))
            has = pd.Series(insidef, index=idx)
            m1 = cell & (has == True)                       # noqa: E712
            m0 = cell & (has == False)                      # noqa: E712
            rows.append(C.describe(f"o{k} cell WITH {evname} in hold",
                                   sig.loc[m1, f"o{k}"]))
            rows.append(C.describe(f"o{k} cell NO   {evname} in hold",
                                   sig.loc[m0, f"o{k}"]))
        C.show(rows)
        print()
except Exception as e:                                    # pragma: no cover
    print(f"  !! macro_events unavailable: {e}")

# ------------------------------------------------------------ (g) the grid
hdr("G. THRESHOLD GRID — ratio cut x near-high band (executable o5), 2018+ ERA ONLY")
grid = []
era18 = sig["era"] == "-0.5x (2018-02+)"
for rc in [0.80, 0.825, 0.85, 0.875, 0.90]:
    for nh in [0.5, 1.0, 3.0]:
        m = base_mask & era18 & (sig["ratio"] <= rc) & (sig["dist"] >= -nh)
        ds = sig.index[m]
        ep = ds[C.declusterize(ds, gap_td=21)] if len(ds) else ds
        x = sig.loc[m, "o5"].dropna()
        xe = sig.loc[ep, "o5"].dropna()
        b = sig.loc[base_mask & era18, "o5"].dropna()
        grid.append({"ratio<=": rc, "nearhigh%": nh, "n_days": len(x),
                     "avg": round(float(x.mean()), 3) if len(x) else np.nan,
                     "t_days": round(C.tstat(x.to_numpy()), 2) if len(x) else np.nan,
                     "lift_vs_all": round(float(x.mean() - b.mean()), 3) if len(x) else np.nan,
                     "n_ep": len(xe),
                     "ep_avg": round(float(xe.mean()), 3) if len(xe) else np.nan,
                     "ep_t": round(C.tstat(xe.to_numpy()), 2) if len(xe) else np.nan})
print(pd.DataFrame(grid).to_string(index=False))

hdr("G2. SAME GRID, FULL SAMPLE (both leverage eras pooled — contaminated)")
grid = []
for rc in [0.80, 0.825, 0.85, 0.875, 0.90]:
    for nh in [0.5, 1.0, 3.0]:
        m = base_mask & (sig["ratio"] <= rc) & (sig["dist"] >= -nh)
        ds = sig.index[m]
        ep = ds[C.declusterize(ds, gap_td=21)] if len(ds) else ds
        x = sig.loc[m, "o5"].dropna()
        xe = sig.loc[ep, "o5"].dropna()
        b = sig.loc[base_mask, "o5"].dropna()
        grid.append({"ratio<=": rc, "nearhigh%": nh, "n_days": len(x),
                     "avg": round(float(x.mean()), 3) if len(x) else np.nan,
                     "t_days": round(C.tstat(x.to_numpy()), 2) if len(x) else np.nan,
                     "lift_vs_all": round(float(x.mean() - b.mean()), 3) if len(x) else np.nan,
                     "n_ep": len(xe),
                     "ep_avg": round(float(xe.mean()), 3) if len(xe) else np.nan,
                     "ep_t": round(C.tstat(xe.to_numpy()), 2) if len(xe) else np.nan})
print(pd.DataFrame(grid).to_string(index=False))

# -------------------------------------------------------- decomposition
hdr("H. DECOMPOSITION — which leg (if either) carries anything? executable o5, 2018+")
rows = []
legs = {
    "ratio<=0.85 ONLY": base_mask & era18 & (sig["ratio"] <= RATIO_MAX),
    "nearhigh<=1% ONLY": base_mask & era18 & (sig["dist"] >= -NEARHIGH_PCT),
    "BOTH (the pitch)": base_mask & era18 & (sig["ratio"] <= RATIO_MAX) & (sig["dist"] >= -NEARHIGH_PCT),
    "NEITHER": base_mask & era18 & ~((sig["ratio"] <= RATIO_MAX) | (sig["dist"] >= -NEARHIGH_PCT)),
    "ALL DAYS 2018+": base_mask & era18,
}
for name, m in legs.items():
    rows.append(C.describe(name, sig.loc[m, "o5"],
                           baseline=sig.loc[base_mask & era18, "o5"]))
C.show(rows)

hdr("I. YEAR TABLE — cell episode PnL by calendar year (gap 21, executable o5)")
ep21 = cell_dates[C.declusterize(cell_dates, gap_td=21)]
yr = pd.DataFrame({"y": [d.year for d in ep21],
                   "o5": sig.loc[ep21, "o5"].to_numpy(),
                   "o10": sig.loc[ep21, "o10"].to_numpy()}).dropna()
print(yr.groupby("y").agg(n=("o5", "size"), sum_o5=("o5", "sum"),
                          avg_o5=("o5", "mean"), worst_o5=("o5", "min"),
                          sum_o10=("o10", "sum")).round(2).to_string())

# ---------------------------------------------- salvage paths, closed off
hdr("J. SALVAGE PATH 1 — 'but the CPI-in-hold sub-cell looked great'. 2018+ ONLY.")
try:
    ev = pd.read_csv(C.ROOT / "data" / "macro_events.csv", parse_dates=["date"])
    idx = sv.index
    pos = {d: i for i, d in enumerate(idx)}
    dts = set(pd.DatetimeIndex(ev.loc[ev["event"] == "cpi", "date"]).normalize())
    rows = []
    for k in HORIZONS:
        insidef = []
        for d in idx:
            i = pos[d]
            insidef.append(np.nan if i + k >= len(idx)
                           else any(w in dts for w in idx[i + 1: i + 1 + k]))
        has = pd.Series(insidef, index=idx)
        for era in ["-1x (pre-2018-02)", "-0.5x (2018-02+)"]:
            m = cell & (has == True) & (sig["era"] == era)      # noqa: E712
            rows.append(C.describe(f"o{k} cell+CPI {era}", sig.loc[m, f"o{k}"]))
            ds = sig.index[m]
            ep = ds[C.declusterize(ds, gap_td=21)] if len(ds) else ds
            rows.append(C.describe(f"o{k} cell+CPI {era} EPISODES",
                                   sig.loc[ep, f"o{k}"]))
    C.show(rows)
    print("\n(registry: 'post-CPI vol crush — the effect died after 2018')")
except Exception as e:                                    # pragma: no cover
    print(f"  !! {e}")

hdr("K. SALVAGE PATH 2 — the trigger's behaviour into Feb 2018 (Volmageddon)")
win = sig.loc["2017-12-15":"2018-02-20", ["ratio", "dist", "svxy", "o3", "o5", "o10"]]
win = win.assign(in_cell=[bool(cell.get(d, False)) for d in win.index])
print(win.round(3).to_string())
print("\nThe trigger was ON (steep contango + SPY pinned at its 252d high) "
      "through the entire run-up to 2018-02-05. The -1x instrument lost ~90% "
      "in one session; a -0.5x would have lost roughly half of that.")

hdr("L. STALENESS — how long has the trigger already been on?")
onstreak = 0
for d in reversed(sig.index[base_mask]):
    if bool(cell.get(d, False)):
        onstreak += 1
    else:
        break
print(f"consecutive trigger-ON sessions ending {C.LAST_BAR.date()}: {onstreak}")
print("recent cell days:", [str(d.date()) for d in sig.index[cell][-12:]])
print("\nDONE c1")
