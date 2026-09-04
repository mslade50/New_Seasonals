"""D1 — adversarial kill check: LONG EEM after a violent 5-day surge off a
dead 3-month base.

Trigger under test: EEM 5d-return rank (252d) >= 95 AND 63d-return rank <= 15
AND EEM > 200d SMA. Today (2026-08-05 close): 99.2 / 4.4 / +9.43% above SMA.

Kill vectors run here:
  (a) is the 63d leg decoration? -> control cells (5d rank alone, +200d only,
      63d alone) and a like-for-like day-matched comparison
  (b) EEM minus SPY excess (is this just beta?)
  (c) era split at 2018 and at 2021 (post-China-2021 EM is different)
  (d) declustered episodes + roster
  (e) dollar conditioning (DX-Y.NYB): washout-inside-uptrend cell
  (f) worst window, worst year, LOYO on episodes
  (g) sister-asset generalization (EFA, FXI) — a cell true of EM should not be
      unique to one ticker

Executable basis LEAD: entry MOO the session after the signal, exit MOC k
sessions later (C.fwd_from_next_open).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _common as C

pd.set_option("display.width", 240)

HORIZONS = [3, 5, 10]
R5_MIN, R63_MAX = 95.0, 15.0


def hdr(s: str) -> None:
    print("\n" + "=" * 104)
    print(s)
    print("=" * 104)


px = C.load(["EEM", "SPY", "DX-Y.NYB", "EFA", "FXI"])


def build(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"]
    f = pd.DataFrame(index=df.index)
    f["close"] = c
    f["r5"] = C.ret(c, 5)
    f["r63"] = C.ret(c, 63)
    f["rk5"] = C.pct_rank(f["r5"], 252)
    f["rk63"] = C.pct_rank(f["r63"], 252)
    f["r21"] = C.ret(c, 21)
    f["rk21"] = C.pct_rank(f["r21"], 252)
    f["sma200"] = c.rolling(200).mean()
    f["above200"] = (c / f["sma200"] - 1.0) * 100.0
    for k in HORIZONS:
        f[f"o{k}"] = C.fwd_from_next_open(df, k)
        f[f"c{k}"] = C.fwd(c, k)
    return f


eem = build(px["EEM"])
spy = build(px["SPY"])
dxy = build(px["DX-Y.NYB"])
efa = build(px["EFA"])
fxi = build(px["FXI"])

for k in HORIZONS:
    eem[f"spy_o{k}"] = spy[f"o{k}"].reindex(eem.index)
    eem[f"xs{k}"] = eem[f"o{k}"] - eem[f"spy_o{k}"]

eem["dx_rk5"] = dxy["rk5"].reindex(eem.index).ffill(limit=3)
eem["dx_rk63"] = dxy["rk63"].reindex(eem.index).ffill(limit=3)

hdr("0. SANITY — reproduce today's stated context from the cache")
last = eem.dropna(subset=["rk5", "rk63", "above200"]).iloc[-1]
ld = eem.dropna(subset=["rk5", "rk63", "above200"]).index[-1]
print(f"EEM bars {eem.index[0].date()} .. {eem.index[-1].date()}  N={len(eem)}")
print(f"{ld.date()}: close {last['close']:.2f}  r5 {last['r5']:+.2f}% "
      f"(rank {last['rk5']:.1f})  r21 {last['r21']:+.2f}% (rank {last['rk21']:.1f})  "
      f"r63 rank {last['rk63']:.1f}  above200 {last['above200']:+.2f}%")
dl = dxy.dropna(subset=["rk5", "rk63"]).iloc[-1]
print(f"DX-Y.NYB {dxy.dropna(subset=['rk5','rk63']).index[-1].date()}: "
      f"5d rank {dl['rk5']:.1f}  63d rank {dl['rk63']:.1f}")
print("(context claimed EEM 99.2 / 4.4 / +9.43%, DX 6.0 / 87.7)")

base = eem["rk5"].notna() & eem["rk63"].notna() & eem["sma200"].notna()
cell = base & (eem["rk5"] >= R5_MIN) & (eem["rk63"] <= R63_MAX) & (eem["above200"] > 0)
print(f"\nusable days {int(base.sum())}, cell days {int(cell.sum())} "
      f"({cell.sum()/base.sum()*100:.2f}% of the tape)")
print(f"trigger today: rk5 {last['rk5']:.1f}>=95 {last['rk5']>=95} | "
      f"rk63 {last['rk63']:.1f}<=15 {last['rk63']<=15} | "
      f"above200 {last['above200']>0}")

# --------------------------------------------------------------- (a) control
hdr("A. IS THE 63d LEG DECORATION? — executable MOO basis, full sample")
controls = {
    "PITCH  rk5>=95 & rk63<=15 & >200d": cell,
    "CTRL-1 rk5>=95 & >200d (no 63d)": base & (eem["rk5"] >= R5_MIN) & (eem["above200"] > 0),
    "CTRL-2 rk5>=95 only": base & (eem["rk5"] >= R5_MIN),
    "CTRL-3 rk63<=15 & >200d (no surge)": base & (eem["rk63"] <= R63_MAX) & (eem["above200"] > 0),
    "CTRL-4 >200d only": base & (eem["above200"] > 0),
    "ALL DAYS": base,
}
for k in HORIZONS:
    rows = [C.describe(f"o{k} {nm}", eem.loc[m, f"o{k}"],
                       baseline=eem.loc[base, f"o{k}"]) for nm, m in controls.items()]
    C.show(rows)
    print()

hdr("A2. CLOSE-BASIS mirror (reference only)")
for k in HORIZONS:
    rows = [C.describe(f"c{k} {nm}", eem.loc[m, f"c{k}"],
                       baseline=eem.loc[base, f"c{k}"]) for nm, m in controls.items()]
    C.show(rows)
    print()

hdr("A3. INCREMENT OF THE 63d LEG — pitch cell vs CTRL-1 minus pitch cell")
rows = []
ctrl1 = controls["CTRL-1 rk5>=95 & >200d (no 63d)"]
for k in HORIZONS:
    x = eem.loc[cell, f"o{k}"].dropna().to_numpy()
    y = eem.loc[ctrl1 & ~cell, f"o{k}"].dropna().to_numpy()
    se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
    rows.append({"h": f"o{k}", "n_pitch": len(x), "n_ctrl1_ex_pitch": len(y),
                 "pitch_avg": round(x.mean(), 3), "other_avg": round(y.mean(), 3),
                 "diff": round(x.mean() - y.mean(), 3),
                 "welch_t": round((x.mean() - y.mean()) / se, 2)})
C.show(rows)

# ------------------------------------------------------------- (b) vs SPY
hdr("B. EEM MINUS SPY EXCESS — is the 'EM turn' anything but beta? (executable)")
rows = []
for k in HORIZONS:
    rows.append(C.describe(f"xs{k} PITCH cell", eem.loc[cell, f"xs{k}"],
                           baseline=eem.loc[base, f"xs{k}"]))
    rows.append(C.describe(f"xs{k} CTRL-1", eem.loc[ctrl1, f"xs{k}"],
                           baseline=eem.loc[base, f"xs{k}"]))
    rows.append(C.describe(f"xs{k} ALL DAYS", eem.loc[base, f"xs{k}"]))
C.show(rows)
print("\nSPY leg on the same cell days (what the beta alone paid):")
C.show([C.describe(f"SPY o{k} on cell days", eem.loc[cell, f"spy_o{k}"],
                   baseline=eem.loc[base, f"spy_o{k}"]) for k in HORIZONS])

# ------------------------------------------------------------------ (c) eras
hdr("C. ERA SPLITS — 2018 and 2021 (post-China-crackdown EM is a different animal)")
for k in HORIZONS:
    print(f"\n-- o{k} split at 2018-01-01")
    C.show(C.era_split(eem.index[cell], eem.loc[cell, f"o{k}"], cut="2018-01-01"))
    print(f"-- o{k} split at 2021-07-01")
    C.show(C.era_split(eem.index[cell], eem.loc[cell, f"o{k}"], cut="2021-07-01"))

print("\nThree-way era table, executable o5 / o10, with matched baselines:")
eras = {"2004-2017": (pd.Timestamp("2000-01-01"), pd.Timestamp("2018-01-01")),
        "2018-2021H1": (pd.Timestamp("2018-01-01"), pd.Timestamp("2021-07-01")),
        "2021H2+": (pd.Timestamp("2021-07-01"), pd.Timestamp("2030-01-01"))}
rows = []
for nm, (a, b) in eras.items():
    win = (eem.index >= a) & (eem.index < b)
    for k in [5, 10]:
        rows.append(C.describe(f"o{k} cell {nm}", eem.loc[cell & win, f"o{k}"],
                               baseline=eem.loc[base & win, f"o{k}"]))
C.show(rows)

# -------------------------------------------------------------- (d) episodes
hdr("D. DECLUSTERED EPISODES + ROSTER")
cd = eem.index[cell]
for gap in [5, 10, 21]:
    ep = cd[C.declusterize(cd, gap_td=gap)]
    rows = [C.describe(f"o{k} episodes gap{gap}", eem.loc[ep, f"o{k}"],
                       baseline=eem.loc[base, f"o{k}"]) for k in HORIZONS]
    rows += [C.describe(f"xs{k} episodes gap{gap}", eem.loc[ep, f"xs{k}"],
                        baseline=eem.loc[base, f"xs{k}"]) for k in HORIZONS]
    print(f"\n--- gap {gap} td: {len(ep)} episodes from {len(cd)} raw days")
    C.show(rows)

ep10 = cd[C.declusterize(cd, gap_td=10)]
print("\nEPISODE ROSTER (gap 10):")
r = pd.DataFrame({
    "date": [d.date() for d in ep10],
    "rk5": eem.loc[ep10, "rk5"].round(1).to_numpy(),
    "rk63": eem.loc[ep10, "rk63"].round(1).to_numpy(),
    "ab200%": eem.loc[ep10, "above200"].round(1).to_numpy(),
    "dx_rk5": eem.loc[ep10, "dx_rk5"].round(1).to_numpy(),
    "dx_rk63": eem.loc[ep10, "dx_rk63"].round(1).to_numpy(),
    "o3": eem.loc[ep10, "o3"].round(2).to_numpy(),
    "o5": eem.loc[ep10, "o5"].round(2).to_numpy(),
    "o10": eem.loc[ep10, "o10"].round(2).to_numpy(),
    "xs5": eem.loc[ep10, "xs5"].round(2).to_numpy(),
    "xs10": eem.loc[ep10, "xs10"].round(2).to_numpy(),
})
print(r.to_string(index=False))

# ---------------------------------------------------------------- (e) dollar
hdr("E. DOLLAR CONDITIONING — DX-Y.NYB washout inside an uptrend is today's setup")
dxcell = eem["dx_rk5"].notna() & eem["dx_rk63"].notna()
splits = {
    "DX 5d rank <= 25 (washout)": dxcell & (eem["dx_rk5"] <= 25),
    "DX 5d rank > 25": dxcell & (eem["dx_rk5"] > 25),
    "DX 63d rank >= 75 (uptrend)": dxcell & (eem["dx_rk63"] >= 75),
    "DX 63d rank < 75": dxcell & (eem["dx_rk63"] < 75),
    "TODAY'S CELL: dx5<=25 & dx63>=75": dxcell & (eem["dx_rk5"] <= 25) & (eem["dx_rk63"] >= 75),
    "complement of today's dx cell": dxcell & ~((eem["dx_rk5"] <= 25) & (eem["dx_rk63"] >= 75)),
}
for k in [5, 10]:
    rows = []
    for nm, m in splits.items():
        rows.append(C.describe(f"o{k} PITCH & {nm}", eem.loc[cell & m, f"o{k}"],
                               baseline=eem.loc[base, f"o{k}"]))
    C.show(rows)
    print()
print("Same dollar split applied to ALL DAYS (is the dollar state alone the driver?):")
rows = []
for nm, m in splits.items():
    rows.append(C.describe(f"o5 ALLDAYS & {nm}", eem.loc[base & m, "o5"],
                           baseline=eem.loc[base, "o5"]))
C.show(rows)

# ------------------------------------------------------ (f) tail / year / LOYO
hdr("F. WORST WINDOW, YEAR TABLE, LOYO ON EPISODES")
for k in HORIZONS:
    x = eem.loc[cell, f"o{k}"].dropna().sort_values()
    print(f"\no{k} worst 5 cell windows:")
    print(pd.DataFrame({"signal_date": [d.date() for d in x.index[:5]],
                        "ret%": x.iloc[:5].round(2).to_numpy()}).to_string(index=False))

yr = pd.DataFrame({"y": [d.year for d in ep10],
                   "o5": eem.loc[ep10, "o5"].to_numpy(),
                   "o10": eem.loc[ep10, "o10"].to_numpy(),
                   "xs5": eem.loc[ep10, "xs5"].to_numpy()}).dropna(subset=["o5"])
print("\nEpisode year table (gap 10, executable):")
print(yr.groupby("y").agg(n=("o5", "size"), sum_o5=("o5", "sum"), avg_o5=("o5", "mean"),
                          worst_o5=("o5", "min"), sum_o10=("o10", "sum"),
                          avg_xs5=("xs5", "mean")).round(2).to_string())

print("\nLOYO on episodes (gap 10) — drop each year, recompute:")
for k in [5, 10]:
    s = eem.loc[ep10, f"o{k}"].dropna()
    rows = []
    for y in sorted({d.year for d in s.index}):
        keep = s[[d.year != y for d in s.index]]
        rows.append({"h": f"o{k}", "dropped": y, "n": len(keep),
                     "avg": round(float(keep.mean()), 3),
                     "t": round(C.tstat(keep.to_numpy()), 2)})
    C.show(rows)
    full = s
    print(f"  full-sample o{k}: n={len(full)} avg={full.mean():.3f} "
          f"t={C.tstat(full.to_numpy()):.2f}\n")

# ---------------------------------------------------- (g) sister assets
hdr("G. SISTER-ASSET GENERALIZATION — same trigger on EFA and FXI")
for nm, f in [("EFA", efa), ("FXI", fxi)]:
    b = f["rk5"].notna() & f["rk63"].notna() & f["sma200"].notna()
    cl = b & (f["rk5"] >= R5_MIN) & (f["rk63"] <= R63_MAX) & (f["close"] > f["sma200"])
    ds = f.index[cl]
    ep = ds[C.declusterize(ds, gap_td=10)] if len(ds) else ds
    rows = []
    for k in HORIZONS:
        rows.append(C.describe(f"{nm} o{k} cell", f.loc[cl, f"o{k}"],
                               baseline=f.loc[b, f"o{k}"]))
        rows.append(C.describe(f"{nm} o{k} episodes", f.loc[ep, f"o{k}"],
                               baseline=f.loc[b, f"o{k}"]))
    print(f"\n{nm}: {int(cl.sum())} cell days, {len(ep)} episodes")
    C.show(rows)

hdr("H. HORIZON SWEEP on the PITCH cell (executable), day-level and episode-level")
rows = []
for k in [1, 2, 3, 4, 5, 7, 10, 15, 21]:
    o = C.fwd_from_next_open(px["EEM"], k)
    sp = C.fwd_from_next_open(px["SPY"], k).reindex(eem.index)
    ep = cd[C.declusterize(cd, gap_td=10)]
    rows.append({"h": k, "n_days": int(o[cell].notna().sum()),
                 "day_avg": round(float(o[cell].mean()), 3),
                 "day_t": round(C.tstat(o[cell].dropna().to_numpy()), 2),
                 "base_avg": round(float(o[base].mean()), 3),
                 "n_ep": int(o.reindex(ep).notna().sum()),
                 "ep_avg": round(float(o.reindex(ep).mean()), 3),
                 "ep_t": round(C.tstat(o.reindex(ep).dropna().to_numpy()), 2),
                 "ep_xs_avg": round(float((o - sp).reindex(ep).mean()), 3),
                 "ep_xs_t": round(C.tstat((o - sp).reindex(ep).dropna().to_numpy()), 2)})
print(pd.DataFrame(rows).to_string(index=False))

hdr("I. STALENESS + FULL CELL-DAY LIST (is today even a fresh trigger?)")
print("all cell days:", [str(d.date()) for d in eem.index[cell]])
streak = 0
for d in reversed(eem.index[base]):
    if bool(cell.get(d, False)):
        streak += 1
    else:
        break
print(f"consecutive trigger-ON sessions ending {C.LAST_BAR.date()}: {streak}")
print("\nSame-cell forward returns measured from the FIRST day of each episode "
      "vs LATER days in the episode (does waiting cost?):")
first = pd.Index(ep10)
later = eem.index[cell].difference(first)
C.show([C.describe(f"o{k} episode-day-1", eem.loc[first, f"o{k}"]) for k in HORIZONS]
       + [C.describe(f"o{k} episode day 2+", eem.loc[later, f"o{k}"]) for k in HORIZONS])

hdr("J. SALVAGE PATH — loosen the 63d leg (is there ANY threshold that works?)")
grid = []
for r5 in [90, 93, 95]:
    for r63 in [10, 15, 25, 40]:
        m = base & (eem["rk5"] >= r5) & (eem["rk63"] <= r63) & (eem["above200"] > 0)
        ds = eem.index[m]
        ep = ds[C.declusterize(ds, gap_td=10)] if len(ds) else ds
        x, xe = eem.loc[m, "o5"].dropna(), eem.loc[ep, "o5"].dropna()
        xs = eem.loc[ep, "xs5"].dropna()
        grid.append({"rk5>=": r5, "rk63<=": r63, "n_days": len(x),
                     "avg_o5": round(float(x.mean()), 3) if len(x) else np.nan,
                     "t_days": round(C.tstat(x.to_numpy()), 2) if len(x) else np.nan,
                     "n_ep": len(xe),
                     "ep_avg": round(float(xe.mean()), 3) if len(xe) else np.nan,
                     "ep_t": round(C.tstat(xe.to_numpy()), 2) if len(xe) else np.nan,
                     "ep_xs_avg": round(float(xs.mean()), 3) if len(xs) else np.nan,
                     "ep_xs_t": round(C.tstat(xs.to_numpy()), 2) if len(xs) else np.nan})
print(pd.DataFrame(grid).to_string(index=False))
print("(baseline o5 all days = "
      f"{eem.loc[base,'o5'].mean():.3f}%)")
print("\nDONE d1")
