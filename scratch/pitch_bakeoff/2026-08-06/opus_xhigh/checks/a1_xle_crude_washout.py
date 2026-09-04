"""A1 -- ADVERSARIAL check of "LONG XLE outright after a crude washout".

Origin cell (from another checker, not validated by them):
    trigger = USO 5d return rank (252d) <= 5th pctile
    measure = XLE forward 10 sessions
    claimed N=289 days avg +3.32% t=5.99; episodes N=93 avg +2.02% t=2.19
    unconditional XLE 10-session drift +0.414%

Kill vectors run here:
  (a) modern era  -- split 2018 and 2021+
  (b) today's configuration -- USO ABOVE its 200d SMA (+13.8%) and XLE ABOVE its
      200d SMA (+9.3%). The washout-inside-an-uptrend subset.
  (c) market beta -- XLE fwd MINUS SPY fwd, and beta-adjusted excess
  (d) episode clustering gap 10 / 21, LOYO on episodes, worst window, worst year
  (e) horizon 3/5/10 on the EXECUTABLE (MOO next open) basis
  (f) calendar -- NFP/CPI/PPI inside the hold window
  (g) book overlap -- structural note

Everything runs off C.load() which truncates strictly before 2026-08-06.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _common as C

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)

HL = "=" * 100
hl = "-" * 100

TICKERS = ["USO", "XLE", "SPY", "XOP"]
D = C.load(TICKERS)
uso, xle, spy, xop = D["USO"], D["XLE"], D["SPY"], D["XOP"]

HORIZONS = [3, 5, 10]

# ----------------------------------------------------------------------------
# frames
# ----------------------------------------------------------------------------
u = pd.DataFrame(index=uso.index)
u["r5"] = C.ret(uso["Close"], 5)
u["r5_rank"] = C.pct_rank(u["r5"], 252)
u["r21"] = C.ret(uso["Close"], 21)
u["r63"] = C.ret(uso["Close"], 63)
u["sma200"] = uso["Close"].rolling(200).mean()
u["uso_above200"] = uso["Close"] > u["sma200"]
u["uso_pct_above200"] = (uso["Close"] / u["sma200"] - 1.0) * 100.0

x = pd.DataFrame(index=xle.index)
_xsma = xle["Close"].rolling(200).mean()
x["xle_above200"] = xle["Close"] > _xsma
x["xle_pct_above200"] = (xle["Close"] / _xsma - 1.0) * 100.0
for k in HORIZONS:
    x[f"cc{k}"] = C.fwd(xle["Close"], k)
    x[f"mo{k}"] = C.fwd_from_next_open(xle, k)

s = pd.DataFrame(index=spy.index)
for k in HORIZONS:
    s[f"spy_cc{k}"] = C.fwd(spy["Close"], k)
    s[f"spy_mo{k}"] = C.fwd_from_next_open(spy, k)

xo = pd.DataFrame(index=xop.index)
for k in HORIZONS:
    xo[f"xop_mo{k}"] = C.fwd_from_next_open(xop, k)

uu = pd.DataFrame(index=uso.index)
for k in HORIZONS:
    uu[f"uso_mo{k}"] = C.fwd_from_next_open(uso, k)

df = u.join(x, how="inner").join(s, how="inner").join(uu, how="inner").join(xo, how="left")
df["year"] = df.index.year

# full-sample beta of XLE to SPY (daily)
rx = xle["Close"].pct_change()
rs = spy["Close"].pct_change()
both = pd.concat([rx, rs], axis=1).dropna()
both.columns = ["xle", "spy"]
BETA_FULL = float(np.polyfit(both["spy"], both["xle"], 1)[0])
# beta on the USO-overlap era only (2006+), which is the cell's sample
both06 = both[both.index >= "2006-04-10"]
BETA_06 = float(np.polyfit(both06["spy"], both06["xle"], 1)[0])

# ----------------------------------------------------------------------------
# macro calendar
# ----------------------------------------------------------------------------
ev = pd.read_csv(C.ROOT / "data" / "macro_events.csv", parse_dates=["date"])
ev = ev[ev["date"] < C.ASOF_EXCL]
NFP = set(ev.loc[ev["event"] == "nfp", "date"])
CPI = set(ev.loc[ev["event"] == "cpi", "date"])
PPI = set(ev.loc[ev["event"] == "ppi", "date"])
BIG = NFP | CPI | PPI

idx = df.index


def window_has_event(d, k, evset):
    """Does an event land in (signal_date, exit_date] i.e. inside the hold?"""
    i = idx.get_loc(d)
    if i + k >= len(idx):
        return np.nan
    lo, hi = idx[i], idx[i + k]
    return any((e > lo) and (e <= hi) for e in evset)


# ----------------------------------------------------------------------------
# trigger
# ----------------------------------------------------------------------------
TRIG = (df["r5_rank"] <= 5.0)
cell = df[TRIG].copy()

print(HL)
print("A1  LONG XLE AFTER A CRUDE WASHOUT  (USO 5d return rank <= 5th pctile)")
print(HL)
print(f"sample: {df.index[0].date()} .. {df.index[-1].date()}   rows={len(df)}")
print(f"XLE daily beta to SPY: full-sample {BETA_FULL:.3f} | 2006+ {BETA_06:.3f}")
print(f"trigger days N={len(cell)}  ({len(cell)/len(df)*100:.1f}% of sessions)")
print(f"TODAY 2026-08-05: USO r5={df['r5'].iloc[-1]:.2f}% rank={df['r5_rank'].iloc[-1]:.1f} "
      f"r21={df['r21'].iloc[-1]:.2f}% r63={df['r63'].iloc[-1]:.2f}%")
print(f"TODAY: USO {df['uso_pct_above200'].iloc[-1]:+.2f}% vs 200dSMA (above={bool(df['uso_above200'].iloc[-1])}) | "
      f"XLE {df['xle_pct_above200'].iloc[-1]:+.2f}% vs 200dSMA (above={bool(df['xle_above200'].iloc[-1])})")
print(f"TODAY fires the trigger? {bool(TRIG.iloc[-1])}")

# ============================================================================
print()
print(HL)
print("(0) REPRODUCTION of the origin cell + UNCONDITIONAL BASELINE, both bases")
print(HL)
rows = []
for k in HORIZONS:
    rows.append({**C.describe(f"XLE cc{k}  CELL", cell[f"cc{k}"], df[f"cc{k}"]), "basis": "close->close"})
    rows.append({**C.describe(f"XLE MOO{k} CELL", cell[f"mo{k}"], df[f"mo{k}"]), "basis": "MOO next open"})
C.show(rows)
print()
print("unconditional baselines (all sessions):")
C.show([C.describe(f"XLE cc{k} ALL", df[f"cc{k}"]) for k in HORIZONS]
       + [C.describe(f"XLE MOO{k} ALL", df[f"mo{k}"]) for k in HORIZONS])
print()
print("overnight give-up: how much of the cc edge is lost by entering at the NEXT OPEN")
for k in HORIZONS:
    a = cell[f"cc{k}"].mean()
    b = cell[f"mo{k}"].mean()
    print(f"  h={k:2d}   cc {a:+.3f}%   MOO {b:+.3f}%   give-up {b-a:+.3f}pp "
          f"({(1-b/a)*100 if a else float('nan'):.1f}% of the edge)")

# ============================================================================
print()
print(HL)
print("(a) MODERN ERA -- is this a 2008/2014-15/2020 crisis-rebound artifact?")
print(HL)
for k in HORIZONS:
    print(f"\n--- horizon {k} sessions, MOO basis (executable) ---")
    rows = []
    for lo, hi, lab in [(None, "2018-01-01", "pre-2018"),
                        ("2018-01-01", None, "2018+"),
                        ("2021-01-01", None, "2021+"),
                        ("2023-01-01", None, "2023+")]:
        sub = cell
        if lo:
            sub = sub[sub.index >= lo]
        if hi:
            sub = sub[sub.index < hi]
        base = df
        if lo:
            base = base[base.index >= lo]
        if hi:
            base = base[base.index < hi]
        rows.append(C.describe(f"{lab}", sub[f"mo{k}"], base[f"mo{k}"]))
    C.show(rows)

print("\nper-YEAR breakdown, h=10 MOO (signal days):")
g = cell.groupby("year")["mo10"]
tab = pd.DataFrame({"n": g.size(), "avg": g.mean().round(3), "med": g.median().round(3),
                    "hit%": (g.apply(lambda v: (v > 0).mean() * 100)).round(1),
                    "sum": g.sum().round(1), "worst": g.min().round(2)})
print(tab.to_string())

# ============================================================================
print()
print(HL)
print("(b) TODAY'S CONFIGURATION -- washout INSIDE an uptrend")
print(HL)
print("today: USO ABOVE 200d (+13.8%), XLE ABOVE 200d (+9.3%)")
for k in [3, 5, 10]:
    print(f"\n--- horizon {k} sessions, MOO basis ---")
    rows = []
    rows.append(C.describe("ALL cell", cell[f"mo{k}"], df[f"mo{k}"]))
    rows.append(C.describe("USO >200d", cell[cell["uso_above200"]][f"mo{k}"], df[f"mo{k}"]))
    rows.append(C.describe("USO <200d", cell[~cell["uso_above200"]][f"mo{k}"], df[f"mo{k}"]))
    rows.append(C.describe("XLE >200d", cell[cell["xle_above200"]][f"mo{k}"], df[f"mo{k}"]))
    rows.append(C.describe("XLE <200d", cell[~cell["xle_above200"]][f"mo{k}"], df[f"mo{k}"]))
    rows.append(C.describe("BOTH >200d (TODAY)",
                           cell[cell["uso_above200"] & cell["xle_above200"]][f"mo{k}"], df[f"mo{k}"]))
    rows.append(C.describe("BOTH <200d",
                           cell[(~cell["uso_above200"]) & (~cell["xle_above200"])][f"mo{k}"], df[f"mo{k}"]))
    C.show(rows)

today_cell = cell[cell["uso_above200"] & cell["xle_above200"]]
print(f"\nBOTH>200d subset dates ({len(today_cell)} signal days):")
if len(today_cell):
    kp = C.declusterize(today_cell.index, gap_td=10)
    print("  episodes (gap 10td):", ", ".join(str(d.date()) for d in today_cell.index[kp]))
    print("  per-year n:", today_cell.groupby("year").size().to_dict())
    for k in HORIZONS:
        e = today_cell[kp][f"mo{k}"].dropna()
        print(f"  episodes h={k:2d} MOO: N={len(e)} avg={e.mean():+.3f}% med={np.median(e) if len(e) else float('nan'):+.3f}% "
              f"t={C.tstat(e):.2f} hit={((e>0).mean()*100) if len(e) else float('nan'):.0f}% "
              f"worst={e.min() if len(e) else float('nan'):+.2f}% best={e.max() if len(e) else float('nan'):+.2f}%")

# today's 21d is POSITIVE (+5.47%) -- sharp 5d drop inside a rising 21d
print("\nEXTRA today-realism: USO 21d return POSITIVE (today +5.47%) on top of the washout")
for k in HORIZONS:
    sub = cell[cell["r21"] > 0]
    print(f"  h={k:2d} MOO  USO r21>0: {C.describe('x', sub[f'mo{k}'])} ")
    sub2 = cell[cell["r21"] <= 0]
    print(f"  h={k:2d} MOO  USO r21<=0: {C.describe('x', sub2[f'mo{k}'])}")

# ============================================================================
print()
print(HL)
print("(c) MARKET BETA -- is this just 'buy equities after a dip' in an energy hat?")
print(HL)
for k in HORIZONS:
    print(f"\n--- horizon {k} sessions, MOO basis ---")
    exc = cell[f"mo{k}"] - cell[f"spy_mo{k}"]
    exc_b = cell[f"mo{k}"] - BETA_06 * cell[f"spy_mo{k}"]
    rows = [C.describe("XLE", cell[f"mo{k}"], df[f"mo{k}"]),
            C.describe("SPY same days", cell[f"spy_mo{k}"], df[f"spy_mo{k}"]),
            C.describe("XLE - SPY (excess)", exc, df[f"mo{k}"] - df[f"spy_mo{k}"]),
            C.describe(f"XLE - {BETA_06:.2f}*SPY", exc_b, df[f"mo{k}"] - BETA_06 * df[f"spy_mo{k}"]),
            C.describe("USO same days", cell[f"uso_mo{k}"], df[f"uso_mo{k}"]),
            C.describe("XOP same days", cell[f"xop_mo{k}"], df[f"xop_mo{k}"])]
    C.show(rows)

print("\nexcess (XLE-SPY) by era, h=10 MOO:")
rows = []
for lo, hi, lab in [(None, "2018-01-01", "pre-2018"), ("2018-01-01", None, "2018+"), ("2021-01-01", None, "2021+")]:
    sub = cell
    if lo:
        sub = sub[sub.index >= lo]
    if hi:
        sub = sub[sub.index < hi]
    rows.append(C.describe(lab, sub["mo10"] - sub["spy_mo10"]))
C.show(rows)

print("\nexcess (XLE-SPY) in TODAY'S config (both >200d), MOO:")
rows = []
for k in HORIZONS:
    rows.append(C.describe(f"h={k}", today_cell[f"mo{k}"] - today_cell[f"spy_mo{k}"]))
C.show(rows)

# ============================================================================
print()
print(HL)
print("(d) EPISODE CLUSTERING / LOYO / WORST WINDOW / WORST YEAR")
print(HL)
for gap in [5, 10, 21]:
    keep = C.declusterize(cell.index, gap_td=gap)
    ep = cell[keep]
    print(f"\n--- decluster gap_td={gap}: {len(ep)} episodes ---")
    rows = []
    for k in HORIZONS:
        rows.append({**C.describe(f"XLE MOO h={k}", ep[f"mo{k}"], df[f"mo{k}"])})
        rows.append({**C.describe(f"XLE-SPY h={k}", ep[f"mo{k}"] - ep[f"spy_mo{k}"])})
    C.show(rows)

keep10 = C.declusterize(cell.index, gap_td=10)
ep = cell[keep10].copy()
print(f"\nLOYO on episodes (gap 10), h=10 MOO  [{len(ep)} episodes]")
rows = []
for yr in sorted(ep["year"].unique()):
    sub = ep[ep["year"] != yr]["mo10"].dropna()
    rows.append({"drop_year": yr, "n": len(sub), "avg": round(sub.mean(), 3), "t": round(C.tstat(sub), 2)})
loyo = pd.DataFrame(rows)
print(loyo.to_string(index=False))
print(f"LOYO t floor = {loyo['t'].min():.2f}   (full-episode t = {C.tstat(ep['mo10'].dropna()):.2f})")

print("\nLOYO on episodes, EXCESS (XLE-SPY) h=10 MOO")
rows = []
exc_ep = (ep["mo10"] - ep["spy_mo10"])
for yr in sorted(ep["year"].unique()):
    sub = exc_ep[ep["year"] != yr].dropna()
    rows.append({"drop_year": yr, "n": len(sub), "avg": round(sub.mean(), 3), "t": round(C.tstat(sub), 2)})
loyo2 = pd.DataFrame(rows)
print(loyo2.to_string(index=False))
print(f"LOYO excess t floor = {loyo2['t'].min():.2f}  (full excess t = {C.tstat(exc_ep.dropna()):.2f})")

print("\nepisode-level per-year, h=10 MOO:")
g = ep.groupby("year")["mo10"]
print(pd.DataFrame({"n": g.size(), "avg": g.mean().round(3), "sum": g.sum().round(2),
                    "worst": g.min().round(2), "best": g.max().round(2)}).to_string())

print("\n10 WORST episode windows (h=10 MOO):")
w = ep.nsmallest(10, "mo10")[["mo10", "spy_mo10", "r5", "uso_above200", "xle_above200"]]
print(w.round(2).to_string())
print("\n10 BEST episode windows (h=10 MOO):")
b = ep.nlargest(10, "mo10")[["mo10", "spy_mo10", "r5", "uso_above200", "xle_above200"]]
print(b.round(2).to_string())
print("\ndrop-best-episode t (h=10 MOO):")
srt = ep["mo10"].dropna().sort_values()
print(f"  full     N={len(srt)} avg={srt.mean():+.3f} t={C.tstat(srt):.2f}")
print(f"  drop 1 best N={len(srt)-1} avg={srt[:-1].mean():+.3f} t={C.tstat(srt[:-1]):.2f}")
print(f"  drop 3 best N={len(srt)-3} avg={srt[:-3].mean():+.3f} t={C.tstat(srt[:-3]):.2f}")

# ============================================================================
print()
print(HL)
print("(e) HORIZON on the EXECUTABLE (MOO) basis -- signal days AND episodes")
print(HL)
rows = []
for k in HORIZONS:
    rows.append({"h": k, "basis": "signal days", "N": int(cell[f"mo{k}"].notna().sum()),
                 "avg": round(cell[f"mo{k}"].mean(), 3), "t": round(C.tstat(cell[f"mo{k}"].dropna()), 2),
                 "hit%": round((cell[f"mo{k}"] > 0).mean() * 100, 1),
                 "base": round(df[f"mo{k}"].mean(), 3),
                 "excess_vs_SPY": round((cell[f"mo{k}"] - cell[f"spy_mo{k}"]).mean(), 3),
                 "excess_t": round(C.tstat((cell[f"mo{k}"] - cell[f"spy_mo{k}"]).dropna()), 2)})
    rows.append({"h": k, "basis": "episodes g10", "N": int(ep[f"mo{k}"].notna().sum()),
                 "avg": round(ep[f"mo{k}"].mean(), 3), "t": round(C.tstat(ep[f"mo{k}"].dropna()), 2),
                 "hit%": round((ep[f"mo{k}"] > 0).mean() * 100, 1),
                 "base": round(df[f"mo{k}"].mean(), 3),
                 "excess_vs_SPY": round((ep[f"mo{k}"] - ep[f"spy_mo{k}"]).mean(), 3),
                 "excess_t": round(C.tstat((ep[f"mo{k}"] - ep[f"spy_mo{k}"]).dropna()), 2)})
print(pd.DataFrame(rows).to_string(index=False))

# ============================================================================
print()
print(HL)
print("(f) CALENDAR -- NFP / CPI / PPI inside the hold window")
print(HL)
for k in [5, 10]:
    col_n = cell.index.to_series().apply(lambda d: window_has_event(d, k, NFP))
    col_c = cell.index.to_series().apply(lambda d: window_has_event(d, k, CPI))
    col_b = cell.index.to_series().apply(lambda d: window_has_event(d, k, BIG))
    print(f"\n--- h={k} MOO ---")
    print(f"  share of cell windows containing NFP: {np.nanmean(col_n.astype(float))*100:.1f}%  "
          f"CPI: {np.nanmean(col_c.astype(float))*100:.1f}%  any(NFP|CPI|PPI): {np.nanmean(col_b.astype(float))*100:.1f}%")
    rows = [C.describe("window HAS NFP", cell.loc[col_b.index[col_n == True], f"mo{k}"]),
            C.describe("window NO NFP", cell.loc[col_b.index[col_n == False], f"mo{k}"]),
            C.describe("window HAS CPI", cell.loc[col_b.index[col_c == True], f"mo{k}"]),
            C.describe("window NO CPI", cell.loc[col_b.index[col_c == False], f"mo{k}"]),
            C.describe("HAS NFP+CPI both", cell.loc[col_b.index[(col_n == True) & (col_c == True)], f"mo{k}"]),
            C.describe("HAS none of 3", cell.loc[col_b.index[col_b == False], f"mo{k}"])]
    C.show(rows)

# today: NFP 8/7, CPI 8/12, PPI 8/13 -- a 10-session hold from 8/6 open contains all three
print("\nTODAY: NFP 2026-08-07, CPI 2026-08-12, PPI 2026-08-13 all inside a >=6 session hold.")
print("So the relevant historical comparison is the 'HAS NFP+CPI both' cohort above.")

# ============================================================================
print()
print(HL)
print("(g) BOOK OVERLAP (structural)")
print(HL)
print("XLE is in LIQUID_PLUS_COMMODITIES, so the systematic scanner already looks at it")
print("every morning. Mean-reversion strategies that can fire on an energy-complex dip:")
print("  - Oversold Low Volume (OLV)          -- persistent limit close-0.25ATR, T+1..T+3, 10d hold")
print("  - LT Trend ST OS                     -- long-term uptrend + short-term oversold (TODAY XLE IS +9.3% ABOVE ITS 200d SMA, i.e. exactly its setup)")
print("  - Indices Oversold Bounce / Sector BO -- index/sector dip and breakout carriers")
print("  - Monday Dip / SPY QQQ MonFri        -- SPY/QQQ only, no XLE, but same market-beta exposure")
print("A long XLE MOO card is therefore NOT orthogonal to the book: it is the same")
print("long-equity-dip factor the scanner is already sized for, minus the limit entry")
print("that gives the book its edge, and with no stop.")

# ============================================================================
print()
print(HL)
print("SUMMARY NUMBERS FOR THE VERDICT")
print(HL)
for k in HORIZONS:
    a = C.describe("sig", cell[f"mo{k}"])
    e = C.describe("ep", ep[f"mo{k}"])
    xs = C.describe("exc", ep[f"mo{k}"] - ep[f"spy_mo{k}"])
    t2018 = cell[cell.index >= "2018-01-01"][f"mo{k}"]
    t2021 = cell[cell.index >= "2021-01-01"][f"mo{k}"]
    tc = today_cell[f"mo{k}"]
    print(f"h={k:2d} MOO | sig N={a['n']} avg={a['avg']:+.3f} t={a['t']:+.2f} | ep N={e['n']} avg={e['avg']:+.3f} t={e['t']:+.2f} "
          f"| ep excess avg={xs['avg']:+.3f} t={xs['t']:+.2f} | 2018+ N={t2018.notna().sum()} avg={t2018.mean():+.3f} t={C.tstat(t2018.dropna()):+.2f} "
          f"| 2021+ N={t2021.notna().sum()} avg={t2021.mean():+.3f} t={C.tstat(t2021.dropna()):+.2f} "
          f"| TODAY-cfg N={tc.notna().sum()} avg={tc.mean():+.3f} t={C.tstat(tc.dropna()):+.2f}")

# ============================================================================
print()
print(HL)
print("ADDENDUM 1 -- the modern-era result on EPISODES (the 2021+ t was signal-day inflated)")
print(HL)
rows = []
for lo, lab in [(None, "all"), ("2018-01-01", "2018+"), ("2021-01-01", "2021+")]:
    sub = ep if lo is None else ep[ep.index >= lo]
    for k in HORIZONS:
        rows.append({"era": lab, "h": k, "N_ep": int(sub[f"mo{k}"].notna().sum()),
                     "avg": round(sub[f"mo{k}"].mean(), 3),
                     "t": round(C.tstat(sub[f"mo{k}"].dropna()), 2),
                     "hit%": round((sub[f"mo{k}"] > 0).mean() * 100, 1),
                     "exc_avg": round((sub[f"mo{k}"] - sub[f"spy_mo{k}"]).mean(), 3),
                     "exc_t": round(C.tstat((sub[f"mo{k}"] - sub[f"spy_mo{k}"]).dropna()), 2)})
print(pd.DataFrame(rows).to_string(index=False))

print()
print(HL)
print("ADDENDUM 2 -- TODAY'S FULL JOINT CELL: washout + both >200d + NFP&CPI in a 10-session window")
print(HL)
has_nfp10 = cell.index.to_series().apply(lambda d: window_has_event(d, 10, NFP))
has_cpi10 = cell.index.to_series().apply(lambda d: window_has_event(d, 10, CPI))
joint = cell[(cell["uso_above200"]) & (cell["xle_above200"]) &
             (has_nfp10.reindex(cell.index) == True) & (has_cpi10.reindex(cell.index) == True)]
print(f"signal days N={len(joint)}")
if len(joint):
    kj = C.declusterize(joint.index, gap_td=10)
    print("  episodes:", ", ".join(str(d.date()) for d in joint.index[kj]))
    for k in HORIZONS:
        v = joint[kj][f"mo{k}"].dropna()
        print(f"  h={k:2d} MOO episodes N={len(v)} avg={v.mean():+.3f}% t={C.tstat(v):.2f} "
              f"hit={((v>0).mean()*100) if len(v) else float('nan'):.0f}% worst={v.min() if len(v) else float('nan'):+.2f}%")
    for k in HORIZONS:
        v = joint[f"mo{k}"].dropna()
        print(f"  h={k:2d} MOO signaldays N={len(v)} avg={v.mean():+.3f}% t={C.tstat(v):.2f}")

# ============================================================================
print()
print(HL)
print("ADDENDUM 3 -- RECENCY: this exact trigger has already fired 5x in 2026")
print(HL)
c26 = cell[cell.index >= "2026-01-01"]
print(f"2026 signal days N={len(c26)}")
k26 = C.declusterize(c26.index, gap_td=10)
cols = ["r5", "uso_above200", "xle_above200", "mo3", "mo5", "mo10", "spy_mo10"]
t26 = c26[k26][cols].copy()
t26.index = t26.index.date
print(t26.round(2).to_string())
for k in HORIZONS:
    v = c26[k26][f"mo{k}"].dropna()
    x = (c26[k26][f"mo{k}"] - c26[k26][f"spy_mo{k}"]).dropna()
    print(f"  2026 episodes h={k:2d}: N={len(v)} avg={v.mean():+.3f}% t={C.tstat(v):.2f} | "
          f"excess vs SPY avg={x.mean():+.3f}% t={C.tstat(x):.2f}")
