"""B1 -- ADVERSARIAL check of "Gold thrust out of a downtrend".

Trigger under test: GLD one-day return >= 2.0x Wilder-14 ATR (price terms)
WHILE GLD closes BELOW its 200d SMA. Claim = CONTINUATION (positioning flush),
measured against GLD's unconditional drift.

Today 2026-08-05: GLD +4.14%, ATR ~1.93% of price -> ~2.14 ATR, 5.36% BELOW
the 200d SMA. SLV +4.14%, GDX +7.39%.

Kill vectors run here:
  (a) sign -- continuation or fade, honestly
  (b) N + clustering, with the episode dates NAMED
  (c) era split 2018; drop-2008 and drop-2020 sensitivity
  (d) threshold grid 1.5/1.75/2.0/2.5 ATR x {below 200d, no trend filter}
  (e) SLV >2% same-day confirmation variant
  (f) EXECUTABLE basis -- the thrust day is 8/5, the card enters at the 8/6 OPEN.
      Quantify the overnight give-up.  <-- most likely kill vector
  (g) vehicle -- GLD vs GDX vs SLV in the same cell
  (h) CPI (8/12) inside the hold window
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _common as C

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)

HL = "=" * 100

D = C.load(["GLD", "SLV", "GDX", "SPY"])
gld, slv, gdx, spy = D["GLD"], D["SLV"], D["GDX"], D["SPY"]

HORIZONS = [3, 5, 10]


def frame(px: pd.DataFrame, tag: str) -> pd.DataFrame:
    f = pd.DataFrame(index=px.index)
    f[f"{tag}_close"] = px["Close"]
    f[f"{tag}_r1"] = px["Close"].pct_change() * 100.0
    atr = C.wilder_atr(px, 14)
    f[f"{tag}_atr"] = atr
    f[f"{tag}_atr_pct"] = atr / px["Close"] * 100.0
    f[f"{tag}_atr_move"] = (px["Close"] - px["Close"].shift(1)) / atr
    sma = px["Close"].rolling(200).mean()
    f[f"{tag}_sma200"] = sma
    f[f"{tag}_below200"] = px["Close"] < sma
    f[f"{tag}_pct_vs200"] = (px["Close"] / sma - 1.0) * 100.0
    # overnight leg: signal close -> next open
    f[f"{tag}_on"] = (px["Open"].shift(-1) / px["Close"] - 1.0) * 100.0
    for k in HORIZONS:
        f[f"{tag}_cc{k}"] = C.fwd(px["Close"], k)
        f[f"{tag}_mo{k}"] = C.fwd_from_next_open(px, k)
    return f


g = frame(gld, "gld")
s = frame(slv, "slv")
m = frame(gdx, "gdx")
p = frame(spy, "spy")

df = g.join(s, how="left").join(m, how="left").join(p, how="left")
df["year"] = df.index.year

ev = pd.read_csv(C.ROOT / "data" / "macro_events.csv", parse_dates=["date"])
ev = ev[ev["date"] < C.ASOF_EXCL]
CPI = set(ev.loc[ev["event"] == "cpi", "date"])
NFP = set(ev.loc[ev["event"] == "nfp", "date"])
FOMC = set(ev.loc[ev["event"] == "fomc_decision", "date"])
idx = df.index


def window_has_event(d, k, evset):
    i = idx.get_loc(d)
    if i + k >= len(idx):
        return np.nan
    lo, hi = idx[i], idx[i + k]
    return any((e > lo) and (e <= hi) for e in evset)


print(HL)
print("B1  GOLD THRUST OUT OF A DOWNTREND  (GLD 1d move >= 2.0 ATR while BELOW 200d SMA)")
print(HL)
last = df.iloc[-1]
print(f"sample: {df.index[0].date()} .. {df.index[-1].date()}  rows={len(df)}")
print(f"TODAY 2026-08-05: GLD r1={last['gld_r1']:+.2f}%  ATR={last['gld_atr_pct']:.2f}% of price  "
      f"move={last['gld_atr_move']:.2f} ATR  vs200d={last['gld_pct_vs200']:+.2f}% (below={bool(last['gld_below200'])})")
print(f"TODAY: SLV r1={last['slv_r1']:+.2f}%  GDX r1={last['gdx_r1']:+.2f}%")

TRIG = (df["gld_atr_move"] >= 2.0) & (df["gld_below200"])
print(f"TODAY fires the trigger? {bool(TRIG.iloc[-1])}")
cell = df[TRIG].copy()
print(f"trigger signal days N={len(cell)} ({len(cell)/len(df)*100:.2f}% of sessions)")

# ============================================================================
print()
print(HL)
print("(a) SIGN -- continuation or fade?  CELL vs UNCONDITIONAL, both bases")
print(HL)
rows = []
for k in HORIZONS:
    rows.append(C.describe(f"GLD cc{k}  CELL", cell[f"gld_cc{k}"], df[f"gld_cc{k}"]))
    rows.append(C.describe(f"GLD MOO{k} CELL", cell[f"gld_mo{k}"], df[f"gld_mo{k}"]))
C.show(rows)
print()
C.show([C.describe(f"GLD cc{k} ALL", df[f"gld_cc{k}"]) for k in HORIZONS] +
       [C.describe(f"GLD MOO{k} ALL", df[f"gld_mo{k}"]) for k in HORIZONS])

# ============================================================================
print()
print(HL)
print("(f) EXECUTABLE BASIS -- the thrust is 8/5, entry is the 8/6 OPEN. Overnight give-up.")
print(HL)
on = cell["gld_on"].dropna()
on_all = df["gld_on"].dropna()
print(f"overnight (signal close -> next open) IN CELL : N={len(on)} avg={on.mean():+.3f}% med={np.median(on):+.3f}% "
      f"t={C.tstat(on):.2f} hit={(on>0).mean()*100:.0f}% worst={on.min():+.2f}% best={on.max():+.2f}%")
print(f"overnight UNCONDITIONAL                        : N={len(on_all)} avg={on_all.mean():+.3f}% med={np.median(on_all):+.3f}%")
print()
for k in HORIZONS:
    a, b = cell[f"gld_cc{k}"].mean(), cell[f"gld_mo{k}"].mean()
    ta, tb = C.tstat(cell[f"gld_cc{k}"].dropna()), C.tstat(cell[f"gld_mo{k}"].dropna())
    frac = (1 - b / a) * 100 if a else float("nan")
    print(f"  h={k:2d}  cc {a:+.3f}% (t={ta:.2f})   MOO {b:+.3f}% (t={tb:.2f})   give-up {b-a:+.3f}pp = {frac:.0f}% of the cc edge")
print()
print("day-1-only decomposition (what the MOO entry actually skips + gets):")
_d1_full = (gld["Close"].shift(-1) / gld["Close"] - 1.0) * 100.0
d1_cc = _d1_full.reindex(cell.index)
print(f"  signal close -> next CLOSE (D+1) : avg={d1_cc.mean():+.3f}% t={C.tstat(d1_cc.dropna()):.2f} hit={(d1_cc>0).mean()*100:.0f}%")
print(f"  of which overnight gap           : avg={on.mean():+.3f}%")
print(f"  intraday D+1 (open->close)       : avg={(d1_cc.mean() - on.mean()):+.3f}%")

# ============================================================================
print()
print(HL)
print("(b) CLUSTERING + NAMED EPISODES")
print(HL)
for gap in [5, 10, 21]:
    keep = C.declusterize(cell.index, gap_td=gap)
    epx = cell[keep]
    print(f"\n--- decluster gap_td={gap}: {len(epx)} episodes ---")
    C.show([C.describe(f"GLD MOO h={k}", epx[f"gld_mo{k}"], df[f"gld_mo{k}"]) for k in HORIZONS] +
           [C.describe(f"GLD cc  h={k}", epx[f"gld_cc{k}"], df[f"gld_cc{k}"]) for k in HORIZONS])

keep10 = C.declusterize(cell.index, gap_td=10)
ep = cell[keep10].copy()
print(f"\nALL {len(cell)} SIGNAL DAYS, named, with forward MOO returns:")
tab = cell[["gld_r1", "gld_atr_move", "gld_pct_vs200", "gld_on", "gld_mo3", "gld_mo5", "gld_mo10",
            "gld_cc10", "slv_r1", "gdx_r1"]].copy()
tab.index = tab.index.date
print(tab.round(2).to_string())
print(f"\nepisode dates (gap 10td), N={len(ep)}:")
print("  " + ", ".join(str(d.date()) for d in ep.index))
print("\nsignal days per year:", cell.groupby("year").size().to_dict())
print("episodes per year:", ep.groupby("year").size().to_dict())

# ============================================================================
print()
print(HL)
print("(c) ERA SPLIT + DROP-2008 / DROP-2020 SENSITIVITY")
print(HL)
for k in HORIZONS:
    print(f"\n--- h={k} MOO ---")
    rows = []
    for lo, hi, lab in [(None, "2018-01-01", "pre-2018"), ("2018-01-01", None, "2018+"),
                        ("2021-01-01", None, "2021+")]:
        sub = cell
        if lo:
            sub = sub[sub.index >= lo]
        if hi:
            sub = sub[sub.index < hi]
        rows.append(C.describe(lab + " (days)", sub[f"gld_mo{k}"], df[f"gld_mo{k}"]))
    rows.append(C.describe("ALL days", cell[f"gld_mo{k}"], df[f"gld_mo{k}"]))
    rows.append(C.describe("drop 2008", cell[cell["year"] != 2008][f"gld_mo{k}"], df[f"gld_mo{k}"]))
    rows.append(C.describe("drop 2020", cell[cell["year"] != 2020][f"gld_mo{k}"], df[f"gld_mo{k}"]))
    rows.append(C.describe("drop 2008+2020", cell[~cell["year"].isin([2008, 2020])][f"gld_mo{k}"], df[f"gld_mo{k}"]))
    rows.append(C.describe("ALL episodes", ep[f"gld_mo{k}"], df[f"gld_mo{k}"]))
    rows.append(C.describe("ep drop 2008", ep[ep["year"] != 2008][f"gld_mo{k}"], df[f"gld_mo{k}"]))
    rows.append(C.describe("ep drop 2020", ep[ep["year"] != 2020][f"gld_mo{k}"], df[f"gld_mo{k}"]))
    C.show(rows)

print("\nLOYO on episodes (gap 10), h=5 and h=10 MOO:")
for k in [5, 10]:
    rows = []
    for yr in sorted(ep["year"].unique()):
        sub = ep[ep["year"] != yr][f"gld_mo{k}"].dropna()
        rows.append({"h": k, "drop_year": yr, "n": len(sub), "avg": round(sub.mean(), 3),
                     "t": round(C.tstat(sub), 2)})
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    print(f"  h={k} LOYO t floor = {t['t'].min():.2f}   full t = {C.tstat(ep[f'gld_mo{k}'].dropna()):.2f}\n")

print("per-year episode table, h=10 MOO:")
gg = ep.groupby("year")["gld_mo10"]
print(pd.DataFrame({"n": gg.size(), "avg": gg.mean().round(3), "sum": gg.sum().round(2),
                    "worst": gg.min().round(2), "best": gg.max().round(2)}).to_string())

# ============================================================================
print()
print(HL)
print("(d) THRESHOLD GRID: {1.5, 1.75, 2.0, 2.5} ATR  x  {below 200d, no filter, above 200d}")
print(HL)
for k in HORIZONS:
    print(f"\n--- h={k} MOO (signal days) ---")
    rows = []
    for thr in [1.5, 1.75, 2.0, 2.5]:
        base = df["gld_atr_move"] >= thr
        for lab, mask in [("below200", base & df["gld_below200"]),
                          ("nofilter", base),
                          ("above200", base & ~df["gld_below200"])]:
            sub = df[mask]
            v = sub[f"gld_mo{k}"].dropna()
            kp = C.declusterize(sub.index, gap_td=10)
            ve = sub[kp][f"gld_mo{k}"].dropna()
            rows.append({"thr_ATR": thr, "trend": lab, "N": len(v),
                         "avg": round(v.mean(), 3) if len(v) else np.nan,
                         "med": round(float(np.median(v)), 3) if len(v) else np.nan,
                         "hit%": round((v > 0).mean() * 100, 1) if len(v) else np.nan,
                         "t": round(C.tstat(v), 2),
                         "N_ep": len(ve), "ep_avg": round(ve.mean(), 3) if len(ve) else np.nan,
                         "ep_t": round(C.tstat(ve), 2),
                         "base": round(df[f"gld_mo{k}"].mean(), 3)})
    print(pd.DataFrame(rows).to_string(index=False))

# ============================================================================
print()
print(HL)
print("(e) SLV CONFIRMATION VARIANT (SLV also up > 2% the same day -- true today)")
print(HL)
conf = cell[cell["slv_r1"] > 2.0]
noconf = cell[cell["slv_r1"] <= 2.0]
for k in HORIZONS:
    rows = [C.describe(f"h={k} SLV>2% (TODAY)", conf[f"gld_mo{k}"], df[f"gld_mo{k}"]),
            C.describe(f"h={k} SLV<=2%", noconf[f"gld_mo{k}"], df[f"gld_mo{k}"])]
    C.show(rows)
print(f"\nSLV>2% subset dates ({len(conf)}): " + ", ".join(str(d.date()) for d in conf.index))
kpc = C.declusterize(conf.index, gap_td=10)
print(f"episodes: {kpc.sum()} -> " + ", ".join(str(d.date()) for d in conf.index[kpc]))
for k in HORIZONS:
    v = conf[kpc][f"gld_mo{k}"].dropna()
    print(f"  h={k:2d} episodes N={len(v)} avg={v.mean():+.3f}% t={C.tstat(v):.2f} "
          f"hit={((v>0).mean()*100) if len(v) else float('nan'):.0f}% worst={v.min() if len(v) else float('nan'):+.2f}%")

# GDX confirmation too (today +7.39%)
print("\nGDX up > 4% same day (today +7.39%):")
confg = cell[cell["gdx_r1"] > 4.0]
for k in HORIZONS:
    print(f"  h={k:2d} N={confg[f'gld_mo{k}'].notna().sum()} avg={confg[f'gld_mo{k}'].mean():+.3f}% "
          f"t={C.tstat(confg[f'gld_mo{k}'].dropna()):.2f}")

# ============================================================================
print()
print(HL)
print("(g) VEHICLE -- GLD vs SLV vs GDX in the SAME cell, MOO basis")
print(HL)
for k in HORIZONS:
    print(f"\n--- h={k} MOO, signal days ---")
    rows = [C.describe("GLD", cell[f"gld_mo{k}"], df[f"gld_mo{k}"]),
            C.describe("SLV", cell[f"slv_mo{k}"], df[f"slv_mo{k}"]),
            C.describe("GDX", cell[f"gdx_mo{k}"], df[f"gdx_mo{k}"]),
            C.describe("SPY", cell[f"spy_mo{k}"], df[f"spy_mo{k}"])]
    C.show(rows)
print("\nepisodes (gap 10), all three vehicles:")
rows = []
for k in HORIZONS:
    for tag in ["gld", "slv", "gdx"]:
        v = ep[f"{tag}_mo{k}"].dropna()
        rows.append({"h": k, "vehicle": tag.upper(), "N": len(v), "avg": round(v.mean(), 3),
                     "med": round(float(np.median(v)), 3) if len(v) else np.nan,
                     "hit%": round((v > 0).mean() * 100, 1), "t": round(C.tstat(v), 2),
                     "worst": round(v.min(), 2), "best": round(v.max(), 2),
                     "base": round(df[f"{tag}_mo{k}"].mean(), 3)})
print(pd.DataFrame(rows).to_string(index=False))

# ============================================================================
print()
print(HL)
print("(h) CALENDAR -- CPI / NFP / FOMC inside the hold window")
print(HL)
for k in [5, 10]:
    hc = cell.index.to_series().apply(lambda d: window_has_event(d, k, CPI))
    hn = cell.index.to_series().apply(lambda d: window_has_event(d, k, NFP))
    hf = cell.index.to_series().apply(lambda d: window_has_event(d, k, FOMC))
    print(f"\n--- h={k} MOO ---  share with CPI={np.nanmean(hc.astype(float))*100:.0f}%  "
          f"NFP={np.nanmean(hn.astype(float))*100:.0f}%  FOMC={np.nanmean(hf.astype(float))*100:.0f}%")
    rows = [C.describe("HAS CPI", cell.loc[hc.index[hc == True], f"gld_mo{k}"]),
            C.describe("NO CPI", cell.loc[hc.index[hc == False], f"gld_mo{k}"]),
            C.describe("HAS NFP", cell.loc[hn.index[hn == True], f"gld_mo{k}"]),
            C.describe("NO NFP", cell.loc[hn.index[hn == False], f"gld_mo{k}"]),
            C.describe("HAS CPI+NFP (TODAY)", cell.loc[hc.index[(hc == True) & (hn == True)], f"gld_mo{k}"])]
    C.show(rows)

print("\nTODAY: NFP 2026-08-07 (D+1 of the hold), CPI 2026-08-12, PPI 2026-08-13.")
print("A 3-session hold contains NFP only; 5 and 10 contain NFP+CPI+PPI.")

# ============================================================================
print()
print(HL)
print("SUMMARY FOR THE VERDICT (MOO basis)")
print(HL)
for k in HORIZONS:
    v = cell[f"gld_mo{k}"].dropna()
    e = ep[f"gld_mo{k}"].dropna()
    print(f"h={k:2d} | days N={len(v)} avg={v.mean():+.3f}% t={C.tstat(v):+.2f} hit={(v>0).mean()*100:.0f}% "
          f"| episodes N={len(e)} avg={e.mean():+.3f}% t={C.tstat(e):+.2f} hit={(e>0).mean()*100:.0f}% worst={e.min():+.2f}% "
          f"| baseline {df[f'gld_mo{k}'].mean():+.3f}%")

# ============================================================================
print()
print(HL)
print("ADDENDUM -- WHY the trend filter hurts: GLD's below-200d regime is a DRAG")
print(HL)
below = df[df["gld_below200"]]
above = df[~df["gld_below200"]]
print(f"share of sessions below the 200d SMA: {len(below)/len(df.dropna(subset=['gld_sma200']))*100:.1f}% "
      f"({len(below)} of {len(df.dropna(subset=['gld_sma200']))})")
rows = []
for k in HORIZONS:
    rows.append(C.describe(f"GLD MOO{k} BELOW 200d", below[f"gld_mo{k}"], df[f"gld_mo{k}"]))
    rows.append(C.describe(f"GLD MOO{k} ABOVE 200d", above[f"gld_mo{k}"], df[f"gld_mo{k}"]))
C.show(rows)
print()
print("i.e. the card's trend filter selects the regime with the WORST unconditional")
print("gold drift, then asks a 2-ATR thrust to overcome it on 11 observations.")
print()
print("the >=2 ATR thrust with NO trend filter (the cell the grid actually supports):")
nf = df[df["gld_atr_move"] >= 2.0]
kpn = C.declusterize(nf.index, gap_td=10)
for k in HORIZONS:
    v = nf[f"gld_mo{k}"].dropna()
    ve = nf[kpn][f"gld_mo{k}"].dropna()
    print(f"  h={k:2d} days N={len(v)} avg={v.mean():+.3f}% t={C.tstat(v):.2f} | "
          f"episodes N={len(ve)} avg={ve.mean():+.3f}% t={C.tstat(ve):.2f} | base {df[f'gld_mo{k}'].mean():+.3f}%")
