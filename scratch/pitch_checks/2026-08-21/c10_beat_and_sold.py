"""C10 round 1: the beat that gets sold -- positive EPS surprise met with a
large adverse next-session move. Both directions measured.

Date convention (established in c10_recon_convention.py, not assumed): the
parquet's `date` is the ANNOUNCEMENT date and the reaction splits between
day 0 (BMO) and day +1 (AMC) -- |ret| / own 63d median is 1.720 at offset 0
and 1.812 at offset +1, volume 1.733 / 1.798, with offsets -1 and +2 near
1.0. So the reaction day is classified per event as argmax|ret| over {0,+1},
and BOTH pure conventions are also reported as the definition-fragility check.

Entry is MOC on the session AFTER the reaction close (lag=1, the repo rule).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from strategy_config import LIQUID_PLUS_COMMODITIES

pd.set_option("display.width", 250)
HS = (1, 2, 3, 5, 10)

mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet")
mp["date"] = pd.to_datetime(mp["date"])
cal = pd.DatetimeIndex(sorted(mp.loc[mp.ticker == "SPY", "date"].unique()))
pos = pd.Series(range(len(cal)), index=cal)
spy = mp.loc[mp.ticker == "SPY"].set_index("date")["Close"].reindex(cal)
spy_f = {h: (spy.shift(-(1 + h)) / spy.shift(-1) - 1.0).values for h in HS}

e = pd.read_parquet(ROOT / "data" / "earnings_calendar.parquet")
e = e[e.eps_est.notna() & e.eps_surprise_pct.notna()]
LIQ = set(LIQUID_PLUS_COMMODITIES)

recs = []
for t, g in mp.groupby("ticker"):
    ev = e.loc[e.ticker == t]
    if ev.empty:
        continue
    g = g.sort_values("date").drop_duplicates("date").set_index("date").reindex(cal)
    c = g["Close"]
    if c.notna().sum() < 300:
        continue
    r1 = c.pct_change(fill_method=None)
    atr = wilder_atr(g["High"], g["Low"], c)
    atrp = atr / c
    # any earnings within +/-5 sessions, for the no-earnings control later
    epos = pos.reindex(cal[np.searchsorted(cal.values, ev.date.values, side="left")
                           .clip(0, len(cal) - 1)]).values
    near = np.zeros(len(cal), dtype=bool)
    for p in epos:
        near[max(0, p - 6):min(len(cal), p + 7)] = True
    cv = c.values
    r1v = r1.values
    atrv = atrp.values
    volv = g["Volume"].values
    vmed = pd.Series(volv).rolling(63).median().values
    for p, sur, rsur, dt in zip(epos, ev.eps_surprise_pct.values,
                                ev.rev_surprise_pct.values, ev.date.values):
        if p + 12 >= len(cal) or p < 70:
            continue
        for lbl, rp in (("d0", p), ("d1", p + 1)):
            pass
        a0 = abs(r1v[p]) if np.isfinite(r1v[p]) else -1
        a1 = abs(r1v[p + 1]) if np.isfinite(r1v[p + 1]) else -1
        rp = p if a0 >= a1 else p + 1
        rec = {"ticker": t, "ann_date": pd.Timestamp(dt), "p": p, "rp": rp,
               "bmo": rp == p, "surp": sur, "rsurp": rsur,
               "react": r1v[rp], "react_d0": r1v[p], "react_d1": r1v[p + 1],
               "atrp": atrv[rp - 1] if np.isfinite(atrv[rp - 1]) else np.nan,
               "volx": volv[rp] / vmed[rp] if np.isfinite(vmed[rp]) and vmed[rp] else np.nan,
               "liq": t in LIQ, "near": near}
        for h in HS:
            if rp + 1 + h < len(cal):
                rec[f"f{h}"] = cv[rp + 1 + h] / cv[rp + 1] - 1.0
                rec[f"s{h}"] = spy_f[h][rp]
            else:
                rec[f"f{h}"] = np.nan
                rec[f"s{h}"] = np.nan
        rec.pop("near")
        rec["rdate"] = cal[rp]
        recs.append(rec)

df = pd.DataFrame(recs)
df = df[df["react"].notna()]
print(f"events with a resolvable reaction: {len(df)}  tickers {df.ticker.nunique()} "
      f" {df.rdate.min().date()}..{df.rdate.max().date()}")
print(f" BMO (day-0 reaction) share: {100*df.bmo.mean():.1f}%")
df.to_parquet(Path(__file__).parent / "_c10_events.parquet")

BEAT = df.surp > 0
print(f" positive EPS surprise: {int(BEAT.sum())} ({100*BEAT.mean():.1f}%)")

print("\n===== 0. the live instances =====")
live = df[(df.rdate >= "2026-08-18") & df.ticker.isin(["WMT", "TJX", "ROST", "TGT"])]
print(live[["ticker", "ann_date", "rdate", "bmo", "surp", "react", "atrp", "volx"]]
      .to_string(index=False))


def blk(sub, h, label, mkt=False):
    v = (sub[f"f{h}"] - sub[f"s{h}"]) if mkt else sub[f"f{h}"]
    v = v.dropna()
    if len(v) < 3:
        return {"label": label, "n": len(v)}
    s = summarize(v.values, label)
    # date-clustered mean: one observation per reaction date
    byd = sub.assign(x=(sub[f"f{h}"] - sub[f"s{h}"]) if mkt else sub[f"f{h}"]) \
             .dropna(subset=["x"]).groupby("rdate")["x"].mean()
    s["n_dates"] = len(byd)
    s["clust_t"] = round(float(byd.mean() / (byd.std(ddof=1) / np.sqrt(len(byd)))), 2) \
        if len(byd) > 2 else np.nan
    return s


# ------------------------------------------------------------------ 1. the cell + controls
print("\n===== 1. THE CELL AND ITS CONTROLS (market-relative, entry lag=1) =====")
TH = -0.05
cell = df[BEAT & (df.react <= TH)]
print(f" trigger def: eps_surprise>0 AND reaction <= {100*TH:.0f}%  -> N={len(cell)}")
for h in HS:
    rows = [blk(cell, h, f"BEAT & sold <= -5%  h={h}", mkt=True),
            blk(df[BEAT], h, "  CTRL all beats", mkt=True),
            blk(df[BEAT & (df.react >= 0)], h, "  CTRL beat & NOT sold (react>=0)", mkt=True),
            blk(df[BEAT & (df.react > TH) & (df.react < 0)], h, "  CTRL beat & mild drop", mkt=True),
            blk(df[~BEAT & (df.react <= TH)], h, "  CTRL MISS & sold <= -5%", mkt=True),
            blk(df[df.react <= TH], h, "  CTRL any print sold <= -5%", mkt=True),
            blk(df, h, "  CTRL all prints", mkt=True)]
    show(rows, f"h={h} (mean_pct = excess over SPY)")

print("\n  raw (not market-relative) for the pitched cell:")
show([blk(cell, h, f"BEAT & sold <=-5% RAW h={h}") for h in HS], "")

# ------------------------------------------------------------------ 2. non-earnings control
print("\n===== 2. IS IT THE EARNINGS, OR JUST A BIG ONE-DAY DROP? =====")
print(" (matched: same names, same -5% one-day drop, NO print within +/-6 sessions)")
ctrl_rows = []
for t, g in mp.groupby("ticker"):
    if t not in set(df.ticker):
        continue
    ev = e.loc[e.ticker == t]
    g = g.sort_values("date").drop_duplicates("date").set_index("date").reindex(cal)
    c = g["Close"]
    if c.notna().sum() < 300:
        continue
    r1 = c.pct_change(fill_method=None).values
    epos = pos.reindex(cal[np.searchsorted(cal.values, ev.date.values, side="left")
                           .clip(0, len(cal) - 1)]).values
    near = np.zeros(len(cal), dtype=bool)
    for p in epos:
        near[max(0, p - 6):min(len(cal), p + 7)] = True
    cv = c.values
    hits = np.where((r1 <= TH) & ~near)[0]
    for p in hits:
        if p + 12 >= len(cal) or p < 70:
            continue
        row = {"ticker": t, "rdate": cal[p], "react": r1[p], "liq": t in LIQ}
        for h in HS:
            row[f"f{h}"] = cv[p + 1 + h] / cv[p + 1] - 1.0
            row[f"s{h}"] = spy_f[h][p]
        ctrl_rows.append(row)
nc = pd.DataFrame(ctrl_rows)
print(f" no-earnings -5% drops: N={len(nc)}")
show([blk(nc, h, f"no-earnings -5% drop h={h}", mkt=True) for h in HS],
     "market-relative")
print(" earnings premium (cell minus no-earnings control), market-relative:")
for h in HS:
    a = blk(cell, h, "", mkt=True)
    b = blk(nc, h, "", mkt=True)
    if a.get("n", 0) and b.get("n", 0):
        print(f"   h={h}: {a['mean_pct']:+.3f}% - {b['mean_pct']:+.3f}% = "
              f"{a['mean_pct']-b['mean_pct']:+.3f}pp")

# ------------------------------------------------------------------ 3. era
print("\n===== 3. ERA SPLIT (post-earnings drift is heavily arbitraged) =====")
for cut in ("2009-01-01", "2013-01-01", "2018-01-01", "2021-01-01"):
    rows = []
    for h in (1, 3, 5, 10):
        a = cell[cell.rdate < cut]
        b = cell[cell.rdate >= cut]
        rows.append(blk(a, h, f"pre-{cut[:4]} h={h}", mkt=True))
        rows.append(blk(b, h, f"{cut[:4]}+   h={h}", mkt=True))
    show(rows, f"cut {cut[:4]}")

print("\n  by calendar year, h=5 market-relative:")
cy = cell.assign(x=cell.f5 - cell.s5).dropna(subset=["x"]).groupby(cell.rdate.dt.year)["x"]
print((cy.agg(["count", "mean"]).assign(mean=lambda d: (100 * d["mean"]).round(3))).to_string())

# ------------------------------------------------------------------ 4. liquidity
print("\n===== 4. LIQUID UNIVERSE ONLY vs EVERYTHING =====")
for h in (1, 3, 5, 10):
    rows = [blk(cell[cell.liq], h, f"LIQUID_PLUS_COMMODITIES h={h}", mkt=True),
            blk(cell[~cell.liq], h, f"everything else        h={h}", mkt=True),
            blk(cell[cell.liq & (cell.rdate >= "2013-01-01")], h, f"  liquid, 2013+ h={h}", mkt=True),
            blk(cell[cell.liq & (cell.rdate >= "2018-01-01")], h, f"  liquid, 2018+ h={h}", mkt=True)]
    show(rows, f"h={h}")

# ------------------------------------------------------------------ 5. threshold walk
print("\n===== 5. DEFINITION FRAGILITY: the adverse-move threshold =====")
rows = []
for th in (-0.03, -0.05, -0.07, -0.09, -0.12):
    for h in (1, 3, 5, 10):
        s = blk(df[BEAT & (df.react <= th)], h, f"react <= {100*th:.0f}% h={h}", mkt=True)
        rows.append(s)
show(rows, "percent thresholds (all names)")
rows = []
for k in (1.0, 1.5, 2.0, 3.0, 4.0):
    m = BEAT & (df.react <= -k * df.atrp)
    for h in (1, 3, 5, 10):
        rows.append(blk(df[m], h, f"react <= -{k} x ATR%  h={h}", mkt=True))
show(rows, "ATR-unit thresholds")

print("\n  where does WMT's -9.15% sit in the trigger population?")
pp = cell.react.dropna()
print(f"   trigger reactions: median {100*pp.median():.2f}%, "
      f"p10 {100*pp.quantile(0.10):.2f}%, p05 {100*pp.quantile(0.05):.2f}%")
for nm, x in [("WMT -9.15%", -0.0915), ("ROST", float(live.loc[live.ticker == 'ROST', 'react'].iloc[0]) if (live.ticker == 'ROST').any() else np.nan),
              ("TJX", float(live.loc[live.ticker == 'TJX', 'react'].iloc[0]) if (live.ticker == 'TJX').any() else np.nan)]:
    if np.isfinite(x):
        print(f"   {nm}: {100*(pp <= x).mean():.1f}th percentile by depth "
              f"(i.e. {100*(pp<=x).mean():.1f}% of triggers were worse)")

# ------------------------------------------------------------------ 6. convention fragility
print("\n===== 6. CONVENTION FRAGILITY (reaction = day0 only vs day+1 only) =====")
d0 = df.copy()
d0["react"] = d0["react_d0"]
d1 = df.copy()
d1["react"] = d1["react_d1"]
# forward returns for the pure conventions need re-anchoring; recompute quickly
print("  (argmax convention above; here the SIGN of the cell under each pure rule)")
for nm, sub in (("day-0 (BMO) reaction", df[df.bmo]), ("day+1 (AMC) reaction", df[~df.bmo])):
    show([blk(sub[(sub.surp > 0) & (sub.react <= TH)], h, f"{nm} h={h}", mkt=True)
          for h in (1, 3, 5, 10)], "")

# ------------------------------------------------------------------ 7. concentration
print("\n===== 7. CONCENTRATION by name and date, h=5 market-relative =====")
x = cell.assign(v=cell.f5 - cell.s5).dropna(subset=["v"])
byname = x.groupby("ticker")["v"].agg(["count", "sum"]).sort_values("sum")
print(" worst 5 names:", byname.head(5).round(3).to_dict("index"))
print(" best 5 names :", byname.tail(5).round(3).to_dict("index"))
tot = x["v"].sum()
top = x.reindex(x["v"].abs().sort_values(ascending=False).index).head(5)
print(f" top-5 |events| = {100*top['v'].sum():+.2f}pp of {100*tot:+.2f}pp total "
      f"({100*top['v'].sum()/tot:.0f}%)")
byyr = x.groupby(x.rdate.dt.year)["v"].sum()
print(f" best 2 years: {byyr.nlargest(2).mul(100).round(1).to_dict()}  "
      f"worst 2: {byyr.nsmallest(2).mul(100).round(1).to_dict()}")

# ------------------------------------------------------------------ 8. cost
print("\n===== 8. COST =====")
for h in (1, 3, 5, 10):
    a = blk(cell[cell.liq & (cell.rdate >= "2013-01-01")], h, "", mkt=True)
    if a.get("n", 0):
        print(f" h={h} liquid 2013+: {a['mean_pct']*100:+.1f} bps vs 10 bps long "
              f"round trip -> {a['mean_pct']*100/10:+.1f}x ; short adds borrow "
              f"(-{a['mean_pct']*100:+.1f} bps for the short side)")
