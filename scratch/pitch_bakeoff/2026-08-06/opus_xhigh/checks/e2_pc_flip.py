"""E2 - Put/call complacency FLIP (flow_mechanics).

Cell: daily equity P/C trailing-252d pctile <= 10 WHILE the 10d-MA of the same
ratio sits at >= its 50th pctile (fast flip from residual fear to complacency).
Controls: each leg alone. Measure: forward SPY 3/5/10 sessions, MOO basis LEADS.

Adversarial brief:
 - lag discipline: the live pitch sees a P/C row dated D-1 bday relative to the
   signal bar (== entry date - 2 bdays). Everything is built with that lag; a
   no-lag version is reported as the contrast.
 - is the interaction anything beyond the single-leg "daily P/C low" cell?
 - is this the already-REJECTED "Equity P/C Complacency" dial finding in new
   clothes (day-level edge = overlap inflation, episode t wrong-signed)?
 - is it just "SPY ripped for a week" restated?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _common as C

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)

PC_PATH = C.ROOT / "data" / "cboe_putcall.parquet"
HORIZONS = [3, 5, 10]
RANK_WINDOW = 252
MA_DAYS = 10


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


def rolling_pct_rank(s: pd.Series, window: int = RANK_WINDOW) -> pd.Series:
    """EXACT pc_fear._rolling_pct_rank convention: (w <= w[-1]).mean()*100."""
    return s.rolling(window, min_periods=window).apply(
        lambda w: (w <= w[-1]).mean() * 100.0, raw=True)


# ----------------------------------------------------------------- data
hdr("E2.0  DATA + LAG CONSTRUCTION")
pc = pd.read_parquet(PC_PATH)
pc.index = pd.to_datetime(pc.index)
pc = pc[pc.index < C.ASOF_EXCL].sort_index()
eq = pc["equity"].dropna()
print(f"  cboe_putcall equity: {len(eq)} rows {eq.index.min().date()} .. {eq.index.max().date()}")
print(f"  newest row {eq.index[-1].date()} value {eq.iloc[-1]:.2f}  "
      f"(signal bar is {C.LAST_BAR.date()}, entry {C.ASOF_EXCL.date()})")

ma10 = eq.rolling(MA_DAYS, min_periods=MA_DAYS).mean()
daily_pct = rolling_pct_rank(eq)
ma_pct = rolling_pct_rank(ma10)
pcdf = pd.DataFrame({"eq": eq, "ma10": ma10,
                     "daily_pct": daily_pct, "ma_pct": ma_pct}).dropna()
print(f"  usable pctile history: {len(pcdf)} rows "
      f"{pcdf.index.min().date()} .. {pcdf.index.max().date()}")
print(f"  TODAY (newest row {pcdf.index[-1].date()}): eq={pcdf['eq'].iloc[-1]:.2f} "
      f"daily_pct={pcdf['daily_pct'].iloc[-1]:.1f}  ma10={pcdf['ma10'].iloc[-1]:.3f} "
      f"ma_pct={pcdf['ma_pct'].iloc[-1]:.1f}")

px = C.load(["SPY"])
spy = px["SPY"]
bars = spy.index

FWD = {}
for k in HORIZONS:
    FWD[("moo", k)] = C.fwd_from_next_open(spy, k)
    FWD[("close", k)] = C.fwd(spy["Close"], k)

spy_r5_rank = C.pct_rank(C.ret(spy["Close"], 5), 252)


def align(lag_bd: int) -> pd.DataFrame:
    """For each SPY bar D, the newest P/C row dated <= D - lag_bd bdays."""
    left = pd.DataFrame({"bar": bars})
    if lag_bd > 0:
        left["cutoff"] = left["bar"] - pd.tseries.offsets.BDay(lag_bd)
    else:
        left["cutoff"] = left["bar"]
    right = pcdf.reset_index().rename(columns={"date": "pc_date", "index": "pc_date"})
    if "pc_date" not in right.columns:
        right = right.rename(columns={right.columns[0]: "pc_date"})
    out = pd.merge_asof(left.sort_values("cutoff"), right.sort_values("pc_date"),
                        left_on="cutoff", right_on="pc_date", direction="backward")
    out = out.set_index("bar").sort_index()
    out["age_bd"] = [np.busday_count(pd.Timestamp(p).date(), pd.Timestamp(b).date())
                     if pd.notna(p) else np.nan
                     for p, b in zip(out["pc_date"], out.index)]
    return out


AL = {lag: align(lag) for lag in (0, 1, 2)}
for lag, a in AL.items():
    v = a.dropna(subset=["daily_pct"])
    print(f"  lag {lag} bd: {len(v)} usable bars, median age {v['age_bd'].median():.0f} bd, "
          f"stale(>3bd) {int((v['age_bd'] > 3).sum())} bars "
          f"({(v['age_bd'] > 3).mean()*100:.1f}%)")
a1 = AL[1]
print(f"\n  LIVE CHECK - alignment for signal bar {C.LAST_BAR.date()} at lag 1:")
print(a1.loc[[C.LAST_BAR]][["pc_date", "eq", "daily_pct", "ma10", "ma_pct", "age_bd"]].to_string())


# ----------------------------------------------------------------- cells
def cells_for(a: pd.DataFrame) -> dict:
    ok = a["daily_pct"].notna() & (a["age_bd"] <= 3)
    dp, mp = a["daily_pct"], a["ma_pct"]
    return {
        "FLIP  daily<=10 & ma>=50": ok & (dp <= 10) & (mp >= 50),
        "LEG1  daily<=10 (alone)": ok & (dp <= 10),
        "LEG2  ma>=50 (alone)": ok & (mp >= 50),
        "ANTI  daily<=10 & ma<50": ok & (dp <= 10) & (mp < 50),
        "REJECTED-DIAL ma<10": ok & (mp < 10),
        "ALL usable bars": ok,
    }


hdr("E2.1  CELL SIZES + does today fire? (lag 1 bd = the live construction)")
cs = cells_for(a1)
for nm, m in cs.items():
    print(f"  {nm:28s} n={int(m.sum()):5d}  rate={m.mean()*100:5.2f}%  "
          f"fires_on_{C.LAST_BAR.date()}={bool(m.loc[C.LAST_BAR])}")

hdr("E2.2  MAIN GRID - forward SPY, MOO basis LEADS (lag 1 bd)")
rows = []
for nm, m in cs.items():
    if nm == "ALL usable bars":
        continue
    dts = a1.index[m.values]
    for basis in ("moo", "close"):
        for k in HORIZONS:
            f = FWD[(basis, k)]
            v = f.reindex(dts).to_numpy()
            base = f.reindex(a1.index[cs["ALL usable bars"].values]).dropna().to_numpy()
            d = C.describe(f"{nm} | {basis} k={k}", v, base)
            for g in (10, 21):
                ed, ev = episodes(dts, v, g)
                d[f"ep{g}_n"] = int(np.isfinite(ev).sum())
                d[f"ep{g}_t"] = round(C.tstat(ev), 2)
            rows.append(d)
C.show(rows)

hdr("E2.3  THE INTERACTION TEST - does ma>=50 add anything on top of daily<=10?")
for basis in ("moo", "close"):
    for k in HORIZONS:
        f = FWD[(basis, k)]
        m_hi = cs["FLIP  daily<=10 & ma>=50"]
        m_lo = cs["ANTI  daily<=10 & ma<50"]
        x = f.reindex(a1.index[m_hi.values]).dropna().to_numpy()
        y = f.reindex(a1.index[m_lo.values]).dropna().to_numpy()
        if len(x) > 2 and len(y) > 2:
            se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
            welch = (x.mean() - y.mean()) / se
        else:
            welch = np.nan
        print(f"  {basis} k={k:2d}: FLIP n={len(x):4d} avg={x.mean():+.3f} | "
              f"ANTI n={len(y):4d} avg={y.mean():+.3f} | "
              f"difference {x.mean()-y.mean():+.3f}pp Welch t={welch:+.2f}")

hdr("E2.4  LAG SENSITIVITY - how much does the honest 1-2 bd lag cost?")
rows = []
for lag in (0, 1, 2):
    a = AL[lag]
    c = cells_for(a)["FLIP  daily<=10 & ma>=50"]
    dts = a.index[c.values]
    for k in HORIZONS:
        f = FWD[("moo", k)]
        v = f.reindex(dts).to_numpy()
        d = C.describe(f"FLIP lag={lag}bd moo k={k}", v)
        ed, ev = episodes(dts, v, 10)
        d["ep10_n"] = int(np.isfinite(ev).sum())
        d["ep10_t"] = round(C.tstat(ev), 2)
        rows.append(d)
C.show(rows)

hdr("E2.5  DEEP DIVE - FLIP cell, MOO basis, lag 1")
dts = a1.index[cs["FLIP  daily<=10 & ma>=50"].values]
print(f"  n signal days = {len(dts)}  {dts.min().date()} .. {dts.max().date()}")
for k in HORIZONS:
    f = FWD[("moo", k)]
    v = f.reindex(dts).to_numpy()
    print(f"\n  --- k={k} MOO ---")
    rr = [C.describe(f"FLIP k={k} signal-days", v)]
    for g in (10, 21):
        ed, ev = episodes(dts, v, g)
        rr.append(C.describe(f"FLIP k={k} episodes gap{g}", ev))
    C.show(rr)
    ed, ev = episodes(dts, v, 10)
    floor, tbl = loyo_floor(ed, ev)
    print(f"  LOYO (episodes gap10) floor t = {floor}")
    print("   " + " ".join(f"{y}:{t}" for y, _, _, t in tbl))
    print("  era split (signal-days, cut 2018):")
    C.show(C.era_split(dts, v))
    print("  per-year:")
    print(per_year(dts, v).to_string())

hdr("E2.6  CONFOUND - is the FLIP cell just 'SPY ripped for a week'?")
r5r = spy_r5_rank
print(f"  today's SPY 5d rank = {r5r.iloc[-1]:.1f}")
mflip = cs["FLIP  daily<=10 & ma>=50"]
dts = a1.index[mflip.values]
print(f"  mean SPY 5d rank in FLIP cell = {r5r.reindex(dts).mean():.1f} "
      f"(unconditional {r5r.dropna().mean():.1f})")
print(f"  share of FLIP days with SPY 5d rank >= 90 = "
      f"{(r5r.reindex(dts) >= 90).mean()*100:.1f}% "
      f"(unconditional {(r5r.dropna() >= 90).mean()*100:.1f}%)")
ok_all = cs["ALL usable bars"]
for k in HORIZONS:
    f = FWD[("moo", k)]
    m_hot = ok_all & (r5r >= 90).reindex(a1.index).fillna(False)
    d_hot = a1.index[m_hot.values]
    both = a1.index[(mflip & (r5r >= 90).reindex(a1.index).fillna(False)).values]
    flip_only = a1.index[(mflip & ~(r5r >= 90).reindex(a1.index).fillna(False)).values]
    hot_only = a1.index[(m_hot & ~mflip).values]
    rr = [C.describe(f"k={k} SPY r5rank>=90 (no P/C)", f.reindex(d_hot).to_numpy()),
          C.describe(f"k={k} FLIP & hot", f.reindex(both).to_numpy()),
          C.describe(f"k={k} FLIP & NOT hot", f.reindex(flip_only).to_numpy()),
          C.describe(f"k={k} hot & NOT flip", f.reindex(hot_only).to_numpy())]
    C.show(rr)

hdr("E2.7  IS THIS THE REJECTED 'Equity P/C Complacency' FINDING?")
m_rej = cs["REJECTED-DIAL ma<10"]
m_flip = cs["FLIP  daily<=10 & ma>=50"]
inter = int((m_rej & m_flip).sum())
print(f"  rejected-dial cell (ma10 pctile < 10) days: {int(m_rej.sum())}")
print(f"  FLIP cell days:                            {int(m_flip.sum())}")
print(f"  overlap:                                   {inter}  "
      "(zero by construction - the FLIP cell REQUIRES ma>=50)")
print("  ... but both are 'low put/call = complacency = short equity' in economics.")
print("  Sign check - the rejected finding's failure mode was day-level edge that")
print("  vanished / flipped on episodes. Same table for BOTH cells, MOO k=5:")
rr = []
for nm in ("REJECTED-DIAL ma<10", "FLIP  daily<=10 & ma>=50", "LEG1  daily<=10 (alone)"):
    dd = a1.index[cs[nm].values]
    for k in HORIZONS:
        f = FWD[("moo", k)]
        v = f.reindex(dd).to_numpy()
        d = C.describe(f"{nm} k={k}", v)
        ed, ev = episodes(dd, v, 10)
        d["ep10_n"] = int(np.isfinite(ev).sum())
        d["ep10_t"] = round(C.tstat(ev), 2)
        d["day_vs_ep_sign_flip"] = bool(np.sign(d["avg"]) != np.sign(np.nanmean(ev)))
        rr.append(d)
C.show(rr)

hdr("E2.8  DAILY-PCTILE SENSITIVITY - is <=10 a cliff or a knob?")
rows = []
for thr in (5, 10, 15, 20, 25):
    for k in (3, 5, 10):
        ok = a1["daily_pct"].notna() & (a1["age_bd"] <= 3)
        m = ok & (a1["daily_pct"] <= thr) & (a1["ma_pct"] >= 50)
        dd = a1.index[m.values]
        f = FWD[("moo", k)]
        v = f.reindex(dd).to_numpy()
        d = C.describe(f"daily<={thr} & ma>=50 k={k}", v)
        ed, ev = episodes(dd, v, 10)
        d["ep10_n"] = int(np.isfinite(ev).sum())
        d["ep10_t"] = round(C.tstat(ev), 2)
        rows.append(d)
C.show(rows)

hdr("E2.9  MA-THRESHOLD SENSITIVITY (with daily<=10 fixed)")
rows = []
for thr in (40, 50, 60, 70):
    for k in (3, 5, 10):
        ok = a1["daily_pct"].notna() & (a1["age_bd"] <= 3)
        m = ok & (a1["daily_pct"] <= 10) & (a1["ma_pct"] >= thr)
        dd = a1.index[m.values]
        f = FWD[("moo", k)]
        v = f.reindex(dd).to_numpy()
        d = C.describe(f"daily<=10 & ma>={thr} k={k}", v)
        ed, ev = episodes(dd, v, 10)
        d["ep10_n"] = int(np.isfinite(ev).sum())
        d["ep10_t"] = round(C.tstat(ev), 2)
        rows.append(d)
C.show(rows)

hdr("E2.10  TIGHTEST ANALOGUES to today (daily<=10 & ma_pct in 60..85)")
ok = a1["daily_pct"].notna() & (a1["age_bd"] <= 3)
m = ok & (a1["daily_pct"] <= 10) & (a1["ma_pct"] >= 60) & (a1["ma_pct"] <= 85)
dd = a1.index[m.values]
tab = pd.DataFrame({
    "pc_date": a1["pc_date"].reindex(dd),
    "eq": a1["eq"].reindex(dd).round(2),
    "daily_pct": a1["daily_pct"].reindex(dd).round(1),
    "ma_pct": a1["ma_pct"].reindex(dd).round(1),
    "spy_r5rank": spy_r5_rank.reindex(dd).round(0),
    "fwd3_moo": FWD[("moo", 3)].reindex(dd).round(2),
    "fwd5_moo": FWD[("moo", 5)].reindex(dd).round(2),
    "fwd10_moo": FWD[("moo", 10)].reindex(dd).round(2),
})
print(f"  n={len(dd)}")
print(tab.to_string())
for k in HORIZONS:
    v = FWD[("moo", k)].reindex(dd).to_numpy()
    ed, ev = episodes(dd, v, 10)
    print(f"  k={k}: days n={int(np.isfinite(v).sum())} avg={np.nanmean(v):+.3f} "
          f"t={C.tstat(v):+.2f} | episodes gap10 n={len(ev)} avg={np.nanmean(ev):+.3f} "
          f"t={C.tstat(ev):+.2f}")

hdr("E2.END")
