"""C1 RED TEAM r2 -- the remaining four attack surfaces.

  E. REGIME. The cell is one era (52/55 post-2018) BY CONSTRUCTION. That era is
     also the inflation-shock era. If the edge only exists when rate vol is
     high, and today's rate vol is low, the sample does not describe tonight.
  F. DEFINITION FRAGILITY. Perturb the cell definition and see if the number
     survives: 2018+ only, drop the 3 stragglers, CPI-on-eve by CALENDAR day
     vs by SESSION, and the effect of a one-session error in the calendar.
  G. BOOK OVERLAP. Actual correlation of TLT's one-day return with SPY (the
     staged book is 4 SELLs), with GDX (a live pitch to 08-17), and the
     conditional co-move on the cell's own dates.
  H. COST, TAIL and SIZE, at the developed entry form and today's numbers.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

ROOT_P = Path(__file__).resolve().parents[3]
mp = pd.read_parquet(ROOT_P / "data" / "master_prices.parquet")


def frame(t):
    g = mp[mp["ticker"] == t].copy()
    g["date"] = pd.to_datetime(g["date"])
    return g.sort_values("date").drop_duplicates("date", keep="last").set_index("date")


tl = frame("TLT")
idx = tl.index
c = tl["Close"].values.astype(float)
N = len(c)
d1 = np.full(N, np.nan)
d1[1:] = c[1:] / c[:-1] - 1.0
ok = ~np.isnan(d1)
base_hit = float((d1[ok] > 0).mean())

ecsv = pd.read_csv(ROOT_P / "data" / "macro_events.csv")
ecsv["date"] = pd.to_datetime(ecsv["date"])
sessd = lambda k: {int(idx.searchsorted(x, "left"))
                   for x in ecsv.loc[ecsv["event"] == k, "date"]
                   if 0 <= int(idx.searchsorted(x, "left")) < N}
PPI, CPI = sessd("ppi"), sessd("cpi")
ppi_l = sorted(p for p in PPI if 1 <= p < N and ok[p])
v = np.array([d1[p] for p in ppi_l])
dt = pd.DatetimeIndex([idx[p] for p in ppi_l])
mo, yr = dt.month.values, dt.year.values
L = np.array([(p - 1) in CPI for p in ppi_l])


def rep(x, lbl):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {"cell": lbl, "N": 0}
    w = int((x > 0).sum())
    sd = x.std(ddof=1) if len(x) > 1 else np.nan
    return {"cell": lbl, "N": len(x), "mean_bps": round(1e4 * x.mean(), 2),
            "hit": round(100 * w / len(x), 1),
            "t": round(x.mean() / (sd / np.sqrt(len(x))), 2),
            "signp": round(sign_test(w, len(x), base_hit), 4)}


# --------------------------------------------------------------------------
print("=" * 100)
print("E. REGIME: is the edge an inflation-shock artifact?")
print("=" * 100)
rv = pd.Series(d1, index=idx).rolling(21).std() * np.sqrt(252) * 100
rv_eve = np.array([rv.iloc[p - 1] for p in ppi_l])
today_rv = float(rv.iloc[-1])
print(f"  TLT 21d annualised realised vol, TODAY (2026-08-11 close) = {today_rv:.2f}%")
q = np.nanpercentile(rv_eve[L], [33, 67])
print(f"  live-cell terciles of eve-day TLT RV: <{q[0]:.1f} | {q[0]:.1f}-{q[1]:.1f} "
      f"| >{q[1]:.1f}")
lv = v[L]
lr = rv_eve[L]
print(pd.DataFrame([
    rep(lv[lr <= q[0]], "live cell, LOW rate vol"),
    rep(lv[(lr > q[0]) & (lr <= q[1])], "live cell, MID rate vol"),
    rep(lv[lr > q[1]], "live cell, HIGH rate vol"),
]).to_string(index=False))
tercile = "LOW" if today_rv <= q[0] else ("MID" if today_rv <= q[1] else "HIGH")
print(f"  -> today sits in the {tercile} tercile.")

try:
    mv = frame("^MOVE")["Close"].reindex(idx).ffill()
    mv_eve = np.array([mv.iloc[p - 1] for p in ppi_l])
    m_ok = ~np.isnan(mv_eve)
    med = np.nanmedian(mv_eve[L & m_ok])
    print(f"\n  ^MOVE today = {float(mv.iloc[-1]):.1f}; live-cell median eve MOVE "
          f"= {med:.1f}")
    print(pd.DataFrame([
        rep(v[L & m_ok & (mv_eve <= med)], "live cell, MOVE below its cell median"),
        rep(v[L & m_ok & (mv_eve > med)], "live cell, MOVE above its cell median"),
    ]).to_string(index=False))
except Exception as e:  # noqa
    print(f"  ^MOVE unavailable: {e}")

print("\n  by sub-era inside the one era:")
for lbl, m in [("2018-2019 (pre-shock)", (yr >= 2018) & (yr <= 2019)),
               ("2020-2022 (shock)", (yr >= 2020) & (yr <= 2022)),
               ("2023-2024 (disinflation)", (yr >= 2023) & (yr <= 2024)),
               ("2025-2026 (today's regime)", yr >= 2025)]:
    print(f"  {lbl:28s} {rep(v[L & m], '')}")

# --------------------------------------------------------------------------
print("\n" + "=" * 100)
print("F. DEFINITION FRAGILITY")
print("=" * 100)
cal_eve = np.array([any((idx[p] - x).days == 1 for x in ecsv.loc[ecsv.event == "cpi", "date"])
                    for p in ppi_l])
print(pd.DataFrame([
    rep(v[L], "baseline: CPI on the prior SESSION"),
    rep(v[cal_eve], "CPI on the prior CALENDAR day"),
    rep(v[L & (yr >= 2018)], "2018+ only (drops 3 stragglers)"),
    rep(v[L & (yr >= 2019)], "2019+ only"),
    rep(v[L & (yr <= 2024)], "through 2024 (drops the newest 9)"),
]).to_string(index=False))
print("\n  one-session calendar-error stress: shift EVERY ppi date by +/-1 session")
for sh in (-1, 1):
    ppi_s = sorted(p + sh for p in PPI if 1 <= p + sh < N and ok[p + sh])
    vs = np.array([d1[p] for p in ppi_s])
    Ls = np.array([(p - 1) in CPI for p in ppi_s])
    print(f"  shift {sh:+d}: {rep(vs[Ls], '')}")
print("  -> a cell that survives its own definition being wrong by a session is")
print("     a calendar artifact; one that dies is anchored to the release.")

print("\n  DECLUSTERING: prints are ~21td apart, so h=1 windows never overlap.")
gaps = np.diff([p for p, f in zip(ppi_l, L) if f])
print(f"  min gap between live-cell prints = {gaps.min()} sessions "
      f"(overlap needs < 1)")

# --------------------------------------------------------------------------
print("\n" + "=" * 100)
print("G. BOOK OVERLAP (measured, not asserted)")
print("=" * 100)
peers = ["SPY", "GDX", "IWM", "QQQ"]
pf = {t: frame(t)["Close"].reindex(idx) for t in peers}
pr = {t: pf[t].pct_change().values for t in peers}
recent = np.arange(N - 252, N)
print("  corr(TLT 1d, X 1d):")
rows = []
for t in peers:
    x = pr[t]
    m_all = ok & ~np.isnan(x)
    m_rec = m_all & np.isin(np.arange(N), recent)
    m_18 = m_all & np.array([d.year >= 2018 for d in idx])
    cell_i = [p for p, f in zip(ppi_l, L) if f]
    m_cell = np.zeros(N, bool)
    m_cell[cell_i] = True
    m_cell &= m_all
    rows.append({"ticker": t,
                 "full": round(np.corrcoef(d1[m_all], x[m_all])[0, 1], 3),
                 "2018+": round(np.corrcoef(d1[m_18], x[m_18])[0, 1], 3),
                 "last252d": round(np.corrcoef(d1[m_rec], x[m_rec])[0, 1], 3),
                 "on_cell_dates": round(np.corrcoef(d1[m_cell], x[m_cell])[0, 1], 3)})
print(pd.DataFrame(rows).to_string(index=False))
sp = pr["SPY"]
cell_i = np.array([p for p, f in zip(ppi_l, L) if f])
print(f"\n  On the cell's own 55 print sessions: TLT mean {1e4*d1[cell_i].mean():+.1f} bps, "
      f"SPY mean {1e4*np.nanmean(sp[cell_i]):+.1f} bps")
both = (d1[cell_i] > 0) & (sp[cell_i] > 0)
print(f"  TLT up AND SPY up on {int(np.nansum(both))}/{len(cell_i)} of them "
      f"({100*np.nansum(both)/len(cell_i):.0f}%)")
print("  The staged book today is FOUR SELLS (short equity). A long-TLT leg is")
print("  additive risk to a short-equity book only if TLT and SPY move together.")
print(f"  Last-252d TLT/SPY correlation is the number that decides it: "
      f"{np.corrcoef(d1[ok & ~np.isnan(sp) & np.isin(np.arange(N), recent)], sp[ok & ~np.isnan(sp) & np.isin(np.arange(N), recent)])[0,1]:+.3f}")

# --------------------------------------------------------------------------
print("\n" + "=" * 100)
print("H. COST, TAIL, SIZE")
print("=" * 100)
atr = wilder_atr(tl["High"].values, tl["Low"].values, tl["Close"].values, 14)
px_now = float(c[-1])
atr_now = float(atr[-1])
print(f"  TLT close 2026-08-11 = {px_now:.2f}   Wilder-14 ATR = {atr_now:.4f} "
      f"({100*atr_now/px_now:.3f}% of price)")

NAV = 750_000
for bps_risk in (30, 20, 15):
    risk_d = NAV * bps_risk / 1e4
    sh = int(risk_d / (1.0 * atr_now))
    notional = sh * px_now
    print(f"\n  at {bps_risk} bps NAV risk, 1.0 ATR unit: risk ${risk_d:,.0f}  "
          f"shares {sh:,}  notional ${notional:,.0f} = {100*notional/NAV:.1f}% NAV")
    for lbl, mv_pct in [("cell mean +25.8 bps", 0.2584),
                        ("month-adj estimate +16.0 bps", 0.1598),
                        ("cell worst day -2.33%", -2.331),
                        ("parent worst day -2.60%", -2.60),
                        ("1-sigma bad (-0.64%)", -0.638)]:
        pnl = notional * mv_pct / 100
        print(f"      {lbl:32s} ${pnl:+9,.0f}  = {pnl/risk_d:+.2f}R  "
              f"= {1e4*pnl/NAV:+.1f} bps NAV")

sh30 = int((NAV * 30 / 1e4) / atr_now)
notional30 = sh30 * px_now
print(f"\n  COST at 30 bps (notional ${notional30:,.0f}, {sh30:,} shares):")
comm = 0.0035 * sh30 * 2
print(f"    IBKR tiered commission ~$0.0035/sh both ways = ${comm:.2f} = "
      f"{1e4*comm/notional30:.2f} bps")
print(f"    TLT NBBO spread ~$0.01 on {px_now:.2f} = "
      f"{1e4*0.01/px_now:.2f} bps; MOC prints at the auction so pay ~half each "
      f"side = {1e4*0.01/px_now:.2f} bps round trip")
print(f"    auction/impact slack on {sh30:,} shares of a ~$1bn-ADV ETF: ~0.5 bps")
allin = 1e4 * comm / notional30 + 1e4 * 0.01 / px_now + 0.5
print(f"    ALL-IN ROUND TRIP ~= {allin:.1f} bps  (the prior report assumed 2.5)")
print(f"    vs month-adjusted edge +16.0 bps  -> {16.0/allin:.1f}x cost")
print(f"    vs raw cell edge      +25.8 bps  -> {25.84/allin:.1f}x cost")

print("\n  the losing tail, live cell:")
lv2 = v[L]
ld = dt[L]
for i in np.argsort(lv2)[:5]:
    p = [q for q, f in zip(ppi_l, L) if f][i]
    o = float(tl["Open"].values[p])
    print(f"    {ld[i].date()}  close-to-close {100*lv2[i]:+.2f}%  "
          f"overnight gap {100*(o/c[p-1]-1):+.2f}%  "
          f"intraday {100*(c[p]/o-1):+.2f}%")
print(f"  losers {int((lv2<=0).sum())}/{len(lv2)}  mean {100*lv2[lv2<=0].mean():+.3f}%  "
      f"worst {100*lv2.min():+.2f}%")
gapv = np.array([tl["Open"].values[p] / c[p - 1] - 1 for p, f in zip(ppi_l, L) if f])
print(f"  the overnight gap carries {1e4*gapv.mean():+.1f} of the "
      f"{1e4*lv2.mean():+.1f} bps ({100*gapv.mean()/lv2.mean():.0f}%); intraday "
      f"adds {1e4*(lv2.mean()-gapv.mean()):+.1f} bps")
print(f"  gap hit rate {100*(gapv>0).mean():.1f}%  worst gap {100*gapv.min():+.2f}%")
