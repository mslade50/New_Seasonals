"""C3 round 1 - SHORT SLV on the three-name metals break out of a PARABOLIC
miner run (GDX 21d rank >= 90).  Signal 2026-09-01, entry MOC 2026-09-02.

This candidate inherits watchlist entry 29's UNPAID DEBT verbatim: the parked
cell pays +0.516% at lag=1 but only +0.039% at lag=0 and +0.035% at lag=2, an
effect one session wide that STARTS A SESSION LATE, which no forced-
deleveraging continuation story predicts.  The parked entry says a future
instance must ALSO show the effect at lag=0 or lag=2 before it is tradeable.

The only genuinely NEW element today is the parabolic conditioner (GDX r21 >=
90) that watchlist 29 never carried.  So the decisive question is exactly one
thing: DOES CONDITIONING ON THE PRECEDING PARABOLIC RUN CHANGE THE LAG
PROFILE?  Everything else here is subordinate to that.

  S0. Live verify off the bars + watchlist 29's depth arm (needs SLV <= -4%).
  S1. THE LAG PROFILE, parabolic cell vs parent, lag 0/1/2/3 at h=1,2,3.
  S2. Gate attribution: does GDX r21>=90 ADD, or does it just remove rows?
  S3. Concentration: 2 of 12 episodes are the LIVE August-2026 episode.
  S4. Vehicle: SLV vs GLD vs GDX, and the GLD-beta-neutral residual.
  S5. Weekday / month-end (today is Wednesday, tdom 2 - confirm not binding).
  S6. Fragility dial on trigger episodes vs today's 87.5.
  S7. Cost + borrow.
  S8. Definition neighbours on the GLD leg (-1.5 the anchor map used vs -2.0
      the 08-31 teardown used) and on the r21 rank cut.
  S9. Book overlap.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 260)

BAR = pd.Timestamp("2026-09-01")
GAP = 5
COST_BPS = 6.0
BASE = ["GLD", "SLV", "GDX", "NEM", "SPY", "DX-Y.NYB"]

px = close_panel(BASE)
px = px[px.index <= BAR]
px = px.dropna(subset=["GLD", "SLV", "GDX"])
D = px.index
print(f"panel {D[0].date()} .. {D[-1].date()}  n={len(D)}")


def d1(t):
    return px[t] / px[t].shift(1) - 1.0


r1 = {t: d1(t) for t in BASE}
gdx_r21 = pct_rank(px["GDX"], 21).reindex(D)
gdx_r5 = pct_rank(px["GDX"], 5).reindex(D)

# ------------------------------------------------------------- S0. live verify
print("\n" + "=" * 100)
print("S0. LIVE VERIFY off the bars")
print("=" * 100)
print(f"  {D[-1].date()}:  SLV {100*r1['SLV'].iloc[-1]:+.2f}%  "
      f"GLD {100*r1['GLD'].iloc[-1]:+.2f}%  GDX {100*r1['GDX'].iloc[-1]:+.2f}%  "
      f"NEM {100*r1['NEM'].iloc[-1]:+.2f}%")
print(f"  GDX r21 rank {gdx_r21.iloc[-1]:.1f}   GDX r5 rank {gdx_r5.iloc[-1]:.1f}")
print(f"  WATCHLIST 29 DEPTH ARM: needs SLV <= -4.00%, live "
      f"{100*r1['SLV'].iloc[-1]:.2f}%  -> ARM MET: "
      f"{bool(r1['SLV'].iloc[-1] <= -0.04)}")

# the anchor map's definition (what selection used) and the 08-31 definition
BRK_MAP = (r1["SLV"] < -0.02) & (r1["GLD"] < -0.015) & (r1["GDX"] < -0.02)
BRK_831 = (r1["SLV"] <= -0.02) & (r1["GLD"] <= -0.02) & (r1["GDX"] <= -0.02)
PARA = BRK_MAP & (gdx_r21 >= 90)
print(f"\n  bare break (anchor-map def, GLD<-1.5%): {int(BRK_MAP.sum())} days")
print(f"  bare break (08-31 def, GLD<=-2%):       {int(BRK_831.sum())} days")
print(f"  parabolic cell (map def + GDX r21>=90): {int(PARA.sum())} days")
print(f"  LIVE today: bare {bool(BRK_MAP.iloc[-1])}  parabolic {bool(PARA.iloc[-1])}")


def epi_of(mask, gap=GAP, ret=None):
    m = mask.reindex(D, fill_value=False).fillna(False)
    days = D[m.values] if ret is None else D[m.values & ret.notna().values]
    return declusters(days, gap, D), days


def cell(mask, h, label, legs=(("SLV", -1.0),), lag=1, gap=GAP):
    r = vehicle_ret(px, list(legs), h, lag)
    e, days = epi_of(mask, gap, r)
    if len(e) == 0:
        return {"label": label, "n": 0}
    v = r.loc[e].values
    base = r.dropna()
    out = summarize(v, label)
    out["n_days"] = len(days)
    out["ctrl_pct"] = round(100 * base.mean(), 3)
    out["edge_pp"] = round(out["mean_pct"] - 100 * base.mean(), 3)
    w = int((v > 0).sum())
    out["rec"] = f"{w}-{len(v)-w}"
    out["p_vs_base"] = round(sign_test(w, len(v), float((base > 0).mean())), 4)
    return out


# ------------------------------------------------------- S1. THE LAG PROFILE
print("\n" + "=" * 100)
print("S1. THE LAG PROFILE - the debt watchlist 29 says must be paid")
print("=" * 100)
for h in (1, 2, 3):
    rows = []
    for lag in (0, 1, 2, 3):
        rows.append(cell(PARA, h, f"PARABOLIC lag={lag}", lag=lag))
        rows.append(cell(BRK_MAP, h, f"  bare break lag={lag}", lag=lag))
    show(rows, f"SHORT SLV, h={h} - parabolic cell vs bare break, by entry lag")

print("\n  SIDE BY SIDE at h=1 (the horizon watchlist 29 traded):")
for lag in (0, 1, 2, 3):
    a = cell(PARA, 1, "", lag=lag)
    b = cell(BRK_MAP, 1, "", lag=lag)
    print(f"    lag={lag}:  parabolic {a.get('mean_pct', float('nan')):+7.3f}% "
          f"(N={a.get('n', 0)}, {a.get('rec', '-')})   "
          f"bare {b.get('mean_pct', float('nan')):+7.3f}% "
          f"(N={b.get('n', 0)}, {b.get('rec', '-')})")

# -------------------------------------------------------- S2. gate attribution
print("\n" + "=" * 100)
print("S2. GATE ATTRIBUTION - does GDX r21>=90 ADD, or just remove rows?")
print("=" * 100)
for h in (1, 3, 5):
    rows = [
        cell(BRK_MAP, h, "PARENT bare 3-name break"),
        cell(PARA, h, "AND GDX r21>=90 (the pitch)"),
        cell(BRK_MAP & (gdx_r21 < 90), h, "AND GDX r21<90 (the DISCARDS)"),
        cell(BRK_MAP & (gdx_r21 >= 80), h, "AND GDX r21>=80 (looser)"),
        cell(BRK_MAP & (gdx_r21 >= 95), h, "AND GDX r21>=95 (tighter)"),
        cell(PARA & (gdx_r5 <= 25), h, "AND GDX r5<=25 (the 4-episode rung)"),
    ]
    show(rows, f"SHORT SLV h={h}, gate ladder")
    a = cell(PARA, h, "")
    b = cell(BRK_MAP & (gdx_r21 < 90), h, "")
    p = cell(BRK_MAP, h, "")
    if a.get("n") and b.get("n"):
        va = vehicle_ret(px, [("SLV", -1.0)], h, 1).loc[epi_of(PARA, GAP,
             vehicle_ret(px, [("SLV", -1.0)], h, 1))[0]].values
        vb = vehicle_ret(px, [("SLV", -1.0)], h, 1).loc[epi_of(
             BRK_MAP & (gdx_r21 < 90), GAP,
             vehicle_ret(px, [("SLV", -1.0)], h, 1))[0]].values
        se = np.sqrt(va.var(ddof=1) / len(va) + vb.var(ddof=1) / len(vb))
        print(f"  h={h}: gate ON {a['mean_pct']:+.3f}% vs OFF {b['mean_pct']:+.3f}% "
              f"vs PARENT {p['mean_pct']:+.3f}%  -> gate adds "
              f"{a['mean_pct']-p['mean_pct']:+.3f}pp over the parent, "
              f"welch t(on-off) {(va.mean()-vb.mean())/se:+.2f}")

# --------------------------------------------------------- S3. concentration
print("\n" + "=" * 100)
print("S3. CONCENTRATION - 2 of 12 episodes are the LIVE 2026 episode")
print("=" * 100)
for h in (1, 3):
    r = vehicle_ret(px, [("SLV", -1.0)], h, 1)
    e, _ = epi_of(PARA, GAP, r)
    v = r.loc[e].values
    print(f"\n  h={h}: {len(e)} episodes, mean {100*v.mean():+.3f}%, "
          f"total {100*v.sum():+.2f}pp")
    for d, x in zip(e, v):
        print(f"     {d.date()}  {100*x:+7.2f}%")
    yr = pd.DatetimeIndex(e).year
    for y in sorted(set(yr)):
        m = yr == y
        print(f"     year {y}: N={int(m.sum())} mean {100*v[m].mean():+.3f}% "
              f"contributes {100*v[m].sum():+.2f}pp of {100*v.sum():+.2f}pp "
              f"({100*v[m].sum()/v.sum()*100 if v.sum() else float('nan'):.0f}%)")
    m26 = yr == 2026
    print(f"    ex-2026 (drop the LIVE episode): mean "
          f"{100*v[~m26].mean():+.3f}% on N={int((~m26).sum())}, "
          f"record {int((v[~m26]>0).sum())}-{int((v[~m26]<=0).sum())}")
    order = np.argsort(-v)
    print(f"    drop-best-2: {100*np.delete(v, order[:2]).mean():+.3f}%  "
          f"median {100*np.median(v):+.3f}%")
    print(f"    bootstrap P(mean<=0) = {bootstrap_p_le0(v):.3f}")

# ------------------------------------------------------------- S4. vehicle
print("\n" + "=" * 100)
print("S4. VEHICLE - SLV vs GLD vs GDX, and the GLD-beta-neutral residual")
print("=" * 100)
for h in (1, 3):
    rows = []
    for t in ("SLV", "GLD", "GDX", "NEM"):
        rows.append(cell(PARA, h, f"SHORT {t}", legs=(("{}".format(t), -1.0),)))
    show(rows, f"vehicle sweep, parabolic cell, h={h}")

print("\n  PIT trailing-252 beta residual, SHORT SLV hedged with GLD:")
dslv = px["SLV"].pct_change()
for bench in ("GLD", "SPY", "DX-Y.NYB"):
    dbe = px[bench].pct_change()
    beta = (dslv.rolling(252).cov(dbe) / dbe.rolling(252).var()).shift(1)
    rows = []
    for h in (1, 2, 3):
        fs = fwd_lag(px["SLV"], h, 1)
        fb = fwd_lag(px[bench], h, 1)
        resid = -(fs - beta * fb)
        e, _ = epi_of(PARA, GAP, resid)
        rr = summarize(resid.loc[e].values, f"h={h} SHORT resid vs {bench}")
        rr["ctrl_pct"] = round(100 * resid.dropna().mean(), 3)
        rr["edge_pp"] = round(rr["mean_pct"] - 100 * resid.dropna().mean(), 3)
        rows.append(rr)
    show(rows, f"beta-neutral vs {bench} (mean beta {beta.dropna().mean():.2f}, "
               f"today {beta.iloc[-1]:.2f})")

# --------------------------------------------------- S5. weekday / month-end
print("\n" + "=" * 100)
print("S5. WEEKDAY / MONTH-END (today Wed 2026-09-02 entry, tdom 2)")
print("=" * 100)
r = vehicle_ret(px, [("SLV", -1.0)], 1, 1)
e, _ = epi_of(PARA, GAP, r)
v = r.loc[e].values
wd = pd.DatetimeIndex(e).dayofweek
print("  trigger weekday (0=Mon):", dict(pd.Series(wd).value_counts().sort_index()))
for k in sorted(set(wd)):
    m = wd == k
    print(f"    trigger wd={k}: N={int(m.sum())} mean {100*v[m].mean():+.3f}%")
pos = pd.Series(range(len(D)), index=D)
ent = [D[pos[d] + 1] for d in e if pos[d] + 1 < len(D)]
me = np.array([1 if (i + 1 >= len(D) or D[pos[d] + 1].month
                     != D[min(pos[d] + 2, len(D) - 1)].month) else 0
               for i, d in enumerate(e) if pos[d] + 1 < len(D)])
print(f"  entry sessions that are month-END: {int(me.sum())} of {len(me)}")
tdom = []
for d in e:
    p = pos[d] + 1
    if p >= len(D):
        continue
    ed = D[p]
    tdom.append(int((pd.DatetimeIndex([x for x in D if x.year == ed.year
                                       and x.month == ed.month]) <= ed).sum()))
print(f"  entry trading-day-of-month values: {tdom}  (today's entry = tdom 2)")

# ---------------------------------------------------------- S6. fragility dial
print("\n" + "=" * 100)
print("S6. FRAGILITY DIAL on trigger episodes vs today's ma10-63d 87.5")
print("=" * 100)
fr = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
fr.index = pd.to_datetime(fr.index)
ma10 = fr["63d"].rolling(10).mean()
print(f"  dial series {fr.index[0].date()} .. {fr.index[-1].date()}; "
      f"today ma10-63d {ma10.iloc[-1]:.1f}")
cov = [(d, float(ma10.reindex([d]).iloc[0])) for d in e]
cov = [(d, x) for d, x in cov if np.isfinite(x)]
print(f"  episodes with a dial reading: {len(cov)} of {len(e)}")
for d, x in cov:
    print(f"     {d.date()}  dial {x:.1f}")
if cov:
    print(f"  episode dial MAX {max(x for _, x in cov):.1f} vs TODAY "
          f"{ma10.iloc[-1]:.1f}")

# ---------------------------------------------------------------- S7. cost
print("\n" + "=" * 100)
print("S7. COST + BORROW")
print("=" * 100)
for h in (1, 3):
    c = cell(PARA, h, "")
    edge = 100 * c["mean_pct"]
    for borrow in (0.0, 0.5, 1.0):
        carry = borrow / 100.0 * (h / 252.0) * 1e4
        print(f"  h={h}: edge {edge:7.1f} bp | {COST_BPS} bp round trip + "
              f"{borrow:.1f}%/yr borrow ({carry:.2f} bp) -> "
              f"{edge/(COST_BPS+carry):.2f}x cost")

# ------------------------------------------------------ S8. definition ladder
print("\n" + "=" * 100)
print("S8. DEFINITION NEIGHBOURS")
print("=" * 100)
for h in (1, 3):
    rows = []
    for gl in (-0.010, -0.015, -0.020, -0.025):
        m = (r1["SLV"] < -0.02) & (r1["GLD"] < gl) & (r1["GDX"] < -0.02) \
            & (gdx_r21 >= 90)
        rows.append(cell(m, h, f"GLD leg < {100*gl:.1f}% (map used -1.5)"))
    for sl in (-0.015, -0.020, -0.025, -0.030):
        m = (r1["SLV"] < sl) & (r1["GLD"] < -0.015) & (r1["GDX"] < -0.02) \
            & (gdx_r21 >= 90)
        rows.append(cell(m, h, f"SLV leg < {100*sl:.1f}% (map used -2.0)"))
    show(rows, f"definition neighbours, h={h}")

# ------------------------------------------------------------ S9. book overlap
print("\n" + "=" * 100)
print("S9. BOOK OVERLAP")
print("=" * 100)
led = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
led["Signal Date"] = pd.to_datetime(led["Signal Date"])
METALS = {"SLV", "GLD", "GDX", "NEM", "AEM", "KGC", "AU", "GOLD", "GFI",
          "PAAS", "WPM", "FNV", "RGLD", "HL", "CDE", "AG", "EXK", "SIL",
          "GDXJ", "IAU", "DUST", "NUGT", "JDST", "JNUG", "AGQ", "ZSL"}
tdays = set(D[PARA.fillna(False).values])
mm = led[led["Ticker"].isin(METALS)]
on = mm[mm["Signal Date"].isin(tdays)]
print(f"  ledger rows {len(led)}, metals rows {len(mm)}, "
      f"metals signals ON a parabolic trigger day: {len(on)}")
if len(on):
    print(on.groupby(["Strategy", "Direction"])["R_Multiple"]
          .agg(["count", "mean"]).round(3).to_string())
slvled = led[led["Ticker"] == "SLV"]
print(f"  SLV ledger rows {len(slvled)}: "
      f"{slvled['Direction'].value_counts().to_dict() if len(slvled) else {}}")
