"""C19 teardown - SHORT SLV after a whole-metals-complex break (GLD, SLV, GDX
each <= -2% on the same session).  Fired 2026-08-28, entry MOC 2026-08-31.

The 5:10 AM run's b2_c3 / b2_c3b left this standing at h=1/h=3.  This script
does NOT re-derive round 1; it charges the eight objections that decide it:

  0.  LIVE VERIFY the trigger off the bars, and today's SLV state.
  1.  RECORD scored against SLV's OWN unconditional DOWN-rate, plus a proper
      right-tail concentration measure (cluster_note picks top-k by ABSOLUTE
      value and NETS a big winner against a big loser -- the "3% of total"
      figure in _out_c3.txt is that artifact).
  2.  GATE ATTRIBUTION: does the 3-way conjunction beat "SLV alone fell 2%"?
      Plus SLV+GLD (no GDX) and SLV+GDX (no GLD), and the INCREMENTAL cell
      (SLV broke, the others did NOT).
  3.  DEFINITION LADDER on the -2% threshold AND a split at TODAY'S LIVE DEPTH.
  4.  MIDTERM split at every horizon.
  5.  LOCAL +/-126td control explicitly, and PIT beta residuals vs GLD, the
      dollar and SPY.
  6.  Is today's state INSIDE the sample?  r63 / dist-200d / dist-52wh across
      the episodes vs today.
  7.  COST + borrow at the horizon actually traded.
  8.  BOOK OVERLAP with ASSERTED column names.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)

BAR = pd.Timestamp("2026-08-28")
GAP = 5
COST_BPS = 6.0
BASE = ["GLD", "SLV", "GDX", "SPY", "DX-Y.NYB", "^TNX"]

px = close_panel(BASE).dropna()
px = px.loc[:BAR]
print(f"panel {px.index[0].date()} .. {px.index[-1].date()}  n={len(px)}")


def dret(s):
    return s / s.shift(1) - 1.0


r1 = {t: dret(px[t]) for t in BASE}

# --------------------------------------------------------------- 0. live verify
print("\n" + "=" * 100)
print("0. LIVE VERIFY - did the trigger actually fire on the freshest bar?")
print("=" * 100)
last = px.index[-1]
print(f"  freshest bar {last.date()}:  GLD {100*r1['GLD'].iloc[-1]:+.2f}%  "
      f"SLV {100*r1['SLV'].iloc[-1]:+.2f}%  GDX {100*r1['GDX'].iloc[-1]:+.2f}%")
FIRED = (r1["GLD"].iloc[-1] <= -0.02 and r1["SLV"].iloc[-1] <= -0.02
         and r1["GDX"].iloc[-1] <= -0.02)
print(f"  trigger fired on {last.date()}: {FIRED}   -> entry MOC next session "
      f"(2026-08-31)")

slv = px["SLV"]
r63_slv = slv.pct_change(63)
sma200 = slv.rolling(200).mean()
hi52 = slv.rolling(252).max()
print(f"  TODAY'S SLV STATE: r63 {100*r63_slv.iloc[-1]:+.2f}%   "
      f"vs 200d {100*(slv.iloc[-1]/sma200.iloc[-1]-1):+.2f}%   "
      f"vs 52w high {100*(slv.iloc[-1]/hi52.iloc[-1]-1):+.2f}%   "
      f"r21 rank {pct_rank(slv, 21).iloc[-1]:.1f}")

trig = (r1["GLD"] <= -0.02) & (r1["SLV"] <= -0.02) & (r1["GDX"] <= -0.02)
print(f"  day-level triggers in sample: {int(trig.sum())}")

# ------------------------------------------------------------------- machinery
LEGS = [("SLV", -1.0)]


def ret_at(h, lag=1, legs=LEGS):
    return vehicle_ret(px, legs, h, lag)


def base_downrate(h):
    r = ret_at(h).dropna()
    return float((r > 0).mean())  # short wins = SLV fell


def cellstats(mask, h, label, gap=GAP, legs=LEGS):
    r = ret_at(h, legs=legs)
    v = r.notna()
    days = px.index[mask.reindex(px.index, fill_value=False).values & v.values]
    if len(days) == 0:
        return {"label": label, "n": 0}, pd.DatetimeIndex([]), np.array([])
    epi = declusters(days, gap, px.index)
    vals = r.loc[epi].values
    base = r[v]
    w = int((vals > 0).sum())
    p0 = float((base > 0).mean())
    out = summarize(vals, label)
    out["n_days"] = len(days)
    out["edge_pp"] = out["mean_pct"] - 100 * base.mean()
    out["rec"] = f"{w}-{len(vals)-w}"
    out["p_vs_base"] = round(sign_test(w, len(vals), p0), 4)
    out["p_vs_coin"] = round(sign_test(w, len(vals)), 4)
    out["x_cost"] = round(100 * out["mean_pct"] / COST_BPS, 2)
    return out, epi, vals


print("\n" + "=" * 100)
print("1. THE RECORD against SLV'S OWN DOWN-RATE, and honest concentration")
print("=" * 100)
rows = []
for h in (1, 2, 3, 4, 5, 6, 8, 10):
    s, epi, vals = cellstats(trig, h, f"SHORT SLV h={h}")
    s["slv_downrate_pct"] = round(100 * base_downrate(h), 2)
    rows.append(s)
show(rows, "episode-level (gap 5), record scored vs SLV's own unconditional down-rate")

for h in (1, 3, 5):
    s, epi, vals = cellstats(trig, h, "")
    order = np.argsort(-vals)
    tot = vals.sum()
    print(f"\n  h={h}: total {100*tot:+.2f}pp over {len(vals)} episodes, "
          f"mean {100*vals.mean():+.3f}%")
    for k in (1, 2, 3, 5):
        top = vals[order[:k]].sum()
        print(f"    drop-best-{k}: mean {100*np.delete(vals, order[:k]).mean():+.3f}%"
              f"   (top-{k} by VALUE = {100*top:+.2f}pp = {100*top/tot:.0f}% of total)")
    print(f"    median {100*np.median(vals):+.3f}%   trimmed-10% "
          f"{100*float(pd.Series(vals).sort_values().iloc[len(vals)//10: len(vals)-len(vals)//10].mean()):+.3f}%")
    print(f"    cluster_note (top-k by ABS, the netting artifact): "
          f"{cluster_note(epi, vals, 2)}")

# ------------------------------------------------------- 2. GATE ATTRIBUTION
print("\n" + "=" * 100)
print("2. GATE ATTRIBUTION - does the CONJUNCTION beat 'SLV alone fell 2%'?")
print("=" * 100)
g = {"GLD": r1["GLD"] <= -0.02, "SLV": r1["SLV"] <= -0.02, "GDX": r1["GDX"] <= -0.02}
gates = {
    "SLV alone <= -2% (PARENT)": g["SLV"],
    "SLV & GLD (no GDX req)": g["SLV"] & g["GLD"],
    "SLV & GDX (no GLD req)": g["SLV"] & g["GDX"],
    "ALL THREE (the pitch)": g["SLV"] & g["GLD"] & g["GDX"],
    "SLV broke, GLD & GDX did NOT (anti-cell)": g["SLV"] & ~(g["GLD"] | g["GDX"]),
    "SLV broke, NOT all three (discards)": g["SLV"] & ~(g["GLD"] & g["GDX"]),
    "GLD & GDX broke, SLV did NOT": (~g["SLV"]) & g["GLD"] & g["GDX"],
}
for h in (1, 3, 5):
    rows = []
    for lbl, m in gates.items():
        rows.append(cellstats(m, h, lbl)[0])
    r = ret_at(h)
    rows.append(summarize(r.dropna().values, "CTRL all days"))
    show(rows, f"gate attribution, SHORT SLV, h={h}")
    par = cellstats(gates["SLV alone <= -2% (PARENT)"], h, "")[0]
    con = cellstats(gates["ALL THREE (the pitch)"], h, "")[0]
    dis = cellstats(gates["SLV broke, NOT all three (discards)"], h, "")[0]
    print(f"  h={h}: conjunction {con['mean_pct']:+.3f}%  vs parent "
          f"{par['mean_pct']:+.3f}%  -> the two extra names add "
          f"{con['mean_pct']-par['mean_pct']:+.3f}pp; "
          f"the DISCARDED half pays {dis['mean_pct']:+.3f}%")
    print(f"      conjunction keeps {con['n_days']} of {par['n_days']} parent days "
          f"({100*con['n_days']/max(par['n_days'],1):.1f}%)")

# ----------------------------------------------------------- 3. DEFINITION LADDER
print("\n" + "=" * 100)
print("3. DEFINITION LADDER on the -2% threshold, and TODAY'S LIVE DEPTH")
print("=" * 100)
for h in (1, 3, 5):
    rows = []
    for thr in (-0.010, -0.015, -0.020, -0.025, -0.030, -0.040):
        m = (r1["GLD"] <= thr) & (r1["SLV"] <= thr) & (r1["GDX"] <= thr)
        rows.append(cellstats(m, h, f"all three <= {100*thr:.1f}%")[0])
    show(rows, f"symmetric threshold ladder, h={h}")

print("\n  ASYMMETRIC ladder on the SLV leg only (the traded leg), others held at -2%:")
for h in (1, 3, 5):
    rows = []
    for thr in (-0.020, -0.025, -0.030, -0.035, -0.040, -0.045):
        m = g["GLD"] & g["GDX"] & (r1["SLV"] <= thr)
        rows.append(cellstats(m, h, f"SLV <= {100*thr:.1f}% (today -4.38)")[0])
    show(rows, f"SLV-leg depth ladder, h={h}")

print("\n  DEPTH-MATCHED BUCKET SPLIT (registry 2026-08-26: split at the LIVE value):")
slv_break = r1["SLV"]
buckets = [(-1.0, -0.04), (-0.04, -0.03), (-0.03, -0.025), (-0.025, -0.02)]
for h in (1, 3, 5):
    rows = []
    for lo, hi in buckets:
        m = trig & (slv_break > lo) & (slv_break <= hi)
        mark = " ***TODAY (-4.38%)" if lo == -1.0 else ""
        rows.append(cellstats(m, h, f"SLV break ({100*lo:.0f},{100*hi:.1f}]{mark}")[0])
    show(rows, f"SLV break-depth buckets, h={h}")

# --------------------------------------------------------------- 4. MIDTERM
print("\n" + "=" * 100)
print("4. MIDTERM SPLIT (2026 is a midterm year)")
print("=" * 100)
for h in (1, 2, 3, 5):
    s, epi, vals = cellstats(trig, h, "")
    yr = pd.DatetimeIndex(epi).year
    mid = (yr % 4) == 2
    r = ret_at(h).dropna()
    b_mid = (pd.DatetimeIndex(r.index).year % 4) == 2
    rows = [summarize(vals[mid], f"h={h} MIDTERM (N={int(mid.sum())})"),
            summarize(vals[~mid], f"h={h} non-midterm (N={int((~mid).sum())})"),
            summarize(r.values[b_mid], f"h={h} CTRL all days, midterm"),
            summarize(r.values[~b_mid], f"h={h} CTRL all days, non-midterm")]
    show(rows)
    w = int((vals[mid] > 0).sum())
    p0 = float((r.values[b_mid] > 0).mean())
    print(f"   midterm record {w}-{int(mid.sum())-w}, sign p vs midterm down-rate "
          f"{100*p0:.1f}% = {sign_test(w, int(mid.sum()), p0):.4f}   |  "
          f"midterm edge {100*vals[mid].mean()-100*r.values[b_mid].mean():+.3f}pp, "
          f"non-midterm edge "
          f"{100*vals[~mid].mean()-100*r.values[~b_mid].mean():+.3f}pp")
    print(f"   midterm episode years: {sorted(set(yr[mid]))}")

# ------------------------------------------- 5. LOCAL CONTROL + BETA RESIDUALS
print("\n" + "=" * 100)
print("5. LOCAL +/-126td CONTROL and PIT BETA RESIDUALS")
print("=" * 100)
for h in (1, 3, 5):
    r = ret_at(h)
    v = r.notna()
    days = px.index[trig.values & v.values]
    epi = declusters(days, GAP, px.index)
    loc = local_control(px.index[v.values], days)
    ep = r.loc[epi].values
    lc = r.loc[loc].values
    se = np.sqrt(ep.var(ddof=1) / len(ep) + lc.var(ddof=1) / len(lc))
    print(f"  h={h}: episodes {100*ep.mean():+.3f}%  local ctrl {100*lc.mean():+.3f}% "
          f"(N={len(lc)})  diff {100*(ep.mean()-lc.mean()):+.3f}pp  welch t "
          f"{(ep.mean()-lc.mean())/se:+.2f}")

print("\n  PIT trailing-252 beta residuals (short SLV hedged with the benchmark):")
d_slv = px["SLV"].pct_change()
for bench in ("GLD", "DX-Y.NYB", "SPY"):
    d_b = px[bench].pct_change()
    beta = (d_slv.rolling(252).cov(d_b) / d_b.rolling(252).var()).shift(1)
    rows = []
    for h in (1, 3, 5):
        f_slv = fwd_lag(px["SLV"], h, 1)
        f_b = fwd_lag(px[bench], h, 1)
        resid = -(f_slv - beta * f_b)          # SHORT the residual
        rv = resid.notna()
        days = px.index[trig.values & rv.values]
        epi = declusters(days, GAP, px.index)
        rr = summarize(resid.loc[epi].values, f"h={h} SHORT resid vs {bench}")
        rr["ctrl_pct"] = round(100 * resid[rv].mean(), 3)
        rr["edge_pp"] = round(rr["mean_pct"] - 100 * resid[rv].mean(), 3)
        rr["raw_short_pct"] = round(100 * (-f_slv).loc[epi].mean(), 3)
        rows.append(rr)
    show(rows, f"beta-neutral vs {bench} (mean beta {beta.dropna().mean():.2f}, "
               f"today {beta.iloc[-1]:.2f})")

# ---------------------------------------------------- 6. is today in-sample?
print("\n" + "=" * 100)
print("6. IS TODAY'S STATE INSIDE THE SAMPLE?")
print("=" * 100)
state = pd.DataFrame({
    "r63_pct": 100 * r63_slv,
    "vs200d_pct": 100 * (slv / sma200 - 1.0),
    "vs52wh_pct": 100 * (slv / hi52 - 1.0),
    "r21rank": pct_rank(slv, 21),
})
_, epi5, _ = cellstats(trig, 5, "")
st = state.loc[epi5].dropna()
print(st.describe().round(2).to_string())
print("\n  TODAY:", state.iloc[-1].round(2).to_dict())
for c in state.columns:
    pctl = 100.0 * (st[c] <= state[c].iloc[-1]).mean()
    print(f"   today's {c:10s} sits at the {pctl:5.1f}th percentile of the "
          f"{len(st)} historical episodes  "
          f"(min {st[c].min():+.2f}, max {st[c].max():+.2f})")

print("\n  cell restricted to state-matched buckets (today: r63 -12.2, "
      "vs200d -7.7, vs52wh -43.2):")
for h in (1, 3, 5):
    rows = []
    for lbl, m in [
        ("r63 <= -5%", trig & (r63_slv <= -0.05)),
        ("r63 <= -10%", trig & (r63_slv <= -0.10)),
        ("below 200d", trig & (slv < sma200)),
        ("below 200d by >5%", trig & (slv / sma200 - 1 <= -0.05)),
        (">30% below 52wh", trig & (slv / hi52 - 1 <= -0.30)),
        (">40% below 52wh", trig & (slv / hi52 - 1 <= -0.40)),
        ("ALL FOUR of today's states", trig & (r63_slv <= -0.10)
         & (slv / sma200 - 1 <= -0.05) & (slv / hi52 - 1 <= -0.30)),
        ("COMPLEMENT (not today's state)", trig & ~((r63_slv <= -0.10)
         & (slv / sma200 - 1 <= -0.05) & (slv / hi52 - 1 <= -0.30))),
    ]:
        rows.append(cellstats(m, h, lbl)[0])
    show(rows, f"state-matched, h={h}")

# ------------------------------------------------------------- 7. cost/borrow
print("\n" + "=" * 100)
print("7. COST AND BORROW")
print("=" * 100)
for h in (1, 3, 5):
    s = cellstats(trig, h, "")[0]
    edge = 100 * s["mean_pct"]
    for borrow in (0.0, 0.5, 1.0, 2.0):
        carry = borrow / 100.0 * (h / 252.0) * 1e4
        print(f"  h={h}: edge {edge:6.1f} bp | 6 bp round trip + {borrow:.1f}%/yr "
              f"borrow ({carry:.2f} bp) -> {edge/(6.0+carry):.1f}x")

# --------------------------------------------------------- 8. book overlap
print("\n" + "=" * 100)
print("8. BOOK OVERLAP (columns ASSERTED)")
print("=" * 100)
led = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
assert "Signal Date" in led.columns, f"missing 'Signal Date': {list(led.columns)[:20]}"
assert "Strategy" in led.columns, f"missing 'Strategy': {list(led.columns)[:20]}"
assert "Ticker" in led.columns, f"missing 'Ticker': {list(led.columns)[:20]}"
led["Signal Date"] = pd.to_datetime(led["Signal Date"])
METALS = {"SLV", "GLD", "GDX", "NEM", "AEM", "KGC", "AU", "GOLD", "GFI", "PAAS",
          "WPM", "FNV", "RGLD", "HL", "CDE", "AG", "EXK", "SIL", "GDXJ", "IAU"}
mm = led[led["Ticker"].isin(METALS)]
print(f"  ledger rows {len(led)}, metals-name rows {len(mm)}")
_, epi5, _ = cellstats(trig, 5, "")
trig_days = set(px.index[trig.values])
on = mm[mm["Signal Date"].isin(trig_days)]
print(f"  metals trades signalled ON a trigger day: {len(on)}")
if len(on):
    print(on.groupby(["Strategy", "Direction"])["R_Multiple"].agg(["count", "mean"])
          .round(3).to_string() if "R_Multiple" in on.columns
          else on.groupby(["Strategy", "Direction"]).size().to_string())
# SLV specifically
slv_led = led[led["Ticker"] == "SLV"]
print(f"  SLV ledger rows: {len(slv_led)}   directions: "
      f"{slv_led['Direction'].value_counts().to_dict() if len(slv_led) else {}}")
