"""C2 round 3 DEVELOPMENT. The survivor is NOT the pitched form.

Pitched: long QQQ / short SPY equal-dollar, "held through the print" (the
recon's headline was h=3). That form is dead -- beta-neutral h=3 is 52.4% hit,
sign p 0.21, and 50.4% hit / p 0.47 once the 2020-21 tech and 2023-25 AI eras
are removed.

Survivor: BETA-NEUTRAL, h=1, i.e. exit at the print-session close.
  N=311, +0.115%, hit 59.8%, t 2.76, sign p 0.00032, boot P(mean<=0) 0.0036
  positive in 22/27 calendar years; drop-3-best-years +0.065% (p 0.0009)
  ^NDX/^GSPC replication +0.097%, 60.5%, p 0.00014
  PPI eve -0.002 / NFP eve -0.041 / FOMC eve +0.065 / all days +0.007

Development, in the order the runbook asks for:
  (a) horizon scan 1..10 -- pick the horizon FROM the table
  (b) MOC vs MOO vs close-anchored LIMIT, as WHOLE variants with fill rates
  (c) target / stop sensitivity, or the reason time-only fits
  (d) the losing episodes: what do they have in common
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    load_prices, close_panel, load_events, fwd_lag, declusters, summarize,
    sign_test, bootstrap_p_le0, horizon_scan, wilder_atr,
)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)

OHLC = load_prices(["SPY", "QQQ"])
px = close_panel(["SPY", "QQQ"]).dropna()
all_dates = px.index
pos = pd.Series(np.arange(len(all_dates)), index=all_dates)

O = pd.DataFrame({t: OHLC[t]["Open"] for t in OHLC}).reindex(all_dates)
Hh = pd.DataFrame({t: OHLC[t]["High"] for t in OHLC}).reindex(all_dates)
L = pd.DataFrame({t: OHLC[t]["Low"] for t in OHLC}).reindex(all_dates)
C = px
ATR = pd.DataFrame({t: wilder_atr(Hh[t], L[t], C[t]) for t in ("SPY", "QQQ")},
                   index=all_dates)

rq, rs = C["QQQ"].pct_change(), C["SPY"].pct_change()
BETA = rq.rolling(126).cov(rs) / rs.rolling(126).var()

ev = load_events(["cpi"])
anch = []
for d in pd.DatetimeIndex(sorted(ev["date"].unique())):
    loc = all_dates.searchsorted(d)
    if loc >= len(all_dates):
        continue
    j = loc - 2
    if 0 <= j < len(all_dates):
        anch.append(all_dates[j])
anch = declusters(pd.DatetimeIndex(sorted(set(anch))), 5, all_dates)


def rep(lbl, vals, extra=""):
    v = np.asarray(vals, float)
    v = v[~np.isnan(v)]
    st = summarize(v)
    if not st["n"]:
        return dict(variant=lbl, N=0)
    w = int((v > 0).sum())
    return dict(variant=lbl, N=st["n"], mean_pct=round(st["mean_pct"], 3),
                med=round(st["median_pct"], 3), hit=round(st["hit"], 1),
                t=round(st["t"], 2), signp=round(sign_test(w, st["n"]), 5),
                sd=round(st["sd_pct"], 2), worst=round(st["worst_pct"], 2),
                bootP=round(bootstrap_p_le0(v), 3), note=extra)


print("=" * 120)
print("(a) HORIZON SCAN 1..10, beta-neutral (PIT 126d beta), entry lag=1")
print("=" * 120)
rows = []
for h in range(1, 11):
    ser = fwd_lag(C["QQQ"], h, 1) - BETA * fwd_lag(C["SPY"], h, 1)
    v = ser.reindex(anch).dropna()
    base = ser.dropna()
    r = rep(f"h={h}", v.values)
    r["all_days"] = round(100 * base.mean(), 3)
    r["excess"] = round(r["mean_pct"] - 100 * base.mean(), 3)
    r["per_day_bps"] = round(100 * r["mean_pct"] / h, 1)
    rows.append(r)
print(pd.DataFrame(rows).drop(columns=["note"]).to_string(index=False))
print("\n  the table picks h=1: it is the only horizon whose sign test clears")
print("  0.01, it carries the most edge PER SESSION HELD, and it is the one")
print("  horizon with a mechanism (the print itself is inside it).")
print("\n  equal-dollar horizon_scan from the lab, for the record:")
print(pd.DataFrame(horizon_scan(px, anch, [("QQQ", 1.0), ("SPY", -1.0)],
                                hs=(1, 2, 3, 5, 10), lag=1,
                                min_gap=5)).to_string(index=False))

print("\n" + "=" * 120)
print("(b) WHOLE VARIANTS: MOC / MOO / close-anchored LIMIT, with fill rates")
print("=" * 120)
# indices: anchor a at position p. entry session = p+1, print session = p+2.
p_a = pos.reindex(anch).dropna().astype(int)
p_a = p_a[p_a + 2 < len(all_dates)]
i_e, i_p = p_a.values + 1, p_a.values + 2
b = BETA.values[p_a.values]                      # beta known at the anchor close

cq, cs = C["QQQ"].values, C["SPY"].values
oq, os_ = O["QQQ"].values, O["SPY"].values
lq, hs_ = L["QQQ"].values, Hh["SPY"].values
aq, as_ = ATR["QQQ"].values, ATR["SPY"].values

variants = []
# V1 MOC entry (the measured cell): close(entry) -> close(print)
v1 = (cq[i_p] / cq[i_e] - 1) - b * (cs[i_p] / cs[i_e] - 1)
variants.append(rep("V1 MOC in (entry close) -> MOC out (print close)", v1,
                    "fill 100%"))
# V2 MOO on the print session -> MOC same session (intraday only)
v2 = (cq[i_p] / oq[i_p] - 1) - b * (cs[i_p] / os_[i_p] - 1)
variants.append(rep("V2 MOO in (print open) -> MOC out (print close)", v2,
                    "fill 100%, gives up the overnight"))
# V3 the overnight alone: close(entry) -> open(print)
v3 = (oq[i_p] / cq[i_e] - 1) - b * (os_[i_p] / cs[i_e] - 1)
variants.append(rep("V3 MOC in -> MOO out (the OVERNIGHT alone)", v3,
                    "decomposition, not a trade"))
# V4 close-anchored LIMIT on BOTH legs, live on the print session, exit MOC
for k in (0.10, 0.25, 0.50):
    lim_q = cq[i_e] - k * aq[i_e]      # buy QQQ cheaper
    lim_s = cs[i_e] + k * as_[i_e]     # short SPY higher
    fill = (lq[i_p] <= lim_q) & (hs_[i_p] >= lim_s)
    r = (cq[i_p] / lim_q - 1) - b * (cs[i_p] / lim_s - 1)
    r = np.where(fill, r, np.nan)
    variants.append(rep(f"V4 LIMIT both legs at {k:.2f} ATR -> MOC out", r,
                        f"JOINT fill {100*np.nanmean(fill):.0f}% "
                        f"({int(fill.sum())}/{len(fill)})"))
print(pd.DataFrame(variants).to_string(index=False))
print("""
  Read: V1 = V3 + V2 by construction. The overnight leg (V3) and the intraday
  leg (V2) split the edge; whichever dominates tells you whether the MOC entry
  the night before is load-bearing or whether you could wait for the open.
  V4 is priced as a WHOLE variant -- only sessions where BOTH legs filled are
  trades, no marginal-fill decomposition (registry rule). A limit that only
  fills one leg leaves a naked index position through a CPI print, which is
  the practical argument against it whatever the table says.""")

print("\n" + "=" * 120)
print("(c) TARGET / STOP SENSITIVITY -- and why time-only fits")
print("=" * 120)
lo_q, hi_q = L["QQQ"].values, Hh["QQQ"].values
lo_s, hi_s2 = L["SPY"].values, Hh["SPY"].values
ent_q, ent_s = cq[i_e], cs[i_e]
# an approximate intraday excursion band for the spread: best/worst case
# orderings of the two legs' extremes. This BRACKETS the true path.
best = (hi_q[i_p] / ent_q - 1) - b * (lo_s[i_p] / ent_s - 1)
worst = (lo_q[i_p] / ent_q - 1) - b * (hi_s2[i_p] / ent_s - 1)
atr_sp = (aq[i_e] / ent_q) + np.abs(b) * (as_[i_e] / ent_s)   # 1 "spread ATR"
print(f"  spread-ATR (QQQ ATR% + |beta| x SPY ATR%) median "
      f"{100*np.nanmedian(atr_sp):.2f}% ; the h=1 edge is "
      f"{100*np.nanmean(v1):.3f}% = {np.nanmean(v1)/np.nanmedian(atr_sp):.2f} "
      f"spread-ATR")
for k in (0.5, 1.0):
    hit_t = best >= k * atr_sp
    hit_s = worst <= -k * atr_sp
    both = hit_t & hit_s
    print(f"  at {k:.1f} spread-ATR: an upper-bound target would have been "
          f"touched on {100*np.nanmean(hit_t):.0f}% of sessions, a lower-bound "
          f"stop on {100*np.nanmean(hit_s):.0f}%, and BOTH on "
          f"{100*np.nanmean(both):.0f}% -- and on those days a daily bar "
          f"cannot say which came first.")
print("""
  Verdict: TIME-ONLY. Two reasons, both structural rather than statistical.
  (1) The whole position is one session long with a scheduled macro print
      inside it, so a stop is a bet on the ORDER of two legs' intraday
      extremes, which daily bars cannot resolve and which the 09:30 print
      reaction routinely reverses.
  (2) A spread stop needs both legs unwound together; a single-leg stop
      converts a market-neutral position into a naked index bet at exactly
      the worst moment. The risk unit is the spread ATR above, and the time
      exit at the print close is the control.""")

print("\n" + "=" * 120)
print("(d) THE LOSING SESSIONS: what do they have in common?")
print("=" * 120)
dates_e = all_dates[i_e]
dates_p = all_dates[i_p]
D = pd.DataFrame({
    "entry": [d.date() for d in dates_e],
    "print": [d.date() for d in dates_p],
    "spread_pct": 100 * v1,
    "qqq_pct": 100 * (cq[i_p] / cq[i_e] - 1),
    "spy_pct": 100 * (cs[i_p] / cs[i_e] - 1),
    "beta": np.round(b, 2),
    "spy_atr_pct": 100 * as_[i_e] / cs[i_e],
}).dropna()
D["spy_move_atr"] = (D["spy_pct"] / D["spy_atr_pct"]).round(2)
worst10 = D.nsmallest(10, "spread_pct")
print(worst10.round(3).to_string(index=False))
big = D[D["spy_move_atr"].abs() >= 1.5]
calm = D[D["spy_move_atr"].abs() < 1.5]
print(f"\n  split by the SIZE of the print-session SPY move (in SPY ATRs):")
print(pd.DataFrame([
    rep(f"|SPY move| >= 1.5 ATR (violent print)", big["spread_pct"].values / 100),
    rep(f"|SPY move| <  1.5 ATR (ordinary print)", calm["spread_pct"].values / 100),
]).to_string(index=False))
print(f"\n  worst 5 losses by year: "
      f"{[(str(r.entry), round(r.spread_pct, 2)) for r in worst10.head(5).itertuples()]}")
print(f"  loss tail: 5th pctile {np.nanpercentile(v1, 5)*100:+.2f}%, "
      f"1st pctile {np.nanpercentile(v1, 1)*100:+.2f}%, "
      f"worst {np.nanmin(v1)*100:+.2f}%")
print("""
  The losing sessions are the violent prints, and they are violent in BOTH
  directions -- which is the honest statement of the risk: the spread is short
  a dispersion event between two indices on the one session of the month when
  a macro surprise can re-rate duration. The time exit at the print close is
  what bounds it to a single session.""")

print("\n" + "=" * 120)
print("TODAY'S PARAMETERS")
print("=" * 120)
t = all_dates[-1]
print(f"  anchor close {t.date()}; entry MOC 2026-08-11; exit MOC 2026-08-12 "
      f"(the CPI session)")
print(f"  PIT 126d beta {BETA.loc[t]:.3f}  ->  short {BETA.loc[t]:.2f} dollars "
      f"of SPY per dollar of QQQ")
print(f"  QQQ Wilder-14 ATR {ATR['QQQ'].loc[t]:.2f} on {C['QQQ'].loc[t]:.2f} "
      f"({100*ATR['QQQ'].loc[t]/C['QQQ'].loc[t]:.2f}%)")
print(f"  SPY Wilder-14 ATR {ATR['SPY'].loc[t]:.2f} on {C['SPY'].loc[t]:.2f} "
      f"({100*ATR['SPY'].loc[t]/C['SPY'].loc[t]:.2f}%)")
print(f"  1 spread-ATR today = {100*(ATR['QQQ'].loc[t]/C['QQQ'].loc[t] + BETA.loc[t]*ATR['SPY'].loc[t]/C['SPY'].loc[t]):.2f}% "
      f"of the QQQ leg's notional -- that is the risk unit")
print("\n  CAVEAT THE PITCH MUST CARRY: today's own joint state (QQQ rel63 rank")
print("  1.6 AND SPY at its 52w high) has 23 historical instances at +0.051%")
print("  (boot P(mean<=0) 0.277), and 5 of those in a midterm year at -0.063%.")
print("  The evidence is for the CPI-eve cell, NOT for today's conditioner.")
