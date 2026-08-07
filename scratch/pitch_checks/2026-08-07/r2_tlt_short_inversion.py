"""RED TEAM / INVERSION 2: SHORT TLT on the "bond floor" trigger.

The prior checker (c3) measured LONG TLT and found it dead-to-negative, with
2018+ h=10 at -1.310% (N=12 episodes, t=-2.18) -- i.e. the SHORT side made
money. This script asks whether that is a trade or an artifact of a post-hoc
sign flip on top of a 2018-2023 bond bear market.

Convention matches c3/_engine: signal on close D, entry MOC D+1, exit MOC
D+1+h. Trade returns are reported SHORT-side (positive = short TLT wins).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import *  # noqa

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

px = close_panel(["TLT", "IEF", "^TNX", "SPY"]).dropna()
idx = px.index
lo52 = px["TLT"].rolling(252).min()
off_lo = px["TLT"] / lo52 - 1.0
tnx63 = pct_rank(px["^TNX"], 63)


def fwd_lag(s, h, lag=1):
    return s.shift(-(lag + h)) / s.shift(-lag) - 1.0


def short(tkr, h, lag=1):
    """SHORT the vehicle. Positive = the short makes money."""
    return -fwd_lag(px[tkr], h, lag)


def mask(d=0.015, rk=85.0):
    return (off_lo <= d) & (tnx63 >= rk)


def dates_of(m, h, lag=1, tkr="TLT", era=None, exclude_years=()):
    ok = short(tkr, h, lag).notna()
    s = idx[m.fillna(False).values & ok.values]
    if era == "2018+":
        s = s[s >= pd.Timestamp("2018-01-01")]
    elif era == "pre2018":
        s = s[s < pd.Timestamp("2018-01-01")]
    if exclude_years:
        s = s[~np.isin(s.year, list(exclude_years))]
    return s


BASE = mask()

print("=" * 80)
print("INVERSION 2 -- SHORT TLT  (trigger: TLT within 1.5% of 52w low & ^TNX rank63>=85)")
print("=" * 80)
print(f"panel {idx[0].date()} .. {idx[-1].date()}  n={len(idx)}")
print(f"TODAY (2026-08-06 close): TLT {100*off_lo.iloc[-1]:+.2f}% above 52w low, "
      f"^TNX rank63={tnx63.iloc[-1]:.1f}  fires={bool(BASE.iloc[-1])}")

# --- the real hold window, on a US-federal-holiday business calendar ----------
cbd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
entry = pd.Timestamp("2026-08-07")
sess = [entry + cbd * k for k in range(0, 12)]
print(f"\nREAL ORDER: entry MOC {entry.date()} (= trigger day 2026-08-06 + 1 session)")
print("  hold sessions:", ", ".join(f"+{k}={d.date()}" for k, d in enumerate(sess) if k <= 11))
cpi_d, ppi_d = pd.Timestamp("2026-08-12"), pd.Timestamp("2026-08-13")
print(f"  CPI {cpi_d.date()} = session +{sess.index(cpi_d) if cpi_d in sess else '?'}, "
      f"PPI {ppi_d.date()} = session +{sess.index(ppi_d) if ppi_d in sess else '?'}, "
      f"exit +10 = {sess[10].date()}")

# --------------------------------------------------- 1) confirm the 2018+ cell
print("\n" + "=" * 80)
print("1) CONFIRM the 2018+ h=10 cell for SHORT TLT")
print("=" * 80)
rows = []
for era in (None, "pre2018", "2018+"):
    s = dates_of(BASE, 10, era=era)
    if len(s) == 0:
        continue
    e = declusters(s, 10, idx)
    v = short("TLT", 10).reindex(e).to_numpy()
    r = summarize(v, f"{era or 'full'} episodes")
    r["n_days"] = len(s)
    r["boot_p_le0"] = bootstrap_p_le0(v)
    rows.append(r)
    rows.append(summarize(short("TLT", 10).reindex(s).to_numpy(), f"{era or 'full'} day-level"))
show(rows, "SHORT TLT h=10 (positive = short wins)")

S18 = dates_of(BASE, 10, era="2018+")
E18 = declusters(S18, 10, idx)
V18 = short("TLT", 10).reindex(E18).to_numpy()
print(f"\n  2018+ episode dates (gap=10), N={len(E18)}:")
for d, v in zip(E18, V18):
    print(f"    {d.date()}  short-TLT {100*v:+7.2f}%   (TLT {100*off_lo[d]:+.2f}% off 52w low, "
          f"^TNX rk63={tnx63[d]:.0f})")
print(f"  2018+ episodes by year: {dict(pd.Series(1, index=E18).groupby(E18.year).sum())}")
print(f"  2018+ TRIGGER DAYS by year: {dict(pd.Series(1, index=S18).groupby(S18.year).sum())}")
SALL = dates_of(BASE, 10)
print(f"  FULL-sample trigger days by year: {dict(pd.Series(1, index=SALL).groupby(SALL.year).sum())}")

# ------------------------------------- 2) drop best/worst + leave-one-year-out
print("\n" + "=" * 80)
print("2) 2018+ ROBUSTNESS: drop-best / drop-worst / leave-one-YEAR-out")
print("=" * 80)
o = np.argsort(V18)
rows = [summarize(V18, "all 2018+ episodes"),
        summarize(np.delete(V18, o[-1]), f"drop BEST ({E18[o[-1]].date()} {100*V18[o[-1]]:+.2f}%)"),
        summarize(np.delete(V18, o[0]), f"drop WORST ({E18[o[0]].date()} {100*V18[o[0]]:+.2f}%)"),
        summarize(np.delete(V18, [o[-1], o[-2]]), "drop BEST 2")]
show(rows)
loyo = []
for y in sorted(set(E18.year)):
    m = E18.year != y
    s = summarize(V18[m], f"drop {y} (n_out={int((~m).sum())})")
    loyo.append(s)
show(loyo, "leave-one-YEAR-out, 2018+ episodes")

# ----------------------------------------------------------- 3) bootstrap
print("\n" + "=" * 80)
print("3) BOOTSTRAP on the SHORT-side 2018+ episode returns")
print("=" * 80)
print(f"  P(mean <= 0) = {bootstrap_p_le0(V18):.4f}  (N={len(V18)})")
EF = declusters(SALL, 10, idx)
VF = short("TLT", 10).reindex(EF).to_numpy()
print(f"  P(mean <= 0) = {bootstrap_p_le0(VF):.4f}  (FULL sample, N={len(VF)})")

# ----------------------------------------------------------- 4) threshold grid
print("\n" + "=" * 80)
print("4) THRESHOLD GRID, h=10, 2018+, SHORT TLT (episodes gap=10)")
print("=" * 80)
grid = []
for d in (0.010, 0.015, 0.020, 0.030):
    for rk in (80, 85, 90):
        s = dates_of(mask(d, rk), 10, era="2018+")
        if len(s) == 0:
            grid.append({"cell": f"d<={100*d:.1f}% rk>={rk}", "n_day": 0})
            continue
        e = declusters(s, 10, idx)
        v = short("TLT", 10).reindex(e).to_numpy()
        r = summarize(v, "")
        grid.append({"cell": f"d<={100*d:.1f}% rk>={rk}", "n_day": len(s), "n_epi": r["n"],
                     "mean_pct": r["mean_pct"], "t": r["t"], "hit": r["hit"],
                     "worst_pct": r["worst_pct"]})
show(grid, "grid (PITCHED CELL = d<=1.5% rk>=85)")

# ------------------------------------------------------------ 5) horizon curve
print("\n" + "=" * 80)
print("5) HORIZON CURVE h=1..21, 2018+ subsample, SHORT TLT")
print("=" * 80)
rows = []
for h in range(1, 22):
    s = dates_of(BASE, h, era="2018+")
    if len(s) == 0:
        continue
    e = declusters(s, max(h, 5), idx)
    v = short("TLT", h).reindex(e).to_numpy()
    r = summarize(v, "")
    dd = summarize(short("TLT", h).reindex(s).to_numpy(), "")
    # excess over TLT's own unconditional short drift on the same span
    u = short("TLT", h)
    span = u[(idx >= pd.Timestamp("2018-01-01")) & (idx <= s[-1])]
    rows.append({"h": h, "n_day": dd["n"], "day_mean_pct": dd["mean_pct"], "day_t": dd["t"],
                 "n_epi": r["n"], "epi_mean_pct": r["mean_pct"], "epi_t": r["t"],
                 "epi_hit": r["hit"], "uncond_pct": 100 * span.mean(),
                 "excess_pct": r["mean_pct"] - 100 * span.mean()})
show(rows, "horizon curve, 2018+")

# ============================================================================
# 6) THE DECISIVE TEST: unconditional 2018+ drift, and the conditional's excess
# ============================================================================
print("\n" + "=" * 80)
print("6) DECISIVE: is this just TLT's 2018+ downtrend?")
print("=" * 80)
for tkr in ("TLT", "IEF"):
    u = short(tkr, 10)
    for lbl, m in (("2018+ all days", idx >= pd.Timestamp("2018-01-01")),
                   ("2018+ ex-2022", (idx >= pd.Timestamp("2018-01-01")) & (idx.year != 2022)),
                   ("full sample", np.ones(len(idx), bool)),
                   ("pre-2018", idx < pd.Timestamp("2018-01-01"))):
        v = u[m].dropna().to_numpy()
        r = summarize(v, f"{tkr} UNCONDITIONAL short h=10, {lbl}")
        print(f"  {r['label']:<48s} n={r['n']:>5d}  mean={r['mean_pct']:+7.3f}%  t={r['t']:+6.2f}")

print()
for lbl, exyr in (("2018+ (all)", ()), ("2018+ EX-2022", (2022,)),
                  ("2018+ EX-2022,2023", (2022, 2023))):
    s = dates_of(BASE, 10, era="2018+", exclude_years=exyr)
    if len(s) == 0:
        print(f"  {lbl}: N=0 trigger days")
        continue
    e = declusters(s, 10, idx)
    v = short("TLT", 10).reindex(e).to_numpy()
    r = summarize(v, f"COND {lbl}")
    m = (idx >= pd.Timestamp("2018-01-01")) & (~np.isin(idx.year, list(exyr)))
    unc = short("TLT", 10)[m].dropna()
    # welch t of conditional vs matched unconditional
    se = np.sqrt(np.var(v, ddof=1) / len(v) + unc.var(ddof=1) / len(unc)) if len(v) > 1 else np.nan
    exc = v.mean() - unc.mean()
    print(f"  {lbl:<20s} N_epi={r['n']:>3d} n_day={len(s):>4d}  cond={r['mean_pct']:+7.3f}% "
          f"t={r['t']:+5.2f} hit={r['hit']:.0f}%  |  uncond={100*unc.mean():+7.3f}%  "
          f"EXCESS={100*exc:+7.3f}%  welch_t={exc/se:+5.2f}  bootP(mean<=0)={bootstrap_p_le0(v):.3f}")
    print(f"      episode dates: {', '.join(str(d.date()) for d in e)}")

# --------------------------------------------------------------- 7) IEF
print("\n" + "=" * 80)
print("7) IEF as the vehicle (half the duration, ATR% 0.37 vs TLT 0.72)")
print("=" * 80)
rows = []
for era in (None, "pre2018", "2018+"):
    s = dates_of(BASE, 10, era=era, tkr="IEF")
    if len(s) == 0:
        continue
    e = declusters(s, 10, idx)
    v = short("IEF", 10).reindex(e).to_numpy()
    r = summarize(v, f"IEF {era or 'full'} episodes")
    r["boot_p_le0"] = bootstrap_p_le0(v)
    rows.append(r)
show(rows, "SHORT IEF h=10")
s = dates_of(BASE, 10, era="2018+", tkr="IEF", exclude_years=(2022,))
if len(s):
    e = declusters(s, 10, idx)
    v = short("IEF", 10).reindex(e).to_numpy()
    m = (idx >= pd.Timestamp("2018-01-01")) & (idx.year != 2022)
    unc = short("IEF", 10)[m].dropna()
    r = summarize(v, "IEF 2018+ EX-2022 episodes")
    print(f"  IEF 2018+ EX-2022: N={r['n']} cond={r['mean_pct']:+.3f}% t={r['t']:+.2f}  "
          f"uncond={100*unc.mean():+.3f}%  EXCESS={r['mean_pct']-100*unc.mean():+.3f}%")
# risk-normalized
print(f"\n  TLT 2018+ episode mean {V18.mean()*100:+.3f}% / ATR%0.72 = "
      f"{V18.mean()*100/0.72:+.2f} ATR ; sd {np.std(V18, ddof=1)*100:.2f}%")

# --------------------------------------------------------- 8) CPI split + tail
print("\n" + "=" * 80)
print("8) CPI-IN-WINDOW SPLIT + LOSS TAIL (2018+ episodes, h=10)")
print("=" * 80)
ev = load_events()
print("  event kinds available:", sorted(set(ev["event"])))
cpi = set(pd.to_datetime(ev.loc[ev.event.str.contains("cpi", case=False, na=False), "date"]))
posn = pd.Series(range(len(idx)), index=idx)


def in_win(dates, h=10, lag=1, evset=cpi):
    out = []
    for d in dates:
        i = posn[d]
        if i + lag + h >= len(idx):
            out.append(False)
            continue
        lo, hi = idx[i + lag], idx[i + lag + h]
        out.append(any(lo < c <= hi for c in evset))
    return np.array(out, bool)


fl = in_win(E18)
show([summarize(V18[fl], f"CPI IN window (N={fl.sum()})"),
      summarize(V18[~fl], f"CPI OUT (N={(~fl).sum()})")], "2018+ episodes, CPI split")
flf = in_win(EF)
show([summarize(VF[flf], f"full-sample CPI IN (N={flf.sum()})"),
      summarize(VF[~flf], f"full-sample CPI OUT (N={(~flf).sum()})")], "full sample, CPI split")

print("\n  LOSS TAIL (worst windows AGAINST the short = biggest TLT rallies):")
srt = pd.Series(V18, index=E18).sort_values()
for d, v in srt.head(5).items():
    print(f"    {d.date()}  short-TLT {100*v:+7.2f}%  (TLT rallied {-100*v:.2f}% over the 10 td)")
print("  full-sample worst 5:")
srtf = pd.Series(VF, index=EF).sort_values()
for d, v in srtf.head(5).items():
    print(f"    {d.date()}  short-TLT {100*v:+7.2f}%")
print(f"\n  day-level worst 2018+: {100*short('TLT',10).reindex(S18).min():+.2f}%")
print(f"  1.5% adverse move on TLT ATR%0.72 = {1.5/0.72:.1f} ATR")
