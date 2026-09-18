"""A1 ROUND 1 -- long TLT, h=1 MOC, with TLT/IEF/LQD all pinned at trailing-252
lows and the trigger day being the FIRST in >= 10 trading sessions.

This is watchlist entry 5, parked 2026-08-12, re-read 2026-09-07. The parked
arithmetic is:
    tight rung TLT <= 0.5% / IEF <= 1.0% / LQD <= 1.0% above the trailing-252 low
    episode-first = first trigger day in >= 10 td
    +0.354pp excess, 82.4% hit, 17 episodes, sign p 0.0101
    ex-2022 +0.339% at 90%, local-control t +2.15
    TLT <= 0.5% ALONE = -0.120% at a 50% hit
    later days inside the same episode = -0.079pp at a 50.0% hit (N=52, p 0.67)
2026-09-07 re-read: +0.385pp excess, 83.3% hit, 18 episodes, sign p 0.0038.

Job here: reproduce it EXACTLY, or report that it does not reproduce.
Everything else in this script is a separate attack:
  1. the six-part battery
  2. episode-first vs later-day split on TODAY's definition
  3. the freshness leg verified independently (gap distribution)
  4. tdom control -- MANDATORY for any rates cell (registry 2026-08-10:
     TLT's own unconditional return swings from -0.202% at tdom 2 to +0.215%
     at tdom 14 with no event anywhere, so an all-days control is invalid)
  5. print-day entry split (today's entry IS a CPI session)
  6. the dose: where does today's yield thrust sit in the episode distribution
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

ASOF = pd.Timestamp("2026-09-10")
TK = ["TLT", "IEF", "LQD", "SPY", "^TNX", "HYG"]
px_raw = load_prices(TK)
C = {t: px_raw[t]["Close"] for t in px_raw}

# the rates calendar: TLT/IEF/LQD share it exactly
IDX = px_raw["TLT"].index
assert IDX.equals(px_raw["IEF"].index) and IDX.equals(px_raw["LQD"].index)
print(f"rates calendar: {len(IDX)} sessions {IDX[0].date()} .. {IDX[-1].date()}")

PX = pd.DataFrame({t: C[t].reindex(IDX) for t in ["TLT", "IEF", "LQD", "SPY"]})
PX["TNX"] = C["^TNX"].reindex(IDX).ffill()


def above_low(t: str, n: int = 252) -> pd.Series:
    s = C[t]
    return ((s / s.rolling(n).min() - 1.0) * 100).reindex(IDX)


TL, IE, LQ = above_low("TLT"), above_low("IEF"), above_low("LQD")

print("\nLIVE READING 2026-09-10:  TLT %.3f%%  IEF %.3f%%  LQD %.3f%%"
      % (TL.loc[ASOF], IE.loc[ASOF], LQ.loc[ASOF]))


def rung(tlt_thr=0.5, ief_thr=1.0, lqd_thr=1.0) -> pd.Series:
    return ((TL <= tlt_thr) & (IE <= ief_thr) & (LQ <= lqd_thr)).fillna(False)


def first_in(mask: pd.Series, gap: int = 10) -> pd.DatetimeIndex:
    """FIRST trigger day in >= gap trading sessions -- computed directly from
    trading-day POSITIONS on IDX, independent of pitch_lab.declusters, so the
    freshness leg is verified rather than borrowed."""
    days = IDX[mask.reindex(IDX, fill_value=False).values]
    pos = pd.Series(range(len(IDX)), index=IDX)
    keep, last = [], -10 ** 9
    for d in days:
        p = int(pos[d])
        if p - last >= gap:
            keep.append(d)
        last = p          # NOTE: last updates on EVERY trigger day, not only kept
    return pd.DatetimeIndex(keep)


def later_days(mask: pd.Series, gap: int = 10) -> pd.DatetimeIndex:
    days = IDX[mask.reindex(IDX, fill_value=False).values]
    return pd.DatetimeIndex(days).difference(first_in(mask, gap))


TIGHT = rung()
EPI = first_in(TIGHT, 10)
LATER = later_days(TIGHT, 10)

print("\n" + "=" * 78)
print("0. REPRODUCE THE PARKED ARITHMETIC")
print("=" * 78)
days = IDX[TIGHT.values]
print(f"  tight-rung trigger DAYS all history : {len(days)}  "
      f"({days[0].date()} .. {days[-1].date()})")
print(f"  episode-first (>= 10 td)            : {len(EPI)}")
print(f"  later days inside an episode        : {len(LATER)}")
print(f"  episode dates: {[str(d.date()) for d in EPI]}")

r1 = fwd_lag(PX["TLT"], 1, 1)          # signal close D, MOC D+1, exit D+2
allv = r1.dropna()
epi_r = r1.loc[EPI].dropna()
lat_r = r1.loc[LATER].dropna()

base_all = allv.mean()
rows = [summarize(epi_r.values, f"EPISODE-FIRST (N={len(epi_r)})"),
        summarize(lat_r.values, f"LATER days (N={len(lat_r)})"),
        summarize(r1.loc[days].dropna().values, f"ALL trigger days (N={len(days)})"),
        summarize(allv.values, "CTRL all days")]
show(rows, "TLT h=1 lag=1, episode-first vs later vs all-days")
w = int((epi_r > 0).sum())
print(f"  EXCESS vs all-days = {100*(epi_r.mean()-base_all):+.3f}pp   "
      f"record {w}-{len(epi_r)-w}  sign p {sign_test(w, len(epi_r)):.4f}   "
      f"bootstrap P(mean<=0) {bootstrap_p_le0(epi_r.values):.4f}")
wl = int((lat_r > 0).sum())
print(f"  LATER excess = {100*(lat_r.mean()-base_all):+.3f}pp  {wl}-{len(lat_r)-wl} "
      f"sign p {sign_test(wl, len(lat_r)):.4f}")
ex22 = epi_r[epi_r.index.year != 2022]
w22 = int((ex22 > 0).sum())
print(f"  ex-2022: {100*ex22.mean():+.3f}% hit {100*(ex22>0).mean():.1f}% "
      f"N={len(ex22)} ({w22}-{len(ex22)-w22}, sign p {sign_test(w22, len(ex22)):.4f})")
print(f"  concentration: {cluster_note(epi_r.index, epi_r.values)}")

print("\n  PARKED SAYS: 17 episodes, +0.354pp excess, 82.4% hit, sign p 0.0101, "
      "ex-2022 +0.339% at 90%")
print("  2026-09-07 SAYS: 18 episodes, +0.385pp excess, 83.3% hit, sign p 0.0038")

print("\n" + "=" * 78)
print("1. THE FULL BATTERY (episode-first mask)")
print("=" * 78)
epi_mask = pd.Series(False, index=IDX)
epi_mask.loc[EPI] = True
battery(PX, epi_mask, [("TLT", 1.0)], h=1, title="A1 long TLT | tight rung, episode-first",
        cost_bps=3.0, min_gap=10, event_kinds=("cpi", "ppi", "nfp", "fomc_decision"))

print("\n" + "=" * 78)
print("2. FRESHNESS LEG VERIFIED INDEPENDENTLY: the gap distribution")
print("=" * 78)
pos = pd.Series(range(len(IDX)), index=IDX)
gaps = np.diff([int(pos[d]) for d in days])
print(f"  gaps between consecutive trigger DAYS: "
      f"n={len(gaps)} median {np.median(gaps):.0f} "
      f"share >= 10 td = {100*(gaps >= 10).mean():.1f}%")
print(f"  live gap 2026-08-18 -> 2026-09-10: "
      f"{int(pos[ASOF]) - int(pos[pd.Timestamp('2026-08-18')])} td")
for g in (5, 10, 15, 21, 42):
    e = first_in(TIGHT, g)
    v = r1.loc[e].dropna()
    ww = int((v > 0).sum())
    print(f"  gap={g:3d}: N={len(v):3d}  mean {100*v.mean():+.3f}%  "
          f"excess {100*(v.mean()-base_all):+.3f}pp  hit {100*(v>0).mean():.1f}%  "
          f"sign p {sign_test(ww, len(v)):.4f}")

print("\n" + "=" * 78)
print("3. TDOM CONTROL (registry 2026-08-10: an all-days control on a rates cell "
      "is INVALID)")
print("=" * 78)
tdom = pd.Series(index=IDX, dtype=float)
for (y, m), grp in pd.Series(IDX, index=IDX).groupby([IDX.year, IDX.month]):
    tdom.loc[grp.index] = np.arange(1, len(grp) + 1)
prof = pd.DataFrame({"r": r1, "tdom": tdom}).dropna()
tab = prof.groupby("tdom")["r"].agg(["mean", "count"])
print("  TLT h=1 lag=1 unconditional by trading-day-of-month (pp):")
print("   " + "  ".join(f"{int(k)}:{100*v:+.3f}" for k, v in
                        tab["mean"].head(23).items()))
epi_tdom = tdom.loc[EPI]
print(f"  episode tdom values: {sorted(int(x) for x in epi_tdom.dropna())}")
print(f"  episode median tdom = {epi_tdom.median():.1f};  today's entry tdom = "
      f"{int(tdom.loc[ASOF]) + 1} (entry is 2026-09-11)")
# tdom-matched control: for each episode, the unconditional mean at that tdom
matched = np.array([tab.loc[int(t), "mean"] for t in epi_tdom.dropna()])
print(f"  TDOM-MATCHED control mean = {100*matched.mean():+.3f}%  vs all-days "
      f"{100*base_all:+.3f}%")
print(f"  EXCESS vs tdom-matched   = {100*(epi_r.mean()-matched.mean()):+.3f}pp  "
      f"(vs +{100*(epi_r.mean()-base_all):.3f}pp against all-days)")

print("\n" + "=" * 78)
print("4. PRINT-DAY ENTRY SPLIT -- today's ENTRY session carries an 08:30 CPI")
print("=" * 78)
ev = load_events(["cpi", "ppi", "nfp", "fomc_decision"])
ev_by_kind = {k: set(pd.DatetimeIndex(g["date"]).normalize())
              for k, g in ev.groupby("event")}
all_print = set().union(*ev_by_kind.values())
entry_dates, entry_kind = [], []
for d in EPI:
    p = int(pos[d])
    e = IDX[p + 1] if p + 1 < len(IDX) else pd.NaT
    entry_dates.append(e)
    ks = [k for k, s in ev_by_kind.items() if e in s]
    entry_kind.append("+".join(sorted(ks)) if ks else "")
edf = pd.DataFrame({"sig": EPI, "entry": entry_dates, "kind": entry_kind})
edf["r"] = [r1.get(d, np.nan) for d in EPI]
edf = edf.dropna(subset=["r"])
print(edf.assign(r_pct=(100 * edf["r"]).round(3)).drop(columns=["r"]).to_string(index=False))
has = edf["kind"] != ""
show([summarize(edf.loc[has, "r"].values, f"entry ON a print (N={int(has.sum())})"),
      summarize(edf.loc[~has, "r"].values, f"entry CLEAN (N={int((~has).sum())})")],
     "print-session entry split")
iscpi = edf["kind"].str.contains("cpi")
show([summarize(edf.loc[iscpi, "r"].values, f"entry on a CPI (N={int(iscpi.sum())})"),
      summarize(edf.loc[~iscpi, "r"].values, f"entry not CPI (N={int((~iscpi).sum())})")],
     "CPI-specific entry split")
# unconditional: what does TLT h=1 lag=1 do when the ENTRY lands on a CPI?
entry_is_cpi_all = pd.Series([IDX[min(i + 1, len(IDX) - 1)] in ev_by_kind["cpi"]
                              for i in range(len(IDX))], index=IDX)
show([summarize(r1[entry_is_cpi_all.values].dropna().values, "ALL days, entry on CPI"),
      summarize(r1[~entry_is_cpi_all.values].dropna().values, "ALL days, entry not CPI")],
     "unconditional CPI-entry profile for TLT h=1 lag=1")

print("\n" + "=" * 78)
print("5. THE DOSE -- 2026-08-07 registry: a rank gate in a quiet tape buys a "
      "fraction of the historical force")
print("=" * 78)
tnx = PX["TNX"]
tnx_chg = (tnx - tnx.shift(252)) * 100          # bp over 252 sessions
tnx_21 = (tnx - tnx.shift(21)) * 100
tnx_rank = rolling_on_valid(tnx, lambda x: x.rolling(252).rank(pct=True) * 100)
dose = pd.DataFrame({"tnx_252bp": tnx_chg, "tnx_21bp": tnx_21,
                     "tnx_rank": tnx_rank, "tlt_above_low": TL,
                     "ief": IE, "lqd": LQ})
d_epi = dose.loc[EPI].copy()
d_epi["r_pct"] = (100 * r1.loc[EPI]).round(3)
print(d_epi.round(2).to_string())
live = dose.loc[ASOF]
print(f"\n  LIVE 2026-09-10: 252-session change {live['tnx_252bp']:+.1f} bp, "
      f"21-session {live['tnx_21bp']:+.1f} bp, ^TNX rank {live['tnx_rank']:.1f}, "
      f"TLT {live['tlt_above_low']:.3f}% above its low")
q = (d_epi["tnx_252bp"] < live["tnx_252bp"]).mean() * 100
print(f"  today's 252-session dose sits at the {q:.0f}th percentile of the "
      f"{len(d_epi)} episode doses (min {d_epi['tnx_252bp'].min():+.0f}, "
      f"max {d_epi['tnx_252bp'].max():+.0f})")
# dose response
hi = d_epi["tnx_252bp"] >= d_epi["tnx_252bp"].median()
show([summarize(r1.loc[d_epi.index[hi.values]].values, "episodes, dose ABOVE median"),
      summarize(r1.loc[d_epi.index[~hi.values]].values, "episodes, dose BELOW median")],
     "dose response of the 252-session yield change")
deep = d_epi["tlt_above_low"] <= 0.05
show([summarize(r1.loc[d_epi.index[deep.values]].values,
                f"TLT EXACTLY at the low (<=0.05%, N={int(deep.sum())})"),
      summarize(r1.loc[d_epi.index[~deep.values]].values,
                f"TLT merely near (N={int((~deep).sum())})")],
     "depth response -- today TLT is exactly AT the low")

print("\n" + "=" * 78)
print("6. TAIL: what sits inside a 1-session hold entered 2026-09-11?")
print("=" * 78)
fut = load_events(["cpi", "ppi", "nfp", "fomc_decision", "opex",
                   "quad_witching", "vix_expiry"])
nxt = fut[(fut["date"] >= "2026-09-11") & (fut["date"] <= "2026-09-30")]
print(nxt.to_string(index=False))
print("  entry 2026-09-11 MOC, exit 2026-09-14 MOC (h=1 = next session close).")
