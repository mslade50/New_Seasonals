"""Monday Dip setup across IWM/DIA/SMH (live universe) vs SPY/QQQ (carved out).

Questions (2026-07-20):
  1. How does the Monday Dip setup perform per-ETF: IWM, DIA, SMH vs SPY, QQQ?
  2. When several fire the SAME Monday, is the outcome better or worse?
  3. Diversification value: is there independent alpha in IWM/DIA/SMH, or are we
     better off sizing up SPY/QQQ and ignoring the other three?
  4. When IWM/DIA/SMH fire INDEPENDENTLY (SPY & QQQ silent), is there still edge?
  5. When ALL FIVE fire on a Monday whose day-return was > -0.2 ATR across the
     board (a shallow-dip co-fire), what happens — and has it ever happened?

Method: run the LIVE Monday Dip config (2d rank<50, VIX>=13, 200SMA-15-consec,
range 0-15%, Mondays, limit Open+/-0.25 ATR, 2d hold, 1 ATR stop, 2 ATR target)
identically on all five ETFs. Flat sizing so R is apples-to-apples across names.
Co-fire counting is at the SIGNAL (mask) level; edge stats are on FILLED trades.
"""
import copy
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import data_provider
from strategy_config import STRATEGY_BOOK, ACCOUNT_VALUE
from pages.strat_backtester import (
    load_seasonal_map,
    load_atr_seasonal_map,
    precompute_all_indicators,
    generate_candidates_fast,
    process_signals_fast,
    get_historical_mask,
)

START = pd.Timestamp("2001-01-01")
TICKS = ["SPY", "QQQ", "IWM", "DIA", "SMH"]
FAMILY_OTHER = ["IWM", "DIA", "SMH"]   # the carved-out three
FAMILY_SPYQQQ = ["SPY", "QQQ"]

book = {s["name"]: s for s in STRATEGY_BOOK}
mdip = copy.deepcopy(book["Monday Dip"])
mdip["universe_tickers"] = TICKS   # apply the SAME setup to all five

md = data_provider.get_history(TICKS + ["^VIX"], start="1999-01-01")
vix_df = md.get("^VIX")
vix_series = None
if vix_df is not None and not vix_df.empty:
    vd = vix_df.copy()
    if isinstance(vd.columns, pd.MultiIndex):
        vd.columns = vd.columns.get_level_values(0)
    vd.columns = [c.capitalize() for c in vd.columns]
    vix_series = vd["Close"]

sznl_map = load_seasonal_map()
atr_sznl_map = load_atr_seasonal_map()
if not atr_sznl_map:
    print("WARNING: atr_seasonal_ranks.parquet missing — 5d ATR-sznl filter "
          "fails closed; Monday Dip will fire ~never.")

processed = precompute_all_indicators(md, [mdip], sznl_map, vix_series, atr_sznl_map)

# ---------------- signal masks per ticker (ex-ante) ----------------
# For each ticker collect: signal dates (Mondays) + today_return_atr on the bar.
sig = {}   # ticker -> DataFrame(index=signal dates) with 'ret_atr'
date_range = {}
for tk in TICKS:
    df = processed.get(tk)
    if df is None or df.empty:
        print(f"WARNING: no data for {tk}")
        sig[tk] = pd.DataFrame(columns=["ret_atr"])
        continue
    m = get_historical_mask(df, mdip["settings"], sznl_map, tk)
    m = m[m.index >= START]
    dates = m.index[m.values]
    ra = df["today_return_atr"].reindex(dates)
    sig[tk] = pd.DataFrame({"ret_atr": ra.values}, index=dates)
    date_range[tk] = (df.index.min(), df.index.max())

print("\n" + "=" * 82)
print("DATA COVERAGE (bar history available in cache)")
print("=" * 82)
for tk in TICKS:
    lo, hi = date_range.get(tk, (None, None))
    n = len(sig[tk])
    print(f"  {tk:<5} bars {str(lo.date()) if lo is not None else '?':>10} .. "
          f"{str(hi.date()) if hi is not None else '?':>10}   signals(Mon)={n}")

# ---------------- trade outcomes (filled) ----------------
def run(strat):
    cands, sd = generate_candidates_fast(processed, [strat], sznl_map, START)
    tr = process_signals_fast(cands, sd, processed, [strat], ACCOUNT_VALUE,
                              flat_sizing=True)
    if tr.empty:
        return tr
    tr["Date"] = pd.to_datetime(tr["Date"])
    tr["R"] = tr["PnL"] / tr["Risk $"].replace(0, np.nan)
    return tr

tr = run(mdip)
# collapse tranche rows if any (Monday Dip has none, but be safe)
tr = tr.dropna(subset=["R"]).copy()

def stats(r, label, extra=""):
    r = pd.Series(r).dropna()
    if len(r) == 0:
        print(f"{label:<50} N=   0")
        return
    pf = r[r > 0].sum() / max(1e-9, -(r[r < 0].sum()))
    print(f"{label:<50} N={len(r):>4}  win={(r > 0).mean():6.1%}  "
          f"avgR={r.mean():+.3f}  medR={r.median():+.3f}  "
          f"totR={r.sum():+7.1f}  PF={pf:5.2f}{extra}")

print("\n" + "=" * 82)
print(f"PART 1 — PER-ETF TRADE OUTCOMES (filled, flat sizing, {START.date()}->present)")
print("=" * 82)
for tk in TICKS:
    stats(tr.loc[tr.Ticker == tk, "R"], f"  {tk}")
print("  " + "-" * 78)
stats(tr.loc[tr.Ticker.isin(FAMILY_SPYQQQ), "R"], "  SPY+QQQ (sibling strat's names)")
stats(tr.loc[tr.Ticker.isin(FAMILY_OTHER), "R"], "  IWM+DIA+SMH (live Monday Dip)")
stats(tr["R"], "  ALL FIVE")

# ---------------- co-firing (signal level) ----------------
# Build a Monday x ticker matrix of signals; count how many fired each day.
all_sig_dates = sorted(set().union(*[set(sig[tk].index) for tk in TICKS]))
fire = pd.DataFrame(False, index=all_sig_dates, columns=TICKS)
retatr = pd.DataFrame(np.nan, index=all_sig_dates, columns=TICKS)
for tk in TICKS:
    fire.loc[sig[tk].index, tk] = True
    retatr.loc[sig[tk].index, tk] = sig[tk]["ret_atr"].values
fire["n"] = fire[TICKS].sum(axis=1)

print("\n" + "=" * 82)
print("PART 2 — CO-FIRING DISTRIBUTION (how many of the 5 signal the same Monday)")
print("=" * 82)
vc = fire["n"].value_counts().sort_index()
for k, v in vc.items():
    print(f"  exactly {int(k)} fire: {v:>4} Mondays")
print(f"  total distinct signal-Mondays: {len(fire)}")

# map each filled trade to that day's co-fire count
tr["cofire_n"] = tr["Date"].map(fire["n"]).fillna(1).astype(int)
print("\n  Filled-trade edge by co-fire count on the signal day:")
for k in range(1, 6):
    stats(tr.loc[tr.cofire_n == k, "R"], f"    co-fire = {k}")
stats(tr.loc[tr.cofire_n >= 3, "R"], "    co-fire >= 3")
stats(tr.loc[tr.cofire_n == 1, "R"], "    co-fire = 1 (fired alone)")

# ---------------- independent alpha in IWM/DIA/SMH ----------------
# Days where SPY AND QQQ were both SILENT (no signal). Do IWM/DIA/SMH still work?
spyqqq_fire_dates = set(fire.index[fire[["SPY", "QQQ"]].any(axis=1)])
tr["spyqqq_silent"] = ~tr["Date"].isin(spyqqq_fire_dates)

print("\n" + "=" * 82)
print("PART 3/4 — INDEPENDENT ALPHA: IWM/DIA/SMH when SPY & QQQ are BOTH silent")
print("=" * 82)
other = tr[tr.Ticker.isin(FAMILY_OTHER)]
stats(other.loc[other.spyqqq_silent, "R"],
      "  IWM+DIA+SMH  (SPY&QQQ silent that day)")
stats(other.loc[~other.spyqqq_silent, "R"],
      "  IWM+DIA+SMH  (SPY or QQQ also fired)")
for tk in FAMILY_OTHER:
    g = tr[tr.Ticker == tk]
    stats(g.loc[g.spyqqq_silent, "R"], f"    {tk} independent (SPY&QQQ silent)")

# and SPY/QQQ when the small-caps are silent (symmetry check)
other_fire_dates = set(fire.index[fire[FAMILY_OTHER].any(axis=1)])
tr["other_silent"] = ~tr["Date"].isin(other_fire_dates)
sq = tr[tr.Ticker.isin(FAMILY_SPYQQQ)]
stats(sq.loc[sq.other_silent, "R"], "  SPY+QQQ  (IWM/DIA/SMH all silent)")

# ---------------- diversification vs concentration ----------------
# Daily-basket comparison: equal-weight R per day for the two families.
print("\n" + "=" * 82)
print("PART 3b — DIVERSIFICATION: daily equal-weight basket R, family vs family")
print("=" * 82)
tr["day"] = tr["Date"].dt.normalize()
def basket_daily(sub):
    return sub.groupby("day")["R"].mean()   # avg R across whichever names filled that day
b_other = basket_daily(tr[tr.Ticker.isin(FAMILY_OTHER)])
b_sq = basket_daily(tr[tr.Ticker.isin(FAMILY_SPYQQQ)])
joined = pd.concat([b_sq.rename("spyqqq"), b_other.rename("other")], axis=1)
overlap = joined.dropna()
print(f"  trading days IWM/DIA/SMH active: {b_other.notna().sum()}")
print(f"  trading days SPY/QQQ active:     {b_sq.notna().sum()}")
print(f"  days BOTH families active:       {len(overlap)}")
if len(overlap) > 5:
    print(f"  corr(SPY/QQQ dayR, IWM/DIA/SMH dayR) on shared days: "
          f"{overlap['spyqqq'].corr(overlap['other']):+.2f}")
    print(f"  mean dayR  SPY/QQQ={overlap['spyqqq'].mean():+.3f}  "
          f"IWM/DIA/SMH={overlap['other'].mean():+.3f}")
# marginal contribution: R that lands on days SPY/QQQ did NOT trade at all
sq_days = set(b_sq.index)
other_only_days = b_other[~b_other.index.isin(sq_days)]
print(f"  IWM/DIA/SMH trading days with NO SPY/QQQ trade: {len(other_only_days)}  "
      f"mean dayR={other_only_days.mean():+.3f}  totR contribution="
      f"{tr[(tr.Ticker.isin(FAMILY_OTHER)) & (~tr.day.isin(sq_days))]['R'].sum():+.1f}")

# ---------------- the shallow all-five co-fire ----------------
print("\n" + "=" * 82)
print("PART 5 — ALL FIVE FIRE + day-return > -0.2 ATR across the board")
print("=" * 82)
all5 = fire[fire["n"] == 5].index
print(f"  Mondays where ALL FIVE signalled: {len(all5)}")
shallow = []
for d in all5:
    row = retatr.loc[d, TICKS]
    if (row > -0.2).all():
        shallow.append(d)
print(f"  ...of which every name's day-return > -0.2 ATR (shallow dip): {len(shallow)}")
if len(all5):
    print("\n  All-five dates (ret_atr per name; * = shallow >-0.2 all):")
    for d in all5:
        row = retatr.loc[d, TICKS]
        tag = "  *SHALLOW" if d in shallow else ""
        vals = "  ".join(f"{tk}={row[tk]:+.2f}" for tk in TICKS)
        print(f"    {d.date()}  {vals}{tag}")
    # outcomes of trades whose signal landed on an all-five day / shallow day
    print()
    stats(tr.loc[tr.Date.isin(set(all5)), "R"], "  trades on ALL-FIVE signal days")
    if shallow:
        stats(tr.loc[tr.Date.isin(set(shallow)), "R"], "  trades on SHALLOW all-five days")

# ---------------- how deep is the dip at each breadth level? ----------------
print("\n" + "=" * 82)
print("PART 5b — DIP DEPTH vs BREADTH: mean day-return (ATR) of the FIRING names")
print("=" * 82)
for k in range(1, 6):
    days_k = fire.index[fire["n"] == k]
    if len(days_k) == 0:
        continue
    depths = []
    for d in days_k:
        row = retatr.loc[d, TICKS]
        depths.extend(row.dropna().values)
    depths = np.array(depths)
    print(f"  co-fire={k}: firing-name day-return mean={depths.mean():+.2f} ATR  "
          f"median={np.median(depths):+.2f}  min={depths.min():+.2f}  "
          f"share shallower than -0.2: {(depths > -0.2).mean():5.1%}")

# any multi-name co-fire (n>=2) where EVERY firing name was shallow (>-0.2)?
print("\n  Multi-name co-fire days (n>=2) that were SHALLOW across all firing names:")
found = 0
for d in fire.index[fire["n"] >= 2]:
    row = retatr.loc[d, TICKS].dropna()
    if (row > -0.2).all():
        found += 1
        vals = "  ".join(f"{tk}={row[tk]:+.2f}" for tk in row.index)
        print(f"    {d.date()}  (n={int(fire.loc[d,'n'])})  {vals}")
if not found:
    print("    NONE — no co-fire of 2+ names was ever shallow across the board.")

# ---------------- LIVE-universe (IWM/DIA/SMH only) breadth ----------------
print("\n" + "=" * 82)
print("PART 6 — LIVE UNIVERSE breadth: co-fire within {IWM,DIA,SMH} only")
print("        (this is the breadth actually observable to the live strategy)")
print("=" * 82)
LIVE = FAMILY_OTHER  # IWM, DIA, SMH
live_dates = sorted(set().union(*[set(sig[tk].index) for tk in LIVE]))
fire3 = pd.DataFrame(False, index=live_dates, columns=LIVE)
for tk in LIVE:
    fire3.loc[sig[tk].index, tk] = True
fire3["n"] = fire3[LIVE].sum(axis=1)
vc3 = fire3["n"].value_counts().sort_index()
for k, v in vc3.items():
    print(f"  exactly {int(k)} of 3 fire: {v:>4} Mondays")
print(f"  total distinct signal-Mondays (live universe): {len(fire3)}")

trL = tr[tr.Ticker.isin(LIVE)].copy()
trL["cofire3"] = trL["Date"].map(fire3["n"]).fillna(1).astype(int)
print("\n  Filled-trade edge by 3-name co-fire count:")
for k in range(1, 4):
    stats(trL.loc[trL.cofire3 == k, "R"], f"    co-fire = {k} of 3")
stats(trL.loc[trL.cofire3 >= 2, "R"], "    co-fire >= 2 of 3 (breadth gate ON)")
solo = trL.loc[trL.cofire3 == 1]
tot = trL["R"].sum()
print(f"\n  Solo (1-of-3) bucket: {len(solo)}/{len(trL)} trades "
      f"({len(solo)/len(trL):.0%}), totR {solo['R'].sum():+.1f} of "
      f"{tot:+.1f} book ({solo['R'].sum()/tot:+.0%} of total return).")
print(f"  Removing solo days -> avgR {trL.loc[trL.cofire3>=2,'R'].mean():+.3f} "
      f"(vs {trL['R'].mean():+.3f} all), but you forgo {solo['R'].sum():+.1f}R "
      f"of realized profit unless you REALLOCATE that risk.")

print("\nDone.")
