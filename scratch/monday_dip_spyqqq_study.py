"""Monday Dip on SPY/QQQ vs SPY QQQ MonFri Reversion — carve-out audit.

Question (2026-07-07): the scan email notes SPY/QQQ are excluded from Monday
Dip because MonFri Reversion handles them. What would Monday Dip signals on
SPY/QQQ return on days when MonFri Reversion does NOT also trigger?

Config read: on a Monday, Monday Dip's entry conditions look like a strict
subset of MonFri's (2d rank <50 implies <85; identical range 0-15 / VIX>=13 /
price / vol / age / ATR%% gates; Monday Dip only ADDS the 200SMA-15-consec and
5d-ATR-seasonal>15 filters). If that holds, "Dip fires but MonFri doesn't" is
empty by construction. This script verifies it with the production engine and
also grades the overlap: do Monday-Dip-qualified Mondays beat the rest of
MonFri's Monday signals? Both strategies share identical execution (limit
Open+/-0.25 ATR, 2d hold, 1 ATR stop, 2 ATR target), so per-(ticker, date)
trade outcomes are strategy-invariant and R comparisons are apples-to-apples.
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

book = {s["name"]: s for s in STRATEGY_BOOK}
monfri = copy.deepcopy(book["SPY QQQ MonFri Reversion"])
mdip = copy.deepcopy(book["Monday Dip"])
mdip["name"] = "Monday Dip (SPY/QQQ test)"
mdip["universe_tickers"] = ["SPY", "QQQ"]
strategies = [monfri, mdip]

md = data_provider.get_history(["SPY", "QQQ", "^VIX"], start="1999-01-01")
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
    print("WARNING: atr_seasonal_ranks.parquet missing — Monday Dip's "
          "5d ATR-sznl filter fails closed and it will fire ~never.")

processed = precompute_all_indicators(md, strategies, sznl_map, vix_series, atr_sznl_map)

# ---------------- signal masks ----------------
masks = {}
for tk in ["SPY", "QQQ"]:
    df = processed[tk]
    m_mf = get_historical_mask(df, monfri["settings"], sznl_map, tk)
    m_md = get_historical_mask(df, mdip["settings"], sznl_map, tk)
    m_mf = m_mf[m_mf.index >= START]
    m_md = m_md[m_md.index >= START]
    masks[tk] = (m_mf, m_md, df)

print("\n" + "=" * 78)
print("SIGNAL SET COMPARISON (Mondays only, per ticker)")
print("=" * 78)
dip_only_all = []
for tk in ["SPY", "QQQ"]:
    m_mf, m_md, df = masks[tk]
    mon = m_mf.index.weekday == 0
    mf_mon = set(m_mf.index[m_mf.values & mon])
    md_mon = set(m_md.index[m_md.values & (m_md.index.weekday == 0)])
    both = mf_mon & md_mon
    dip_only = md_mon - mf_mon
    mf_only = mf_mon - md_mon
    dip_only_all.extend((tk, d) for d in sorted(dip_only))
    print(f"\n{tk}: MonFri Monday signals={len(mf_mon)}  MondayDip signals={len(md_mon)}")
    print(f"  both fire: {len(both)}   dip-ONLY (MonFri silent): {len(dip_only)}   "
          f"MonFri-only: {len(mf_only)}")

    # why the MonFri-only Mondays fail Monday Dip's extra gates
    if mf_only:
        dd = df.loc[sorted(mf_only)]
        r2 = dd["rank_ret_2d"].values >= 50.0
        sma_ok = pd.Series(df["Close"].values > df["SMA200"].values, index=df.index)
        sma_consec = sma_ok.rolling(15).sum().reindex(dd.index).values < 15
        asz = ~(dd["atr_sznl_5d"].values > 15.0)  # fail or NaN
        print(f"  MonFri-only failure reasons (not mutually exclusive): "
              f"2d rank >=50: {r2.sum()}  |  <15d above 200SMA: {sma_consec.sum()}"
              f"  |  ATR-sznl<=15/NaN: {asz.sum()}")

if dip_only_all:
    print("\nDIP-ONLY DATES (Monday Dip fires, MonFri silent) — investigate:")
    for tk, d in dip_only_all:
        print(f"  {tk} {d.date()}")
else:
    print("\n>> Subset CONFIRMED: every SPY/QQQ Monday Dip signal is also a "
          "MonFri Reversion signal. 'Dip fires, MonFri doesn't' = empty set.")

# ---------------- trade outcomes ----------------
def run(strat):
    cands, sd = generate_candidates_fast(processed, [strat], sznl_map, START)
    tr = process_signals_fast(cands, sd, processed, [strat], ACCOUNT_VALUE,
                              flat_sizing=True)
    if tr.empty:
        return tr
    tr["Date"] = pd.to_datetime(tr["Date"])
    tr["R"] = tr["PnL"] / tr["Risk $"].replace(0, np.nan)
    return tr

def stats(r, label):
    r = pd.Series(r).dropna()
    if len(r) == 0:
        print(f"{label:<52} N=   0")
        return
    pf = r[r > 0].sum() / max(1e-9, -(r[r < 0].sum()))
    print(f"{label:<52} N={len(r):>4}  win={(r > 0).mean():6.1%}  "
          f"avgR={r.mean():+.3f}  medR={r.median():+.3f}  "
          f"totR={r.sum():+8.1f}  PF={pf:5.2f}")

tr_mf = run(monfri)
tr_md = run(mdip)

tr_mf["wd"] = tr_mf["Date"].dt.weekday
mf_mon_tr = tr_mf[tr_mf["wd"] == 0].copy()
mf_fri_tr = tr_mf[tr_mf["wd"] == 4].copy()

# label each MonFri Monday trade with dip qualification
def dip_qual(row):
    m_md = masks[row["Ticker"]][1]
    return bool(m_md.reindex([row["Date"]]).fillna(False).iloc[0])

mf_mon_tr["dip_qual"] = mf_mon_tr.apply(dip_qual, axis=1)

print("\n" + "=" * 78)
print(f"TRADE OUTCOMES (filled trades, {START.date()} -> present, "
      f"identical execution both strats)")
print("=" * 78)
stats(tr_mf["R"], "MonFri Reversion — ALL (Mon+Fri, SPY+QQQ)")
stats(mf_fri_tr["R"], "MonFri Reversion — Fridays only")
stats(mf_mon_tr["R"], "MonFri Reversion — Mondays only")
stats(mf_mon_tr.loc[mf_mon_tr.dip_qual, "R"],
      "  Mondays ALSO qualifying as Monday Dip (overlap)")
stats(mf_mon_tr.loc[~mf_mon_tr.dip_qual, "R"],
      "  Mondays NOT dip-qualified (MonFri-only)")
stats(tr_md["R"], "Monday Dip run standalone on SPY/QQQ")

# per-ticker split of the overlap
for tk in ["SPY", "QQQ"]:
    g = mf_mon_tr[(mf_mon_tr.Ticker == tk) & mf_mon_tr.dip_qual]
    stats(g["R"], f"  overlap, {tk} only")

print("\nNote: Monday Dip standalone should track the overlap bucket nearly "
      "1:1 (same fills); small diffs = max_one_pos/position-mgmt effects.")
