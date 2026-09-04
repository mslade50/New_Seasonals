"""Family-only engine re-run over 2026 so the post-2026-07-29 window (which the
local pcfear shadow, built 2026-08-07, does not cover) is scored from bars by
the production engine itself. Three passes on identical candidates:

  pcfear_on   production rule (parity check against data/backtest_trades_full)
  pcfear_off  incumbent 0.25x table everywhere (= build_pcfear_shadow)
  bands_off   frag_risk_bands + pc_fear_bands removed (1.0x at every dial)

Writes shadow_2026_<pass>.parquet into this folder. Nothing under data/ is
written. Mirrors scripts/build_trade_ledger.py step for step (imports, not
reimplementation)."""
from __future__ import annotations

import copy
import datetime
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))


# The dev interpreter's streamlit cannot import (protobuf descriptor error);
# the strat_backtester import only needs inert decorators here. Same stub as
# scratch/ultracode_sizing_2026-09-02/dd_pit/02_pit_dial_extended.py.
class _NoOp:
    def __getattr__(self, name): return self
    def __call__(self, *a, **k): return self
    def __bool__(self): return False
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def cache_data(self, *a, **k):
        # handles both @st.cache_data and @st.cache_data(ttl=...)
        if len(a) == 1 and callable(a[0]) and not k:
            return a[0]
        def deco(fn): return fn
        return deco
    cache_resource = cache_data


sys.modules["streamlit"] = _NoOp()

from strategy_config import STRATEGY_BOOK, ACCOUNT_VALUE  # noqa: E402
from pages.strat_backtester import (  # noqa: E402
    load_seasonal_map, load_atr_seasonal_map, precompute_all_indicators,
    generate_candidates_fast, process_signals_fast,
)
from build_trade_ledger import (  # noqa: E402
    load_data, shape_flat_trades, POOLED_LONG_CAP_BPS, POOLED_SHORT_CAP_BPS,
)

BT_START = datetime.date(2026, 1, 1)
FAMILY = [s for s in STRATEGY_BOOK if s.get("execution", {}).get("pc_fear_bands")]
print("family:", [s["name"] for s in FAMILY])
assert len(FAMILY) == 6

tickers = set()
for s in FAMILY:
    tickers.update(s["universe_tickers"])
tickers.update(["SPY", "^VIX"])
md = load_data(tickers)
vix_df = md.get("^VIX")
vix_series = None
if vix_df is not None and not vix_df.empty:
    vd = vix_df.copy()
    if isinstance(vd.columns, pd.MultiIndex):
        vd.columns = vd.columns.get_level_values(0)
    vd.columns = [c.capitalize() for c in vd.columns]
    vix_series = vd["Close"]
last_bar = max(pd.to_datetime(df.index.max()) for t, df in md.items() if df is not None and not df.empty)
print("last bar in loaded data:", last_bar.date())

sznl_map = load_seasonal_map()
atr_sznl_map = load_atr_seasonal_map()


def run(book, pc_fear_enabled, tag):
    processed = precompute_all_indicators(md, book, sznl_map, vix_series, atr_sznl_map)
    candidates, signal_data = generate_candidates_fast(processed, book, sznl_map, BT_START)
    print(f"[{tag}] {len(candidates)} candidate signal-dates")
    sig = process_signals_fast(
        candidates, signal_data, processed, book, ACCOUNT_VALUE,
        cap_bps=250, overflow_active=True, flat_sizing=True,
        max_long_risk_bps=POOLED_LONG_CAP_BPS, max_short_risk_bps=POOLED_SHORT_CAP_BPS,
        pc_fear_enabled=pc_fear_enabled,
    )
    df = shape_flat_trades(sig)
    df = df[df["Strategy"].isin([s["name"] for s in book])].reset_index(drop=True)
    out = os.path.join(HERE, f"shadow_2026_{tag}.parquet")
    df.to_parquet(out)
    print(f"[{tag}] {len(df)} trades -> {out}; signal dates {df['Signal Date'].min().date()} .. {df['Signal Date'].max().date()}")
    return df


book_prod = copy.deepcopy(FAMILY)
on = run(book_prod, True, "pcfear_on")
off = run(copy.deepcopy(FAMILY), False, "pcfear_off")
book_nb = copy.deepcopy(FAMILY)
for s in book_nb:
    s["execution"].pop("frag_risk_bands", None)
    s["execution"].pop("pc_fear_bands", None)
nb = run(book_nb, False, "bands_off")

# --- parity of the pcfear_on pass vs the production ledger, 2026 family rows ---
led = pd.read_parquet(os.path.join(ROOT, "data", "backtest_trades_full.parquet"))
fam = led[led["Strategy"].isin([s["name"] for s in FAMILY])].copy()
fam["Signal Date"] = pd.to_datetime(fam["Signal Date"])
fam = fam[fam["Signal Date"] >= pd.Timestamp(BT_START)]
key = ["Strategy", "Ticker", "Signal Date"]
a = fam.groupby(key).agg(pnl=("PnL_flat_750k", "sum"), risk=("Risk_flat_750k", "sum")).reset_index()
b = on.groupby(key).agg(pnl=("PnL_flat_750k", "sum"), risk=("Risk_flat_750k", "sum")).reset_index()
m = a.merge(b, on=key, how="outer", suffixes=("_led", "_rerun"), indicator=True)
print("\n=== parity: production ledger vs pcfear_on re-run (2026 family positions) ===")
print(m["_merge"].value_counts().to_string())
both = m[m["_merge"] == "both"].copy()
both["R_led"] = both.pnl_led / both.risk_led
both["R_rerun"] = both.pnl_rerun / both.risk_rerun
print(f"positions in both: {len(both)}; max |R diff| {(both.R_led - both.R_rerun).abs().max():.4f}; "
      f"max |PnL diff| ${(both.pnl_led - both.pnl_rerun).abs().max():,.0f}")
odd = m[m["_merge"] != "both"]
if len(odd):
    print(odd.to_string())

# --- parity of the pcfear_off pass vs the local Aug-07 shadow over the overlap ---
sh = pd.read_parquet(os.path.join(ROOT, "data", "backtest_trades_pcfear_shadow.parquet"))
sh["Signal Date"] = pd.to_datetime(sh["Signal Date"])
sh = sh[sh["Signal Date"] >= pd.Timestamp(BT_START)]
c = sh.groupby(key).agg(pnl=("PnL_flat_750k", "sum"), risk=("Risk_flat_750k", "sum")).reset_index()
d = off.groupby(key).agg(pnl=("PnL_flat_750k", "sum"), risk=("Risk_flat_750k", "sum")).reset_index()
d = d[d["Signal Date"] <= sh["Signal Date"].max()]
m2 = c.merge(d, on=key, how="outer", suffixes=("_shadow", "_rerun"), indicator=True)
print("\n=== parity: local 2026-08-07 shadow vs pcfear_off re-run (2026, through the shadow's last signal) ===")
print(m2["_merge"].value_counts().to_string())
b2 = m2[m2["_merge"] == "both"].copy()
print(f"positions in both: {len(b2)}; max |R diff| {((b2.pnl_shadow / b2.risk_shadow) - (b2.pnl_rerun / b2.risk_rerun)).abs().max():.4f}")
odd2 = m2[m2["_merge"] != "both"]
if len(odd2):
    print(odd2.to_string())

# --- the zeroed set: in OFF, not in ON ---
z = off.merge(on[key].drop_duplicates(), on=key, how="left", indicator=True)
z = z[z["_merge"] == "left_only"].drop(columns="_merge")
print(f"\n=== zeroed by the live table in 2026 (present in pcfear_off, absent in pcfear_on): {len(z)} rows ===")
cols = ["Strategy", "Ticker", "Signal Date", "Entry Date", "Exit Date", "Exit Type", "R_Multiple", "PnL_flat_750k", "Risk_flat_750k", "Size_Mult"]
print(z[cols].sort_values("Signal Date").to_string())
