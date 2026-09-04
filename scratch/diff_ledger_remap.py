"""Trade-level diff of the ledger across the 2026-08-07 ticker repairs.

Totals matching is not proof: a rebuild can lose a trade on one ticker and gain
one elsewhere and still land on the same count. This compares by IDENTITY.

The identity key MUST include Tranche. OVS scale-outs book two rows per fill
(near 40% at 1 ATR, far 60% at 2 ATR) sharing a strategy, tier, ticker and
signal date. A first pass keyed without it silently compared near tranches
against far ones and reported 122 "changed" exit prices whose returns were
exact halves and doubles of each other. 1192 rows were also being dropped by a
duplicate-index guard. Both were artifacts of the key, not the ledger.

Repaired tickers are joined under their NEW name against the old ledger's OLD
name, otherwise every one would read as a delete plus an insert.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BEFORE = ROOT / "scratch" / "_ledger_before_remap.parquet"
AFTER = ROOT / "data" / "backtest_trades_full.parquet"
PAIRS = {"BK": "BNY", "ASGN": "EFOR", "IAC": "PPLI", "SATS": "ECHO",
         "MMC": "MRSH", "ARMN": "ARIS", "ATGE": "CVSA"}
NEW_NAMES = set(PAIRS.values())
KEY = ["Strategy", "Tier", "Ticker", "Signal Date", "Tranche"]

a = pd.read_parquet(BEFORE)
b = pd.read_parquet(AFTER)
a["Ticker"] = a["Ticker"].replace(PAIRS)
print(f"before {len(a)} trades | after {len(b)} trades | delta {len(b) - len(a):+d}")
for f, d in (("before", a), ("after", b)):
    assert not d.duplicated(KEY).any(), f"{f}: key is not unique"
print(f"identity key unique in both: {KEY}\n")

ka, kb = a.set_index(KEY).sort_index(), b.set_index(KEY).sort_index()
gone, added = ka.index.difference(kb.index), kb.index.difference(ka.index)
both = ka.index.intersection(kb.index)
print(f"vanished {len(gone)} | appeared {len(added)} | common {len(both)}\n")


def show(idx, label):
    if not len(idx):
        return
    df = pd.DataFrame(list(idx), columns=KEY)
    df["repaired"] = df.Ticker.isin(NEW_NAMES)
    print(f"{label} ({len(idx)}) — on repaired tickers: {int(df.repaired.sum())}, "
          f"elsewhere: {int((~df.repaired).sum())}")
    other = df[~df.repaired]
    if len(other):
        print("  NOT on a repaired ticker (these need explaining):")
        print(other.head(15).to_string(index=False))
    print()


show(gone, "VANISHED")
show(added, "APPEARED")

# --- common trades: did any number move? -----------------------------------
print("=" * 78)
print("COMMON TRADES — did any number move?")
print("=" * 78)
ca, cb = ka.loc[both], kb.loc[both]
num = [c for c in ("Return_Pct", "PnL_flat_750k", "PnL_compounded", "Shares",
                   "Entry Price", "Exit Price") if c in ca.columns]
rows = []
for c in num:
    x = pd.to_numeric(ca[c], errors="coerce")
    y = pd.to_numeric(cb[c], errors="coerce")
    d = (y - x).abs()
    rel = d / x.abs().replace(0, np.nan)
    rows.append({"field": c, "n": int(d.notna().sum()),
                 "rows_changed": int((d > 1e-9).sum()),
                 "max_abs_diff": round(float(np.nan_to_num(d.max())), 8),
                 "max_rel_pct": round(100 * float(np.nan_to_num(rel.max())), 6)})
print(pd.DataFrame(rows).to_string(index=False))

changed_price = int((pd.to_numeric(cb["Exit Price"], errors="coerce")
                     - pd.to_numeric(ca["Exit Price"], errors="coerce")).abs()
                    .gt(1e-9).sum())
changed_ret = int((pd.to_numeric(cb["Return_Pct"], errors="coerce")
                   - pd.to_numeric(ca["Return_Pct"], errors="coerce")).abs()
                  .gt(1e-9).sum())

# --- repaired tickers ------------------------------------------------------
print("\n" + "=" * 78)
print("REPAIRED TICKERS")
print("=" * 78)
for t in sorted(NEW_NAMES):
    na, nb = int((a.Ticker == t).sum()), int((b.Ticker == t).sum())
    if na or nb:
        print(f"  {t:<6} before {na:>3}  after {nb:>3}  "
              f"{'(new history recovered)' if nb > na else ''}")

# --- book totals -----------------------------------------------------------
print("\n" + "=" * 78)
print("BOOK TOTALS")
print("=" * 78)
for c in ("Return_Pct", "PnL_flat_750k", "PnL_compounded"):
    if c in a.columns:
        x = pd.to_numeric(a[c], errors="coerce").sum()
        y = pd.to_numeric(b[c], errors="coerce").sum()
        print(f"  {c:<16} before {x:>16,.2f}   after {y:>16,.2f}   "
              f"delta {y - x:+,.2f}")

# --- verdict ---------------------------------------------------------------
gone_other = [k for k in gone if k[2] not in NEW_NAMES]
added_other = [k for k in added if k[2] not in NEW_NAMES]
print("\n" + "=" * 78)
ok = (changed_price == 0 and changed_ret == 0 and not added_other)
print(f"exit prices moved on a pre-existing trade : {changed_price}")
print(f"returns moved on a pre-existing trade     : {changed_ret}")
print(f"trades appeared on an UNrepaired ticker   : {len(added_other)}")
print(f"trades vanished on an UNrepaired ticker   : {len(gone_other)} "
      f"(expected: the delisted names removed from the universe)")
print(f"\n{'CLEAN — no booked trade moved' if ok else 'INSPECT — something moved'}")
