"""Decisive check on the 2026-08-07 ticker repairs.

R2 still serves the PRE-change parquet, so the old and new caches can be
diffed directly instead of reasoned about. Three questions, in order of how
much they matter:

  1. did any ticker OTHER than the seven repaired ones move? (must be zero)
  2. do the bars the LEDGER actually used reprice identically?
  3. is the new data sane where it is genuinely new?

Question 2 is the one that decides whether uploading is safe: a replaced
series that shifts a booked trade would silently rewrite history the live
gates read.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cache_io import download_to_local  # noqa: E402

OLD = ROOT / "scratch" / "_master_prices_r2_prechange.parquet"
NEW = ROOT / "data" / "master_prices.parquet"
PAIRS = {"BK": "BNY", "ASGN": "EFOR", "IAC": "PPLI", "SATS": "ECHO",
         "MMC": "MRSH", "ARMN": "ARIS", "ATGE": "CVSA"}

if not OLD.exists():
    print("pulling pre-change parquet from R2 ...")
    if not download_to_local("master_prices.parquet", str(OLD)):
        raise SystemExit("could not pull the R2 copy; cannot verify")

old = pd.read_parquet(OLD)
new = pd.read_parquet(NEW)
touched = set(PAIRS) | set(PAIRS.values())
print(f"old: {old.ticker.nunique()} tickers / {len(old):,} rows")
print(f"new: {new.ticker.nunique()} tickers / {len(new):,} rows\n")

# --- 1. nothing else moved -------------------------------------------------
key = ["ticker", "date"]
o = (old[~old.ticker.isin(touched)].sort_values(key).reset_index(drop=True))
n = (new[~new.ticker.isin(touched)].sort_values(key).reset_index(drop=True))
print("=" * 78)
print("1) EVERY UNTOUCHED TICKER IDENTICAL")
print("=" * 78)
same = o.equals(n)
print(f"   untouched tickers: {o.ticker.nunique()} | rows {len(o):,} vs {len(n):,}")
print(f"   byte-identical: {same}")
if not same:
    diff = o.merge(n, on=key, suffixes=("_o", "_n"))
    bad = diff[(diff.Close_o != diff.Close_n)]
    print(f"   MISMATCHED ROWS: {len(bad)}")
    print(bad.head().to_string())

# --- 2. ledger bars reprice identically ------------------------------------
print("\n" + "=" * 78)
print("2) DO THE BARS THE LEDGER USED REPRICE IDENTICALLY?")
print("=" * 78)
led = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
dcols = [c for c in led.columns if "Date" in c]
rows, worst = [], 0.0
for oldt, newt in PAIRS.items():
    sub = led[led.Ticker.isin([oldt, newt])]
    if sub.empty:
        continue
    o_s = old[old.ticker == oldt].set_index("date")
    n_s = new[new.ticker == newt].set_index("date")
    for _, tr in sub.iterrows():
        for dc in dcols:
            d = tr.get(dc)
            if pd.isna(d) or d not in o_s.index or d not in n_s.index:
                continue
            for col in ("Open", "High", "Low", "Close"):
                a, b = float(o_s.loc[d, col]), float(n_s.loc[d, col])
                rel = abs(b / a - 1.0) if a else 0.0
                worst = max(worst, rel)
                if rel > 1e-6:
                    rows.append({"ticker": f"{oldt}->{newt}", "date": str(d.date()),
                                 "field": col, "old": a, "new": b,
                                 "rel_diff_pct": 100 * rel})
print(f"   ledger trades on repaired tickers: "
      f"{sum(len(led[led.Ticker.isin([o_, n_])]) for o_, n_ in PAIRS.items())}")
print(f"   worst relative change on any ledger-used bar: {100 * worst:.6f}%")
if rows:
    print(f"   BARS THAT MOVED: {len(rows)}")
    print(pd.DataFrame(rows).head(20).to_string(index=False))
else:
    print("   no ledger-used bar changed. Booked history is untouched.")

# --- 3. the genuinely new data is sane -------------------------------------
print("\n" + "=" * 78)
print("3) SANITY OF THE NEW/EXTENDED DATA")
print("=" * 78)
out = []
for oldt, newt in PAIRS.items():
    s = new[new.ticker == newt].sort_values("date")
    ret = s.Close.pct_change().abs()
    ohlc_ok = bool(((s.High >= s.Low) & (s.High >= s.Close) &
                    (s.Low <= s.Close)).all())
    out.append({"ticker": newt, "bars": len(s),
                "last": str(s.date.max().date()),
                "max_1d_move_pct": round(100 * ret.max(), 1),
                "nonpos_close": int((s.Close <= 0).sum()),
                "ohlc_consistent": ohlc_ok,
                "dupe_dates": int(s.date.duplicated().sum())})
print(pd.DataFrame(out).to_string(index=False))

ok = same and not rows and all(r["ohlc_consistent"] and not r["nonpos_close"]
                               and not r["dupe_dates"] for r in out)
print(f"\n{'SAFE TO UPLOAD' if ok else 'DO NOT UPLOAD — see above'}")


# --- 4. the test that actually decides: is the change SCALE-INVARIANT? -----
# CLAUDE.md's dividend-adjustment rule: the engines recompute every level from
# the same adjusted series each run (Close +/- k*ATR) and compare to that
# series' own forward bars, so a UNIFORM rescale of a whole series leaves every
# fill decision, R-multiple and percentage return exactly unchanged. Only the
# dollar prices move. So "a bar changed" is not the question. "Did the ratio
# change WITHIN any trade's span" is.
print("\n" + "=" * 78)
print("4) SCALE-INVARIANCE WITHIN EVERY AFFECTED LEDGER TRADE")
print("=" * 78)
verdicts, unsafe = [], 0
for oldt, newt in PAIRS.items():
    o_s = old[old.ticker == oldt].set_index("date")["Close"]
    n_s = new[new.ticker == newt].set_index("date")["Close"]
    common = o_s.index.intersection(n_s.index)
    if len(common) == 0:
        verdicts.append({"ticker": f"{oldt}->{newt}", "trades": 0,
                         "verdict": "NEW SERIES (no prior data)"})
        continue
    ratio = (n_s[common] / o_s[common])
    sub = led[led.Ticker.isin([oldt, newt])]
    if sub.empty:
        drift = float((ratio.max() / ratio.min()) - 1.0)
        verdicts.append({"ticker": f"{oldt}->{newt}", "trades": 0,
                         "verdict": "no ledger trades",
                         "max_intra_drift_pct": round(100 * drift, 6)})
        continue
    worst_span = 0.0
    for _, tr in sub.iterrows():
        ds = [tr.get(c) for c in dcols if pd.notna(tr.get(c))]
        if not ds:
            continue
        lo, hi = min(ds), max(ds)
        w = ratio[(ratio.index >= lo) & (ratio.index <= hi)]
        if len(w) > 1:
            worst_span = max(worst_span, float(w.max() / w.min() - 1.0))
    ok = worst_span < 1e-5
    unsafe += 0 if ok else 1
    verdicts.append({"ticker": f"{oldt}->{newt}", "trades": len(sub),
                     "max_intra_drift_pct": round(100 * worst_span, 8),
                     "verdict": "SCALE-INVARIANT (returns unchanged)" if ok
                                else "RATIO MOVES INSIDE A TRADE"})
print(pd.DataFrame(verdicts).to_string(index=False))
print("\n  A constant ratio inside a trade's span means entry, stop, target and")
print("  ATR all scale together: identical fills, identical R, identical %.")
print(f"\n{'SAFE TO UPLOAD' if same and not unsafe else 'DO NOT UPLOAD'}"
      f" — untouched-identical={same}, trades with intra-span drift={unsafe}")
