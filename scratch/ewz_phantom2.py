"""Forensic part 2: yfinance raw vs adjusted + dividend, and the ledger trade."""
from pathlib import Path
import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

ROOT = Path(r"C:\Users\McKinley Slade\dev\New_Seasonals")
LIMIT = 33.51

print("=" * 70)
print("LEDGER: EWZ trades in backtest_trades_full.parquet")
print("=" * 70)
led = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
print("ledger cols:", list(led.columns))
tcol = next((c for c in led.columns if c.lower() == "ticker"), None)
if tcol:
    ewz_t = led[led[tcol] == "EWZ"].copy()
    print(f"\nEWZ trades: {len(ewz_t)}")
    # show recent ones
    datecols = [c for c in led.columns if "date" in c.lower() or "entry" in c.lower() or "exit" in c.lower()]
    print("date-ish cols:", datecols)
    if len(ewz_t):
        print(ewz_t.tail(8).to_string())

print("\n" + "=" * 70)
print("YFINANCE: raw (auto_adjust=False) + dividends + adjusted")
print("=" * 70)
try:
    import yfinance as yf
    tk = yf.Ticker("EWZ")
    div = tk.dividends
    div = div[div.index >= "2026-01-01"] if len(div) else div
    print("Dividends since 2026-01-01:")
    print(div.to_string() if len(div) else "  (none returned)")
    acts = tk.actions
    if acts is not None and len(acts):
        acts = acts[acts.index >= "2026-05-01"]
        print("\nActions since 2026-05-01:")
        print(acts.to_string() if len(acts) else "  (none)")

    raw = yf.download("EWZ", start="2026-05-25", end="2026-06-18",
                      auto_adjust=False, progress=False)
    adj = yf.download("EWZ", start="2026-05-25", end="2026-06-18",
                      auto_adjust=True, progress=False)

    def flat(d):
        if isinstance(d.columns, pd.MultiIndex):
            d = d.copy()
            d.columns = d.columns.get_level_values(0)
        return d

    raw, adj = flat(raw), flat(adj)
    print("\nRAW (auto_adjust=False) OHLC + Adj Close:")
    rc = [c for c in ["Open", "High", "Low", "Close", "Adj Close"] if c in raw.columns]
    print(raw[rc].round(4).to_string())
    print("\nADJUSTED (auto_adjust=True) OHLC:")
    ac = [c for c in ["Open", "High", "Low", "Close"] if c in adj.columns]
    print(adj[ac].round(4).to_string())

    # adjustment factor on a pre-ex day
    if "Adj Close" in raw.columns and "Close" in raw.columns:
        f = (raw["Adj Close"] / raw["Close"]).round(6)
        print("\nf = AdjClose/Close per day:")
        print(f.to_string())

    print(f"\n--- DECISION on LIMIT {LIMIT} ---")
    if "Low" in raw.columns:
        rdip = raw.loc["2026-06-09":, "Low"]
        print(f"RAW min Low 6/9..: {rdip.min():.4f}  -> traded <= {LIMIT}? {bool((rdip <= LIMIT).any())}")
    if "Low" in adj.columns:
        adip = adj.loc["2026-06-09":, "Low"]
        print(f"ADJ(now) min Low 6/9..: {adip.min():.4f}  -> dips <= {LIMIT}? {bool((adip <= LIMIT).any())}")
except Exception as e:
    print("YFINANCE FAILED:", repr(e))
