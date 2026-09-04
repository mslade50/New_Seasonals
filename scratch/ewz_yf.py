"""Pull EWZ from yfinance: raw (auto_adjust=False) + adjusted + dividends.
Degrade gracefully if network fails."""
import sys
import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

try:
    import yfinance as yf
except Exception as e:
    print("yfinance import failed:", e); sys.exit(0)

START = "2026-05-15"
END = "2026-06-18"

def show(label, df):
    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)
    if df is None or df.empty:
        print("  (empty)"); return
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs("EWZ", level=1, axis=1)
        except Exception:
            df.columns = df.columns.get_level_values(0)
    cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    print(df[cols].round(4).to_string())

try:
    raw = yf.download("EWZ", start=START, end=END, auto_adjust=False, progress=False)
    show("RAW (auto_adjust=False) — actual traded prices", raw)
except Exception as e:
    print("raw download failed:", e)

try:
    adj = yf.download("EWZ", start=START, end=END, auto_adjust=True, progress=False)
    show("ADJUSTED (auto_adjust=True) — what the cache/backtester store", adj)
except Exception as e:
    print("adj download failed:", e)

try:
    tk = yf.Ticker("EWZ")
    divs = tk.dividends
    if divs is not None and len(divs):
        recent = divs[divs.index >= "2026-01-01"]
        print("\n" + "=" * 70)
        print("DIVIDENDS 2026+ (ex-date -> amount)")
        print("=" * 70)
        print(recent.to_string())
except Exception as e:
    print("dividends fetch failed:", e)
