"""Live yfinance pull to recover EWZ raw vs adjusted lows and the actual
ex-div date/factor. This is the load-bearing check for the phantom theory:
does the ADJUSTED low dip below 33.51 while the RAW low stayed above?"""
import yfinance as yf
import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 60)

START = "2026-06-01"
END = "2026-06-17"

print("=== EWZ dividends (yfinance) ===")
try:
    tk = yf.Ticker("EWZ")
    divs = tk.dividends
    print(divs.tail(8))
except Exception as e:
    print("dividends pull failed:", e)

print("\n=== RAW (auto_adjust=False) ===")
try:
    raw = yf.download("EWZ", start=START, end=END, auto_adjust=False, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    print(raw[["Open", "High", "Low", "Close", "Adj Close"]].to_string())
except Exception as e:
    print("raw pull failed:", e)
    raw = None

print("\n=== ADJUSTED (auto_adjust=True) ===")
try:
    adj = yf.download("EWZ", start=START, end=END, auto_adjust=True, progress=False)
    if isinstance(adj.columns, pd.MultiIndex):
        adj.columns = adj.columns.get_level_values(0)
    print(adj[["Open", "High", "Low", "Close"]].to_string())
except Exception as e:
    print("adj pull failed:", e)
    adj = None

LIMIT = 33.51
print(f"\n=== Phantom check vs limit {LIMIT} ===")
if raw is not None and adj is not None:
    for d in ["2026-06-09", "2026-06-10"]:
        try:
            rl = raw.loc[d, "Low"]
            al = adj.loc[d, "Low"]
            print(f"{d}: RAW low={rl:.4f} (<= {LIMIT}? {rl <= LIMIT})   ADJ low={al:.4f} (<= {LIMIT}? {al <= LIMIT})")
        except KeyError:
            print(f"{d}: not in data")
    # factor from a pre-ex day
    try:
        d0 = "2026-06-09"
        f = adj.loc[d0, "Close"] / raw.loc[d0, "Close"]
        print(f"\nadjustment factor f (adj close / raw close on {d0}) = {f:.6f}")
        print(f"f * limit_raw(33.51) = {f*LIMIT:.4f}  <- self-consistent adj limit")
    except Exception as e:
        print("factor calc failed:", e)
