"""Try a live yfinance pull: EWZ dividends + adjusted vs raw close to confirm
the back-adjustment factor and ex-div date. Degrade gracefully on no network."""
import sys
try:
    import yfinance as yf
    import pandas as pd
    pd.set_option("display.width", 200)

    tk = yf.Ticker("EWZ")
    print("=== EWZ dividends (last 6) ===")
    try:
        div = tk.dividends
        print(div.tail(6))
    except Exception as e:
        print("dividends failed:", e)

    print("\n=== RAW (auto_adjust=False) 2026-05-26..06-16 ===")
    raw = yf.download("EWZ", start="2026-05-26", end="2026-06-17",
                      auto_adjust=False, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    print(raw[["Open", "High", "Low", "Close", "Adj Close"]].round(4) if "Adj Close" in raw.columns
          else raw.round(4))

    print("\n=== ADJUSTED (auto_adjust=True) same window ===")
    adj = yf.download("EWZ", start="2026-05-26", end="2026-06-17",
                      auto_adjust=True, progress=False)
    if isinstance(adj.columns, pd.MultiIndex):
        adj.columns = adj.columns.get_level_values(0)
    print(adj[["Open", "High", "Low", "Close"]].round(4))

    # implied factor on the 6/9 close (raw close vs adjusted close)
    try:
        idx = pd.Timestamp("2026-06-09")
        rclose = float(raw.loc[idx, "Close"])
        aclose = float(adj.loc[idx, "Close"])
        print(f"\n6/9 raw close={rclose:.4f}  adj close={aclose:.4f}  factor f={aclose/rclose:.6f}")
        rlow = float(raw.loc[idx, "Low"]); alow = float(adj.loc[idx, "Low"])
        print(f"6/9 raw low={rlow:.4f}  adj low={alow:.4f}")
        print(f"6/10 raw low={float(raw.loc[pd.Timestamp('2026-06-10'),'Low']):.4f}  "
              f"adj low={float(adj.loc[pd.Timestamp('2026-06-10'),'Low']):.4f}")
        print(f"\nLive staged limit was 33.51 (a FIXED number from the pre-ex basis).")
        print(f"Would 33.51 'fill' against the ADJUSTED 6/10 low {float(adj.loc[pd.Timestamp('2026-06-10'),'Low']):.4f}? "
              f"{float(adj.loc[pd.Timestamp('2026-06-10'),'Low']) <= 33.51}")
        print(f"Did it fill against the RAW/live 6/10 low {float(raw.loc[pd.Timestamp('2026-06-10'),'Low']):.4f}? "
              f"{float(raw.loc[pd.Timestamp('2026-06-10'),'Low']) <= 33.51}")
    except Exception as e:
        print("factor calc failed:", e)

except Exception as e:
    print("NO NETWORK / yfinance unavailable:", repr(e))
    sys.exit(0)
