"""Independent scale-invariance + phantom proof.
Uses live raw and adjusted bars to reproduce the 4 cases the synthesis claims."""
import yfinance as yf
import pandas as pd

START = "2026-05-01"
END = "2026-06-17"

raw = yf.download("EWZ", start=START, end=END, auto_adjust=False, progress=False)
adj = yf.download("EWZ", start=START, end=END, auto_adjust=True, progress=False)
for d in (raw, adj):
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)

def atr_sma14(df):
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(14).mean()

raw["ATR"] = atr_sma14(raw)
adj["ATR"] = atr_sma14(adj)

sd = "2026-06-08"
fwd = ["2026-06-09", "2026-06-10"]
OFF = 0.25

c_raw = raw.loc[sd, "Close"]; a_raw = raw.loc[sd, "ATR"]
c_adj = adj.loc[sd, "Close"]; a_adj = adj.loc[sd, "ATR"]
lim_raw = round(c_raw - OFF * a_raw, 2)
lim_adj = round(c_adj - OFF * a_adj, 2)
print(f"RAW  6/8 close={c_raw:.4f} ATR={a_raw:.4f} -> limit_raw={lim_raw:.2f}")
print(f"ADJ  6/8 close={c_adj:.4f} ATR={a_adj:.4f} -> limit_adj={lim_adj:.2f}")

def fill(limit, df):
    for d in fwd:
        lo = df.loc[d, "Low"]
        if lo <= limit:
            return d, lo, limit
    return None

print("\n[A] RAW self-consistent (limit_raw vs raw lows):", fill(lim_raw, raw))
print("[B] ADJ self-consistent (limit_adj vs adj lows):", fill(lim_adj, adj))
print("[C] FROZEN 33.51 vs ADJ lows (THE PHANTOM):      ", fill(33.51, adj))
print("[D] FROZEN 33.51 vs RAW lows (live truth):       ", fill(33.51, raw))
