from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet", columns=["ticker", "date", "Close"])
last = mp.groupby("ticker")["date"].max()
cutoff = pd.Timestamp("2026-06-01")
dead = last[last < cutoff]
print("tickers total:", len(last), "| last-date < 2026-06-01 (possible delisted):", len(dead))
print(dead.sort_values().tail(20).to_string())

# RSX sanity
rsx = mp[mp["ticker"] == "RSX"].set_index("date")["Close"]
print("\nRSX 2022:", rsx.loc["2022-01":"2022-12"].resample("ME").last().to_string())
print("RSX tail:", rsx.tail(5).to_string())

from strategy_config import LIQUID_PLUS_COMMODITIES, CSV_UNIVERSE
print("\nLIQUID_PLUS_COMMODITIES:", len(LIQUID_PLUS_COMMODITIES))
print("CSV_UNIVERSE:", len(CSV_UNIVERSE))
liq = [t for t in LIQUID_PLUS_COMMODITIES if t in set(last.index)]
print("liquid in master_prices:", len(liq))
# first dates for liquid universe: how far back does it go
firsts = mp[mp["ticker"].isin(liq)].groupby("ticker")["date"].min()
print("liquid first-date distribution:")
print(firsts.dt.year.value_counts().sort_index().to_string())
