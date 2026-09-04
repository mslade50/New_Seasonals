from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\Users\McKinley Slade\dev\New_Seasonals")
led = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
tot = led["PnL_flat_750k"].sum()
NARROW = {"GLD", "SLV", "USO", "UNG", "DBC", "DBA", "DBO", "UGA", "CORN", "WEAT",
          "SOYB", "CPER", "PALL", "PPLT",
          "TLT", "IEF", "SHY", "AGG", "BND", "LQD", "HYG", "JNK", "TBT", "TIP",
          "EMB", "MUB", "ZROZ", "EDV", "TMF", "TMV",
          "UUP", "UDN", "FXE", "FXY", "FXB", "FXA", "FXC", "FXF"}
m = led["Ticker"].isin(NARROW)
print("narrow tickers present:", sorted(led.loc[m, "Ticker"].unique()))
print(f"narrow CBFX share: {led.loc[m,'PnL_flat_750k'].sum()/tot:.2%} "
      f"(${led.loc[m,'PnL_flat_750k'].sum():,.0f})")
# per-ticker
print(led.loc[m].groupby("Ticker")["PnL_flat_750k"].sum().sort_values().to_string())
