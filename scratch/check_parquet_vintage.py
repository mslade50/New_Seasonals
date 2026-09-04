import pandas as pd

PATH = r"C:\Users\McKinley Slade\dev\New_Seasonals\data\master_prices.parquet"
m = pd.read_parquet(PATH)
m["date"] = pd.to_datetime(m["date"])

last_dates = m.groupby("ticker")["date"].max()
earliest_stale = last_dates.min()
overall_max = last_dates.max()

print(f"rows: {len(m):,}  tickers: {m['ticker'].nunique()}")
print(f"overall max date:   {overall_max.date()}")
print(f"earliest_stale:     {earliest_stale.date()}  (min of per-ticker max)")
print(f"fetch_start (buf5): {(earliest_stale - pd.Timedelta(days=5)).date()}")
print()
# How many tickers are 'current' vs lagging?
cur = (last_dates == overall_max).sum()
print(f"tickers at overall max: {cur} / {len(last_dates)}")
print("oldest 8 per-ticker max dates:")
print(last_dates.sort_values().head(8).to_string())
print()
# EWZ bars around the 6/8-6/12 window
ewz = m[m["ticker"].str.upper() == "EWZ"].set_index("date").sort_index()
win = ewz.loc["2026-06-05":"2026-06-16", ["Open", "High", "Low", "Close"]]
print("EWZ stored bars 6/5-6/16:")
print(win.to_string())
