import pandas as pd

PATH = r"C:\Users\McKinley Slade\dev\New_Seasonals\data\master_prices.parquet"
m = pd.read_parquet(PATH)
m["date"] = pd.to_datetime(m["date"])
last_dates = m.groupby("ticker")["date"].max()

# Use the parquet's own max as "today" so the result is meaningful against this
# (stale, Jun-10) snapshot rather than the real wall-clock date.
today = last_dates.max().normalize()
buffer_days = 5
max_lookback = 120

earliest_stale = last_dates.min()
old_start = (earliest_stale - pd.Timedelta(days=buffer_days))
floor = today - pd.Timedelta(days=max_lookback)
new_start = max(earliest_stale - pd.Timedelta(days=buffer_days), floor)
stale = last_dates[last_dates < floor]

print(f"today (parquet max): {today.date()}")
print(f"earliest_stale:      {earliest_stale.date()}")
print(f"OLD fetch_start:     {old_start.date()}   span ~ {(today - old_start).days} days")
print(f"NEW fetch_start:     {new_start.date()}   span ~ {(today - new_start).days} days")
print(f"stale > {max_lookback}d (excluded from deep refresh): {len(stale)}")
print(stale.sort_values().to_string())
