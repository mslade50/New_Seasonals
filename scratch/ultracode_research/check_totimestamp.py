import pandas as pd
pi = pd.period_range("2020-01", periods=3, freq="M")
print(pi.to_timestamp("M"))
print(pi.to_timestamp())
