"""Engine: same trading-day-of-year as today, next session, midterm years: SPY/QQQ/IWM/^GSPC 6 of 6 down.
Which years and dates, how big, is it the Thursday after September quad witching, and does it survive
the post-expiry-week theme already told (Thu 9/17, Sun 9/20, Tue 9/22)?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from seasonal_edge import seasonal_window_returns  # noqa

prices = load_prices(["SPY", "QQQ", "IWM", "^GSPC", "TLT", "^TNX"])
asof = pd.Timestamp("2026-09-23")
for t in ("^GSPC", "SPY", "IWM"):
    for filt in (None, 2):
        st = seasonal_window_returns(prices[t], asof, 1, cycle_phase_filter=filt)
        yrs = st["years"]
        print(t, "phase", filt, "n", st["n"], "mean", round(100 * st["mean"], 3), "down", st["n_down"],
              list(zip(yrs, [round(100 * x, 2) for x in st["rets"]]))[-30:])

# which calendar dates were the h1 sessions in the midterm years, and what weekday/position vs quad witching
gspc = prices["^GSPC"]["Close"].dropna()
idx = gspc.index
ret = gspc.pct_change()
def third_friday(y, m):
    d = pd.Timestamp(y, m, 15)
    while d.weekday() != 4:
        d += pd.Timedelta(days=1)
    return d
print("\nThursday after September quad witching (quad = 3rd Fri Sep; Thursday = 4 sessions later), ^GSPC close-to-close:")
rows = []
for y in range(1999, 2026):
    q = third_friday(y, 9)
    qi = idx.searchsorted(q)
    if qi >= len(idx) or idx[qi] != q:
        # holiday-shifted expiry
        qi = idx.searchsorted(q) - 1
    for k in (4,):
        if qi + k < len(idx):
            d = idx[qi + k]
            rows.append({"year": y, "date": d.date(), "wd": d.day_name()[:3], "k": k, "ret": ret.iloc[qi + k],
                         "wk5": gspc.iloc[qi + 5] / gspc.iloc[qi] - 1, "mid": y % 4 == 2})
df = pd.DataFrame(rows)
print(df.assign(ret=lambda x: (100 * x.ret).round(2), wk5=lambda x: (100 * x.wk5).round(2)).to_string(index=False))
v = df["ret"].values
s = summarize(v, "Thu after Sept quad, all")
print(s, "down", int((v < 0).sum()), "of", len(v), "sign p", sign_test(int((v < 0).sum()), len(v)))
for part in era_split(pd.DatetimeIndex(df["date"]), v):
    print(part)
m = df["mid"]
print("midterm", df[m][["year", "date", "ret"]].assign(ret=lambda x: (100 * x.ret).round(2)).to_string(index=False))
# control: all Thursdays in September, and all Thursdays
thu = idx[(idx.weekday == 3)]
thu_sep = thu[thu.month == 9]
print("all Thursdays:", summarize(ret.loc[thu].values, "thu"))
print("Sept Thursdays:", summarize(ret.loc[thu_sep].values, "sept thu"))
# midterm-year h1 at this doy, removing the 2 worst
mid = df[m]["ret"].values
print("midterm mean ex worst two:", round(100 * np.sort(mid)[2:].mean(), 3))
# the whole 5-session post-expiry window so far this year
q26 = pd.Timestamp("2026-09-18")
qi = idx.get_loc(q26)
print("\n2026 post-expiry so far: S&P", round(100 * (gspc.iloc[-1] / gspc.iloc[qi] - 1), 2))
iwm, spy = prices["IWM"]["Close"], prices["SPY"]["Close"]
print("IWM", round(100 * (iwm.iloc[-1] / iwm.loc[q26] - 1), 2), "SPY", round(100 * (spy.iloc[-1] / spy.loc[q26] - 1), 2))
