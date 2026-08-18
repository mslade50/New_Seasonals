"""C1 round 5. The anchor did not fire on 2026-08-17 in a1e. Check the actual
US trading calendar: how many sessions is the 2026-08-31 close after the
2026-08-18 entry close? The brief says 8. Then re-price the headline cells at
the CORRECT horizon.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np, pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

cal = CustomBusinessDay(calendar=USFederalHolidayCalendar())
sess = pd.date_range("2026-08-17", "2026-09-02", freq=cal)
print("US sessions 08-17..09-02:", [str(d.date()) for d in sess])
entry = pd.Timestamp("2026-08-18"); me = pd.Timestamp("2026-08-31")
n = int(((sess > entry) & (sess <= me)).sum())
print("sessions from the %s ENTRY close to the %s month-end close = %d  (brief said 8)"
      % (entry.date(), me.date(), n))
aug = pd.date_range("2026-08-01", "2026-08-31", freq=cal)
print("August 2026 sessions: %d | tdom of 2026-08-17 = %d | tdom of 2026-08-18 = %d"
      % (len(aug), list(aug).index(pd.Timestamp("2026-08-17"))+1,
         list(aug).index(pd.Timestamp("2026-08-18"))+1))

# --- re-price the headline at the CORRECT h
LAG = 1
raw = load_prices(["SPY", "TLT"])
spy, tlt = raw["SPY"]["Close"], raw["TLT"]["Close"]
idx = spy.index.intersection(tlt.index)
px = pd.DataFrame({"SPY": spy.reindex(idx), "TLT": tlt.reindex(idx)}).dropna()
idx = px.index
div = (spy.pct_change(21).reindex(idx) - tlt.pct_change(21).reindex(idx))
div_rk = div.rolling(1260).rank(pct=True)*100
T21 = tlt.pct_change(21).reindex(idx)
ymv = pd.Series(idx.year*100+idx.month, index=idx)
is_last = ymv.ne(ymv.shift(-1)); is_last.iloc[-1] = False
pos = pd.Series(range(len(idx)), index=idx)

def anchor(h):
    t = pos + LAG + h
    m = pd.Series(False, index=idx); ok = t < len(idx)
    m.loc[idx[ok.values]] = is_last.values[t[ok.values].values]
    return m

for h in (8, n):
    r_tlt = fwd_lag(px["TLT"], h, LAG); r_spy = fwd_lag(px["SPY"], h, LAG)
    sp = r_tlt - r_spy; v = sp.notna(); A = anchor(h)
    rows = []
    for lab, g in (("gate OFF", pd.Series(True, index=idx)),
                   ("div >= +5pp", (div >= 0.05).fillna(False)),
                   ("div >= +7.32pp (LIVE)", (div >= 0.0732).fillna(False)),
                   ("div rank1260 >= 88 (LIVE)", (div_rk >= 88).fillna(False)),
                   ("TLT21d <= -2.5%", (T21 <= -0.025).fillna(False))):
        d = idx[(A & g).values & v.values]
        rr = summarize(sp.loc[d].values, f"SPREAD {lab}")
        rr["TLTonly_pct"] = round(100*r_tlt.loc[d].mean(), 3)
        rows.append(rr)
    show(rows, f"h={h} sessions to the month-end close  {'<-- BRIEF ASSUMED' if h==8 else '<-- ACTUAL 2026-08-18 -> 2026-08-31'}")
