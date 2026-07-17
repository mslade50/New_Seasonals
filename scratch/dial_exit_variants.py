"""Price-exit variants for the dial-gated SPY sleeve (T=20 frozen).

Entry (all variants): close within 5% of 252d high AND dial 10d-MA < 20,
executed next open. Dial exit (all variants except A): 10d-MA >= 25.
Price-exit leg varies:
  A  no-hysteresis reference (dial exit >= 20, single close < 95% band)
  B  FROZEN SPEC: dial hysteresis 20/25, single close outside band exits
  C  B but price exit needs 2 CONSECUTIVE closes outside the band
  D  B but first close outside band only ARMS at that candle's low;
     exit on a later CLOSE below the armed low; disarm on close back in band
  E  B but armed low becomes an intraday STOP from the next session
     (fill at stop, or at open if gapped through — book stop-fill convention);
     disarm on close back in band
Execution: next-open MOO, 2 bps/side both ways; intraday stop fills charged
the same 2 bps. In-sample variant selection — results amend the prereg,
they do not validate it.
"""
import os

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLIP = 2 / 1e4

frag = pd.read_parquet(os.path.join(_ROOT, "data", "rd2_fragility.parquet"))
s63 = frag["63d"].dropna().sort_index()
ma10 = s63.rolling(10, min_periods=1).mean()

mp = pd.read_parquet(os.path.join(_ROOT, "data", "master_prices.parquet"),
                     filters=[("ticker", "in", ["SPY", "^IRX"])])
mp["date"] = pd.to_datetime(mp["date"])
px = (mp[mp["ticker"] == "SPY"].set_index("date")[["Open", "High", "Low", "Close"]]
      .sort_index().reindex(s63.index).ffill())
irx = (mp[mp["ticker"] == "^IRX"].set_index("date")["Close"]
       .sort_index().reindex(s63.index).ffill().fillna(4.0))
tbill = (irx / 100.0) / 252.0

close, open_, low = px["Close"], px["Open"], px["Low"]
hi252 = close.rolling(252, min_periods=60).max()
near5 = (close / hi252 - 1) >= -0.05
n = len(px)
YEARS = (px.index[-1] - px.index[0]).days / 365.25

DIAL_IN, DIAL_OUT = 20.0, 25.0


def run(variant: str):
    """Event loop. Signals read at close t; MOO actions at open t+1.
    Variant E's stop can fill intraday on day t itself."""
    r = np.zeros(n)
    in_pos = False
    pending = None            # 'buy' | 'sell' staged for next open
    armed_low = None
    entry_switches = exit_switches = 0
    dial_exit_lvl = DIAL_IN if variant == "A" else DIAL_OUT

    for t in range(1, n):
        # --- execute pending order at today's open, then accrue today's pnl
        stopped_today = False
        if pending == "buy":
            in_pos = True
            armed_low = None
            entry_switches += 1
            r[t] = open_.iloc[t + 1] / open_.iloc[t] - 1 if t + 1 < n else 0.0
            r[t] -= SLIP
        elif pending == "sell":
            in_pos = False
            armed_low = None
            exit_switches += 1
            r[t] = tbill.iloc[t] - SLIP
        elif in_pos:
            # intraday stop (variant E) — active only when armed
            if variant == "E" and armed_low is not None:
                if open_.iloc[t] < armed_low:
                    fill = open_.iloc[t]
                    stopped_today = True
                elif low.iloc[t] <= armed_low:
                    fill = armed_low
                    stopped_today = True
            if stopped_today:
                r[t] = fill / open_.iloc[t] - 1 - SLIP
                in_pos = False
                armed_low = None
                exit_switches += 1
            else:
                r[t] = open_.iloc[t + 1] / open_.iloc[t] - 1 if t + 1 < n else 0.0
        else:
            r[t] = tbill.iloc[t]
        pending = None

        # --- read signals at close t, stage next-open action
        dial = ma10.iloc[t]
        nh = bool(near5.iloc[t])
        if in_pos:
            exit_now = dial >= dial_exit_lvl
            if variant in ("A", "B"):
                exit_now = exit_now or not nh
            elif variant == "C":
                exit_now = exit_now or (not nh and not bool(near5.iloc[t - 1]))
            elif variant in ("D", "E"):
                if nh:
                    armed_low = None
                elif armed_low is None:
                    armed_low = low.iloc[t]      # trigger candle arms its low
                elif variant == "D" and close.iloc[t] < armed_low:
                    exit_now = True
            if exit_now:
                pending = "sell"
        else:
            if nh and dial < DIAL_IN:
                pending = "buy"

    r = pd.Series(r, index=px.index)
    pos_days = r.index[(r != tbill.reindex(r.index)) & (r != 0)]  # approx
    eq = (1 + r).cumprod()
    ex = r - tbill
    # in-market mask: rebuild from returns is fragile; track via simulation
    return r, eq, entry_switches, exit_switches


pd.set_option("display.width", 150)
rows = []
for v, label in [("A", "no hysteresis, single close"),
                 ("B", "FROZEN: hysteresis + single close"),
                 ("C", "hysteresis + 2 consec closes"),
                 ("D", "hysteresis + close < trigger low"),
                 ("E", "hysteresis + intraday stop at trigger low")]:
    r, eq, n_in, n_out = run(v)
    ex = r - tbill
    cagr = eq.iloc[-1] ** (1 / YEARS) - 1
    sharpe = ex.mean() / r.std() * np.sqrt(252)
    dd = (eq / eq.cummax() - 1).min()
    # in-market fraction: days where return != tbill (holds except zero-return coincidences)
    in_mkt = (r != tbill).mean()
    inmkt_sharpe = (ex[r != tbill].mean() / ex[r != tbill].std() * np.sqrt(252)
                    if (r != tbill).sum() > 20 else np.nan)
    rows.append({"variant": v, "exit": label,
                 "CAGR%": round(cagr * 100, 2),
                 "Sharpe": round(sharpe, 2),
                 "inmkt_Sharpe": round(inmkt_sharpe, 2),
                 "maxDD%": round(dd * 100, 1),
                 "in_mkt%": round(in_mkt * 100),
                 "roundtrips/yr": round(n_out / YEARS, 1)})
print(pd.DataFrame(rows).set_index("variant").to_string())
