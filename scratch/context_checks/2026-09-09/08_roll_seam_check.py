"""Are tonight's commodity moves prices or continuous-contract roll seams?

2026-09-08's brief found coffee at -10.58% on 24,202 contracts against a 20-day
median of 180 and dropped every soft/metal on that basis. Tonight the same
complex is loud again, so the same test has to run before anything publishes.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import load_prices  # noqa: E402

TICKERS = ["KC=F", "SB=F", "CT=F", "CC=F", "ZC=F", "ZS=F", "ZW=F", "HG=F",
           "GC=F", "SI=F", "PL=F", "PA=F", "CL=F", "NG=F", "HE=F", "LE=F"]

px = load_prices(TICKERS)
rows = []
for t in TICKERS:
    df = px.get(t)
    if df is None or df.empty:
        continue
    df = df.dropna(subset=["Close"]).sort_index()
    for d in (pd.Timestamp("2026-09-08"), pd.Timestamp("2026-09-09")):
        if d not in df.index:
            continue
        i = df.index.get_loc(d)
        if i < 21:
            continue
        bar = df.iloc[i]
        prev_close = float(df["Close"].iloc[i - 1])
        vol = float(bar.get("Volume", float("nan")))
        med20 = float(df["Volume"].iloc[i - 20:i].median())
        ret = bar["Close"] / prev_close - 1.0
        # a genuine session trades through the gap; a roll seam opens away and
        # never revisits the prior close
        lo, hi = float(bar["Low"]), float(bar["High"])
        spans_prev = lo <= prev_close <= hi
        rows.append({
            "ticker": t,
            "date": d.date().isoformat(),
            "ret_pct": round(100 * ret, 2),
            "prev_close": round(prev_close, 4),
            "open": round(float(bar["Open"]), 4),
            "low": round(lo, 4),
            "high": round(hi, 4),
            "close": round(float(bar["Close"]), 4),
            "bar_spans_prev_close": spans_prev,
            "volume": int(vol) if vol == vol else None,
            "vol_med20": int(med20) if med20 == med20 else None,
            "vol_x_med": round(vol / med20, 1) if med20 else None,
        })

out = pd.DataFrame(rows).sort_values(["date", "ticker"])
pd.set_option("display.width", 220)
print(out.to_string(index=False))

print("\nVERDICT HEURISTIC: a volume multiple far off 1x together with a bar "
      "that never trades back through the prior close is a contract roll, not "
      "a session. Both conditions together are the kill.")
for _, r in out[out.date == "2026-09-09"].iterrows():
    flag = []
    if r.vol_x_med is not None and (r.vol_x_med > 5 or r.vol_x_med < 0.2):
        flag.append(f"volume {r.vol_x_med}x median")
    if not r.bar_spans_prev_close:
        flag.append("bar never touches prior close")
    print(f"  {r.ticker:<6} {r.ret_pct:+7.2f}%  "
          f"{'SEAM SUSPECT: ' + '; '.join(flag) if len(flag) == 2 else ('watch: ' + '; '.join(flag) if flag else 'clean')}")
