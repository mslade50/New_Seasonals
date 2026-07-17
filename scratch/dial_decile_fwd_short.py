"""SPY forward 2/5/10/21d returns by fragility-dial decile (10d MA of 63d).
Companion to dial_decile_fwd.py. Same vintage caveat (pre-2026-07-02 =
recompute)."""
import os

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

frag = pd.read_parquet(os.path.join(_ROOT, "data", "rd2_fragility.parquet"))
s63 = frag["63d"].dropna().sort_index()
ma10 = s63.rolling(10, min_periods=1).mean()

mp = pd.read_parquet(os.path.join(_ROOT, "data", "master_prices.parquet"),
                     filters=[("ticker", "==", "SPY")])
spy = (mp.assign(date=pd.to_datetime(mp["date"]))
         .set_index("date")["Close"].sort_index())
spy = spy.reindex(s63.index).ffill()

HORIZONS = [2, 5, 10, 21]
fwd = {h: spy.shift(-h) / spy - 1 for h in HORIZONS}

df = pd.DataFrame({"score": ma10, **{f"f{h}": fwd[h] for h in HORIZONS}}).dropna()
df["decile"] = pd.qcut(df["score"], 10, labels=False, duplicates="drop")

pd.set_option("display.width", 170)
rows = []
for d, g in df.groupby("decile"):
    row = {"decile": d, "n": len(g),
           "lo": round(g["score"].min(), 1), "hi": round(g["score"].max(), 1)}
    for h in HORIZONS:
        row[f"f{h}_mean"] = round(g[f"f{h}"].mean() * 100, 2)
        row[f"f{h}_med"] = round(g[f"f{h}"].median() * 100, 2)
        row[f"f{h}_neg%"] = round((g[f"f{h}"] < 0).mean() * 100, 0)
    rows.append(row)
print(pd.DataFrame(rows).set_index("decile").to_string())

print("\nspearman(score, fwd) per horizon:")
for h in HORIZONS:
    print(f"  {h:>2}d: {df['score'].corr(df[f'f{h}'], method='spearman'):+.3f}")

# annualized-ish per-day drag: mean daily fwd return by above/below the break
print("\nabove/below the ~27 break (score threshold from the 63d study):")
for h in HORIZONS:
    lo = df.loc[df["score"] < 27, f"f{h}"]
    hi = df.loc[df["score"] >= 27, f"f{h}"]
    print(f"  {h:>2}d: below +{lo.mean()*100:.2f}% (med {lo.median()*100:+.2f}%, "
          f"{(lo<0).mean()*100:.0f}% neg, N={len(lo)}) | "
          f"above {hi.mean()*100:+.2f}% (med {hi.median()*100:+.2f}%, "
          f"{(hi<0).mean()*100:.0f}% neg, N={len(hi)})")
