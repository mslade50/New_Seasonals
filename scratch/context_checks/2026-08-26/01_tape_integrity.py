import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import pandas as pd

m = pd.read_parquet("data/master_prices.parquet")
m["date"] = pd.to_datetime(m["date"])

loud = ["KC=F", "ZW=F", "ZC=F", "ZS=F", "NG=F", "LE=F", "^MOVE", "SB=F", "CT=F", "HE=F"]
for t in loud:
    d = m[m["ticker"] == t].sort_values("date").tail(6)
    if d.empty:
        print(f"{t}: absent")
        continue
    print(f"\n=== {t} ===")
    prev_c = None
    for _, r in d.iterrows():
        o, h, l, c = r["Open"], r["High"], r["Low"], r["Close"]
        gap = (o / prev_c - 1) * 100 if prev_c else float("nan")
        sess = (c / o - 1) * 100 if o else float("nan")
        tot = (c / prev_c - 1) * 100 if prev_c else float("nan")
        flag = ""
        if o == 0 or pd.isna(o):
            flag += " OPEN=0"
        if not pd.isna(c) and not pd.isna(h) and c > h * 1.0001:
            flag += " CLOSE>HIGH"
        if not pd.isna(c) and not pd.isna(l) and c < l * 0.9999:
            flag += " CLOSE<LOW"
        print(f"  {r['date'].date()} O{o:10.3f} H{h:10.3f} L{l:10.3f} C{c:10.3f} "
              f"gap{gap:7.2f}% intraday{sess:7.2f}% total{tot:7.2f}%{flag}")
        prev_c = c
