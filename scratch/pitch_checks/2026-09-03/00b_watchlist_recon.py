"""Watchlist verdict numbers for 2026-09-03."""
import sys, warnings, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import load_prices
warnings.filterwarnings("ignore")

tnx = load_prices(["^TNX"])["^TNX"]["Close"].dropna()
print("TNX now %.3f  252 sessions ago %.3f  change %+0.1f bp" %
      (tnx.iloc[-1], tnx.iloc[-253], 100*(tnx.iloc[-1]-tnx.iloc[-253])))

tape = json.load(open("data/pitch_tape.json"))["tickers"]
hold = [(k, v["rank_21d"], v["rank_63d"], v["rank_5d"]) for k, v in tape.items()
        if v.get("rank_21d") is not None and v["rank_21d"] >= 90 and v["rank_63d"] <= 10]
print("watchlist28 holders (r21>=90 & r63<=10):", hold or "NONE")

# watchlist 34 / C4: which names hold the triple rank floor (5/21/63 all low)?
tri = [(k, v["rank_5d"], v["rank_21d"], v["rank_63d"]) for k, v in tape.items()
       if v.get("rank_5d") is not None and v["rank_5d"] <= 10 and v["rank_21d"] <= 10 and v["rank_63d"] <= 20]
print("\ntriple-floor names (r5<=10, r21<=10, r63<=20):")
for t in sorted(tri, key=lambda x: x[1]):
    print("   %-6s r5 %5.1f r21 %5.1f r63 %5.1f" % t)

# XLV - XLK one day gap for watchlist 14
print("\nXLV 1d %+0.2f  XLK 1d %+0.2f  gap %+0.2fpp" %
      (tape["XLV"]["ret_1d"], tape["XLK"]["ret_1d"], tape["XLV"]["ret_1d"]-tape["XLK"]["ret_1d"]))
for t in ["PCG","EIX","XLU","CMS","PEG","AEP","D","SO","DUK","ED","EXC"]:
    v = tape.get(t)
    if v: print("%-5s 1d %+6.2f 5d %+7.2f 21d %+7.2f r5 %5.1f 52wh %+7.2f 200d %+7.2f" %
                (t, v["ret_1d"], v["ret_5d"], v["ret_21d"], v["rank_5d"], v["dist_52w_high_pct"], v["dist_sma200_pct"]))
