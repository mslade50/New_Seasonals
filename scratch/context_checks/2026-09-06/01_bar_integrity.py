"""Roll-seam / bad-bar screen for the commodity and metal subjects that fired.

The 2026-09-02 brief excluded grains, coffee and cotton as continuous-contract
roll seams. Re-check on the 09-04 bar before any of them can carry a nugget.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import load_prices  # noqa

TK = ["KC=F","ZC=F","ZS=F","ZW=F","CT=F","CC=F","SB=F","SI=F","GC=F","HG=F",
      "CL=F","NG=F","PL=F","PA=F","HE=F","LE=F"]
px = load_prices(TK)
print(f"{'tk':8} {'date':11} {'open':>10} {'close':>10} {'prevclose':>10} "
      f"{'gap%':>8} {'intraday%':>9} {'sess%':>8} {'vol':>12} {'prevvol':>12}")
for t in TK:
    df = px.get(t)
    if df is None or df.empty:
        print(f"{t:8} MISSING"); continue
    d = df.tail(2)
    if len(d) < 2:
        print(f"{t:8} short"); continue
    prev, cur = d.iloc[0], d.iloc[1]
    gap = (cur["Open"]/prev["Close"] - 1) * 100
    intra = (cur["Close"]/cur["Open"] - 1) * 100
    sess = (cur["Close"]/prev["Close"] - 1) * 100
    print(f"{t:8} {str(d.index[1].date()):11} {cur['Open']:>10.3f} {cur['Close']:>10.3f} "
          f"{prev['Close']:>10.3f} {gap:>8.2f} {intra:>9.2f} {sess:>8.2f} "
          f"{cur.get('Volume', float('nan')):>12,.0f} {prev.get('Volume', float('nan')):>12,.0f}")
