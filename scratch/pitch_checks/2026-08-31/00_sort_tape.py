"""Sort the whole tape for the surface map. Not a check, an enumeration."""
import json
from pathlib import Path

T = json.load(open(Path(__file__).resolve().parents[3] / "data" / "pitch_tape.json"))["tickers"]
rows = [dict(tk=k, **v) for k, v in T.items()]

def show(key, n=14, rev=True, fmt="{:.2f}"):
    print(f"\n===== {key} {'HIGH' if rev else 'LOW'} =====")
    s = sorted([r for r in rows if r.get(key) is not None], key=lambda r: r[key], reverse=rev)[:n]
    for r in s:
        print(f"  {r['tk']:<12} {key}={fmt.format(r[key]):>9}  r5={r['rank_5d']:>5.1f} r21={r['rank_21d']:>5.1f} r63={r['rank_63d']:>5.1f} z10={r['z10']:>6.2f} 52wh={r['dist_52w_high_pct']:>7.2f} 52wl={r['dist_52w_low_pct']:>8.2f} 200d={r['dist_sma200_pct']:>7.2f}")

for k in ["dist_52w_high_pct", "dist_52w_low_pct", "dist_sma200_pct", "z10", "rank_5d", "rank_21d", "rank_63d", "ret_1d", "ret_5d", "ret_21d", "ret_63d", "atr_pct", "vol_vs_63d"]:
    show(k, rev=True)
    show(k, rev=False)
