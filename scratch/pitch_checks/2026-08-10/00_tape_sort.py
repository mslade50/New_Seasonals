"""Sort the whole tape so the survey sees every name, not the ones I walked in with."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
tape = json.loads((ROOT / "data" / "pitch_tape.json").read_text())
rows = tape["tickers"] if isinstance(tape, dict) and "tickers" in tape else tape
if isinstance(rows, dict):
    rows = [dict(v, ticker=k) for k, v in rows.items()]

print(f"n={len(rows)}  keys={sorted(rows[0].keys())}\n")


def show(field, n=15, rev=True, label=None):
    ok = [r for r in rows if r.get(field) is not None]
    ok.sort(key=lambda r: r[field], reverse=rev)
    print(f"--- {label or field} {'high' if rev else 'low'} ---")
    for r in ok[:n]:
        print(f"  {r['ticker']:<10} {r[field]:>8.2f}   r5={r.get('rank_5d')} r21={r.get('rank_21d')} "
              f"r63={r.get('rank_63d')} z10={r.get('z10')} d52h={r.get('dist_52w_high_pct')} "
              f"d200={r.get('dist_sma200_pct')}")
    print()


for f in ["ret_5d", "ret_21d", "ret_63d", "z10", "dist_52w_high_pct", "dist_sma200_pct", "rank_63d"]:
    show(f, rev=True)
    show(f, rev=False)
