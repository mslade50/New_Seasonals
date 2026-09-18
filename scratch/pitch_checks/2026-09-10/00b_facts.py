"""Concrete tape facts for the surface map. Stage B1 input."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
tape = json.load(open(ROOT / "data" / "pitch_tape.json"))
rows = tape["tickers"] if isinstance(tape, dict) and "tickers" in tape else tape
if isinstance(rows, dict):
    rows = [dict(ticker=k, **v) for k, v in rows.items()]
by = {r["ticker"]: r for r in rows}

sec = ["XLB","XLC","XLE","XLF","XLI","XLK","XLP","XLRE","XLU","XLV","XLY",
       "SMH","XBI","KRE","IBB","OIH","XOP","ITA","IYR","VNQ","ITB","IHI","CEF","GDX","XME","XRT"]
print(f"{'tkr':7s} {'r5':>6s} {'r21':>6s} {'r63':>6s} {'z10':>6s} {'d52h':>7s} {'d52l':>7s} {'d200':>7s} {'21d%':>7s}")
for t in sec:
    r = by.get(t)
    if not r: print(t, "MISSING"); continue
    print(f"{t:7s} {r['rank_5d']:>6.1f} {r['rank_21d']:>6.1f} {r['rank_63d']:>6.1f} {r['z10']:>6.2f} {r['dist_52w_high_pct']:>7.2f} {r['dist_52w_low_pct']:>7.2f} {r['dist_sma200_pct']:>7.2f} {r['ret_21d']:>7.2f}")

print()
# fresh 52w lows / highs counts on the survivor tape
lows = [r["ticker"] for r in rows if r.get("dist_52w_low_pct") is not None and r["dist_52w_low_pct"] <= 0.01]
highs = [r["ticker"] for r in rows if r.get("dist_52w_high_pct") is not None and r["dist_52w_high_pct"] >= -0.01]
print("at 52w LOW (<=0.01%):", len(lows), lows)
print("at 52w HIGH (>=-0.01%):", len(highs), highs)
print()
n = len([r for r in rows if r.get("z10") is not None])
print("z10 <= -2:", len([r for r in rows if (r.get('z10') or 0) <= -2]), "/", n)
print("z10 >= +2:", len([r for r in rows if (r.get('z10') or 0) >= 2]), "/", n)
print("rank_21d <= 5:", len([r for r in rows if (r.get('rank_21d') if r.get('rank_21d') is not None else 50) <= 5]))
print("rank_21d >= 95:", len([r for r in rows if (r.get('rank_21d') if r.get('rank_21d') is not None else 50) >= 95]))
print()
for t in ["^VIX","^VIX3M","^SKEW","^MOVE","^TNX","SVXY","UVXY","TLT","IEF","LQD","HYG","GLD","SLV","GDX","USO","DBC","UUP","DX-Y.NYB","EWZ","EWJ","FXI","EEM","EFA","VNQ","SPY","QQQ","IWM"]:
    r = by.get(t)
    if r: print(f"{t:10s} close={r['close']:>10.3f} 5d%={r['ret_5d']:>7.2f} 21d%={r['ret_21d']:>7.2f} r5={r['rank_5d']:>5.1f} r21={r['rank_21d']:>5.1f} r63={r['rank_63d']:>5.1f} z10={r['z10']:>6.2f} d52h={r['dist_52w_high_pct']:>7.2f} d52l={r['dist_52w_low_pct']:>7.2f}")
print()
print("VIX/VIX3M =", round(by['^VIX']['close']/by['^VIX3M']['close'], 4))
print("SKEW/VIX  =", round(by['^SKEW']['close']/by['^VIX']['close'], 3))
