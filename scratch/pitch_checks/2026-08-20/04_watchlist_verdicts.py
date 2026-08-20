"""Every active watchlist entry owes a verdict with today's number. This computes
the ones the tape file cannot answer directly."""
import sys, warnings, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
warnings.filterwarnings("ignore")
from pitch_lab import *  # noqa

tape = json.load(open(ROOT / "data/pitch_tape.json"))["tickers"]
def T(t, f): return tape[t][f]

print("=== tape-answerable triggers (value today -> gate) ===")
rows = [
 ("W4  GLD miner-led thrust", f"GDX r5={T('GDX','rank_5d')} (need>=95), GLD r5={T('GLD','rank_5d')} (need<95)"),
 ("W5  XLE crude pop band", f"USO 1d={T('USO','ret_1d')}% (need +5..6% and >=1.50 ATR)"),
 ("W6  TLT tight IG rung", f"TLT off low={T('TLT','dist_52w_low_pct')}% (need<=0.5), IEF={T('IEF','dist_52w_low_pct')}, LQD={T('LQD','dist_52w_low_pct')}"),
 ("W7  SPY skew-alone", f"SKEW r5={T('^SKEW','rank_5d')} (need>=95); SPY off high={T('SPY','dist_52w_high_pct')}% (need< -1, CLEARS); midterm=YES (structural fail)"),
 ("W8  USO thrust fade", f"USO r5={T('USO','rank_5d')} (need>=90), r63={T('USO','rank_63d')} (need<=20, CLEARS)"),
 ("W9  IHI 21d rank", f"IHI r21={T('IHI','rank_21d')} (need 100)"),
 ("W10 FXI break in thrust", f"FXI r5={T('FXI','rank_5d')} (need<=20), r21={T('FXI','rank_21d')} (need>=80), EEM 5d={T('EEM','ret_5d')}% (need>0)"),
 ("W13 SPY high / TLT low", f"SPY off high={T('SPY','dist_52w_high_pct')}% (need>=-0.5), TLT off low={T('TLT','dist_52w_low_pct')}% (need<=1.0)"),
 ("W15 SPY vol pop calm tape", f"VIX r21={T('^VIX','rank_21d')} (need<=25 CLEARS), VIX 1d={T('^VIX','ret_1d')}% (need>=+5 FAILS), SPY 1d={T('SPY','ret_1d')}%"),
 ("W16 gold unconfirmed rate rise", f"DX r21={T('DX-Y.NYB','rank_21d')} (need<=15 CLEARS)"),
 ("W17 tech vs healthcare rotation", f"XLV 1d={T('XLV','ret_1d')} - XLK 1d={T('XLK','ret_1d')} = {round(T('XLV','ret_1d')-T('XLK','ret_1d'),2)}pp (need>=3.0), SPY off high={T('SPY','dist_52w_high_pct')} (need>-3), SPY atr%={T('SPY','atr_pct')} (need<1.2)"),
 ("W18 short dollar on rate rise", f"TNX r21={T('^TNX','rank_21d')} (need>=65), DX r21={T('DX-Y.NYB','rank_21d')} (need<=20 CLEARS)"),
]
for a, b in rows:
    print(f" {a:<34} {b}")

px = close_panel(["TLT", "IEF", "LQD", "HYG", "^TNX", "XLV", "XLK", "SPY", "GDX", "GLD"])
d = px.index

print("\n=== W16: TNX 21-session LEVEL change (need >= +0.20pt) ===")
tnx = px["^TNX"].dropna()
print(" today", round(tnx.iloc[-1], 3), " 21d chg", round(tnx.iloc[-1] - tnx.iloc[-22], 3), "pt")

print("\n=== W2: HYG/LQD joint extreme, declustered episode count (need >=8 over >=3 years ex-2018) ===")
hyg, lqd = px["HYG"].dropna(), px["LQD"].dropna()
hh = hyg / hyg.rolling(252).max() - 1
ll = lqd / lqd.rolling(252).min() - 1
print(f" today HYG off 52w high {hh.iloc[-1]*100:+.2f}% (need >=-0.5), LQD off 52w low {ll.iloc[-1]*100:+.2f}% (need <=2.0)")
m = (hh >= -0.005) & (ll <= 0.02)
m = m.reindex(d).fillna(False).astype(bool)
ep = declusters(d[m], 21, d)
print(f" days={int(m.sum())} episodes(gap21)={len(ep)} years={sorted(set(ep.year))}")

print("\n=== W11: industry breadth washout (need >=70% of an industry at 5d rank <=20, median 63d rank < 70) ===")
GROUPS = {
 "semis": ["NVDA","AVGO","AMD","INTC","MU","AMAT","ADI","TXN","QCOM","SMH"],
 "megacap_tech": ["AAPL","MSFT","GOOG","AMZN","META","ORCL","CRM","ADBE","CSCO","IBM"],
 "banks": ["JPM","BAC","C","WFC","GS","MS","USB","KEY","RF","STT","SCHW"],
 "insurers": ["ALL","TRV","PGR","HIG","AIG","MET","AON","MRSH","CB" ],
 "staples": ["KO","PEP","PG","CL","KMB","GIS","CAG","CPB","HSY","SYY","TSN","HRL","MO"],
 "utilities": ["NEE","DUK","SO","D","AEP","EXC","XEL","ED","PEG","SRE","ETR","FE","CMS","DTE","PPL","PNW","CNP","EIX","PCG"],
 "healthcare": ["JNJ","PFE","MRK","ABT","AMGN","BMY","LLY","GILD","UNH","HUM","CVS","MDT","SYK","BDX","BAX","TMO","REGN"],
 "energy": ["XOM","CVX","COP","EOG","SLB","HAL","OXY","VLO","WMB"],
 "industrials": ["HON","MMM","GE","CAT","DE","EMR","ITW","PH","ROK","DOV","SWK","SNA","NSC","UNP","CSX","FDX","LMT","NOC","GD","RTX","BA"],
 "retail": ["WMT","TGT","COST","LOW","HD","TJX","ROST","KR","DIS","SBUX","MCD","NKE"],
}
for g, names in GROUPS.items():
    have = [n for n in names if n in tape]
    if len(have) < 5: continue
    r5 = [tape[n]["rank_5d"] for n in have]
    r63 = [tape[n]["rank_63d"] for n in have]
    frac = sum(1 for x in r5 if x <= 20) / len(r5)
    print(f" {g:<14} n={len(have):>2}  frac r5<=20 = {frac*100:>5.1f}%  median r63 = {np.median(r63):>5.1f}")

print("\n=== W17: rotation subclass, does today arm it? ===")
gap = px["XLV"].pct_change() - px["XLK"].pct_change()
spy = px["SPY"]
offhigh = spy / spy.rolling(252).max() - 1
atr = wilder_atr(load_prices(["SPY"])["SPY"], 14) / spy
sub = (gap >= 0.03) & (offhigh >= -0.03) & (atr < 0.012)
sub = sub.reindex(d).fillna(False).astype(bool)
ep17 = declusters(d[sub], 3, d)
print(f" subclass days={int(sub.sum())} episodes(gap3)={len(ep17)}")
v = vehicle_ret(px, [("XLK", 1.0), ("XLV", -1.0)], 3).reindex(ep17).dropna()
print(f" h=3 pair N={len(v)} mean={v.mean()*100:+.3f}% record {(v>0).sum()}-{(v<=0).sum()}")
if len(v) > 2:
    order = np.argsort(v.values)[::-1]
    drop2 = np.delete(v.values, order[:1])
    print(f" drop-best mean={drop2.mean()*100:+.3f}% (arm needs >= +0.50% at N>=24 and >= 15-9)")
    yr = pd.Series(v.values, index=v.index).groupby(v.index.year).agg(["count", "mean"])
    yr["mean"] = (yr["mean"] * 100).round(3)
    print(yr)
    ex26 = v[v.index.year != 2026]
    print(f" ex-2026 N={len(ex26)} mean={ex26.mean()*100:+.3f}% record {(ex26>0).sum()}-{(ex26<=0).sum()}")
