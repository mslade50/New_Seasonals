"""Stage B1 recon part 2: cross-section, dial history, retail complex."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

ASOF = pd.Timestamp("2026-08-26")
tape = json.load(open(ROOT / "data/pitch_tape.json"))["tickers"]
names = sorted(tape)
frames = load_prices(names)
print(f"loaded {len(frames)} of {len(names)}")

rows = []
for t, df in frames.items():
    s = df["Close"].dropna()
    s = s[s.index <= ASOF]
    if len(s) < 300:
        continue
    hi = s.rolling(252).max().iloc[-1]
    lo = s.rolling(252).min().iloc[-1]
    rows.append({"t": t, "dl": 100 * (s.iloc[-1] / lo - 1), "dh": 100 * (s.iloc[-1] / hi - 1)})
R = pd.DataFrame(rows).set_index("t")
print(f"\n=== D. names near a 52w LOW while SPY is {R.loc['SPY','dh']:+.2f}% off its high  (n={len(R)}) ===")
for thr in [1, 2, 3, 5]:
    sub = R[R.dl <= thr]
    print(f"  within {thr}% of 52w low: {len(sub):3d} ({100*len(sub)/len(R):4.1f}%)  {sorted(sub.index)[:16]}")
print(f"  names within 1% of a 52w HIGH: {(R.dh >= -1).sum()}  {sorted(R[R.dh>=-1].index)[:16]}")

print("\n=== I. fragility dial history ===")
f = pd.read_parquet(ROOT / "data/rd2_fragility.parquet")
print(" cols", list(f.columns), " rows", len(f), f.index[0].date(), "->", f.index[-1].date())
ma = f["63d"].rolling(10).mean()
cur = ma.iloc[-1]
print(f"  ma10(63d) now {cur:.1f}  |  days >= 85 ever: {(ma >= 85).sum()}  >= 80: {(ma>=80).sum()}  >= 70: {(ma>=70).sum()}  of {ma.notna().sum()}")
hi = ma[ma >= 85]
print(f"  >=85 episodes (year counts): {hi.groupby(hi.index.year).size().to_dict()}")
print(f"  raw63 now {f['63d'].iloc[-1]:.1f}  raw21 {f['21d'].iloc[-1]:.1f}")

print("\n=== J. consumer / retail complex washout ===")
CONS = ["TJX", "ROST", "WMT", "TGT", "LOW", "HD", "NKE", "COST", "DG", "KSS", "M",
        "BBY", "GPS", "F", "GM", "SBUX", "MCD", "XRT", "XLY", "XLP", "LULU", "ULTA"]
have = [t for t in CONS if t in frames]
out = []
for t in have:
    s = frames[t]["Close"].dropna(); s = s[s.index <= ASOF]
    if len(s) < 300: continue
    out.append({"t": t, "r21": pct_rank(s, 21).iloc[-1], "r5": pct_rank(s, 5).iloc[-1],
                "z10": zscore(s, 10).iloc[-1],
                "dl": 100*(s.iloc[-1]/s.rolling(252).min().iloc[-1]-1)})
O = pd.DataFrame(out).set_index("t").sort_values("r21")
print(O.round(2).to_string())
print(f"  count r21 <= 10: {(O.r21 <= 10).sum()} of {len(O)}")

print("\n=== K. calendar sanity: Jackson Hole + month end ===")
ev = load_events()
jh = ev[ev.event == "jackson_hole"]
print("  JH dates recent:", [str(d.date()) for d in jh.date.tail(4)])
spy = frames["SPY"]["Close"].dropna()
d = spy.index
aug = d[(d >= "2026-08-01") & (d <= "2026-08-31")]
print("  Aug 2026 sessions in cache (through cache end):", [str(x.date()) for x in aug[-5:]])
