"""Stage B1 recon: how rare is each of today's candidate states, PIT.

Every percentile here is a TRAILING-252d PIT rank (the 2026-08-17 method trap:
a full-history percentile is lookahead).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

ASOF = pd.Timestamp("2026-08-26")

NAMES = ["SPY", "QQQ", "IWM", "SMH", "XLK", "XLV", "XLP", "XLF", "XLE", "XLI",
         "XLY", "XLU", "XLB", "XLRE", "XLC", "IBB", "GDX", "GLD", "SLV", "NEM",
         "TLT", "IEF", "HYG", "LQD", "UUP", "DX-Y.NYB", "^VIX", "^VIX3M",
         "SVXY", "^MOVE", "^SKEW", "USO", "XLE", "EEM", "EFA", "FXI", "^TNX",
         "XME", "FCX", "NKE", "TJX", "WMT", "XRT", "OIH", "XOP", "DBC", "UNG"]
px = close_panel(sorted(set(NAMES)))
px = px[px.index <= ASOF]
print("panel", px.shape, px.index[0].date(), "->", px.index[-1].date())


def pit_rank_of_last(s, n, lookback=252):
    r = pct_rank(s.dropna(), n, lookback)
    return r.dropna().iloc[-1]


def dist_52w(s):
    s = s.dropna()
    hi = s.rolling(252).max().iloc[-1]
    lo = s.rolling(252).min().iloc[-1]
    return 100 * (s.iloc[-1] / hi - 1), 100 * (s.iloc[-1] / lo - 1)


print("\n=== A. 63d return-rank spread, semis vs healthcare (PIT) ===")
for a, b in [("SMH", "XLV"), ("SMH", "IBB"), ("XLK", "XLV"), ("QQQ", "XLV")]:
    ra = px[a].pct_change(63)
    rb = px[b].pct_change(63)
    sp = (ra - rb).dropna()
    pit = rolling_on_valid(sp, lambda w: w.rank(pct=True).iloc[-1] * 100) if False else None
    # PIT percentile of the spread within its own trailing 252 obs
    pitser = sp.rolling(252).apply(lambda w: (w[:-1] < w[-1]).mean() * 100, raw=True)
    print(f"  {a}-{b} 63d spread {sp.iloc[-1]*100:+7.2f}pp   PIT pctile {pitser.iloc[-1]:5.1f}"
          f"   n_days_at_or_below_2.5pct_since2005 {(pitser[pitser.index>='2005'] <= 2.5).sum()}")

print("\n=== B. GDX / NEM / GLD 21d thrust rarity (PIT trailing-252 rank of the 21d return) ===")
for t in ["GDX", "NEM", "GLD", "SLV", "XME", "FCX"]:
    s = px[t].dropna()
    r21 = s.pct_change(21)
    pit = r21.rolling(252).apply(lambda w: (w[:-1] < w[-1]).mean() * 100, raw=True)
    fullpct = 100 * (r21.dropna() <= r21.dropna().iloc[-1]).mean()
    print(f"  {t:<10} 21d {r21.iloc[-1]*100:+7.2f}%  PIT pctile {pit.iloc[-1]:5.1f}  "
          f"full-hist pctile {fullpct:5.1f}  1d {s.pct_change().iloc[-1]*100:+6.2f}%")

print("\n=== C. vol term structure ===")
v, v3 = px["^VIX"].dropna(), px["^VIX3M"].dropna()
j = pd.concat([v, v3], axis=1).dropna()
j.columns = ["vix", "vix3m"]
ratio = j.vix / j.vix3m
print(f"  VIX {j.vix.iloc[-1]:.2f}  VIX3M {j.vix3m.iloc[-1]:.2f}  ratio {ratio.iloc[-1]:.4f}")
for lbl, s in [("VIX", j.vix), ("VIX3M", j.vix3m), ("ratio", ratio)]:
    hi, lo = dist_52w(s)
    pit = s.rolling(252).apply(lambda w: (w[:-1] < w[-1]).mean() * 100, raw=True)
    print(f"    {lbl:<6} lvl-PIT pctile {pit.iloc[-1]:5.1f}   vs 52wHigh {hi:+7.2f}%  vs 52wLow {lo:+7.2f}%")

print("\n=== D. cross-sectional: names at/near a 52w low while SPY is near its high ===")
big = close_panel(sorted(set(json.load(open(ROOT / "data/pitch_tape.json"))["tickers"])))
big = big[big.index <= ASOF]
last = big.iloc[-1]
lo252 = big.rolling(252).min().iloc[-1]
hi252 = big.rolling(252).max().iloc[-1]
dl = 100 * (last / lo252 - 1)
dh = 100 * (last / hi252 - 1)
n = dl.notna().sum()
print(f"  universe {n} names;  SPY off high {dh['SPY']:+.2f}%")
for thr in [1, 2, 3, 5]:
    k = (dl <= thr).sum()
    print(f"    within {thr}% of a 52w low: {k} names ({100*k/n:.1f}%)  -> "
          f"{sorted(dl[dl<=thr].index.tolist())[:14]}")

print("\n=== E. month position ===")
d = big.index
print("  last 6 sessions:", [str(x.date()) for x in d[-6:]])
print("  today 2026-08-27 is ME-2 (Aug last trading day = 2026-08-31)")

print("\n=== F. sector 63d-rank cross-sectional dispersion ===")
SECT = ["XLK", "XLV", "XLF", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]
r63 = {t: pct_rank(px[t].dropna(), 63) for t in SECT}
r63 = pd.DataFrame(r63).dropna()
disp = r63.max(axis=1) - r63.min(axis=1)
sd = r63.std(axis=1)
print(f"  today spread {disp.iloc[-1]:.1f}  sd {sd.iloc[-1]:.1f}")
for lbl, s in [("spread", disp), ("sd", sd)]:
    pit = s.rolling(252).apply(lambda w: (w[:-1] < w[-1]).mean() * 100, raw=True)
    print(f"    {lbl} PIT pctile {pit.iloc[-1]:5.1f}")
print("  ranks:", {t: round(r63[t].iloc[-1], 1) for t in SECT})

print("\n=== G. credit / duration joint state ===")
for t in ["HYG", "LQD", "TLT", "IEF", "SPY", "XLB", "XLF", "SVXY"]:
    hi, lo = dist_52w(px[t])
    print(f"  {t:<6} off 52wHigh {hi:+7.2f}%   above 52wLow {lo:+8.2f}%")

print("\n=== H. dollar washout-then-bounce ===")
for t in ["DX-Y.NYB", "UUP"]:
    s = px[t].dropna()
    print(f"  {t:<10} r21 PIT {pit_rank_of_last(s,21):5.1f}  r5 PIT {pit_rank_of_last(s,5):5.1f}  "
          f"21d {s.pct_change(21).iloc[-1]*100:+.2f}%  5d {s.pct_change(5).iloc[-1]*100:+.2f}%")
