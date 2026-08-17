"""Stage B1 enumerations 2 and 3: tape extremes made testable, plus the live
seasonal / cycle cells and the fragility-dial state.

Prints the raw material the surface map cites. No thesis, no verdict.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

BAR = pd.Timestamp("2026-08-14")
ROOT_ = Path(__file__).resolve().parents[3]

T = ["SPY", "QQQ", "IWM", "SMH", "SOXX", "XLK", "XLF", "KRE", "XLV", "IHI", "XLE",
     "XOP", "OIH", "USO", "GLD", "GDX", "SLV", "TLT", "IEF", "LQD", "HYG",
     "EEM", "EWZ", "EFA", "FXI", "XRT", "UUP", "SVXY", "^VIX", "^VIX3M", "^SKEW", "XLU", "XLP"]
px = load_prices(T)
C = pd.DataFrame({t: px[t]["Close"] for t in T if t in px}).sort_index()
ALL = C.index

print("=" * 100)
print("A. RANK DIVERGENCES (5d / 21d / 63d rank, live values)")
print("=" * 100)
print(f"{'tkr':<7} {'r5':>7} {'r21':>7} {'r63':>7} {'ret63%':>8} {'52wh%':>8} {'200d%':>8}")
for t in C.columns:
    s = C[t].dropna()
    if BAR not in s.index:
        continue
    r5, r21, r63 = (pct_rank(s, n).loc[BAR] for n in (5, 21, 63))
    ret63 = s.pct_change(63).loc[BAR] * 100
    dh = (s.loc[BAR] / s.rolling(252).max().loc[BAR] - 1) * 100
    d200 = (s.loc[BAR] / s.rolling(200).mean().loc[BAR] - 1) * 100
    print(f"{t:<7} {r5:>7.1f} {r21:>7.1f} {r63:>7.1f} {ret63:>8.2f} {dh:>8.2f} {d200:>8.2f}")

print("\n" + "=" * 100)
print("B. LATE-AUGUST SEASONAL CELL: entry on the analogue of Aug 17, by cycle year")
print("=" * 100)
spy = C["SPY"].dropna()
qqq = C["QQQ"].dropna()
tlt = C["TLT"].dropna()
for name, s in [("SPY", spy), ("QQQ", qqq), ("TLT", tlt)]:
    aug = pd.DatetimeIndex([d for d in s.index if d.month == 8 and 15 <= d.day <= 19])
    aug = declusters(aug, 5, s.index)
    mid = pd.DatetimeIndex([d for d in aug if d.year % 4 == 2])
    for h in (5, 8, 10):
        f = fwd_lag(s, h, 1)
        a = f.reindex(aug).dropna()
        m = f.reindex(mid).dropna()
        d = f.mean() * 100
        print(f"{name} h={h:<3} all-Aug N={len(a):<3} {a.mean()*100:+7.3f}%  "
              f"excess {a.mean()*100-d:+7.3f}  | midterm N={len(m):<3} "
              f"{m.mean()*100 if len(m) else float('nan'):+7.3f}%  excess "
              f"{(m.mean()*100-d) if len(m) else float('nan'):+7.3f}")

print("\n" + "=" * 100)
print("C. FRAGILITY DIAL STATE (data/rd2_fragility.parquet)")
print("=" * 100)
fr = pd.read_parquet(ROOT_ / "data" / "rd2_fragility.parquet")
fr.index = pd.to_datetime(fr.index)
print("cols:", list(fr.columns), "| span", fr.index.min().date(), "->", fr.index.max().date(),
      "| rows", len(fr))
ma = fr["63d"].rolling(10).mean()
print(f"today ma10_63d = {ma.iloc[-1]:.1f} | raw63 {fr['63d'].iloc[-1]:.1f} "
      f"| raw21 {fr['21d'].iloc[-1]:.1f} | raw5 {fr['5d'].iloc[-1]:.1f}")
print(f"pctile of ma10_63d in its own history: "
      f"{(ma.dropna() <= ma.iloc[-1]).mean()*100:.1f}")
hi = ma[ma >= 78].index
print(f"days with ma10_63d >= 78: {len(hi)}; declustered(21td): "
      f"{len(declusters(pd.DatetimeIndex(hi), 21, fr.index))}")
print("years represented:", sorted({d.year for d in hi}))

print("\n" + "=" * 100)
print("D. VOL TERM STRUCTURE + SKEW")
print("=" * 100)
v, v3 = C["^VIX"].dropna(), C["^VIX3M"].dropna()
ratio = (v / v3).reindex(ALL).dropna()
print(f"VIX {v.loc[BAR]:.2f} | VIX3M {v3.loc[BAR]:.2f} | ratio {ratio.loc[BAR]:.3f} "
      f"| pctile of ratio (trailing 252) "
      f"{(ratio.tail(252) <= ratio.loc[BAR]).mean()*100:.1f}")
sk = C["^SKEW"].dropna()
print(f"SKEW {sk.loc[BAR]:.2f} | 5d rank {pct_rank(sk,5).loc[BAR]:.1f} "
      f"| level pctile(252) {(sk.tail(252) <= sk.loc[BAR]).mean()*100:.1f}")
