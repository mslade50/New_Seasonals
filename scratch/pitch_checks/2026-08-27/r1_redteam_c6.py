"""Red-team pass over the one survivor (C6), composer-side.

Deliberately re-derives the mask from scratch rather than importing the
checker's construction, so the headline is verified rather than trusted.
Answers the four red-team questions the skill requires (basket correlation is
moot at one idea): book overlap, cost at the DEVELOPED entry form, the
strongest single argument against, and whether that argument would convince.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np

ASOF = pd.Timestamp("2026-08-26")

# ---------------------------------------------------------------- mask, from scratch
ohlc = load_prices(["GDX"])["GDX"]
g = ohlc["Close"].dropna()
g = g[g.index <= ASOF]
r21 = g / g.shift(21) - 1.0
r1 = g / g.shift(1) - 1.0

# PIT trailing-252 percentile of the 21d return, computed the long way:
# share of the PRIOR 252 observations strictly below today's, no lookahead.
rk = r21.rolling(253).apply(lambda w: (w[:-1] < w[-1]).mean() * 100.0, raw=True)

mask = ((rk >= 99) & (r1 <= -0.02)).fillna(False)
trig = g.index[mask]
epi = declusters(trig, 10, g.index)
print("raw trigger days:", len(trig), " declustered episodes:", len(epi))
print("episodes:", [str(d.date()) for d in epi])

live = epi[-1] == ASOF
print(f"\nTRIGGER LIVE ON {ASOF.date()}: {live}")
print("  GDX close %.2f | r21 %+.2f%% | PIT rank %.1f | 1d %+.2f%%"
      % (g.iloc[-1], 100 * r21.iloc[-1], rk.iloc[-1], 100 * r1.iloc[-1]))
atr = pd.Series(wilder_atr(ohlc["High"], ohlc["Low"], ohlc["Close"], 14),
                index=ohlc.index).reindex(g.index)
print("  Wilder-14 ATR %.4f = %.2f%% of price" % (atr.iloc[-1], 100 * atr.iloc[-1] / g.iloc[-1]))

hist = epi[epi < ASOF]
print(f"\nhistorical episodes for stats: {len(hist)}")

# ---------------------------------------------------------------- headline, h=5
px = g.to_frame("GDX")
for h in (3, 5, 10):
    f = fwd_lag(g, h, lag=1)
    vals = f.reindex(hist).dropna().values
    drift = f.dropna()
    s = summarize(vals, f"h={h} cell")
    d = summarize(drift.values, f"h={h} all-days drift")
    wins = int((vals > 0).sum())
    print(f"\nh={h}: N={s['n']} mean={s['mean_pct']:+.3f}%  drift={d['mean_pct']:+.3f}%  "
          f"edge={s['mean_pct']-d['mean_pct']:+.3f}pp  record {wins}-{s['n']-wins}  "
          f"sign_p={sign_test(wins, s['n']):.4f}  bootstrap_P(<=0)={bootstrap_p_le0(vals):.4f}")
    if h == 5:
        print("  per-episode:", [f"{d.date()} {v*100:+.2f}%" for d, v in
                                 zip(hist, f.reindex(hist).values)])
        srt = np.sort(vals)
        print(f"  drop-best-2 mean = {srt[:-2].mean()*100:+.3f}%   worst = {srt[0]*100:+.2f}%")

# ---------------------------------------------------------------- the depth caveat
print("\n=== today's depth bucket (the honest expectation, not the headline) ===")
f5 = fwd_lag(g, 5, lag=1)
for lo, hi, lbl in [(-1e9, -0.04, "<=-4%"), (-0.04, -0.03, "(-4,-3]"),
                    (-0.03, -0.02, "(-3,-2]  <-- TODAY -2.94%")]:
    for rung, rl in [(99, ">=99"), (97, ">=97"), (95, ">=95")]:
        m = ((rk >= rung) & (r1 > lo) & (r1 <= hi)).fillna(False)
        e = declusters(g.index[m], 10, g.index)
        e = e[e < ASOF]
        v = f5.reindex(e).dropna().values
        if len(v):
            w = int((v > 0).sum())
            print(f"  rank{rl:>4} {lbl:<24} N={len(v):2d} mean={v.mean()*100:+7.3f}%  {w}-{len(v)-w}")

# ---------------------------------------------------------------- Jackson Hole in the hold
print("\n=== Jackson Hole inside the hold (today's hold contains it) ===")
jh = load_events(["jackson_hole"])
inw = event_in_window(hist, g.index, 5, lag=1, kinds=("jackson_hole",))
print(f"  episodes with a JH inside the hold: {int(np.sum(inw))} of {len(hist)}")
print(f"  JH dates: {[str(d.date()) for d in jh.date.tail(3)]}  | next 2026-08-28, +1 td")

# ---------------------------------------------------------------- dial support
print("\n=== fragility dial support ===")
fr = pd.read_parquet(ROOT / "data/rd2_fragility.parquet")
ma = fr["63d"].rolling(10).mean()
vals = [(str(d.date()), round(float(ma.reindex([d]).iloc[0]), 1))
        for d in hist if d in ma.index and pd.notna(ma.reindex([d]).iloc[0])]
print("  episode readings:", vals)
print(f"  today ma10(63d) = {ma.iloc[-1]:.1f}")

# ---------------------------------------------------------------- book overlap
print("\n=== book overlap (systematic layers only) ===")
led = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
# NB: the ledger's columns are space-separated ("Signal Date", "Strategy").
# A `Signal_Date`/`Strategy_Name` guess silently falls through to trade_id and
# returns a FALSE ZERO overlap -- caught on the composer pass, 2026-08-27.
assert "Signal Date" in led.columns and "Strategy" in led.columns, led.columns.tolist()
led["Signal Date"] = pd.to_datetime(led["Signal Date"])
sub = led[led["Signal Date"].isin(set(trig))]
print(f"  ledger signals on ANY of the {len(trig)} trigger days: {len(sub)}")
if len(sub):
    print(sub.groupby("Strategy")["R_Multiple"].agg(["count", "mean"]).round(3).to_string())

# The hold, not just the signal day: anything open across a 5-session window.
win = set()
for d in hist:
    i = g.index.get_loc(d)
    win |= set(g.index[i: i + 7])
sub2 = led[led["Signal Date"].isin(win)]
print(f"  ledger signals anywhere inside the 6 historical holds: {len(sub2)}")
if len(sub2):
    print(sub2.groupby("Strategy")["R_Multiple"].agg(["count", "mean"]).round(3).to_string())

for t in ["GDX", "GLD", "NEM", "SLV"]:
    sl = led[led["Ticker"] == t]
    print(f"  ledger trades ever on {t}: {len(sl)}"
          + (f"  avgR {sl['R_Multiple'].mean():+.3f}" if len(sl) else ""))

# ---------------------------------------------------------------- cost at the developed form
print("\n=== cost at the DEVELOPED entry form (MOC, one leg) ===")
edge_bp = (f5.reindex(hist).dropna().mean() - f5.dropna().mean()) * 1e4
for c in (5, 10, 15):
    print(f"  edge {edge_bp:.1f} bp / {c} bp round trip = {edge_bp/c:.1f}x")
