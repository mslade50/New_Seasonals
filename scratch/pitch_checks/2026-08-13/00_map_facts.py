"""Facts the B1 surface map asserts. Cheap lookups, no falsification here."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import pandas as pd
from pitch_lab import ROOT, load_prices, pct_rank  # noqa

px = load_prices(["^MOVE", "^VIX", "^VIX3M", "^SKEW", "TLT", "SPY", "GLD", "SLV",
                  "IHI", "XLU", "XLV", "FXI", "EEM", "SVXY", "UVXY"])

# 1. ^MOVE level percentile of its trailing 252d (the 2026-08-10 kill's premise)
mv = px["^MOVE"]["Close"].dropna()
r = mv.rolling(252).apply(lambda w: (w.iloc[-1] > w[:-1]).mean() * 100, raw=False)
print(f"^MOVE  close {mv.iloc[-1]:.2f}  trailing-252d level pctile {r.iloc[-1]:.1f}"
      f"   (bottom-decile premise needs < 10)")

# VIX/VIX3M term spread percentile
vx, v3 = px["^VIX"]["Close"].dropna(), px["^VIX3M"]["Close"].dropna()
sp = (v3 / vx - 1.0).dropna()
print(f"VIX3M/VIX-1 = {sp.iloc[-1]*100:.1f}%   trailing-252d pctile "
      f"{(sp.iloc[-1] > sp.iloc[-253:-1]).mean()*100:.1f}")

# 2. fragility dial: level and 21d change
frag = pd.read_parquet(ROOT / "data/rd2_fragility.parquet")
frag.index = pd.to_datetime(frag.index)
ma = frag["63d"].rolling(10).mean()
print(f"\ndial ma10-63d {ma.iloc[-1]:.1f} on {frag.index[-1].date()};"
      f" 21td ago {ma.iloc[-22]:.1f}; delta {ma.iloc[-1]-ma.iloc[-22]:+.1f}")
print(f"  PIT rows since 2026-07-02: {(frag.index >= '2026-07-02').sum()};"
      f" total rows {len(frag)} from {frag.index[0].date()}")
d21 = ma.diff(21)
print(f"  delta21 >= +30 days in history: {(d21 >= 30).sum()} of {d21.notna().sum()}")
spy = px["SPY"]["Close"]
h52 = spy.rolling(252).max()
near = (spy / h52 - 1.0) > -0.01
mask = (d21 >= 30) & near.reindex(d21.index).fillna(False)
print(f"  delta21>=+30 AND SPY within 1% of 52w high: {int(mask.sum())} days")

# 3. gold/silver ratio state
gs = (px["GLD"]["Close"] / px["SLV"]["Close"]).dropna()
print(f"\nGLD/SLV {gs.iloc[-1]:.3f}  trailing-252d pctile "
      f"{(gs.iloc[-1] > gs.iloc[-253:-1]).mean()*100:.1f}")

# 4. IHI / XLU / XLV rank state, FXI vs EEM
for tk in ["IHI", "XLU", "XLV", "FXI", "EEM", "SVXY", "UVXY"]:
    c = px[tk]["Close"].dropna()
    r21 = pct_rank(c.pct_change(21), 252)
    print(f"{tk:6s} 21d ret {c.pct_change(21).iloc[-1]*100:+7.2f}%  rank21 {r21.iloc[-1]:5.1f}"
          f"  off-52wh {(c.iloc[-1]/c.rolling(252).max().iloc[-1]-1)*100:+7.2f}%")

# 5. events still ahead
ev = pd.read_csv(ROOT / "data/macro_events.csv", parse_dates=["date"])
fut = ev[(ev["date"] > "2026-08-12") & (ev["date"] <= "2026-09-30")]
print("\nevents Aug 13 - Sep 30:")
for _, row in fut.iterrows():
    print(f"  {row['date'].date()}  {row['event']}")
