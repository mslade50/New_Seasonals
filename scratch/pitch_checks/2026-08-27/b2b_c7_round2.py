"""C7 round 2 — the three things that bear directly on TODAY, after the
reference class already returned family-wise p 0.8805.

(a) the current episode is IN the sample: 2026-07-28 and 2026-08-11 are this
    same SMH drawdown. Drop 2026 and re-read.
(b) today's r5 is 31.35 -> SMH is not in a fresh downdraft. Split the cell on
    r5 and read the sub-cell today actually sits in.
(c) regime: is this a bear-tape cell (the registry's laggard-snapback kill)?
(d) dial support and the midterm split.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

ASOF = pd.Timestamp("2026-08-26")
H = 10
px = close_panel(["SMH", "SPY"])
px = px[px.index <= ASOF]
s = px["SMH"].dropna()


def vret(x, n):
    v = x.dropna()
    return (v / v.shift(n) - 1.0).reindex(x.index)


r63 = pct_rank(s, 63)
r5 = pct_rank(s, 5)
r252 = vret(s, 252)
mask = (r63 <= 5) & (r252 >= 0.40)
f = fwd_lag(s, H, 1)
valid = f.dropna().index
trig = pd.DatetimeIndex(s.index[mask.reindex(s.index, fill_value=False).fillna(False).values]).intersection(valid)
epi = declusters(trig, H, valid)
v = f.loc[epi].values
drift = f.loc[valid].mean()
print(f"base cell h={H}: N={len(epi)} episodes, mean {100*v.mean():+.3f}%, "
      f"own drift {100*drift:+.3f}%, excess {100*(v.mean()-drift):+.3f}pp")
print("  episodes:", [str(d.date()) for d in epi])

print("\n(a) DROP THE CURRENT EPISODE (2026 rows are this same drawdown)")
for cut in [2026, 2024]:
    m = np.array([d.year < cut for d in epi])
    print(f"  drop {cut}+: N={int(m.sum())}  mean {100*v[m].mean():+.3f}%  "
          f"excess {100*(v[m].mean()-drift):+.3f}pp  hit {100*(v[m]>0).mean():.1f}%  "
          f"record {int((v[m]>0).sum())}-{int((v[m]<=0).sum())} sign p "
          f"{sign_test(int((v[m]>0).sum()), int(m.sum())):.4f}")
print(f"  2026 episodes alone: "
      f"{[f'{d.date()} {100*x:+.2f}%' for d, x in zip(epi, v) if d.year == 2026]}")

print("\n(b) r5 SPLIT — today's SMH r5 = %.1f" % r5.iloc[-1])
r5e = r5.reindex(epi)
for lo, hi, lbl in [(0, 15, "r5<15 fresh downdraft"), (15, 25, "r5 15-25"),
                    (25, 101, "r5>=25  <-- TODAY (31.4)")]:
    m = ((r5e >= lo) & (r5e < hi)).values
    if m.sum() == 0:
        print(f"  {lbl:<28} N=0")
        continue
    print(f"  {lbl:<28} N={int(m.sum()):>3}  mean {100*v[m].mean():+7.3f}%  "
          f"excess {100*(v[m].mean()-drift):+7.3f}pp  hit {100*(v[m]>0).mean():5.1f}%  "
          f"record {int((v[m]>0).sum())}-{int((v[m]<=0).sum())}  sign p "
          f"{sign_test(int((v[m]>0).sum()), int(m.sum())):.4f}")

print("\n(c) REGIME — SPY vs its 200d (the laggard-snapback kill's mechanism)")
spy = px["SPY"].dropna()
sma = rolling_on_valid(spy, lambda x: x.rolling(200).mean())
below = (spy < sma)
b = below.reindex(epi).fillna(False).values
print(f"  SPY below 200d on {100*b.mean():.1f}% of trigger episodes vs "
      f"{100*below.mean():.1f}% base rate -> over-selection {100*(b.mean()-below.mean()):+.1f}pp")
for m, lbl in [(b, "SPY<200d"), (~b, "SPY>=200d  <-- TODAY (SPY -1.52% off its high)")]:
    if m.sum() == 0:
        continue
    print(f"  {lbl:<45} N={int(m.sum()):>3}  mean {100*v[m].mean():+7.3f}%  "
          f"excess {100*(v[m].mean()-drift):+7.3f}pp  hit {100*(v[m]>0).mean():5.1f}%")

print("\n(d) DIAL SUPPORT + MIDTERM")
frag = pd.read_parquet(ROOT / "data/rd2_fragility.parquet")
ma10 = frag["63d"].rolling(10).mean()
dl = ma10.reindex(epi).dropna()
print(f"  dial ma10(63d) on trigger episodes: n_with_reading {len(dl)}  "
      f"max {dl.max():.1f}  >=85: {int((dl>=85).sum())}   [today 88.6]")
mid = np.array([d.year % 4 == 2 for d in epi])
print(f"  midterm N={int(mid.sum())} mean {100*v[mid].mean():+.3f}% | "
      f"non-midterm N={int((~mid).sum())} mean {100*v[~mid].mean():+.3f}%")
print(f"  midterm episodes: {[str(d.date()) for d in epi[mid]]} "
      f"(2 of 4 are 2026 = the live drawdown)")

print("\n(e) BOOK OVERLAP — ledger signals on an SMH C7 trigger day")
led = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
led["Signal Date"] = pd.to_datetime(led["Signal Date"])
mm = mask.reindex(s.index, fill_value=False).fillna(False)
tdays = set(s.index[mm.values])
sub = led[led["Signal Date"].isin(tdays)]
print(f"  {len(sub)} ledger trades signalled on one of the {len(tdays)} SMH trigger days")
if len(sub):
    print(sub.groupby("Strategy").agg(n=("R_Multiple", "size"),
                                      avgR=("R_Multiple", "mean")).round(3).to_string())
    print(f"  overall avgR {sub.R_Multiple.mean():+.3f}")
smh = led[led.Ticker == "SMH"]
print(f"  ledger trades in SMH itself, any state: {len(smh)}")
