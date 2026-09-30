"""GLD -3.94% Monday, with the dollar at a 52w high, the 10y at a 19-year high, GLD 21d -10.6% and 24% off its
January high. Raw and vol-scaled one-day drops, forward GLD and SLV, and the dollar-high condition."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

px = load_prices(["GLD", "SLV", "UUP", "^TNX", "SPY", "GC=F"])
g = px["GLD"]["Close"].astype(float)
idx = g.index
r = g.pct_change()
hist = idx[:-1]
sd63 = r.rolling(63).std().shift(1)
z = r / sd63
print(f"today GLD {100 * r.iloc[-1]:.2f}%, z vs prior 63d sd {z.iloc[-1]:.2f}, sd63 {100 * sd63.iloc[-1]:.2f}%")
print("GLD 21d", round(100 * (g.iloc[-1] / g.iloc[-22] - 1), 2), "off 252d high", round(100 * (g.iloc[-1] / g.tail(252).max() - 1), 2))
big = idx[(r <= -0.035).values]
print("GLD <= -3.5% days, count by year:", pd.Series(big.year).value_counts().sort_index().to_dict())
bigz = idx[(z <= -2.5).values]
print("z <= -2.5 days by year:", pd.Series(bigz.year).value_counts().sort_index().to_dict())

sv = px["SLV"]["Close"].astype(float).reindex(idx)
uu = px["UUP"]["Close"].astype(float).reindex(idx)
tnx = px["^TNX"]["Close"].astype(float).reindex(idx)
uhi = rolling_on_valid(uu, lambda x: x.rolling(252).max())
thi = rolling_on_valid(tnx, lambda x: x.rolling(252).max())
at_uhi = (uu >= uhi * 0.995)
at_thi = (tnx >= thi * 0.99)
ret21 = g.pct_change(21)

def cell(dates: pd.DatetimeIndex, label: str) -> None:
    d = dates[dates < idx[-1]]
    dd = declusters(d, 5, idx)
    rows = []
    for nm, s in [("GLD", g), ("SLV", sv)]:
        for h in (1, 5, 21):
            f = fwd_ret(s, h)
            x = summarize(f.reindex(dd).values, f"{nm} h{h}")
            x["up"] = int((f.reindex(dd) > 0).sum())
            rows.append(x)
    show(rows, f"{label}: raw {len(d)}, declustered {len(dd)}")

for thr in (-0.03, -0.035, -0.04):
    cell(idx[(r <= thr).values], f"GLD <= {100 * thr:.1f}%")
cell(bigz, "z <= -2.5")
cell(idx[((r <= -0.03) & at_uhi).values], "GLD <= -3% with UUP within 0.5% of 252d high")
cell(idx[((z <= -2.5) & at_uhi).values], "z <= -2.5 with UUP near 252d high")
cell(idx[((r <= -0.03) & (ret21 <= -0.08)).values], "GLD <= -3% with 21d <= -8%")
cell(idx[((r <= -0.03) & at_thi).values], "GLD <= -3% with 10y within 1% of 252d high")

print("\nall-days GLD h1/h5/h21:", [round(summarize(fwd_ret(g, h).reindex(hist).values)["mean_pct"], 3) for h in (1, 5, 21)],
      "hit", [round(summarize(fwd_ret(g, h).reindex(hist).values)["hit"], 1) for h in (1, 5, 21)])
d35 = declusters(big[big < idx[-1]], 5, idx)
f1 = fwd_ret(g, 1).reindex(d35).dropna()
f5 = fwd_ret(g, 5).reindex(d35).dropna()
show(era_split(f1.index, f1.values), "GLD<=-3.5% h1 era")
show(era_split(f5.index, f5.values), "GLD<=-3.5% h5 era")
print(cluster_note(f5.index, f5.values))
print("episodes (h1, h5):", [(str(d.date()), round(100 * r[d], 2), round(100 * f1.get(d, np.nan), 2), round(100 * f5.get(d, np.nan), 2)) for d in d35])
print("local control h5:", round(summarize(fwd_ret(g, 5).reindex(local_control(idx, d35)).values)["mean_pct"], 3))
