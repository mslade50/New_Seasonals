"""Recompute every number the brief quotes, plus resolutions of prior items."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

px = load_prices(["SPY", "TLT", "IEF", "GLD", "UUP", "EEM", "^TNX", "^VIX", "DX-Y.NYB"])
spy_idx = px["SPY"].index
cp = pd.DataFrame({t: px[t]["Close"].astype(float) for t in px}).loc[spy_idx]

u = cp["UUP"].dropna()
prior = u.iloc[:-1]
print("UUP", round(u.iloc[-1], 3), "last close >= today:", prior[prior >= u.iloc[-1]].index[-1:].date if (prior >= u.iloc[-1]).any() else "none")
t = cp["^TNX"].dropna()
print("10y", t.iloc[-1], "last close >= today:", t.iloc[:-1][t.iloc[:-1] >= t.iloc[-1]].index[-1:].date)
print("VIX", round(cp["^VIX"].iloc[-1], 2), round(100 * cp["^VIX"].pct_change().iloc[-1], 2))

# three-way down day
g = cp.dropna(subset=["SPY", "TLT", "GLD"])
r = g.pct_change()
mask = (r.SPY <= -0.005) & (r.TLT <= -0.005) & (r.GLD <= -0.02)
raw = g.index[mask.fillna(False).values]
print("\nthree-way raw incl today:", len(raw), "by year:", pd.Series(raw.year).value_counts().sort_index().to_dict())
prior_raw = raw[raw < g.index[-1]]
dd = declusters(prior_raw, 5, g.index)
f21 = fwd_ret(g["SPY"], 21).reindex(dd).dropna()
print("SPY h21 up", int((f21 > 0).sum()), "of", len(f21), f"mean {100 * f21.mean():.2f}%", "t", round(summarize(f21.values)["t"], 2),
      "base", round(summarize(fwd_ret(g["SPY"], 21).reindex(g.index[:-1]).values)["mean_pct"], 2))
show(era_split(f21.index, f21.values), "SPY h21 era")
print(cluster_note(f21.index, f21.values))
print("2026 SPY h21:", [(str(d.date()), round(100 * f21.get(d, np.nan), 2)) for d in dd if d.year == 2026])
f1t = fwd_ret(g["TLT"], 1).reindex(dd).dropna()
print("TLT h1 up", int((f1t > 0).sum()), "of", len(f1t))
ex = f21[~f21.index.year.isin([2008, 2020])]
print("SPY h21 ex 2008/2020:", int((ex > 0).sum()), "of", len(ex), f"{100 * ex.mean():.2f}%")
# how rare is the base P9b pair per year (context for the 2026 count)
p9 = g.index[((r.SPY <= -0.005) & (r.TLT <= -0.005)).fillna(False).values]
print("P9b pair 2026:", int((p9.year == 2026).sum()), "avg per year 2005-2025:", round(pd.Series(p9.year).value_counts().loc[2005:2025].mean(), 1))

# gold
gl = cp["GLD"].dropna()
gr = gl.pct_change()
big = gl.index[(gr <= -0.035).values]
print("\nGLD <= -3.5% by year:", pd.Series(big.year).value_counts().sort_index().to_dict())
d35 = declusters(big[big < gl.index[-1]], 5, gl.index)
for h in (1, 5):
    f = fwd_ret(gl, h).reindex(d35).dropna()
    print(f"GLD h{h} after <= -3.5%: up {int((f > 0).sum())} of {len(f)}, mean {100 * f.mean():.2f}%, median {100 * f.median():.2f}%")
print("GLD off Jan high:", round(100 * (gl.iloc[-1] / gl.tail(252).max() - 1), 2), gl.tail(252).idxmax().date(), "21d", round(100 * (gl.iloc[-1] / gl.iloc[-22] - 1), 2))

# resolutions
print("\nIEF since Wed Sep 23 close:", round(100 * (cp["IEF"].iloc[-1] / cp["IEF"].loc["2026-09-23"] - 1), 2))
print("IEF since Tue Sep 22 close:", round(100 * (cp["IEF"].iloc[-1] / cp["IEF"].loc["2026-09-22"] - 1), 2))
print("since Sep 22 close: SPY", round(100 * (cp["SPY"].iloc[-1] / cp["SPY"].loc["2026-09-22"] - 1), 2),
      "TLT", round(100 * (cp["TLT"].iloc[-1] / cp["TLT"].loc["2026-09-22"] - 1), 2))
print("QTD SPY-TLT gap:", round(100 * (cp["SPY"].iloc[-1] / cp["SPY"].loc["2026-06-30"] - 1) - 100 * (cp["TLT"].iloc[-1] / cp["TLT"].loc["2026-06-30"] - 1), 2))
print("EEM Mon", round(100 * cp["EEM"].pct_change().iloc[-1], 2), "UUP Mon", round(100 * cp["UUP"].pct_change().iloc[-1], 2))
