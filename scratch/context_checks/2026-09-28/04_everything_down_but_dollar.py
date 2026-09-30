"""Monday: SPY -0.74, TLT -0.88, GLD -3.94, UUP +0.28 to a 52w high. The engine's P9b (stocks and bonds down
50bp+) is null. Does adding gold sharpen it, and what did the next sessions do?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

px = load_prices(["SPY", "TLT", "GLD", "UUP", "^VIX"])
cp = pd.DataFrame({t: px[t]["Close"].astype(float) for t in px}).loc[px["GLD"].index].dropna(subset=["SPY", "TLT", "GLD"])
idx = cp.index
r = cp.pct_change()
hist = idx[:-1]
print("today:", r.iloc[-1].round(4).to_dict(), "GLD from", idx[0].date())

def cell(mask: pd.Series, label: str, gap: int = 5) -> pd.DatetimeIndex:
    d = idx[mask.fillna(False).values]
    d = d[d < idx[-1]]
    dd = declusters(d, gap, idx)
    print(f"\n### {label}: raw {len(d)}, declustered {len(dd)}")
    rows = []
    for tk in ["SPY", "TLT", "GLD"]:
        for h in (1, 5, 21):
            f = fwd_ret(cp[tk], h)
            s = summarize(f.reindex(dd).values, f"{tk} h{h}")
            s["up"] = int((f.reindex(dd) > 0).sum())
            base = summarize(f.reindex(hist).values)
            s["base_mean"] = base["mean_pct"]
            s["base_hit"] = base["hit"]
            rows.append(s)
    show(rows)
    return dd

base = (r.SPY <= -0.005) & (r.TLT <= -0.005)
cell(base, "P9b base: SPY and TLT both <= -0.5% (GLD era)")
g2 = cell(base & (r.GLD <= -0.02), "plus GLD <= -2%")
g3 = cell(base & (r.GLD <= -0.03), "plus GLD <= -3%")
g15 = cell(base & (r.GLD <= -0.015), "plus GLD <= -1.5%")
print("GLD<=-2% episodes:", [(str(d.date()), round(100 * r.SPY[d], 2), round(100 * r.TLT[d], 2), round(100 * r.GLD[d], 2)) for d in g2])
print("GLD<=-3% episodes:", [str(d.date()) for d in g3])

uu = cp["UUP"]
uhi = rolling_on_valid(uu, lambda x: x.rolling(252).max())
u_up = r.UUP > 0
cell(base & (r.GLD <= -0.015) & u_up, "base + GLD <= -1.5% + UUP up")

# SPY h5 era for the GLD <= -2% cell and GLD h5 detail
f5 = fwd_ret(cp["SPY"], 5).reindex(g2).dropna()
show(era_split(f5.index, f5.values), "SPY h5 era, GLD<=-2% cell")
print(cluster_note(f5.index, f5.values))
g5 = fwd_ret(cp["GLD"], 5).reindex(g2).dropna()
show(era_split(g5.index, g5.values), "GLD h5 era, GLD<=-2% cell")
print("GLD h5 per episode:", [(str(d.date()), round(100 * v, 2)) for d, v in g5.items()])
