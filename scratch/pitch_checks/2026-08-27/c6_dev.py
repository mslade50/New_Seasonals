"""C6 round 3 (dev) -- horizon from the table, entry form as WHOLE variants,
exit sensitivity, and the loser paths that `what_kills_it` has to quote.

No marginal-fill decompositions (registry rule): variants are compared whole,
fill rate plus conditional stats.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np, pandas as pd

ohlc = load_prices(["GDX"])["GDX"]
px = close_panel(["GDX"])
g = px["GDX"]
r21 = g / g.shift(21) - 1.0
rk = rolling_on_valid(r21, lambda x: x.rolling(252).rank(pct=True) * 100.0)
r1 = g.pct_change(fill_method=None)
mask = ((rk >= 99) & (r1 <= -0.02)).fillna(False)
atr = pd.Series(wilder_atr(ohlc["High"], ohlc["Low"], ohlc["Close"], 14),
                index=ohlc.index).reindex(px.index)

print("TODAY: GDX close %.2f  Wilder-14 ATR %.3f (%.2f%% of price)  "
      "r21 %+.1f%%  rank %.1f  1d %+.2f%%"
      % (g.iloc[-1], atr.iloc[-1], 100 * atr.iloc[-1] / g.iloc[-1],
         100 * r21.iloc[-1], rk.iloc[-1], 100 * r1.iloc[-1]))

epi = declusters(px.index[mask], 10, px.index)
epi_hist = epi[epi < px.index[-1]]
print("historical episodes:", [str(d.date()) for d in epi_hist])

# ---------- (a) horizon table ----------
print("\n########## (a) horizon scan, episodes ##########")
show(horizon_scan(px, epi_hist, [("GDX", 1.0)], hs=tuple(range(1, 11)),
                  min_gap=10), "GDX h=1..10")

# ---------- (b) entry form, WHOLE variants ----------
print("\n########## (b) entry form (whole variants, h=5) ##########")
idx = px.index
pos = pd.Series(range(len(idx)), index=idx)
O, H, L, C = (ohlc["Open"].reindex(idx), ohlc["High"].reindex(idx),
              ohlc["Low"].reindex(idx), ohlc["Close"].reindex(idx))

def run_variant(k_atr, h=5, window=1):
    """k_atr None -> MOC at close D+1. Else a LIMIT at signal close - k*ATR
    live for `window` sessions from D+1; unfilled = no trade."""
    fills, misses = [], 0
    for d in epi_hist:
        p = pos[d]
        if p + 1 + h >= len(idx):
            continue
        if k_atr is None:
            e = C.iloc[p + 1]
            x = C.iloc[p + 1 + h]
            fills.append((d, e, x / e - 1.0, 0))
            continue
        lim = C.iloc[p] - k_atr * atr.iloc[p]
        hit = None
        for j in range(1, window + 1):
            if p + j >= len(idx):
                break
            if L.iloc[p + j] <= lim:
                hit = (p + j, min(lim, O.iloc[p + j]))
                break
        if hit is None:
            misses += 1
            continue
        fp, e = hit
        if fp + h >= len(idx):
            continue
        fills.append((d, e, C.iloc[fp + h] / e - 1.0, fp - p))
    v = np.array([f[2] for f in fills])
    n = len(fills) + misses
    return {"n_signals": n, "n_fills": len(fills),
            "fill_rate": round(100 * len(fills) / max(1, n), 1),
            **{k: (round(x, 3) if isinstance(x, float) else x)
               for k, x in summarize(v, "").items() if k != "label"}}

rows = []
for lbl, kw in (("MOC close D+1", dict(k_atr=None)),
                ("LIMIT close-0.25ATR, 1d", dict(k_atr=0.25, window=1)),
                ("LIMIT close-0.25ATR, 2d", dict(k_atr=0.25, window=2)),
                ("LIMIT close-0.50ATR, 2d", dict(k_atr=0.50, window=2)),
                ("LIMIT close-1.00ATR, 2d", dict(k_atr=1.00, window=2))):
    r = run_variant(**kw); r["label"] = lbl; rows.append(r)
print(pd.DataFrame(rows)[["label", "n_signals", "n_fills", "fill_rate", "n",
                          "mean_pct", "median_pct", "hit", "worst_pct",
                          "best_pct"]].to_string(index=False))

# ---------- (c) exit sensitivity ----------
print("\n########## (c) exit sensitivity: time-only vs a target/stop ##########")
for tgt, stp in ((None, None), (2.0, None), (None, 2.0), (2.0, 2.0), (3.0, 2.0)):
    vals = []
    for d in epi_hist:
        p = pos[d]
        if p + 6 >= len(idx):
            continue
        e = C.iloc[p + 1]
        a = atr.iloc[p]
        out = None
        for j in range(p + 2, p + 7):
            if stp is not None and L.iloc[j] <= e - stp * a:
                out = (min(e - stp * a, O.iloc[j]) / e - 1.0); break
            if tgt is not None and H.iloc[j] >= e + tgt * a:
                out = (max(e + tgt * a, O.iloc[j]) / e - 1.0); break
        if out is None:
            out = C.iloc[p + 6] / e - 1.0
        vals.append(out)
    v = np.array(vals)
    print(f"  tgt={tgt} stop={stp}: N={len(v)} mean {100*v.mean():+.3f}%  "
          f"hit {100*(v>0).mean():.1f}%  worst {100*v.min():+.2f}%  "
          f"best {100*v.max():+.2f}%")

# ---------- (d) loser paths: the near-miss population, since the cell is 6-0 ----
print("\n########## (d) what kills it -- the adjacent populations ##########")
ret5 = fwd_lag(g, 5, 1)
near = {
    "rank>=99 & 1d in (-2,-1.5]": ((rk >= 99) & (100*r1 > -2) & (100*r1 <= -1.5)),
    "rank>=95 & 1d in (-3,-2] (TODAY's depth, wide rank)":
        ((rk >= 95) & (100*r1 > -3) & (100*r1 <= -2)),
    "MAG r21>=30% & 1d<=-2": ((r21 >= 0.30) & (r1 <= -0.02)),
}
for nm, m in near.items():
    m = m.fillna(False)
    e = declusters(px.index[m.values & ret5.notna().values], 10, px.index)
    v = ret5.loc[e].values
    print(f"  {nm}: N={len(v)} mean {100*v.mean():+.3f}% hit {100*(v>0).mean():.1f}% "
          f"worst {100*v.min():+.2f}% on "
          f"{e[int(np.argmin(v))].date() if len(v) else '-'}")

print("\nepisode paths h=5 (cum % from the D+1 entry close):")
print((100 * episode_paths(px, epi_hist, [("GDX", 1.0)], 5)).round(2).to_string())
print("\nintra-hold worst DRAWDOWN from the entry close, per episode (h=5):")
for d in epi_hist:
    p = pos[d]
    e = C.iloc[p + 1]
    lo = L.iloc[p + 2:p + 7].min()
    print(f"  {d.date()}: entry {e:.2f}  worst low {lo:.2f}  "
          f"= {100*(lo/e-1):+.2f}%  ({(lo/e-1)/(atr.iloc[p]/e):+.2f} ATR)")
