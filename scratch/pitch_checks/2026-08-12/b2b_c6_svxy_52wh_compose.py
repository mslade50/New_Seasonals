"""C6 round 2: does the PPI-eve cell COMPOSE with SVXY-at-a-52w-high, or is
one of them the other?

b2 computed the 52w-high mask on a close_panel index that included ^VIX, which
carries 3 sessions SVXY does not. Those NaNs poison every rolling(252) window
that spans them, so the mask read False on a day SVXY closed EXACTLY at its
52w high. Recomputed here on SVXY's own bar index.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

mp = pd.read_parquet(PRICES_PATH)
mp["date"] = pd.to_datetime(mp["date"])
sv = mp[mp.ticker == "SVXY"].set_index("date").sort_index()["Close"]
sp = mp[mp.ticker == "SPY"].set_index("date").sort_index()["Close"]
idx = sv.index                                  # SVXY's OWN calendar

hi52 = sv >= sv.rolling(252).max() * 0.999
print(f"2026-08-11: SVXY {sv.iloc[-1]:.2f}, 52w high "
      f"{sv.rolling(252).max().iloc[-1]:.2f}, at-high = {bool(hi52.iloc[-1])}")

ev = load_events(["ppi"])
PPI = pd.DatetimeIndex(sorted(ev.loc[ev.event == "ppi", "date"].unique()))
anch = []
for d in PPI:
    loc = idx.searchsorted(pd.Timestamp(d))
    if loc >= len(idx):
        continue
    p = loc - 2
    if 0 <= p < len(idx):
        anch.append(idx[p])
A2 = pd.DatetimeIndex(sorted(set(anch)))

r = fwd_lag(sv, 1, 1)
m = fwd_lag(sp.reindex(idx), 1, 1)
base = r.dropna()
hi_days = idx[hi52.fillna(False).values]

rows = []
for lbl, t in [("PPI-eve ALL", A2),
               ("PPI-eve AND at 52w high (TODAY'S cell)",
                A2.intersection(hi_days)),
               ("PPI-eve, NOT at 52w high", A2.difference(hi_days)),
               ("at 52w high alone, any day", hi_days),
               ("at 52w high, NOT a PPI eve", hi_days.difference(A2))]:
    v = r.loc[r.index.intersection(t)].dropna()
    if len(v) < 3:
        rows.append({"label": lbl, "n": len(v)})
        continue
    span = (idx >= v.index[0]) & (idx <= v.index[-1])
    b = r[span].dropna()
    w = int((v < 0).sum())                       # the SHORT wins when SVXY falls
    rows.append({"label": lbl, "n": len(v),
                 "svxy_mean_pct": round(100 * v.mean(), 3),
                 "excess_pct": round(100 * (v.mean() - b.mean()), 3),
                 "short_hit": round(100 * (v < 0).mean(), 1),
                 "sign_p_short": round(sign_test(w, len(v),
                                                 float((b < 0).mean())), 4)})
show(rows, "does the PPI eve compose with the 52w-high state?")

joint = A2.intersection(hi_days)
only_hi = hi_days.difference(A2)
only_ppi = A2.difference(hi_days)
print(f"\n  overlap: {len(joint)} of {len(A2)} PPI eves are at-a-high "
      f"({100*len(joint)/len(A2):.0f}%), and {len(joint)} of {len(hi_days)} "
      f"at-a-high days are a PPI eve ({100*len(joint)/len(hi_days):.0f}%)")
jv = r.loc[r.index.intersection(joint)].dropna()
pv = r.loc[r.index.intersection(only_ppi)].dropna()
hv = r.loc[r.index.intersection(only_hi)].dropna()
print(f"  additivity: joint {100*jv.mean():+.3f}%  vs  "
      f"PPI-only {100*pv.mean():+.3f}% + hi-only {100*hv.mean():+.3f}% "
      f"= {100*(pv.mean()+hv.mean()):+.3f}%  -> the joint is NOT the sum, "
      f"gap {100*(jv.mean()-pv.mean()-hv.mean()):+.3f}pp")

# beta-neutral version of the joint cell, since that is the registry's rule
d = pd.concat([r, m], axis=1, keys=["s", "mkt"]).dropna()
b = np.polyfit(d["mkt"], d["s"], 1)
resid = d["s"] - (b[0] * d["mkt"] + b[1])
base_neg = float((resid < 0).mean())
print(f"\n  UNCONDITIONAL residual-negative rate = {100*base_neg:.1f}% "
      f"(N={len(resid)}) -- this, not 0.5, is the null for the sign test")
for lbl, t in [("PPI-eve ALL", A2), ("joint (today)", joint),
               ("52wh alone", hi_days)]:
    rv = resid.loc[resid.index.intersection(t)]
    if len(rv) < 3:
        continue
    w = int((rv < 0).sum())
    print(f"  beta-neutral residual, {lbl:16s}: {100*rv.mean():+.3f}%  "
          f"short record {w}-{len(rv)-w} ({100*w/len(rv):.1f}%)  "
          f"sign p vs 0.5 = {sign_test(w, len(rv)):.4f}  "
          f"sign p vs base rate = {sign_test(w, len(rv), base_neg):.4f}")
