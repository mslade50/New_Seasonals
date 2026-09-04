"""C6 round 2, decisive: run the PLACEBO ANCHOR LADDER on the beta-neutral
RESIDUAL statistic itself, not just on its mean.

b2b found the one number that could have carried C6: on the PPI eve the
SPY-beta-neutral SVXY residual is negative 99 of 177 times (55.9%) against a
46.1% unconditional base rate, sign p = 0.0053. That is the strongest reading
of the morning and it is exactly the kind of number the UNG registry entry
exists to discipline. So: compute the SAME statistic at 20 nonsense anchors.

If a nonsense anchor also produces sign p < 0.01, the anchor is not what is
generating the number, and the cell is a filter that does not filter.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

mp = pd.read_parquet(PRICES_PATH)
mp["date"] = pd.to_datetime(mp["date"])
sv = mp[mp.ticker == "SVXY"].set_index("date").sort_index()["Close"]
sp = mp[mp.ticker == "SPY"].set_index("date").sort_index()["Close"]
idx = sv.index

r = fwd_lag(sv, 1, 1)
m = fwd_lag(sp.reindex(idx), 1, 1)
d = pd.concat([r, m], axis=1, keys=["s", "mkt"]).dropna()
beta, alpha = np.polyfit(d["mkt"], d["s"], 1)
resid = d["s"] - (beta * d["mkt"] + alpha)
base_neg = float((resid < 0).mean())
print(f"beta={beta:.3f} alpha={100*alpha:+.3f}%/day  "
      f"unconditional residual-negative rate={100*base_neg:.1f}%  N={len(resid)}")

ev = load_events(["ppi"])
PPI = pd.DatetimeIndex(sorted(ev.loc[ev.event == "ppi", "date"].unique()))


def anchors(k):
    out = []
    for dt in PPI:
        loc = idx.searchsorted(pd.Timestamp(dt))
        if loc >= len(idx):
            continue
        p = loc - k
        if 0 <= p < len(idx):
            out.append(idx[p])
    return pd.DatetimeIndex(sorted(set(out)))


rows = []
for k in range(-8, 13):
    t = resid.index.intersection(anchors(k))
    if len(t) < 20:
        continue
    rv = resid.loc[t]
    w = int((rv < 0).sum())
    rows.append({"k": k, "n": len(rv),
                 "resid_mean_pct": round(100 * rv.mean(), 3),
                 "short_wins": w, "short_hit": round(100 * w / len(rv), 1),
                 "sign_p_vs_base": round(sign_test(w, len(rv), base_neg), 4),
                 "real": "<<<< REAL" if k == 2 else ""})
show(rows, "placebo ladder on the beta-neutral residual sign test")

s = pd.DataFrame(rows).set_index("k")
real_p = s.loc[2, "sign_p_vs_base"]
better = s[s["sign_p_vs_base"] < real_p]
tied = s[(s["sign_p_vs_base"] < 0.05) & (s.index != 2)]
print(f"\n  real anchor k=2: sign p = {real_p:.4f}")
print(f"  NONSENSE anchors with a SMALLER p: {len(better)} "
      f"-> {dict(better['sign_p_vs_base'])}")
print(f"  NONSENSE anchors clearing p<0.05 at all: {len(tied)} "
      f"-> {dict(tied['sign_p_vs_base'])}")
print(f"  rank of the real anchor: {int((s['sign_p_vs_base'] <= real_p).sum())}"
      f"/{len(s)}  -> empirical p = "
      f"{(s['sign_p_vs_base'] <= real_p).sum()/len(s):.3f}")
print(f"  ladder mean residual {s['resid_mean_pct'].mean():+.3f}%, "
      f"sd {s['resid_mean_pct'].std():.3f}%; real is "
      f"{(s.loc[2,'resid_mean_pct']-s['resid_mean_pct'].mean())/s['resid_mean_pct'].std():+.2f} sd")

# and the honest tradeable version: the residual needs a SPY hedge leg
print("\n=== what trading the residual actually costs ===")
print(f"  the residual is not an instrument. Capturing it = SHORT SVXY + "
      f"SHORT {beta:.2f}x SPY notional (full-sample beta), or {1.484:.2f}x "
      f"on the post-2018 beta -- an unstable hedge ratio.")
print("  2 legs: short-SVXY round trip ~25 bps (borrow is the binding cost) "
      "+ SPY round trip ~4 bps = ~29 bps")
print(f"  residual mean on the real anchor = {100*resid.loc[resid.index.intersection(anchors(2))].mean():+.3f}% "
      f"= {abs(100*100*resid.loc[resid.index.intersection(anchors(2))].mean()):.1f} bps "
      f"-> {abs(100*100*resid.loc[resid.index.intersection(anchors(2))].mean())/29:.1f}x cost (need >=5x)")
