"""C10 probe c: the MIDTERM class-wide washout rebound (found by the round-1 regime split,
so it is a SEARCHED cell). Independence check: per-year share, leave-one-year-out,
cross-name date clusters as the unit, SPY-level analogue, month mix, live-state slices."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

CLASS = ["ITA", "IHI", "IBB", "XBI", "KRE", "XME", "XRT", "XHB", "OIH", "XOP", "ITB",
         "SMH", "IYR", "XLK", "XLF", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLE"]
px = close_panel(CLASS + ["SPY"])
idx = px.index
mid = idx.year % 4 == 2
spy200 = (px["SPY"] > px["SPY"].rolling(200).mean()).values
for h in (5, 10):
    recs = []
    for t in CLASS:
        ret = vehicle_ret(px, [(t, 1.0)], h, 1)
        ok = ret.notna().values & (idx >= "2006-06-01") & mid
        m = ((pct_rank(px[t], 21) <= 2) & (pct_rank(px[t], 63) <= 10)).values & ok
        sig = idx[m]
        if len(sig) < 2:
            continue
        epi = declusters(sig, h, idx)
        drift = ret[ok].mean()
        below = (px[t] < px[t].rolling(200).mean())
        for d in epi:
            recs.append({"t": t, "d": d, "ex": 100 * (ret.loc[d] - drift), "raw": 100 * ret.loc[d],
                         "below200": bool(below.loc[d]), "spy200": bool(spy200[idx.get_loc(d)])})
    df = pd.DataFrame(recs).sort_values("d")
    print(f"\n===== h={h} midterm class episodes N={len(df)} =====")
    yr = df.groupby(df.d.dt.year)["ex"].agg(["count", "mean", "sum"])
    print(yr.round(2).to_string())
    print("LOYO pooled mean:", {y: round(df[df.d.dt.year != y].ex.mean(), 3) for y in yr.index})
    # cross-name date clusters: new cluster when gap > h trading days
    pos = df.d.map(lambda x: idx.get_loc(x)).values
    cl = np.cumsum(np.r_[1, np.diff(pos) > h])
    df["cl"] = cl
    g = df.groupby("cl").agg(start=("d", "min"), names=("t", "count"), ex=("ex", "mean"))
    w = int((g.ex > 0).sum())
    print(f"date clusters: {len(g)}  mean of cluster means {g.ex.mean():+.3f}pp  rec {w}-{len(g)-w} "
          f"sign p {sign_test(w, len(g)):.4f}  t {g.ex.mean()/(g.ex.std(ddof=1)/np.sqrt(len(g))):+.2f}")
    print(g.assign(start=g.start.dt.date).round(2).to_string())
    print("month mix of episodes:", df.d.dt.month.value_counts().sort_index().to_dict())
    print("Sep-Oct only:", round(df[df.d.dt.month.isin([9, 10])].ex.mean(), 3),
          "n", int(df.d.dt.month.isin([9, 10]).sum()))
    print("member below own 200d (ITA live):", round(df[df.below200].ex.mean(), 3), "n", int(df.below200.sum()),
          "| above:", round(df[~df.below200].ex.mean(), 3))
    print("SPY above 200d (live):", round(df[df.spy200].ex.mean(), 3), "n", int(df.spy200.sum()),
          "| SPY below:", round(df[~df.spy200].ex.mean(), 3))
    # SPY-level analogue in midterm years
    ret = vehicle_ret(px, [("SPY", 1.0)], h, 1)
    ok = ret.notna().values & mid
    m = ((pct_rank(px["SPY"], 21) <= 2) & (pct_rank(px["SPY"], 63) <= 10)).values & ok
    s = idx[m]
    if len(s):
        e = declusters(s, h, idx)
        print(f"SPY same rule midterm: n={len(e)} mean {100*ret.loc[e].mean():+.3f}% vs midterm drift "
              f"{100*ret[ok].mean():+.3f}%")
