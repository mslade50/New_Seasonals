"""C10 round-2 probe b: the only strong slice in round 1 was MIDTERM years (ITA h5 +1.72pp,
17-2). Is that ITA-specific or the class-wide midterm-year washout rebound? Same rule on
the 22-ETF class inside midterm years only, control = own drift in midterm years."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
sys.path.insert(0, str(Path(__file__).resolve().parent))
from k2_common import fmt, null_maxk  # noqa

CLASS = ["ITA", "IHI", "IBB", "XBI", "KRE", "XME", "XRT", "XHB", "OIH", "XOP", "ITB",
         "SMH", "IYR", "XLK", "XLF", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLE"]
px = close_panel(CLASS + ["SPY"])
mid = pd.Series(px.index.year % 4 == 2, index=px.index)


def cell_mid(t, m, h, midterm=True):
    ret = vehicle_ret(px, [(t, 1.0)], h, 1)
    ok = ret.notna() & (px.index >= "2006-06-01") & (mid if midterm else ~mid)
    sig = px.index[(m.reindex(px.index, fill_value=False) & ok).values]
    if len(sig) < 2:
        return None
    epi = declusters(sig, h, px.index)
    drift = ret[ok & (px.index >= sig[0])].mean()
    ep = ret.loc[epi].values
    return epi, 100 * (ep - drift), 100 * ep


for h in (5, 10):
    for midterm in (True, False):
        bk = {}
        for t in CLASS:
            m = (pct_rank(px[t], 21) <= 2) & (pct_rank(px[t], 63) <= 10)
            r = cell_mid(t, m, h, midterm)
            if r is not None and len(r[1]) > 1:
                bk[t] = r[1]
                if t == "ITA":
                    epi, ex, ep = r
                    w = int((ep > 0).sum())
                    print(f"\nh={h} midterm={midterm} ITA n={len(ep)} mean {ep.mean():+.3f}% ex "
                          f"{ex.mean():+.3f}pp rec {w}-{len(ep)-w} p {sign_test(w, len(ep)):.4f}")
                    print("  ITA episodes:", ", ".join(f"{d.date()} {v:+.2f}" for d, v in zip(epi, ep)))
        allx = np.concatenate(list(bk.values()))
        print(f"  class pooled excess {allx.mean():+.3f}pp over {len(allx)} episodes; "
              f"positive {sum(1 for v in bk.values() if v.mean() > 0)}/{len(bk)}")
        null_maxk(bk, "ITA", f"h={h} {'MIDTERM' if midterm else 'non-midterm'} excess", nb=5000)
