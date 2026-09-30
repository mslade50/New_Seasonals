"""C5B round 2 (cont.): attribution of the DX leg with parent = GLD dd>=15, and
the live slice (GLD < 200d, 2018+, NFP inside the hold) of the pre-spec cell.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

gp = close_panel(["GLD"]).dropna()
idx = gp.index
g = gp["GLD"]
dxs = close_panel(["DX-Y.NYB"])["DX-Y.NYB"].dropna()
dxr = pct_rank(dxs, 21).reindex(idx, method="ffill", limit=2)
gdd = 100 * (1 - g / g.rolling(252).max())
below200 = g < g.rolling(200).mean()
gate = (gdd >= 15) & (dxr >= 85)


def ep(ret, m, gap):
    s = idx[m.reindex(idx, fill_value=False).values & ret.notna().values]
    e = declusters(s, gap, idx)
    return e, ret.loc[e].values


def line(v, lbl, base):
    r = summarize(v, lbl)
    if len(v):
        w = int((v > 0).sum())
        r["rec"] = f"{w}-{len(v)-w}"
        r["sign_p"] = round(sign_test(w, len(v)), 4)
        r["xs_pp"] = round(r["mean_pct"] - 100 * base, 3)
    return r


for h in (5, 10):
    ret = fwd_lag(g, h)
    base = ret.mean()
    pe = declusters(idx[(gdd >= 15).values], h, idx)
    ce = declusters(idx[gate.values], h, idx)
    pm = pd.Series(False, index=idx); pm.loc[pe] = True
    cm = pd.Series(False, index=idx); cm.loc[ce] = True
    fr = filter_vs_reanchor(ret, pm, cm, idx, 21, f"parent dd>=15 -> child +DX r21>=85, h={h}")
    if fr["shifts"]:
        rn = reanchor_null(ret, [a for a, _, _ in fr["pairs"]], fr["shifts"], idx,
                           ret.reindex(ce).mean())
        print("  reanchor_null:", {k: round(v, 4) if isinstance(v, float) else v for k, v in rn.items()})
    e, v = ep(ret, gate, h)
    b2 = below200.reindex(e).values
    post = e.year >= 2018
    nfp = event_in_window(e, idx, h, 1, ("nfp",))
    rows = [line(v, f"pre-spec all (N={len(e)})", base),
            line(v[b2], "GLD<200d (live)", base),
            line(v[b2 & post], "GLD<200d & 2018+", base),
            line(v[b2 & ~post], "GLD<200d & pre-2018", base),
            line(v[nfp], "NFP inside hold (live)", base),
            line(v[b2 & nfp], "GLD<200d & NFP inside", base)]
    show(rows, f"h={h} live-slice cuts")
    # below-200d parent without DX gate, to see whether DX still adds in the live regime
    e3, v3 = ep(ret, (gdd >= 15) & below200 & (dxr < 85), h)
    print("  dd>=15 & GLD<200d & DX<85 (complement in live regime):", line(v3, "", base))
