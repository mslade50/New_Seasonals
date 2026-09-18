"""C2 probe d: widen the complex reference class with every cached sub-industry ETF
(XLB+COPX, XLRE complex, XLC/XLK/SMH, XLF/KRE/IYR) and re-run max-of-K at h10 on
eq$ and beta-neutral pairs, plus the trailing-63d-beta form on all members."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
sys.path.insert(0, str(Path(__file__).resolve().parent))
from k2_common import cell, null_maxk  # noqa

CPLX = {"XLV": ["XLV", "IBB", "XBI", "IHI"], "XLE": ["XLE", "XOP", "OIH"],
        "XLY": ["XLY", "XRT", "XHB", "ITB"], "XLI": ["XLI", "ITA", "IYT"],
        "XLB": ["XLB", "XME", "GDX", "COPX"], "XLRE": ["XLRE", "IYR", "VNQ"],
        "XLK": ["XLK", "SMH", "XLC"], "XLF": ["XLF", "KRE", "IYR"]}
allt = sorted({t for v in CPLX.values() for t in v} | {"SPY"})
px = close_panel(allt)
r5 = {t: pct_rank(px[t], 5) for t in px.columns}
dr = px.pct_change()
for h in (5, 10):
    bk, bn = {}, {}
    for s, mem in CPLX.items():
        cnt = sum((r5[t] <= 5).astype(int) for t in mem)
        nav = sum(r5[t].notna().astype(int) for t in mem)
        m = (r5[s] <= 1) & (cnt >= np.ceil(0.75 * nav)) & (nav >= 3)
        bs = float(dr[s].cov(dr["SPY"]) / dr["SPY"].var())
        a = cell(px, m, [(s, 1.0), ("SPY", -1.0)], h)
        b = cell(px, m, [(s, 1.0), ("SPY", -bs)], h)
        if a.get("n", 0) > 2:
            bk[s], bn[s] = a["ex"], b["ex"]
    null_maxk(bk, "XLV", f"h={h} pair eq$ (widened)")
    null_maxk(bn, "XLV", f"h={h} pair beta-neutral (widened)")
