"""C5 follow-up: is the September (live-month) post-close USDJPY record anything
but the dollar? Every entry (QE-1, QE) x exit (QE+3..+5) cell for September,
USDJPY vs DX vs USDJPY minus beta*DX (beta = full-sample daily OLS)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kB_common import *  # noqa

px = nyse_panel(["JPY=X", "DX-Y.NYB"], ffill=("JPY=X", "DX-Y.NYB"))
idx = px.index
J, D = px["JPY=X"].values, px["DX-Y.NYB"].values
dj, dd = px["JPY=X"].pct_change(fill_method=None), px["DX-Y.NYB"].pct_change(fill_method=None)
ok = dj.notna() & dd.notna()
beta = float(np.cov(dj[ok], dd[ok])[0, 1] / dd[ok].var())
print(f"USDJPY daily beta on DX = {beta:.3f}")
me = month_end_positions(idx)
out = []
for grp, months in (("September", (9,)), ("Mar+Sep", (3, 9)), ("Jun+Dec", (6, 12)), ("ordinary ME", (1, 2, 4, 5, 7, 8, 10, 11))):
    sel = [m for m in me if idx[m].month in months and m + 5 < len(idx)]
    for e in (-1, 0):
        for x in (3, 4, 5):
            j = np.array([span_ret(J, m + e, m + x) for m in sel])
            d = np.array([span_ret(D, m + e, m + x) for m in sel])
            rj = cell(j, f"{grp} QE{e:+d}->QE+{x} USDJPY")
            rr = cell(j - beta * d, "resid")
            rj["dx_pct"] = round(100 * np.nanmean(d), 3)
            rj["resid_pct"] = round(rr["mean_pct"], 3)
            rj["resid_rec"] = rr["rec"]
            rj["resid_p"] = rr["sign_p"]
            out.append(rj)
show(out, "USDJPY vs DX vs beta-residual after the book close")
