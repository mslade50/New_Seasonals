"""C7 byproduct (flip, owes a flip charge): REITs vs ex-ante beta-SPY after the
QE close are NEGATIVE ungated. By-session, era blocks, September, gate
attribution for the short side, and a beta sanity check (live beta 0.30)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kB_common import nyse_index, me_table, stats_line, exante_beta  # noqa

idx = nyse_index()
px = close_panel(["IYR", "XLRE", "SPY"]).reindex(idx)
T = me_table(idx)
for reit in ("IYR", "XLRE"):
    b = exante_beta(px[reit], px["SPY"]).shift(1)
    rk = pct_rank(px[reit], 63)
    R, S = px[reit].values, px["SPY"].values
    q = T[T.qe & (T.me_pos > 300)] if reit == "IYR" else T[T.qe & (T.me_pos > idx.searchsorted(pd.Timestamp("2016-10-10")))]
    rows = []
    for d in range(1, 6):
        v = np.array([(R[m + d] / R[m + d - 1] - 1) - b.iloc[m] * (S[m + d] / S[m + d - 1] - 1)
                      if m + d < len(idx) else np.nan for m in q.me_pos])
        rows.append(stats_line(v, q.me_date, f"{reit} QE+{d} session resid"))
    show(rows, f"{reit} by session after QE (N={len(q)})")
    v5 = np.array([(R[m + 5] / R[m] - 1) - b.iloc[m] * (S[m + 5] / S[m] - 1) if m + 5 < len(idx) else np.nan
                   for m in q.me_pos])
    yrs = q.year.values
    blocks = pd.Series(v5).groupby(pd.cut(yrs, [1999, 2005, 2010, 2015, 2020, 2027]).astype(str)).agg(
        lambda x: f"{100*np.nanmean(x):+.3f}% ({int((x>0).sum())}-{int((x<=0).sum())})")
    print(blocks.to_string())
    rkq = np.array([rk.iloc[m] for m in q.me_pos])
    show([stats_line(-v5, q.me_date, "SHORT side ungated h=5"),
          stats_line(-v5[q.month.values == 9], q.me_date[q.month.values == 9], "SHORT September"),
          stats_line(-v5[rkq <= 5], q.me_date[rkq <= 5], "SHORT r63<=5"),
          stats_line(-v5[rkq > 5], q.me_date[rkq > 5], "SHORT r63>5 (complement)"),
          stats_line(-v5[q.midterm.values], q.me_date[q.midterm.values], "SHORT midterm")],
         f"{reit} short-side splits")
    print(f"{reit} beta at recent QEs:", {str(idx[m].date()): round(b.iloc[m], 2) for m in q.me_pos[-8:]},
          f"live {exante_beta(px[reit], px['SPY']).iloc[-1]:.2f}")
