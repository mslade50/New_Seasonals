"""c4 round 1: long EWJ QE-9 -> QE in March/September (Japanese fiscal
year-end / half-year dividend-reinvestment demand) vs June/December vs all
month-ends. Yen leg (JPY=X = USDJPY), local Nikkei, sub-window decomposition,
reversal after QE."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from k1_common import *  # noqa

idx = nyse_index()
raw = close_panel(["EWJ", "EFA", "SPY", "JPY=X", "^N225"])
px = raw.reindex(idx.union(raw.index)).ffill().reindex(idx)  # foreign calendars ffilled onto NYSE
E = px["EWJ"].values
A = anchors(idx, -10)
A = A[A.me_pos + 5 < len(idx)].copy()


def win(col, a_off, b_off):
    v = px[col].values
    return [v[m + b_off] / v[m + a_off] - 1 for m in A.me_pos]


A["ewj"] = win("EWJ", -9, 0)
A["efa"] = win("EFA", -9, 0)
A["usdjpy"] = win("JPY=X", -9, 0)
A["n225"] = win("^N225", -9, 0)
A["hedged"] = A.ewj + A.usdjpy  # long EWJ + long USDJPY ~ local-currency
A["ewj_efa"] = A.ewj - A.efa
A["early"] = win("EWJ", -9, -3)
A["last3"] = win("EWJ", -3, 0)
A["last2"] = win("EWJ", -2, 0)
A["post5"] = win("EWJ", 0, 5)
A = A.dropna(subset=["ewj"])
A["ms"] = A.month.isin([3, 9])
A["jd"] = A.month.isin([6, 12])

allw = pd.Series(E).shift(-9) / pd.Series(E) - 1
allw = allw.dropna()


def cells(col, title):
    rows = [stats_line(A[A.ms][col], A[A.ms].sig_date, "Mar+Sep"),
            stats_line(A[A.month == 3][col], A[A.month == 3].sig_date, "March"),
            stats_line(A[A.month == 9][col], A[A.month == 9].sig_date, "September"),
            stats_line(A[A.jd][col], A[A.jd].sig_date, "Jun+Dec"),
            stats_line(A[~A.qe][col], A[~A.qe].sig_date, "non-QE month-ends"),
            stats_line(A[col], A.sig_date, "all month-ends")]
    show(rows, title)


cells("ewj", "EWJ QE-9 -> QE (USD)")
print(f"EWJ all 9-session windows: mean {100*allw.mean():+.3f}% hit {100*(allw>0).mean():.1f}% n {len(allw)}")
ms, ctl = A[A.ms], A[~A.ms]
print(f"Mar+Sep minus all other month-ends: {100*(ms.ewj.mean()-ctl.ewj.mean()):+.3f}pp  "
      f"welch t {(ms.ewj.mean()-ctl.ewj.mean())/np.sqrt(ms.ewj.var()/len(ms)+ctl.ewj.var()/len(ctl)):+.2f}")
cells("ewj_efa", "EWJ minus EFA, same window")
cells("hedged", "EWJ + USDJPY (yen-hedged proxy)")
cells("usdjpy", "USDJPY leg (positive = yen weaker)")
cells("n225", "Nikkei 225 local price (drops on ex-div inside Mar/Sep windows)")
print(f"corr(EWJ, USDJPY) in Mar+Sep windows: {ms[['ewj','usdjpy']].corr().iloc[0,1]:+.3f}; all: {A[['ewj','usdjpy']].corr().iloc[0,1]:+.3f}")

show([stats_line(ms[ms.year < 2013].ewj, ms[ms.year < 2013].sig_date, "Mar+Sep pre-2013"),
      stats_line(ms[ms.year >= 2013].ewj, ms[ms.year >= 2013].sig_date, "Mar+Sep 2013+"),
      stats_line(ms[ms.year < 2018].ewj, ms[ms.year < 2018].sig_date, "Mar+Sep pre-2018"),
      stats_line(ms[ms.year >= 2018].ewj, ms[ms.year >= 2018].sig_date, "Mar+Sep 2018+"),
      stats_line(ctl[ctl.year >= 2013].ewj, ctl[ctl.year >= 2013].sig_date, "other ME 2013+"),
      stats_line(ms[ms.midterm].ewj, ms[ms.midterm].sig_date, "Mar+Sep midterm"),
      stats_line(A[(A.month == 9) & A.midterm].ewj, A[(A.month == 9) & A.midterm].sig_date, "Sep midterm"),
      stats_line(A[(A.month == 9) & (A.year >= 2018)].ewj, A[(A.month == 9) & (A.year >= 2018)].sig_date, "Sep 2018+"),
      stats_line(A[(A.month == 9) & (A.year < 2018)].ewj, A[(A.month == 9) & (A.year < 2018)].sig_date, "Sep pre-2018")],
     "era / midterm (EWJ USD)")

show([stats_line(ms.early, ms.sig_date, "Mar+Sep QE-9->QE-3"),
      stats_line(ctl.early, ctl.sig_date, "other ME-9->ME-3"),
      stats_line(ms.last3, ms.sig_date, "Mar+Sep QE-3->QE (ex-div run-in)"),
      stats_line(ctl.last3, ctl.sig_date, "other ME-3->ME"),
      stats_line(ms.last2, ms.sig_date, "Mar+Sep QE-2->QE"),
      stats_line(ctl.last2, ctl.sig_date, "other ME-2->ME"),
      stats_line(ms.post5, ms.sig_date, "Mar+Sep QE->QE+5 (reversal?)"),
      stats_line(ctl.post5, ctl.sig_date, "other ME->ME+5")],
     "mechanism: where in the window")
s = A[A.month == 9]
print("Sep:", ", ".join(f"{d.year}:{100*r:+.2f}" for d, r in zip(s.me_date, s.ewj)))
s = A[A.month == 3]
print("Mar:", ", ".join(f"{d.year}:{100*r:+.2f}" for d, r in zip(s.me_date, s.ewj)))
print("concentration Mar+Sep:", cluster_note(pd.DatetimeIndex(ms.sig_date), ms.ewj.values))
print(f"cost: ~3bp RT; Mar+Sep mean {1e4*ms.ewj.mean():.1f}bp; excess over other ME "
      f"{1e4*(ms.ewj.mean()-ctl.ewj.mean()):.1f}bp -> {(ms.ewj.mean()-ctl.ewj.mean())*1e4/3:.1f}x")
ewj = load_prices(["EWJ"])["EWJ"]["Close"]
print(f"LIVE EWJ {ewj.index[-1].date()} r5 {pct_rank(ewj,5).iloc[-1]:.1f} r21 {pct_rank(ewj,21).iloc[-1]:.1f} "
      f"r63 {pct_rank(ewj,63).iloc[-1]:.1f} z10 {zscore(ewj).iloc[-1]:.2f} off-high {100*(ewj.iloc[-1]/ewj.iloc[-252:].max()-1):.2f}%")
