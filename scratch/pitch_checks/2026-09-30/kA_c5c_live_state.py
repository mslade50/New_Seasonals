"""C5: where does today's DX run-in (QE-5 close 09-23 -> 09-29, 4 of 5 sessions)
sit against the historical QE-5->QE run-in terciles used in kA_c5b?"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kA_common import *  # noqa

if __name__ == "__main__":
    dx, spy = own_series("DX-Y.NYB"), own_series("SPY")
    me = month_ends(spy.index)
    qe = me[me.month.isin([3, 6, 9, 12])]
    a = pd.DatetimeIndex([dx.index[dx.index.searchsorted(d, side="right") - 1] for d in qe])
    run = fwd(dx, a, 5, start=-5)
    cuts = run.quantile([1 / 3, 2 / 3])
    print(f"QE run-in tercile cuts: {100*cuts.iloc[0]:+.3f}% / {100*cuts.iloc[1]:+.3f}%  (n {len(run)})")
    print("DX last 6 closes:", ", ".join(f"{d.date()} {v:.3f}" for d, v in dx.tail(6).items()))
    live4 = dx.iloc[-1] / dx.iloc[-5] - 1
    print(f"live run-in so far (4 sessions to 09-29): {100*live4:+.3f}%")
    run4 = fwd(dx, a, 4, start=-5)
    print(f"percentile of live 4-session run-in among QE-5->QE-1 history: "
          f"{100*(run4 < live4).mean():.1f}")
