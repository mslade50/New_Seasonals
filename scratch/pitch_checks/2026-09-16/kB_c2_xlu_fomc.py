"""c2 round 1: long XLU MOC on the decision close with XLU at a rank floor and
^TNX at its trailing-252 max, both read on the EVE close.

Live: XLU r5/r21/r63 = 1.2/1.2/0.8; ^TNX 4.996 = 252 max. Reference class over
rate-sensitive equities (XLRE, IYR, VNQ, XHB, ITB) with each member's own floor.
"""
from kB_common import *  # noqa

tnx = ser(TNX)


def gates_for(s: pd.Series) -> dict:
    idx = s.index
    t = tnx.reindex(idx).ffill()
    tmax = rolling_on_valid(t, lambda x: x.rolling(252).max())
    at_max = t >= tmax - 1e-9
    near5 = t >= tmax - 0.05
    near15 = t >= tmax - 0.15
    r5, r21, r63 = pct_rank(s, 5), pct_rank(s, 21), pct_rank(s, 63)
    f5 = (r5 <= 5) & (r21 <= 5) & (r63 <= 5)
    f10 = (r5 <= 10) & (r21 <= 10) & (r63 <= 10)
    f2163 = (r21 <= 5) & (r63 <= 5)
    return {
        "TNX@252max": at_max,
        "floor5 (5/21/63<=5)": f5,
        "floor10 (<=10)": f10,
        "r21&r63<=5": f2163,
        "r21<=10": r21 <= 10,
        "floor5 & TNX@max [LIVE]": f5 & at_max,
        "floor10 & TNX within 5bp": f10 & near5,
        "floor10 & TNX within 15bp": f10 & near15,
        "r21<=10 & TNX within 15bp": (r21 <= 10) & near15,
        "floor10 & NOT TNX near15 (complement)": f10 & ~near15,
    }


s = ser("XLU")
verify_alignment(s.index)
g = gates_for(s)
live = s.index[-1]
print("live eve read:", {k: bool(v.iloc[-1]) for k, v in g.items()})
event_state_cell("XLU", s, g, headline="floor10 & TNX within 15bp", cost_bps=3)

print("\n\n================ REFERENCE CLASS (rate-sensitive equity) ================")
for t in ["XLRE", "IYR", "VNQ", "XHB", "ITB", "XLP"]:
    st = ser(t)
    gt = gates_for(st)
    sub = {k: gt[k] for k in ["TNX@252max", "floor10 (<=10)",
                               "floor10 & TNX within 15bp",
                               "r21<=10 & TNX within 15bp"]}
    print(f"\n{t} live eve:", {k: bool(v.iloc[-1]) for k, v in sub.items()},
          f"r5/r21/r63 {pct_rank(st,5).iloc[-1]:.1f}/{pct_rank(st,21).iloc[-1]:.1f}"
          f"/{pct_rank(st,63).iloc[-1]:.1f}")
    event_state_cell(t, st, sub, hs=(2, 5, 10), headline=None, show_eps=False)
