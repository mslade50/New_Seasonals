"""C2 round 1: SHORT TLT from the ME-0 close, h=1..10, gated on ^TNX within 1%
of its trailing-252 max at the ME close. Pre-specified sign SHORT.
Controls: all month-ends ungated (parent), all days, all TNX-gated days (the
state control: yields at a high = TLT in a downtrend, so the state alone may
pay the short), start-offset placebo ladder, quarter-end split, NFP split."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kA_common import *  # noqa

HS = (1, 2, 3, 5, 7, 10)

if __name__ == "__main__":
    tlt, ief, tnx = own_series("TLT"), own_series("IEF"), own_series("^TNX")
    tmax = rolling_on_valid(tnx, lambda x: x.rolling(252).max())
    gate_s = (tnx >= 0.99 * tmax)
    for name, c in (("TLT", tlt), ("IEF", ief)):
        me = month_ends(c.index)
        g = asof_on(gate_s.astype(float), c.index).fillna(0).astype(bool)
        meg = me[g.reindex(me).values]
        print(f"\n######## {name}: {len(me)} month-ends {me[0].date()}..{me[-1].date()}, "
              f"gated {len(meg)} ########")
        rows = []
        for h in HS:
            s_me = -fwd(c, me, h)
            s_g = -fwd(c, meg, h)
            s_all = -all_windows(c, h)
            gd = c.index[g.values]
            s_gall = s_all.reindex(gd).dropna()
            s_gnon = s_gall.drop(meg, errors="ignore")
            qe = s_me[s_me.index.month.isin([3, 6, 9, 12])]
            qeg = s_g[s_g.index.month.isin([3, 6, 9, 12])]
            rows += [rec(s_me, f"h={h} PARENT all ME"),
                     rec(s_g, f"h={h} GATED ME (TNX<=1% of 252max)"),
                     rec(s_gnon, f"h={h} CTRL gated non-ME days"),
                     rec(s_all, f"h={h} CTRL all days"),
                     rec(qe, f"h={h} QE-only ungated"),
                     rec(qeg, f"h={h} QE-only gated")]
            d, t = welch(s_me, s_all)
            d2, t2 = welch(s_g, s_gnon)
            rows.append({"label": f"   h={h} parent-allday {d:+.3f}pp t{t:+.2f} | gated-vs-gated-days {d2:+.3f}pp t{t2:+.2f}"})
        show(rows, f"{name} short from ME-0 close")

        # start-offset placebo ladder at h=5 (does the ME-0 start matter?)
        lad = []
        for st in (-10, -7, -5, -3, -2, -1, 0, 1, 2, 3, 5):
            v = -fwd(c, me, 5, start=st)
            lad.append(rec(v, f"start ME{st:+d}, h=5"))
        show(lad, f"{name} start-offset placebo ladder (short, h=5, all ME)")

        # eras / midterm on the parent and gated, h=5 and h=10
        for h in (5, 10):
            s_me, s_g = -fwd(c, me, h), -fwd(c, meg, h)
            er = eras(s_me, f"h={h} parent", ("2013-01-01", "2018-01-01", "2020-01-01"))
            er += eras(s_g, f"h={h} gated", ("2018-01-01",))
            mid = s_me.index.year % 4 == 2
            er += [rec(s_me[mid], f"h={h} parent midterm"), rec(s_me[~mid], f"h={h} parent non-mid")]
            midg = s_g.index.year % 4 == 2
            er += [rec(s_g[midg], f"h={h} gated midterm"), rec(s_g[~midg], f"h={h} gated non-mid")]
            er += [rec(s_me[s_me.index.month == 9], f"h={h} parent September"),
                   rec(s_g[s_g.index.month == 9], f"h={h} gated September")]
            show(er, f"{name} era / cycle split h={h}")
            print("  parent concentration:", cluster_note(s_me.index, s_me.values))
            if len(s_g):
                print("  gated concentration:", cluster_note(s_g.index, s_g.values))
                print("  gated dates:", ", ".join(f"{d.date()}:{100*v:+.2f}" for d, v in s_g.items()))
