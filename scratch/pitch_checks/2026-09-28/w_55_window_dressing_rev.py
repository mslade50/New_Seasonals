"""W55 CHECK (adapted copy of 2026-09-17/k1_c2_window_dressing.py, reversal half
only): top2-minus-bottom2 63d SPDRs from the QE close to QE+5. Arm: 2018+
reversal (short-pair payoff = -spread) >= +0.40% and September not wrong-signed."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-09-17"))
from k1_common import *  # noqa

idx = nyse_index()
px = close_panel(SPDR9 + ["SPY"]).reindex(idx)
P = px[SPDR9]
r63 = pd.DataFrame({t: P[t] / P[t].shift(63) - 1 for t in SPDR9})
A = anchors(idx, -10)
Pv, Rv = P.values, r63.values


def rev_at(rank_pos_col: str) -> list[float]:
    out = []
    for _, a in A.iterrows():
        m, s = int(a.me_pos), int(a[rank_pos_col])
        if m + 5 >= len(idx):
            out.append(np.nan)
            continue
        rr = Rv[s]
        f = Pv[m + 5] / Pv[m] - 1
        ok = ~np.isnan(rr) & ~np.isnan(f)
        o = np.argsort(rr[ok])
        out.append(f[ok][o[-2:]].mean() - f[ok][o[:2]].mean())
    return out


A["rev_sig"] = rev_at("sig_pos")   # ranking at QE-10 (the script's convention)
A["rev_qe"] = rev_at("me_pos")     # ranking at the QE close itself
rows = []
for col, tag in (("rev_sig", "rank@QE-10"), ("rev_qe", "rank@QE")):
    q = A[A.qe]
    nq = A[~A.qe]
    for lab, sub in (("QE all", q), ("QE 2018+", q[q.year >= 2018]), ("Sep QE all", q[q.month == 9]),
                     ("Sep QE 2018+", q[(q.month == 9) & (q.year >= 2018)]), ("non-QE ME all", nq),
                     ("non-QE ME 2018+", nq[nq.year >= 2018])):
        # short-pair payoff = -spread
        rows.append(stats_line(-sub[col], sub.sig_date, f"{tag} SHORT pair {lab}"))
show(rows, "W55 post-QE reversal, SHORT top2 / LONG bottom2, QE close -> QE+5 (payoff = -spread)")
s9 = A[(A.month == 9) & (A.year >= 2018)]
print("Sep 2018+ short-pair by year (rank@QE-10):",
      ", ".join(f"{d.year}:{-100*r:+.2f}" for d, r in zip(s9.me_date, s9.rev_sig)))
print("cost bar: 5x an 8 bp two-leg round trip = +0.40%")
qe10 = idx[idx.get_loc(pd.Timestamp("2026-09-30")) - 10] if pd.Timestamp("2026-09-30") in idx else idx[-1 + 3 - 10]
print(f"\nQE-10 for the 2026-09-30 anchor = {qe10.date()} (NYSE sessions; 09-30 not yet in index, so counted back from 09-25 +3)")
print((100 * r63.loc[qe10]).sort_values(ascending=False).round(2).to_string())
print(f"\nLIVE {idx[-1].date()} 63d returns (rank@QE form would use the 09-30 close):")
print((100 * r63.iloc[-1]).sort_values(ascending=False).round(2).to_string())
