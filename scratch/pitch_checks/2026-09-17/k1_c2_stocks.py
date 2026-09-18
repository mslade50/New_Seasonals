"""c2 mechanism test on the tape single-stock cross-section: top-decile minus
bottom-decile 63d return, QE-9 -> QE vs non-QE month-ends vs all days,
plus the QE -> QE+5 reversal. Survivorship: today's survivors only."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from k1_common import *  # noqa

ROOT = Path(__file__).resolve().parents[3]
NON_STOCK = set("""CEF DBC DIA DX-Y.NYB EEM EFA EWJ EWZ FXI GDX GLD HYG IBB IEF IHI ITA
ITB IWM IYR KRE LQD OIH QQQ SLV SMH SPY SVXY TLT UNG USO UUP UVXY VNQ XBI XHB
XLB XLC XLE XLF XLI XLK XLP XLRE XLU XLV XLY XME XOP XRT""".split())
tape = json.load(open(ROOT / "data" / "pitch_tape.json"))["tickers"]
names = [t for t in tape if t not in NON_STOCK and not t.startswith("^")]
idx = nyse_index()
P = close_panel(names).reindex(idx)
print(f"stocks: {P.shape[1]}")
r63 = P / P.shift(63) - 1
H = 9


def dec_spread(sig_pos, ent_off, ex_off):
    rr = r63.values[sig_pos]
    a, b = sig_pos + ent_off, sig_pos + ex_off
    if b >= len(idx):
        return np.nan, 0
    f = P.values[b] / P.values[a] - 1
    ok = ~np.isnan(rr) & ~np.isnan(f)
    n = ok.sum()
    if n < 50:
        return np.nan, n
    k = max(1, n // 10)
    o = np.argsort(rr[ok])
    return f[ok][o[-k:]].mean() - f[ok][o[:k]].mean(), n


A = anchors(idx, -10)
A[["ret", "n"]] = [dec_spread(int(s), 1, 10) for s in A.sig_pos]
A["rev"] = [dec_spread(int(s), 10, 15)[0] for s in A.sig_pos]
A = A.dropna(subset=["ret"])
allpos = np.arange(260, len(idx) - 15, 3)
alld = np.array([dec_spread(int(s), 1, 10)[0] for s in allpos])
allr = np.array([dec_spread(int(s), 10, 15)[0] for s in allpos])
print(f"names in cross-section at first anchor {int(A.n.iloc[0])}, last {int(A.n.iloc[-1])}")

q, nq = A[A.qe], A[~A.qe]
show([stats_line(q.ret, q.sig_date, "stocks QE-9->QE"),
      stats_line(nq.ret, nq.sig_date, "stocks non-QE ME-9->ME"),
      stats_line(A[A.month == 9].ret, A[A.month == 9].sig_date, "stocks Sep"),
      stats_line(A[A.month == 12].ret, A[A.month == 12].sig_date, "stocks Dec"),
      stats_line(A[(A.month == 9) & (A.year >= 2018)].ret, A[(A.month == 9) & (A.year >= 2018)].sig_date, "stocks Sep 2018+"),
      stats_line(A[(A.month == 9) & A.midterm].ret, A[(A.month == 9) & A.midterm].sig_date, "stocks Sep midterm"),
      stats_line(q[q.year >= 2018].ret, q[q.year >= 2018].sig_date, "stocks QE 2018+"),
      stats_line(q[q.year < 2018].ret, q[q.year < 2018].sig_date, "stocks QE pre-2018"),
      stats_line(alld, idx[allpos], "stocks all days (every 3rd)")],
     "c2 single-stock decile spread, run-in")
show([stats_line(q.rev, q.sig_date, "stocks QE->QE+5"),
      stats_line(nq.rev, nq.sig_date, "stocks non-QE ME->ME+5"),
      stats_line(A[A.month == 9].rev, A[A.month == 9].sig_date, "stocks Sep QE->+5"),
      stats_line(allr, idx[allpos], "stocks all days d+10->d+15")],
     "c2 single-stock reversal")
print(f"QE minus non-QE run-in: {100*(q.ret.mean()-nq.ret.mean()):+.3f}pp")
s9 = A[A.month == 9]
print("Sep:", ", ".join(f"{d.year}:{100*r:+.2f}" for d, r in zip(s9.me_date, s9.ret)))
