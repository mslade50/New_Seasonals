"""C4 exit-ladder byproduct (lead for the 10-12 map, NOT today's trade): after a
bank washout, does XLF pay ACROSS the kickoff print (entry k=-1 close, exit
k+0/k+1/k+2)? Gate measured at the k=-1 close. XLF raw and resid vs beta-SPY.
Also the ex-post live number for the 8 gated k-9 episodes split into run-in
and print legs. Also prints XLF r21 pct and the ex-XLE p80 gap for C1's trigger."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kB_common import SPDR9, nyse_index, me_table, stats_line, exante_beta  # noqa

ROOTP = Path(__file__).resolve().parents[3]
idx = nyse_index()
px = close_panel(["XLF", "SPY"]).reindex(idx)
rk21 = pct_rank(px["XLF"], 21)
b = exante_beta(px["XLF"], px["SPY"]).shift(1).values
ec = pd.read_parquet(ROOTP / "data" / "earnings_calendar.parquet", columns=["ticker", "date"])
ec = ec[ec.ticker.isin(["JPM", "C", "WFC"]) & (ec.date >= "2000-01-01")]
ec = ec[ec.date.dt.month.isin([1, 4, 7, 10]) & ec.date.dt.day.between(8, 25)]
kick = ec.groupby(ec.date.dt.to_period("Q")).date.min().sort_values()
p0 = []
for d in kick:
    loc = int(idx.searchsorted(d))
    if loc < len(idx):
        p0.append(loc if idx[loc] == d else loc - 1)
p0 = np.array([p for p in p0 if p - 9 > 260 and p + 2 < len(idx)])
X, S = px["XLF"].values, px["SPY"].values
rows = []
for gate in (None, 20, 10, 5):
    for ex in (0, 1, 2):
        a = p0 - 1
        sel = a if gate is None else a[rk21.values[a] <= gate]
        raw = X[sel + 1 + ex] / X[sel] - 1
        res = raw - b[sel] * (S[sel + 1 + ex] / S[sel] - 1)
        r = stats_line(raw, idx[sel], f"k-1->k{ex:+d} gate {gate}")
        r["resid_mean"] = 100 * np.nanmean(res)
        r["resid_rec"] = f"{int((res>0).sum())}-{int((res<=0).sum())}"
        rows.append(r)
show(rows, "XLF across the kickoff print, entry k=-1 close (gate at k=-1)")
a = p0 - 9
g = a[rk21.values[a] <= 5]
print("gated (at k-9) episodes: run-in k-9->k-1 | print k-1->k+1:")
for p in g:
    q = p + 9
    print(f"  {idx[q].date()}  {100*(X[q-1]/X[p]-1):+.2f}%  {100*(X[q+1]/X[q-1]-1):+.2f}%")

# C1 trigger number: ex-XLE 8-SPDR QE gap p80
cp = close_panel(SPDR9).reindex(idx)
u8 = [t for t in SPDR9 if t != "XLE"]
R8 = pd.DataFrame({t: cp[t] / cp[t].shift(63) - 1 for t in u8}).values
T = me_table(idx)
gaps = []
for m in T[T.qe].me_pos:
    rr = R8[int(m) - 1]
    rr = np.sort(rr[np.isfinite(rr)])
    if len(rr) >= 5:
        gaps.append(rr[-2:].mean() - rr[:2].mean())
print(f"\nC1 trigger: ex-XLE 8-SPDR QE gap p50 {100*np.quantile(gaps,.5):.1f}pp, p80 {100*np.quantile(gaps,.8):.1f}pp; "
      f"live {100*(np.sort(R8[-1])[-2:].mean()-np.sort(R8[-1])[:2].mean()):.1f}pp")
