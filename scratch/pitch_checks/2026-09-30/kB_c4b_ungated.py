"""C4 byproduct (checker's walk, 1 row: the UNGATED kickoff run-in, which was a
control in kB_c4_bank_kickoff.py). Is the +0.574% (65-42) a bank object or the
Jan/Apr/Jul/Oct turn-of-month tape? Same windows (k=-9 close -> k=-1 close):
SPY, XLF resid vs ex-ante beta-SPY, the 8 other SPDRs (reference class), and
tdom-matched non-kickoff months for each, by era."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kB_common import SPDR9, nyse_index, stats_line, exante_beta  # noqa

ROOTP = Path(__file__).resolve().parents[3]
idx = nyse_index()
px = close_panel(SPDR9 + ["SPY"]).reindex(idx)
ec = pd.read_parquet(ROOTP / "data" / "earnings_calendar.parquet", columns=["ticker", "date"])
ec = ec[ec.ticker.isin(["JPM", "C", "WFC"]) & (ec.date >= "2000-01-01")]
ec = ec[ec.date.dt.month.isin([1, 4, 7, 10]) & ec.date.dt.day.between(8, 25)]
kick = ec.groupby(ec.date.dt.to_period("Q")).date.min().sort_values()
p0 = []
for d in kick:
    loc = int(idx.searchsorted(d))
    if loc < len(idx):
        p0.append(loc if idx[loc] == d else loc - 1)
p0 = np.array([p for p in p0 if p - 9 > 260])
H = 8
first = pd.Series(range(len(idx)), index=idx).groupby(idx.to_period("M")).min()
eoff = sorted(set(int(p - 9 - first[idx[p].to_period("M")]) for p in p0))
ctrl = []
for per, f in first.items():
    if per.month in (1, 4, 7, 10):
        continue
    for eo in eoff:
        a = f + eo
        if 260 < a and a + H < len(idx):
            ctrl.append(a)
ctrl = np.array(sorted(set(ctrl)))
ent = p0 - 9
b = exante_beta(px["XLF"], px["SPY"]).shift(1).values


def r(t, a):
    v = px[t].values
    return v[a + H] / v[a] - 1


rows = []
for t in ["XLF", "SPY"] + [s for s in SPDR9 if s != "XLF"]:
    k, c = r(t, ent), r(t, ctrl)
    y, yc = idx[ent].year, idx[ctrl].year
    rows.append({"veh": t, "kick_mean": 100 * np.nanmean(k), "kick_hit": 100 * np.nanmean(k > 0),
                 "ctrl_mean": 100 * np.nanmean(c), "excess": 100 * (np.nanmean(k) - np.nanmean(c)),
                 "kick_18+": 100 * np.nanmean(k[y >= 2018]), "ctrl_18+": 100 * np.nanmean(c[yc >= 2018]),
                 "excess_18+": 100 * (np.nanmean(k[y >= 2018]) - np.nanmean(c[yc >= 2018]))})
rows = pd.DataFrame(rows).sort_values("excess", ascending=False)
show(rows.to_dict("records"), f"reference class: kickoff windows (N={len(ent)}) vs tdom-matched non-kickoff (N={len(ctrl)})")
print(f"XLF excess ranks {1 + list(rows.veh).index('XLF')} of {len(rows)}")

res_k = r("XLF", ent) - b[ent] * r("SPY", ent)
res_c = r("XLF", ctrl) - b[ctrl] * r("SPY", ctrl)
show([stats_line(res_k, idx[ent], "XLF resid vs beta-SPY, kickoff windows"),
      stats_line(res_c, idx[ctrl], "XLF resid, tdom-matched non-kickoff"),
      stats_line(res_k[idx[ent].year >= 2018], idx[ent][idx[ent].year >= 2018], "  kickoff resid 2018+"),
      stats_line(res_k[idx[ent].year < 2018], idx[ent][idx[ent].year < 2018], "  kickoff resid pre-2018")],
     "bank residual")
