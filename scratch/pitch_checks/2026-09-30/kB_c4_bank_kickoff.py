"""C4 round 1: long XLF (and EW JPM/C/WFC/GS) from k=-9 (close 9 sessions
before the season's first money-centre print) to k=-1 (h=8), also to the
print-day close (h=9). Calendar anchor: entry = the k=-9 session's own close.
Gate (pre-specified): XLF 21d return, trailing-252 percentile (pitch_lab.pct_rank,
the tape convention) <= 5 at the entry close. Live: 2.8 on the 09-29 close.
Kickoff = earliest of JPM/C/WFC announcement dates (earnings_calendar.parquet)
falling on day 8..25 of Jan/Apr/Jul/Oct; weekend dates snap to the prior session."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kB_common import nyse_index, stats_line  # noqa

ROOTP = Path(__file__).resolve().parents[3]
BANKS = ["JPM", "C", "WFC", "GS"]
idx = nyse_index()
px = close_panel(["XLF", "SPY"] + BANKS).reindex(idx)
px["BASK"] = np.nan  # placeholder; basket computed from legs
rk21 = pct_rank(px["XLF"], 21)

ec = pd.read_parquet(ROOTP / "data" / "earnings_calendar.parquet", columns=["ticker", "date"])
ec = ec[ec.ticker.isin(["JPM", "C", "WFC"]) & (ec.date >= "2000-01-01")]
ec = ec[ec.date.dt.month.isin([1, 4, 7, 10]) & ec.date.dt.day.between(8, 25)]
ec["q"] = ec.date.dt.to_period("Q")
kick = ec.groupby("q").date.min().sort_values()
pos = []
snapped = 0
for d in kick:
    loc = int(idx.searchsorted(d))
    if loc < len(idx) and idx[loc] == d:
        pos.append(loc)
    elif loc < len(idx):
        pos.append(loc - 1)  # weekend/holiday date: prior session
        snapped += 1
    else:
        pos.append(-1)  # future (2026-10-13)
K = pd.DataFrame({"kick": kick.values, "p0": pos})
live = K[K.p0 < 0]
K = K[K.p0 > 0].reset_index(drop=True)
print(f"kickoffs measured {len(K)} ({K.kick.min().date()}..{K.kick.max().date()}), "
      f"snapped {snapped}, future {len(live)} ({[str(d.date()) for d in live.kick]})")

# date validation: |ret| of the reporting-day name set on p0 vs neighbours
lr = np.log(px[BANKS[:3]]).diff().abs().mean(axis=1).values
for off in (-1, 0, 1):
    print(f"  mean |EW JPM/C/WFC 1d ret| at kickoff{off:+d}: {100*np.nanmean([lr[p+off] for p in K.p0]):.3f}%")
print(f"  mean |same| all days: {100*np.nanmean(lr):.3f}%")

C = {t: px[t].values for t in ["XLF"] + BANKS}


def ret_between(v, a, b):
    if a < 0 or b >= len(idx):
        return np.nan
    return v[b] / v[a] - 1


def veh(a, b, which):
    if which == "XLF":
        return ret_between(C["XLF"], a, b)
    return np.nanmean([ret_between(C[t], a, b) for t in BANKS])


def cell(ent_k, ex_k, which="XLF", gate=None, gate_lag=0):
    out, dates = [], []
    for p0, d in zip(K.p0, K.kick):
        a, b = p0 + ent_k, p0 + ex_k
        g = rk21.iloc[a - gate_lag]
        if gate is not None and not (g <= gate):
            continue
        out.append(veh(a, b, which))
        dates.append(idx[a])
    return np.array(out), pd.DatetimeIndex(dates)


K["g"] = [rk21.iloc[p - 9] for p in K.p0]
K["g_prev"] = [rk21.iloc[p - 10] for p in K.p0]
rows = []
for which in ("XLF", "BASK"):
    for lbl, gate, gl in (("ungated", None, 0), ("gate r21<=5 @entry", 5, 0),
                          ("gate r21<=5 @k-10 (known)", 5, 1), ("gate r21<=10", 10, 0),
                          ("gate r21<=20", 20, 0)):
        for ex in (-1, 0):
            v, d = cell(-9, ex, which, gate, gl)
            rows.append(stats_line(v, d, f"{which} k-9->k{ex:+d} {lbl}"))
show(rows, "C4 kickoff run-in, entry k=-9 close")

# gated episode list
v, d = cell(-9, -1, "XLF", 5)
vb, _ = cell(-9, -1, "BASK", 5)
v0, _ = cell(-9, 0, "XLF", 5)
print("\ngated episodes (XLF k-9->k-1 | basket | XLF to print close):")
for dd, a, b, c in zip(d, v, vb, v0):
    print(f"  {dd.date()}  {100*a:+.2f}%  {100*b:+.2f}%  {100*c:+.2f}%")

# CONTROLS ---------------------------------------------------------------
H = 8
xlf = px["XLF"]
f8 = xlf.shift(-H) / xlf - 1
print(f"\nCTRL XLF all-days h=8 drift (lag 0): {100*f8.mean():+.3f}% (n={f8.notna().sum()}, hit {100*(f8.dropna()>0).mean():.1f}%)")
st = idx[(rk21 <= 5).values & f8.notna().values]
ep = declusters(st, H, idx)
show([summarize(f8.loc[st].values, f"XLF r21<=5 ALL days, day-level (N={len(st)})"),
      summarize(f8.loc[ep].values, f"XLF r21<=5 ALL days, episodes (N={len(ep)})"),
      summarize(f8.loc[ep[ep >= '2018-01-01']].values, "  episodes 2018+"),
      summarize(f8.loc[ep[ep < '2018-01-01']].values, "  episodes pre-2018")],
     "CTRL price state alone (XLF washout bounce, h=8 from the close)")
w = int((f8.loc[ep] > 0).sum())
print(f"  price-state episodes record {w}-{len(ep)-w}, sign p {sign_test(w, len(ep)):.4f}")

# tdom-matched windows in non-kickoff months: same offset from the month's first session
first = pd.Series(range(len(idx)), index=idx).groupby(idx.to_period("M")).min()
K["eoff"] = [p - 9 - first[idx[p].to_period("M")] for p in K.p0]
print(f"\nentry offset vs first session of the kickoff month: {K.eoff.value_counts().sort_index().to_dict()}")
ctrl, cdates, cg = [], [], []
for per, f in first.items():
    if per.month in (1, 4, 7, 10) or per.year < 2000:
        continue
    for eo in K.eoff.unique():
        a = f + eo
        if a < 252 or a + H >= len(idx):
            continue
        ctrl.append(C["XLF"][a + H] / C["XLF"][a] - 1)
        cdates.append(idx[a])
        cg.append(rk21.iloc[a])
ctrl, cdates, cg = np.array(ctrl), pd.DatetimeIndex(cdates), np.array(cg)
show([stats_line(ctrl, cdates, "non-kickoff months, same entry offsets, ungated"),
      stats_line(ctrl[cg <= 5], cdates[cg <= 5], "non-kickoff months, same offsets, r21<=5"),
      stats_line(ctrl[cg <= 10], cdates[cg <= 10], "non-kickoff months, same offsets, r21<=10")],
     "CTRL tdom-matched XLF 8-session windows, non-kickoff months")

# offset placebo ladder
lad = []
for ek in range(-12, -4):
    for gate in (None, 5):
        v, d = cell(ek, -1, "XLF", gate)
        lad.append(stats_line(v, d, f"entry k{ek:+d}->k-1 {'gated' if gate else 'ungated'}"))
show(lad, "entry ladder (exit k=-1)")
lad = []
for xk in range(-4, 3):
    for gate in (None, 5):
        v, d = cell(-9, xk, "XLF", gate)
        lad.append(stats_line(v, d, f"k-9->k{xk:+d} {'gated' if gate else 'ungated'}"))
show(lad, "exit ladder (entry k=-9)")

# era + midterm, ungated and gated
for gate in (None, 10):
    v, d = cell(-9, -1, "XLF", gate)
    show(era_split(d, v) + [summarize(v[(d.year % 4 == 2)], "midterm"),
                            summarize(v[(d.month == 9) | (d.month == 10)], "Oct kickoff (entry late Sep/Oct)")],
         f"era/midterm, XLF k-9->k-1, gate {gate}")
print(f"\ncost: XLF ~2-3 bp RT; basket 4 names ~8 bp.")
print(f"LIVE: XLF r21 pct {rk21.iloc[-1]:.1f} at {idx[-1].date()}; XLF 21d {100*(xlf.iloc[-1]/xlf.iloc[-22]-1):+.2f}%")
