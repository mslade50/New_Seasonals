"""C2 round 2 on the one reading that survived round 1: (b) complex flush, long XLV /
short SPY at h=10 (+0.969%, 14-7). Legs priced separately, beta-neutral residual,
concentration, neighbours, era/regime, FOMC, sector-complex reference class."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
sys.path.insert(0, str(Path(__file__).resolve().parent))
from k2_common import cell, fmt, conc, null_maxk  # noqa

CPLX = {"XLV": ["XLV", "IBB", "XBI", "IHI"], "XLF": ["XLF", "KRE", "KBE", "KIE"],
        "XLE": ["XLE", "XOP", "OIH"], "XLK": ["XLK", "SMH", "IGV", "SOXX"],
        "XLY": ["XLY", "XRT", "XHB", "ITB"], "XLI": ["XLI", "ITA", "IYT"],
        "XLB": ["XLB", "XME", "GDX"], "XLU": ["XLU", "IDU"], "XLP": ["XLP", "PBJ"]}
allt = sorted({t for v in CPLX.values() for t in v} | {"SPY"})
px = close_panel(allt)
px = px.loc[:, px.notna().sum() > 1000]
r5 = {t: pct_rank(px[t], 5) for t in px.columns}
dr = px.pct_change()


def rule(sec, sec_thr=1, mem_thr=5, frac=0.75):
    mem = [t for t in CPLX[sec] if t in px.columns]
    cnt = sum((r5[t] <= mem_thr).astype(int) for t in mem)
    nav = sum(r5[t].notna().astype(int) for t in mem)
    return (r5[sec] <= sec_thr) & (cnt >= np.ceil(frac * nav)) & (nav >= 3)


beta_x = float((dr["XLV"].cov(dr["SPY"]) / dr["SPY"].var()))
print("full-sample beta XLV on SPY", round(beta_x, 2))
B = rule("XLV")
LP = [("XLV", 1.0), ("SPY", -1.0)]
LN = [("XLV", 1.0), ("SPY", -beta_x)]
print("\n=== legs priced separately, h=10 and h=5 ===")
for h in (5, 10):
    print(fmt(cell(px, B, [("XLV", 1.0)], h), f"h{h} XLV leg"))
    print(fmt(cell(px, B, [("SPY", -1.0)], h), f"h{h} SHORT SPY leg"))
    print(fmt(cell(px, B, LP, h), f"h{h} pair eq$"))
    print(fmt(cell(px, B, LN, h), f"h{h} pair beta-neutral ({beta_x:.2f})"))

print("\n=== horizon scan, pair eq$ and beta-neutral ===")
dates = px.index[B.reindex(px.index, fill_value=False).values]
show(horizon_scan(px, dates, LP, hs=tuple(range(1, 11))), "eq$")
show(horizon_scan(px, dates, LN, hs=tuple(range(1, 11))), "beta-neutral")

c = cell(px, B, LP, 10)
print("\nconcentration pair h10:", conc(c))
ep = pd.Series(c["ep"] * 100, index=c["epi"])
print(ep.round(2).to_string())
yrs = ep.groupby(ep.index.year).sum()
print("LOYO min mean:", round(min((ep[ep.index.year != y].mean()) for y in yrs.index), 3),
      " drop 2008+2020:", round(ep[~ep.index.year.isin([2008, 2020])].mean(), 3))
spy200 = px["SPY"] > px["SPY"].rolling(200).mean()
mid = pd.Series(px.index.year % 4 == 2, index=px.index)
for L, nm in ((LP, "eq$"), (LN, "bneu")):
    print(f"-- regime h10 {nm}")
    print(fmt(cell(px, B & spy200, L, 10), "SPY above 200d (live)"))
    print(fmt(cell(px, B & ~spy200, L, 10), "SPY below 200d"))
    print(fmt(cell(px, B & (px.index < "2018-01-01"), L, 10), "pre-2018"))
    print(fmt(cell(px, B & (px.index >= "2018-01-01"), L, 10), "2018+"))
    print(fmt(cell(px, B & mid, L, 10), "midterm"))
    print(fmt(cell(px, B & ~mid, L, 10), "non-midterm"))

fl = event_in_window(c["epi"], px.index, 10, 1, ("fomc_decision",))
print("FOMC in window pair h10:", summarize(c["ep"][fl], "in"), summarize(c["ep"][~fl], "out"))

print("\n=== neighbours, pair eq$ h10 / beta-neutral h10 ===")
for st in (1, 2, 3):
    for mt in (5, 10):
        for fr in (0.5, 0.75, 1.0):
            m = rule("XLV", st, mt, fr)
            a, b = cell(px, m, LP, 10), cell(px, m, LN, 10)
            if a.get("n"):
                print(f"sec<={st} mem<={mt} frac>={fr}: eq$ n={a['n']} ex {a['excess_pp']:+.3f} "
                      f"rec {a['rec']} | bneu ex {b['excess_pp']:+.3f} rec {b['rec']}")

print("\n=== reference class: same complex-flush rule on each sector complex ===")
for h in (5, 10):
    bk, bn = {}, {}
    for s in CPLX:
        mem = [t for t in CPLX[s] if t in px.columns]
        if len(mem) < 3:
            continue
        m = rule(s)
        bs = float(dr[s].cov(dr["SPY"]) / dr["SPY"].var())
        a = cell(px, m, [(s, 1.0), ("SPY", -1.0)], h)
        b = cell(px, m, [(s, 1.0), ("SPY", -bs)], h)
        if a.get("n", 0) > 2:
            bk[s], bn[s] = a["ex"], b["ex"]
    null_maxk(bk, "XLV", f"h={h} pair eq$")
    null_maxk(bn, "XLV", f"h={h} pair beta-neutral")
