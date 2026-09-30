"""Kill check: board SHORT TXN 10d, entry T+5 (Oct 7 close), also T+1 (Oct 1).
Returns are side-signed (positive = the short made money)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitch_lab import *  # noqa
from sb2_common import *  # noqa

px = close_panel(["TXN", "SMH", "QQQ", "SPY"])
t = px["TXN"].dropna()
idx = t.index
anc = anchors(idx)

s5 = full_report(t, anc, 5, 10, -1, "TXN T+5 10td short (board)")
s1 = full_report(t, anc, 1, 10, -1, "TXN T+1 10td short")
s0 = full_report(t, anc, 0, 10, -1, "TXN lag0 10td (board's own measurement)")
print(f"\ncost: T+5 mean {1e4*s5.mean():.0f} bps, T+1 {1e4*s1.mean():.0f} bps vs ~5 bps RT")

print("\n=== neighbours short (all-yrs mean / hit | midterm mean hit dropbest) ===")
for lag in (1, 5):
    for sh in (-3, -2, -1, 0, 1, 2, 3):
        a = anchors(idx, shift=sh)
        for h in (5, 10, 15):
            s = yearly(t, a, lag, h, -1)
            m = s[s.index.isin(MIDTERMS)]
            md = m.sort_values(ascending=False).iloc[1:]
            print(f"lag{lag} shift {sh:+d} h={h}: all {100*s.mean():+.2f}% {int((s>0).sum())}/{len(s)} | "
                  f"mid {100*m.mean():+.2f}% {int((m>0).sum())}/{len(m)} dropbest {100*md.mean():+.2f}%")

print("\n=== gate attribution: 10d pctile >= 85 at T-1 (and any 5/10/21 max) ===")
rows = []
for y, p in anc.items():
    rows.append((y, pit_pctile(t, p - 1, 10),
                 max(pit_pctile(t, p - 1, k) for k in (5, 10, 21)), s5.get(y), s1.get(y)))
g = pd.DataFrame(rows, columns=["yr", "p10", "pany", "r_T5", "r_T1"]).set_index("yr")
print(g.round(3).to_string())
for col in ("p10", "pany"):
    for rc in ("r_T5", "r_T1"):
        on, off = g[g[col] >= 85][rc].dropna(), g[g[col] < 85][rc].dropna()
        print(f"{col}/{rc}: ON n={len(on)} {100*on.mean():+.2f}% hit {int((on>0).sum())}/{len(on)} "
              f"yrs {list(on.index)} | OFF n={len(off)} {100*off.mean():+.2f}% hit {int((off>0).sum())}/{len(off)}")
r10 = -fwd_lag(t, 10, 5)
pr = t.pct_change(10).expanding(250).rank(pct=True) * 100
dd = pd.concat([r10, pr], axis=1, keys=["r", "p"]).dropna()
on, off = dd[dd.p >= 85].r, dd[dd.p < 85].r
print(f"all-days 10d>=85pct -> T+5 10td short: ON {100*on.mean():+.2f}% hit {100*(on>0).mean():.0f}% "
      f"(n={len(on)}) vs OFF {100*off.mean():+.2f}% hit {100*(off>0).mean():.0f}%")

print("\n=== earnings truncation (Oct print; TXN reports after close) ===")
e = pd.read_parquet(ROOT / "data" / "earnings_calendar.parquet")
e = e[e.ticker == "TXN"].copy()
e["date"] = pd.to_datetime(e["date"])
octp = {d.year: d for d in e["date"] if d.month == 10}
c = t.values
tr = []
for y, p in anc.items():
    d = octp.get(y)
    if d is None:
        continue
    pp = int(idx.searchsorted(d))  # print-date bar (AMC -> reaction next bar)
    for lag, h, lab in ((5, 10, "T5"), (1, 10, "T1")):
        a, b = p + lag, p + lag + h
        inside = pp < b  # reaction bar pp+1 <= b
        pre_end = min(b, pp - 1)  # close BEFORE the print date
        pre = -(c[pre_end] / c[a] - 1) if pre_end > a else np.nan
        full = -(c[b] / c[a] - 1)
        rx = -(c[pp + 1] / c[pp] - 1)
        tr.append((y, lab, str(d.date()), inside, full, pre, rx))
T = pd.DataFrame(tr, columns=["yr", "leg", "print", "inside", "full", "pre", "rxn"])
for lab in ("T5", "T1"):
    q = T[T.leg == lab].set_index("yr")
    print(f"{lab}: print inside window {int(q.inside.sum())}/{len(q)} yrs")
    for col in ("full", "pre"):
        s = q[col].dropna()
        m = s[s.index.isin(MIDTERMS)]
        print(f"  {col:5s} all {100*s.mean():+.2f}% hit {int((s>0).sum())}/{len(s)} | midterm {100*m.mean():+.2f}% "
              f"{int((m>0).sum())}/{len(m)} dropbest {100*m.sort_values(ascending=False).iloc[1:].mean():+.2f}%")
    ins = q[q.inside]
    print(f"  inside-yrs: full {100*ins.full.mean():+.2f}% vs pre {100*ins.pre.mean():+.2f}% (n={len(ins)}); "
          f"print-reaction (short-signed) mean {100*q.rxn.mean():+.2f}% hit {int((q.rxn>0).sum())}/{len(q)}")
print(T[T.yr.isin(MIDTERMS)].round(4).to_string(index=False))

print("\n=== sector/market residual (short-signed) ===")
for lag in (5, 1):
    base = yearly(t, anc, lag, 10, -1)
    out = {"txn": base}
    for bm in ("SMH", "QQQ"):
        b = px[bm].dropna()
        ab = anchors(b.index)
        bs = yearly(b, ab, lag, 10, -1)
        out[bm.lower()] = bs
        out[f"txn-{bm.lower()}"] = (base - bs).dropna()
        rr = {}
        for y, p in anc.items():
            if y in bs and y in base:
                beta = beta_at(t, b, b.index.get_loc(idx[p]))
                rr[y] = base[y] - beta * bs[y]
        out[f"resid_b{bm.lower()}"] = pd.Series(rr)
    print(f"-- lag {lag}")
    for k, s in out.items():
        s = s.dropna()
        m = s[s.index.isin(MIDTERMS)]
        print(f"{k:11s} all {100*s.mean():+.2f}% hit {int((s>0).sum())}/{len(s)} | midterm {100*m.mean():+.2f}% "
              f"{int((m>0).sum())}/{len(m)} dropbest {100*m.sort_values(ascending=False).iloc[1:].mean():+.2f}% "
              f"drop2 {100*m.sort_values(ascending=False).iloc[2:].mean():+.2f}%")
    print(pd.DataFrame(out)[lambda d: d.index.isin(MIDTERMS)].mul(100).round(2).to_string())
    print(f"SMH own short drift 10td: {all_day_drift(px['SMH'].dropna(), lag, 10, -1):+.2f}%")
