"""Kill check: board LONG WMT 21d, entry T+2 (Oct 2 close), also T+1."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitch_lab import *  # noqa
from sb2_common import *  # noqa

px = close_panel(["WMT", "XLP", "SPY"])
w = px["WMT"].dropna()
idx = w.index
anc = anchors(idx)
print("anchors:", {y: str(idx[p].date()) for y, p in list(anc.items())[:3]}, "...",
      len(anc), "yrs")
p26 = anchors(idx, last_year=2026)[2026]
print("2026 anchor bar:", idx[p26].date())

# ROUND 1
s2 = full_report(w, anc, 2, 21, +1, "WMT T+2 21td (board)")
s1 = full_report(w, anc, 1, 21, +1, "WMT T+1 21td")
s0 = full_report(w, anc, 0, 21, +1, "WMT lag0 21td (board's own measurement)")
print(f"\ncost: T+2 mean {1e4*s2.mean():.0f} bps vs ~5 bps RT")

# ROUND 2b neighbours
print("\n=== neighbours (all-yrs mean / hit | midterm mean / hit, drop-best midterm) ===")
for sh in (-3, -2, -1, 0, 1, 2, 3):
    a = anchors(idx, shift=sh)
    for h in (15, 21, 26):
        s = yearly(w, a, 2, h, +1)
        m = s[s.index.isin(MIDTERMS)]
        md = m.sort_values(ascending=False).iloc[1:]
        print(f"shift {sh:+d} h={h}: all {100*s.mean():+.2f}% {int((s>0).sum())}/{len(s)} | "
              f"mid {100*m.mean():+.2f}% {int((m>0).sum())}/{len(m)} dropbest {100*md.mean():+.2f}%")

# ROUND 2c gate attribution: 5d return <= 15th pctile at T-1 (and any 5/10/21)
print("\n=== gate attribution (PIT pctile at T-1) ===")
rows = []
for y, p in anc.items():
    g5 = pit_pctile(w, p - 1, 5)
    gany = min(pit_pctile(w, p - 1, k) for k in (5, 10, 21))
    rows.append((y, g5, gany, s2.get(y, np.nan)))
g = pd.DataFrame(rows, columns=["yr", "p5", "pany", "ret"]).set_index("yr").dropna()
print(g.round(3).to_string())
for col in ("p5", "pany"):
    on, off = g[g[col] <= 15]["ret"], g[g[col] > 15]["ret"]
    print(f"{col}: gate ON n={len(on)} mean {100*on.mean():+.2f}% hit {int((on>0).sum())}/{len(on)} "
          f"yrs {list(on.index)} | OFF n={len(off)} mean {100*off.mean():+.2f}% "
          f"hit {int((off>0).sum())}/{len(off)}")
print(f"2026 now: p5 {pit_pctile(w, p26 - 1, 5):.1f}, p10 {pit_pctile(w, p26 - 1, 10):.1f}, "
      f"p21 {pit_pctile(w, p26 - 1, 21):.1f}")
# day-level gate power across all history (not just Oct)
r21 = fwd_lag(w, 21, 2)
pr5 = w.pct_change(5).expanding(250).rank(pct=True) * 100
dd = pd.concat([r21, pr5], axis=1, keys=["r", "p"]).dropna()
on, off = dd[dd.p <= 15].r, dd[dd.p > 15].r
print(f"all-days 5d<=15pct -> T+2 21td: ON {100*on.mean():+.2f}% hit {100*(on>0).mean():.0f}% "
      f"(n={len(on)}) vs OFF {100*off.mean():+.2f}% hit {100*(off>0).mean():.0f}%")
oct_ = dd[(dd.index.month == 9) & (dd.index.day >= 20) | (dd.index.month == 10) & (dd.index.day <= 10)]
on, off = oct_[oct_.p <= 15].r, oct_[oct_.p > 15].r
print(f"late-Sep/early-Oct days only: ON {100*on.mean():+.2f}% (n={len(on)}) vs OFF "
      f"{100*off.mean():+.2f}% (n={len(off)})")

# ROUND 2e sector / market
print("\n=== sector/market residual, T+2 21td ===")
x = yearly(px["XLP"].dropna(), anchors(px["XLP"].dropna().index), 2, 21, +1)
sp = yearly(px["SPY"].dropna(), anchors(px["SPY"].dropna().index), 2, 21, +1)
res = []
for y, p in anc.items():
    if y not in s2 or y not in x or y not in sp:
        continue
    pxl = px["XLP"].index.get_loc(idx[p])
    b = beta_at(w, px["XLP"].dropna(), pxl)
    bs = beta_at(w, px["SPY"].dropna(), px["SPY"].dropna().index.get_loc(idx[p]))
    res.append((y, s2[y], x[y], sp[y], s2[y] - x[y], s2[y] - b * x[y], s2[y] - bs * sp[y]))
R = pd.DataFrame(res, columns=["yr", "wmt", "xlp", "spy", "wmt_xlp", "resid_bxlp", "resid_bspy"]).set_index("yr")
for col in R.columns:
    s = R[col]
    m = s[s.index.isin(MIDTERMS)]
    print(f"{col:11s} all {100*s.mean():+.2f}% hit {int((s>0).sum())}/{len(s)} | midterm "
          f"{100*m.mean():+.2f}% hit {int((m>0).sum())}/{len(m)} dropbest "
          f"{100*m.sort_values(ascending=False).iloc[1:].mean():+.2f}%")
print("XLP own all-day drift 21td:", f"{all_day_drift(px['XLP'].dropna(), 2, 21, 1):+.2f}%")
print(R[R.index.isin(MIDTERMS)].mul(100).round(2).to_string())
