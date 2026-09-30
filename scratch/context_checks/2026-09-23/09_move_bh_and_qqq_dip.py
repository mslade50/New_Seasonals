"""A. Engine BH pass: MOVE 5d return in the top 5% of its year -> lower next day 180 of 290 (overlapping days).
   Declustered, and split by whether the week's rise came from one session of 12%+ (today) or was spread out.
B. Engine P3: QQQ down 0.5%+ the session after a 52w high -> next day 74-55 up. Does a 10bp+ 10y jump
   on the pullback day change that?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

px = close_panel(["^MOVE", "^VIX", "QQQ", "^TNX", "SPY"])
mv = px["^MOVE"].dropna()
mi = mv.index
r1 = mv.pct_change()
rk = pct_rank(mv, 5)
f1, f5 = fwd_ret(mv, 1), fwd_ret(mv, 5)
print("today MOVE 5d rank", round(rk.iloc[-1], 1), "1d", round(100 * r1.iloc[-1], 1))
for label, m in (("top-5% week, all", rk >= 95), ("top-5% week, one 12%+ session today", (rk >= 95) & (r1 >= 0.12)),
                 ("top-5% week, today < 12%", (rk >= 95) & (r1 < 0.12))):
    t = mi[m.fillna(False).values]
    t = t[t < mi[-1]]
    e = declusters(t, 5, mi)
    for nm, f in (("h1", f1), ("h5", f5)):
        v = f.loc[e].dropna()
        dn = int((v < 0).sum())
        print(f"  {label}: raw {len(t)} epi {len(v)} MOVE {nm} {100 * v.mean():+.2f}% down {dn}/{len(v)} p {sign_test(dn, len(v)):.4f}")
    v = f1.loc[e].dropna()
    for part in era_split(v.index, v.values):
        print("     era h1", part["label"], part["n"], round(part.get("mean_pct", np.nan), 2), round(part.get("hit", np.nan), 1))

q = px["QQQ"].dropna()
qi = q.index
tnx = px["^TNX"].reindex(qi)
bp = tnx.diff() * 100
qr = q.pct_change()
hi = q >= q.rolling(252, min_periods=250).max() - 1e-9
after_hi = hi.shift(1, fill_value=False) & (qr <= -0.005)
fq1, fq5 = fwd_ret(q, 1), fwd_ret(q, 5)
print("\nB. today QQQ", round(100 * qr.iloc[-1], 2), "bp", round(bp.iloc[-1], 1))
for label, m in (("QQQ -0.5% after a high, all", after_hi), ("... with 10y +10bp", after_hi & (bp >= 10)),
                 ("... with 10y +5bp", after_hi & (bp >= 5)), ("... 10y not up 5bp", after_hi & (bp < 5))):
    t = qi[m.fillna(False).values]
    t = t[t < qi[-1]]
    ctl = local_control(qi, t, 126)
    for nm, f in (("h1", fq1), ("h5", fq5)):
        v = f.loc[t].dropna()
        up = int((v > 0).sum())
        print(f"  {label}: n {len(v)} QQQ {nm} {100 * v.mean():+.3f}% up {up}/{len(v)} p {sign_test(up, len(v)):.3f} local {100 * f.loc[ctl].mean():+.3f}%")
    if "10bp" in label:
        print("     dates:", [(str(d.date()), round(bp.loc[d], 1), round(100 * fq1.loc[d], 2)) for d in t])
