"""10y yield up 12bp+ in a session to a 52-week closing high (today +14.6bp to 5.114).
What did yields, bonds and stocks do next session / next week / next month?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

px = close_panel(["^TNX", "^FVX", "TLT", "IEF", "SPY", "QQQ", "IWM", "^GSPC", "^MOVE", "DX-Y.NYB", "^VIX"])
tnx = px["^TNX"].dropna()
idx = tnx.index
bp = tnx.diff() * 100
hi252 = tnx.rolling(252, min_periods=200).max()
at_high = tnx >= hi252 - 1e-9
spx = px["^GSPC"].reindex(idx)
spx_r = spx.pct_change()

print("today", idx[-1].date(), "tnx", tnx.iloc[-1], "bp", round(bp.iloc[-1], 1), "rank of |bp| over last 252:",
      int((bp.tail(252).abs() >= abs(bp.iloc[-1])).sum()))
big = bp[bp >= bp.iloc[-1] - 1e-9]
print("sessions since 1999 with bp >= today:", len(big), "last few:", [(str(d.date()), round(v, 1)) for d, v in big.tail(6).items()])
prev_big = bp.loc[: idx[-2]]
last_ge = prev_big[prev_big >= bp.iloc[-1] - 1e-9]
print("previous day with bp >= today's:", last_ge.index[-1].date() if len(last_ge) else None)

def fwd_bp(h):
    return (tnx.shift(-h) - tnx) * 100

def report(mask, label, gap=5):
    trig = idx[mask.reindex(idx).fillna(False).values]
    trig = trig[trig < idx[-1]]
    epi = declusters(trig, gap, idx)
    print(f"\n##### {label}: raw {len(trig)}, declustered({gap}) {len(epi)}")
    print("   dates:", [str(d.date()) for d in epi])
    ctl = local_control(idx, epi, 126)
    for h in (1, 5, 21):
        f = fwd_bp(h)
        v = f.loc[epi].dropna()
        up = int((v > 0).sum())
        print(f"   TNX h{h}: n {len(v)} mean {v.mean():+.1f}bp median {v.median():+.1f} up {up}/{len(v)} "
              f"sign_p(up) {sign_test(up, len(v)):.3f} sign_p(down) {sign_test(len(v)-up, len(v)):.3f} | local {f.loc[ctl].mean():+.2f}bp all {f.mean():+.2f}bp")
    rows = []
    for t in ("TLT", "SPY", "QQQ", "IWM"):
        s = px[t].reindex(idx)
        for h in (1, 5, 21):
            r = fwd_ret(s, h)
            e = [d for d in epi if not np.isnan(r.get(d, np.nan))]
            row = summarize(r.loc[e].values, f"{t} h{h}")
            row["local"] = 100 * r.loc[ctl].mean()
            row["sign_p"] = sign_test(int((r.loc[e] > 0).sum()), len(e)) if row["hit"] >= 50 else sign_test(int((r.loc[e] < 0).sum()), len(e))
            rows.append(row)
    show(rows, label)
    f1 = fwd_bp(1).loc[epi].dropna()
    for part in era_split(f1.index, f1.values / 100):
        print("   era TNX h1 (pp of yield; x100=bp):", part["label"], part["n"], round(part.get("mean_pct", np.nan), 2), round(part.get("hit", np.nan), 1))
    f5 = fwd_bp(5).loc[epi].dropna()
    for part in era_split(f5.index, f5.values / 100):
        print("   era TNX h5:", part["label"], part["n"], round(part.get("mean_pct", np.nan), 2), round(part.get("hit", np.nan), 1))
    s1 = fwd_ret(px["SPY"].reindex(idx), 5).loc[epi].dropna()
    for part in era_split(s1.index, s1.values):
        print("   era SPY h5:", part["label"], part["n"], round(part.get("mean_pct", np.nan), 2), round(part.get("hit", np.nan), 1))
    print("   cluster TNX h5:", cluster_note(f5.index, f5.values / 100))
    print("   cluster SPY h5:", cluster_note(s1.index, s1.values))
    return epi

e1 = report((bp >= 12) & at_high, "10y +12bp or more, closing at a 52w high")
e2 = report((bp >= 12) & at_high & (spx_r <= -0.005), "same, S&P down 0.5%+")
e3 = report((bp >= 10) & at_high, "10y +10bp or more, closing at a 52w high")
sd63 = bp.rolling(63).std()
e4 = report((bp / sd63 >= 2.5) & at_high, "10y move >= 2.5 sd(63d) of daily bp changes, at a 52w high")
print("\ntoday's z vs 63d sd:", round(bp.iloc[-1] / sd63.iloc[-1], 2), "sd63", round(sd63.iloc[-1], 2))
e5 = report((bp >= 12), "10y +12bp or more, any level")
