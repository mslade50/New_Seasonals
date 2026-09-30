"""USO five straight down closes worth ~11%, measured on the ETF so the CL=F roll
seam (October expired today, -4.0% opening gap) never enters the cell."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

px = load_prices(["USO", "SPY", "XLE"])
c = px["USO"]["Close"].dropna()
r1 = c.pct_change()
r5 = c.pct_change(5)
r63 = c.pct_change(63)
sma200 = c.rolling(200).mean()
down = (r1 < 0).astype(int)
streak = down.groupby((down == 0).cumsum()).cumsum()

today = c.index[-1]
print(f"today {today.date()} USO {c.iloc[-1]:.2f} r1 {100*r1.iloc[-1]:+.2f}% "
      f"r5 {100*r5.iloc[-1]:+.2f}% streak {streak.iloc[-1]} r63 {100*r63.iloc[-1]:+.2f}% "
      f"vs200 {100*(c.iloc[-1]/sma200.iloc[-1]-1):+.2f}%")
rank5 = pct_rank(c, 5)
print(f"rank5 {rank5.iloc[-1]:.1f}")

fr = {h: fwd_ret(c, h) for h in (1, 5, 10, 21)}
base = {h: summarize(fr[h].dropna().values, f"all days h{h}") for h in (1, 5, 10, 21)}


def report(mask: pd.Series, label: str, gap: int = 10) -> pd.DatetimeIndex:
    trig = declusters(mask[mask].index, gap, c.index)
    trig = trig[trig < today]
    rows = []
    for h in (1, 5, 10, 21):
        v = fr[h].reindex(trig).dropna()
        s = summarize(v.values, f"h{h}")
        s["ctrl_all"] = base[h]["mean_pct"]
        lc = local_control(c.index, trig)
        s["ctrl_local"] = 100 * fr[h].reindex(lc).dropna().mean()
        s["sign_p_up"] = sign_test(int((v > 0).sum()), len(v))
        rows.append(s)
    show(rows, f"{label}  (declustered {gap}td, N={len(trig)})")
    v5 = fr[5].reindex(trig).dropna()
    show(era_split(v5.index, v5.values), "h5 era split")
    print("  h5", cluster_note(v5.index, v5.values))
    v1 = fr[1].reindex(trig).dropna()
    print("  h1", cluster_note(v1.index, v1.values))
    print("  dates:", [str(d.date()) for d in trig])
    return trig


m_a = (streak >= 5) & (r5 <= -0.08)
report(m_a, "A: 5+ down closes and 5d <= -8%")
m_b = (streak >= 5) & (r5 <= -0.08) & (c > sma200)
tb = report(m_b, "B: A while above the 200d")
m_c = (r5 <= -0.10) & (c > sma200)
tc = report(m_c, "C: 5d <= -10% while above the 200d (no streak req)")
m_d = (r5 <= -0.10) & (r63 >= 0.15)
report(m_d, "D: 5d <= -10% after a 63d gain >= 15%")

# Wednesday (EIA) next-day for C episodes whose next session is a Wednesday
nxt = pd.Series(c.index[1:], index=c.index[:-1])
wed = [d for d in tc if d in nxt.index and nxt[d].weekday() == 2]
v = fr[1].reindex(wed).dropna()
show([summarize(v.values, "C, next session a Wednesday")], "EIA-day subset")
