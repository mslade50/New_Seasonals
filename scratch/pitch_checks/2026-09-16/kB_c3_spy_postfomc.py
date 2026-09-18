"""c3 round 1: long SPY MOC on the FOMC decision close, exit h=1..3.

Anchor = decision session p (entry close p, exit close p+h); gates on eve p-1.
Live state 2026-09-16: midterm, SPY above 200d (+6.15%), r21 rank 7.1, VIX
expiry collision, quad witching at p+2.
"""
from kB_common import *  # noqa

s = ser("SPY")
idx, c = s.index, s.values
verify_alignment(idx)
dp = dec_pos(idx, hmax=10, pre=252)
ds = pd.DatetimeIndex([d for d, _ in dp])
ps = np.array([p for _, p in dp])
print(f"decisions measured: {len(ps)}  {ds[0].date()} .. {ds[-1].date()}")

sma200 = s.rolling(200).mean().values
r21 = pct_rank(s, 21).values
irx = ser("^IRX").reindex(idx).ffill().values
vx = vix_exp()
qw = quad()

eve = ps - 1
mid = ds.year % 4 == 2
above = c[eve] > sma200[eve]
post18 = ds >= "2018-01-01"
coll = np.array([d in vx for d in ds])
weak = r21[eve] <= 20
irx_chg = irx[eve] - irx[np.maximum(eve - 126, 0)]
cyc = np.where(irx_chg > 0.25, "hike", np.where(irx_chg < -0.25, "cut", "hold"))
sept = ds.month == 9

excl = set()
for p in ps:
    excl.update(range(p - 1, p + 4))

for h in (1, 2, 3):
    v = np.array([c[p + h] / c[p] - 1.0 for p in ps])
    ra = fwd_arr(c, h)
    loc = local_control(idx[:-h], idx[ps])
    locv = pd.Series(ra, index=idx).loc[loc].values
    tc = tdom_ctrl(idx, c, ps, h, exclude=excl)
    tmc = tdom_month_ctrl(idx, c, ps, h, exclude=excl)
    qin = np.array([((qw > idx[p]) & (qw <= idx[p + h])).any() for p in ps])
    print(f"\n==================== h={h} ====================")
    print(line("ALL decisions", v))
    print(line("CTRL all days", ra))
    print(line("CTRL local +/-126td", locv))
    print(f"  tdom-matched ctrl mean {100*np.nanmean(tc):+.3f}% -> excess "
          f"{100*(v.mean()-np.nanmean(tc)):+.3f}pp ; tdom+month ctrl "
          f"{100*np.nanmean(tmc):+.3f}% -> excess {100*np.nanmean(v-tmc):+.3f}pp")
    for lbl, m in [("midterm", mid), ("non-midterm", ~mid),
                   ("pre-2018", ~post18), ("2018+", post18),
                   ("eve above 200d", above), ("eve below 200d", ~above),
                   ("midterm & above200", mid & above),
                   ("non-mid & above200", ~mid & above),
                   ("midterm & below200", mid & ~above),
                   ("VIX-exp collision", coll), ("no collision", ~coll),
                   ("midterm & collision", mid & coll),
                   ("quad witching in hold", qin), ("no quad in hold", ~qin),
                   ("midterm & quad in hold", mid & qin),
                   ("eve r21<=20 (weak run-in)", weak),
                   ("midterm & r21<=20", mid & weak),
                   ("midterm & above & r21<=20", mid & above & weak),
                   ("September", sept), ("midterm September", mid & sept),
                   ("hiking cycle", cyc == "hike"), ("cutting cycle", cyc == "cut"),
                   ("hold", cyc == "hold")]:
        print(line("  " + lbl, v[m]))
    print(f"  midterm tdom excess {100*np.nanmean((v-tc)[mid]):+.3f}pp ; "
          f"non-mid {100*np.nanmean((v-tc)[~mid]):+.3f}pp")
    print("  concentration (all):", cluster_note(ds, v))
    if mid.sum():
        print("  concentration (midterm):", cluster_note(ds[mid], v[mid]))
    df, r0 = placebo(c, ps, h)
    dfm, r0m = placebo(c, ps[mid], h)
    print(f"  PLACEBO all: k=0 rank {r0}/11 ; midterm: k=0 rank {r0m}/11")
    print("   k:    " + " ".join(f"{k:+7d}" for k in df.k))
    print("   all:  " + " ".join(f"{m:+7.3f}" for m in df.mean_pct))
    print("   mid:  " + " ".join(f"{m:+7.3f}" for m in dfm.mean_pct))

print("\n==================== CMVJ week-0 day decomposition ====================")
for j in (-1, 0, 1, 2, 3):
    v = np.array([c[p + j] / c[p + j - 1] - 1.0 for p in ps])
    alld = fwd_arr(c, 1)
    print(line(f"day {j:+d} all", v), "| mid:", f"{100*v[mid].mean():+.3f}%",
          f"{int((v[mid]>0).sum())}-{int((v[mid]<=0).sum())}",
          "| non-mid:", f"{100*v[~mid].mean():+.3f}%",
          "| mid&above:", f"{100*v[mid & above].mean():+.3f}%")
print(f"all-days 1d mean {100*np.nanmean(fwd_arr(c,1)):+.3f}%")

print("\n==================== midterm episodes (h=1,2,3) ====================")
for i in np.where(mid)[0]:
    p = ps[i]
    rr = [100 * (c[p + h] / c[p] - 1) for h in (1, 2, 3)]
    print(f"  {ds[i].date()} above200={above[i]!s:5} coll={coll[i]!s:5} r21="
          f"{r21[p-1]:5.1f} cyc={cyc[i]:4} h1 {rr[0]:+.2f} h2 {rr[1]:+.2f} h3 {rr[2]:+.2f}")
print("\n2026 episodes:")
for i in np.where(ds.year == 2026)[0]:
    p = ps[i]
    print(f"  {ds[i].date()} " + " ".join(
        f"h{h} {100*(c[p+h]/c[p]-1):+.2f}" for h in (1, 2, 3)))
