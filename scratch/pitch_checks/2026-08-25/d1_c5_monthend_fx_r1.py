"""C5 round 1 - the month-end FX rebalancing flow, ME-4 close to ME-0 close.

MECHANISM, pre-specified (Melvin-Prins style), NOT a grid I built:
foreign investors hedge their US equity exposure with short-USD forwards of
fixed notional. When US equities OUTPERFORM foreign equities over a month, the
hedged book is under-hedged in USD and hedgers must SELL dollars at the
month-end fix; when foreign equities outperform, the flow reverses and dollars
are BOUGHT. So the signed hypothesis is:

    dollar return over the last sessions of the month  ~  -beta * (US equity
    month return  -  foreign equity month return)

That is ONE signed relationship with a stated sign, so there is no search to
charge. The trade's DIRECTION today falls out of today's reading rather than
out of picking the better-looking tail.

Today (2026-08-25) is ME-4: the last August session is Mon 2026-08-31, so an
entry MOC today held to the month-end close is h=4. Signal is measured at the
ME-5 close (2026-08-24), the last close before the entry decision - lag-1 safe
and an exact mirror of the live situation.

Registry context this must clear:
- 2026-08-24 CLOSED month-end on EQUITIES (SPY ME-1 -> ME-0 pays -0.006% at a
  47.6% hit; 60-cell grid Sidak 0.877) and SUSPENDED it on rates (mechanism
  decayed: TLT ME-1 -> ME-0 fell from +25.65 bp t=3.09 to +3.99 bp t=0.37).
  The lesson attached to both: DECOMPOSE THE WINDOW BY SESSION and check the
  session the mechanism names is the one that pays.
- FX month-end has never been examined in this repo.
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

TK = ["DX-Y.NYB", "UUP", "EURUSD=X", "SPY", "EFA", "EEM", "QQQ", "TLT", "GLD"]
px = close_panel(TK)
print("panel", px.index[0].date(), "->", px.index[-1].date(), px.shape)

# ---------------------------------------------------------------- calendar
idx = px.index
ym = pd.Series(list(zip(idx.year, idx.month)), index=idx)
# sessions remaining to the month's LAST session (0 = the month-end close)
pos = pd.Series(0, index=idx)
for _, grp in ym.groupby(ym):
    d = grp.index
    pos.loc[d] = np.arange(len(d) - 1, -1, -1)

ANCH = 4  # today's offset: entry MOC ME-4, exit MOC ME-0
print(f"\nanchor = ME-{ANCH} (entry MOC), exit MOC at ME-0, hold = {ANCH} sessions")


def me_window(t, anchor=ANCH):
    """Return (anchor_dates, entry->ME0 returns) for one ticker, complete months only."""
    s = px[t].dropna()
    p = pos.reindex(s.index)
    # rebuild pos on the ticker's own session index so a ticker with holes
    # cannot inherit the panel's month boundaries
    ym2 = pd.Series(list(zip(s.index.year, s.index.month)), index=s.index)
    p = pd.Series(0, index=s.index)
    for _, grp in ym2.groupby(ym2):
        d = grp.index
        p.loc[d] = np.arange(len(d) - 1, -1, -1)
    dates, rets, sig_dates = [], [], []
    for a in s.index[p == anchor]:
        i = s.index.get_loc(a)
        j = i + anchor
        if j >= len(s) or i == 0:
            continue
        # exclude the live incomplete month
        if p.iloc[j] != 0:
            continue
        dates.append(a)
        rets.append(s.iloc[j] / s.iloc[i] - 1.0)
        sig_dates.append(s.index[i - 1])       # the ME-5 close = signal date
    return pd.DatetimeIndex(dates), np.array(rets), pd.DatetimeIndex(sig_dates)


# --------------------------------------------------- the unconditional cell
print("\n=== 1. UNCONDITIONAL: ME-4 close -> ME-0 close ===")
for t in ["DX-Y.NYB", "UUP", "EURUSD=X"]:
    d, r, _ = me_window(t)
    show([summarize(r, f"{t} ME-{ANCH}->ME-0  (N={len(r)})")])
    # control: every 4-session window on the same instrument
    s = px[t].dropna()
    allr = (s.shift(-ANCH) / s - 1.0).dropna().values
    show([summarize(allr, f"{t} CTRL-a own drift, all {ANCH}-session windows")])

# ------------------------------------------------------- the signed signal
print("\n=== 2. THE MECHANISM: dollar leg vs relative equity month return ===")
print("signal = SPY month-to-date  minus  EFA month-to-date, measured at the ME-5 close")


def mtd(t, sig_date):
    """month-to-date return at sig_date: prior month's last close -> sig_date close."""
    s = px[t].dropna()
    if sig_date not in s.index:
        return np.nan
    i = s.index.get_loc(sig_date)
    y, m = sig_date.year, sig_date.month
    j = i
    while j > 0 and (s.index[j - 1].year, s.index[j - 1].month) == (y, m):
        j -= 1
    if j == 0:
        return np.nan
    return s.iloc[i] / s.iloc[j - 1] - 1.0


for vt in ["DX-Y.NYB", "UUP"]:
    d, r, sd = me_window(vt)
    rows = []
    for a, ret, sg in zip(d, r, sd):
        us = mtd("SPY", sg)
        fx = mtd("EFA", sg)
        if np.isnan(us) or np.isnan(fx):
            continue
        rows.append({"anchor": a, "ret": ret, "us": us, "fx": fx, "rel": us - fx})
    df = pd.DataFrame(rows).dropna()
    print(f"\n--- {vt}   N={len(df)}  {df.anchor.min().date()} .. {df.anchor.max().date()}")
    x = df["rel"].values
    y = df["ret"].values
    b, a0 = np.polyfit(x, y, 1)
    n = len(x)
    resid = y - (a0 + b * x)
    se = np.sqrt((resid ** 2).sum() / (n - 2) / ((x - x.mean()) ** 2).sum())
    print(f"  regression  dollar_ret = {a0*100:+.4f}% + {b:+.4f} * (SPY_mtd - EFA_mtd)")
    print(f"  slope t = {b/se:+.2f}   (mechanism predicts a NEGATIVE slope)")
    print(f"  R^2 = {1 - (resid**2).sum()/((y-y.mean())**2).sum():.4f}")

    # terciles of the signal
    q = np.quantile(x, [1/3, 2/3])
    for lab, m in [("US LAGS  (rel low)", x <= q[0]),
                   ("middle", (x > q[0]) & (x < q[1])),
                   ("US LEADS (rel high)", x >= q[1])]:
        v = y[m]
        print(f"  {lab:<22} N={m.sum():>3}  mean={v.mean()*100:+.3f}%  med={np.median(v)*100:+.3f}%  hit(up)={(v>0).mean()*100:5.1f}%")

    # today's reading
    live_us = mtd("SPY", px.index[-1])
    live_fx = mtd("EFA", px.index[-1])
    print(f"  LIVE 2026-08-24: SPY mtd {live_us*100:+.2f}%  EFA mtd {live_fx*100:+.2f}%  rel {(live_us-live_fx)*100:+.2f}pp")
    print(f"  live rel percentile within the sample: {(x < (live_us-live_fx)).mean()*100:.1f}")

# ------------------------------------------- session decomposition (the 08-24 lesson)
print("\n=== 3. SESSION DECOMPOSITION - which session actually pays? ===")
print("the mechanism names the FIX, i.e. the last session. Check it.")
for vt in ["DX-Y.NYB", "UUP"]:
    s = px[vt].dropna()
    ym2 = pd.Series(list(zip(s.index.year, s.index.month)), index=s.index)
    p = pd.Series(0, index=s.index)
    for _, grp in ym2.groupby(ym2):
        d = grp.index
        p.loc[d] = np.arange(len(d) - 1, -1, -1)
    ret1 = (s / s.shift(1) - 1.0)
    print(f"\n  --- {vt}")
    for k in range(0, 8):
        m = (p == k)
        v = ret1[m].dropna().values
        if len(v) < 20:
            continue
        # era split
        e = ret1[m].dropna()
        pre = e[e.index < "2013-01-01"].values
        post = e[e.index >= "2020-01-01"].values
        print(f"   ME-{k}: N={len(v):>4} mean={v.mean()*1e4:+7.2f}bp hit={(v>0).mean()*100:5.1f}%"
              f" | pre2013 {pre.mean()*1e4:+7.2f}bp | 2020+ {post.mean()*1e4:+7.2f}bp")
    allv = ret1.dropna().values
    print(f"   ALL DAYS base: mean={allv.mean()*1e4:+7.2f}bp  hit={(allv>0).mean()*100:5.1f}%")
