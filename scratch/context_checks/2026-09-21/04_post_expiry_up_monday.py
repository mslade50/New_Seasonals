"""The post-September-expiry Monday went UP 1.49% (S&P). Thursday priced the whole
five-session window (SPY lower in 19 of 26), Sunday priced the Monday (18 of 26 down).
New question only: when the Monday rallied, what did Tuesday through Friday do?

Anchor = the Monday's close (today's analogue). h1 = Tuesday, h4 = Friday.
Control 1: the same, all 12 monthly expiries. Control 2: every Monday up 1%+.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import close_panel, sign_test, cluster_note, load_events  # noqa

px = close_panel(["^GSPC", "IWM", "QQQ", "SPY"]).dropna(subset=["^GSPC"])
px = px[px.index >= "1999-01-01"]
spx = px["^GSPC"]
ret = spx.pct_change()
idx = px.index
pos = pd.Series(range(len(idx)), index=idx)

ev = load_events(["opex", "quad_witching"])
opex = sorted(set(ev["date"]))


def next_session(d):
    n = idx[idx > d]
    return n[0] if len(n) else None


def fwd(s, d, h):
    p = pos.get(d)
    if p is None or p + h >= len(idx):
        return np.nan
    return s.iloc[p + h] / s.iloc[p] - 1


rows = []
for d in opex:
    if d not in pos.index:
        continue
    m = next_session(d)
    if m is None:
        continue
    rows.append({"expiry": d, "post": m, "month": d.month, "mon_ret": ret.get(m),
                 "h1": fwd(spx, m, 1), "h4": fwd(spx, m, 4),
                 "iwm_h4": fwd(px["IWM"], m, 4), "qqq_h4": fwd(px["QQQ"], m, 4),
                 "is_mon": m.dayofweek == 0})
df = pd.DataFrame(rows).dropna(subset=["mon_ret"])
df = df[df["post"] < idx[-1]]


def rep(v, label):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        print(f"  {label:48} n 0")
        return
    up = int((v > 0).sum())
    t = v.mean() / (v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan
    print(f"  {label:48} n {len(v):3}  mean {100*v.mean():+6.2f}%  median {100*np.median(v):+6.2f}%  "
          f"rec {up}-{len(v)-up}  signp_dn {sign_test(len(v)-up, len(v)):.4f}  signp_up {sign_test(up, len(v)):.4f}  t {t:+.2f}")


sep = df[df.month == 9]
print("=== September expiry: the session after, then Tue-Fri from its close ===")
print(sep[["expiry", "post", "mon_ret", "h1", "h4", "iwm_h4", "qqq_h4"]].assign(
    mon_ret=lambda x: 100 * x.mon_ret, h1=lambda x: 100 * x.h1, h4=lambda x: 100 * x.h4,
    iwm_h4=lambda x: 100 * x.iwm_h4, qqq_h4=lambda x: 100 * x.qqq_h4).round(2).to_string(index=False))
up_m = sep[sep.mon_ret > 0]
dn_m = sep[sep.mon_ret <= 0]
rep(up_m.h1, "Sep: post-expiry session UP -> next day")
rep(up_m.h4, "Sep: post-expiry session UP -> h4")
rep(dn_m.h1, "Sep: post-expiry session DOWN -> next day")
rep(dn_m.h4, "Sep: post-expiry session DOWN -> h4")
rep(sep.h4, "Sep: all -> h4")
big = sep[sep.mon_ret >= 0.01]
rep(big.h4, "Sep: post-expiry session +1%+ -> h4")
print("  those years:", [str(d.date()) for d in big.post])

print("\n=== all 12 expiries: post-expiry session up 1%+ ===")
allbig = df[df.mon_ret >= 0.01]
rep(allbig.h1, "all months, +1%+ -> next day")
rep(allbig.h4, "all months, +1%+ -> h4")
rep(df[df.mon_ret < 0.01].h4, "all months, under +1% -> h4")
rep(df[df.month != 9].h4, "all months ex-Sep, all -> h4")
print("\n=== control: every Monday up 1%+ (S&P), h4 ===")
mon_up = idx[(idx.dayofweek == 0) & (ret >= 0.01).reindex(idx).fillna(False).values]
mon_up = mon_up[mon_up < idx[-1]]
rep([fwd(spx, d, 4) for d in mon_up], "every Monday +1%+ -> h4")
rep([fwd(spx, d, 1) for d in mon_up], "every Monday +1%+ -> h1")
sep_mon_up = [d for d in mon_up if d.month == 9]
rep([fwd(spx, d, 4) for d in sep_mon_up], "September Mondays +1%+ -> h4")
allh4 = pd.Series([fwd(spx, d, 4) for d in idx[:-5]])
print(f"  all sessions h4 mean {100*allh4.mean():+.3f}% hit {100*(allh4>0).mean():.1f}%")
