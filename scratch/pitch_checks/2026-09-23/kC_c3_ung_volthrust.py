"""C3 round 1: long UNG after a volume-confirmed thrust day.
Trigger: UNG 1d >= +5% AND volume >= 3x its 63d mean (mean incl. today, as
_metrics_for defines vol_vs_63d). Pre-specified sign: LONG continuation,
h=1..3, lag=1. Matched control: +5% UNG days WITHOUT the volume leg.
UNG returns already carry the roll drag (actual ETF closes), so the absolute
number is net of it. NG=F front shown for contrast (roll seams)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd


def build(t="UNG"):
    d = load_prices([t])[t]
    c, v = d["Close"].astype(float), d["Volume"].astype(float)
    r1 = c.pct_change()
    vr = v / v.rolling(63).mean()
    return d, pd.DataFrame({t: c}), r1, vr


def eps(ret, mask, h, idx):
    s = idx[mask.reindex(idx, fill_value=False).values & ret.notna().values]
    e = declusters(s, h, idx)
    return e, ret.loc[e].values


if __name__ == "__main__":
    d, px, r1, vr = build()
    print("LIVE STATE CHECK:")
    print(pd.DataFrame({"close": d["Close"], "r1_pct": 100 * r1, "vol_x63": vr,
                        "vol": d["Volume"]}).tail(6).round(3).to_string())
    print(f"UNG close {d['Close'].iloc[-1]:.2f} -> 1 cent = "
          f"{1e4*0.01/d['Close'].iloc[-1]:.1f} bps")
    trig = (r1 >= 0.05) & (vr >= 3)
    ctrl = (r1 >= 0.05) & (vr < 3)
    print(f"trigger days={int(trig.sum())}, no-volume +5% days={int(ctrl.sum())}")
    for h in (1, 3):
        battery(px, trig, [("UNG", 1.0)], h, "C3 long UNG after vol-confirmed thrust",
                cost_bps=6.0, event_kinds=("nfp",))
    idx = px.index
    rows = []
    for h in (1, 2, 3, 5):
        ret = vehicle_ret(px, [("UNG", 1.0)], h)
        e, v = eps(ret, trig, h, idx)
        em, vm = eps(ret, ctrl, h, idx)
        ep_, vp = eps(ret, r1 >= 0.05, h, idx)
        rows += [summarize(v, f"h={h} CHILD +5% & vol>=3x N={len(e)}"),
                 summarize(vm, f"h={h} MATCHED +5% & vol<3x N={len(em)}"),
                 summarize(vp, f"h={h} PARENT any +5% N={len(ep_)}")]
        if h in (1, 3):
            w = int((v > 0).sum())
            print(f"  h={h} child record {w}-{len(v)-w} sign p={sign_test(w, len(v)):.4f}")
            show(era_split(e, v, "2011-01-01"), f"h={h} era split at 2011 (UNG 2009 creation halt)")
            mon = pd.DatetimeIndex(e).month
            so = np.isin(mon, [9, 10])
            show([summarize(v[so], f"Sep-Oct N={int(so.sum())}"),
                  summarize(v[~so], f"other months N={int((~so).sum())}")],
                 f"h={h} shoulder-season split")
    show(rows, "C3 child vs matched (no volume) vs parent (long UNG, episodes)")
    # EIA storage (Thursday) inside h=1: signal on Tuesday -> exit Thursday close
    ret1 = vehicle_ret(px, [("UNG", 1.0)], 1)
    e, v = eps(ret1, trig, 1, idx)
    tue = pd.DatetimeIndex(e).weekday == 1
    show([summarize(v[tue], f"h=1 signal Tue (hold = Thu storage day) N={int(tue.sum())}"),
          summarize(v[~tue], f"h=1 other weekdays N={int((~tue).sum())}")],
         "C3 storage-day split")
    for dd, vv in zip(e, v):
        print(f"   {dd.date()}  h1 {100*vv:+.2f}%")
    # NG=F front contrast
    ng = close_panel(["NG=F"]).dropna()
    r1n = ng["NG=F"].pct_change()
    ngv = load_prices(["NG=F"])["NG=F"]["Volume"].astype(float)
    trig_ung_days = trig[trig].index
    m = pd.Series(ng.index.isin(trig_ung_days), index=ng.index)
    for h in (1, 3):
        r = vehicle_ret(ng, [("NG=F", 1.0)], h)
        e2, v2 = eps(r, m, h, ng.index)
        show([summarize(v2, f"NG=F on UNG trigger days h={h} N={len(e2)}"),
              summarize(r.dropna().values, f"NG=F all days h={h}")], "front-futures contrast")
