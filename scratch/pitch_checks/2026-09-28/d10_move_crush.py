"""d10 round 1: long TLT/IEF from the first >= 5% ^MOVE fall inside 5 sessions of a
top-3% daily ^MOVE rise (the crush after a rates-vol spike).

Pre-written mechanism (C10 block): vol-scaled holders (risk parity, CTA vol targeting,
bank VaR) cut duration on the spike and re-add once implied vol turns down, so TLT/IEF
pay at h=1..5 from the crush close (entry lag 1).

Spike threshold is EX-ANTE: expanding 97th pctile of ^MOVE daily valid-session moves
(min 252 obs), shifted one session. Full-sample quantile shown for contrast.
Sections: A live gate; B pattern vs controls (all days, any >=5% fall, spike day itself);
C era; D filter_vs_reanchor (parent = spike day, child = crush day); E battery.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

px = close_panel(["^MOVE", "TLT", "IEF", "SPY"])
px = px[px["TLT"].notna()]
px = px[px.index >= "2002-11-12"]
idx = px.index

mv = px["^MOVE"].dropna()
mret = mv / mv.shift(1) - 1.0
q97 = mret.expanding(252).quantile(0.97).shift(1)
q97_full = mret.quantile(0.97)
spike = mret >= q97
fall = mret <= -0.05


def crush_mask(spike_s: pd.Series, fall_s: pd.Series, win: int = 5) -> pd.Series:
    out = pd.Series(False, index=spike_s.index)
    sp = spike_s.values
    fa = fall_s.values
    n = len(sp)
    for i in range(n):
        if not fa[i]:
            continue
        lo = max(0, i - win)
        js = [j for j in range(lo, i) if sp[j]]
        if not js:
            continue
        last = js[-1]
        if not fa[last + 1:i].any():
            out.iloc[i] = True
    return out


crush = crush_mask(spike.fillna(False), fall.fillna(False))
crush_full = crush_mask((mret >= q97_full).fillna(False), fall.fillna(False))
spike_px = spike.reindex(idx, fill_value=False)
crush_px = crush.reindex(idx, fill_value=False)
crush_full_px = crush_full.reindex(idx, fill_value=False)
fall_px = fall.reindex(idx, fill_value=False)

print("A. LIVE GATE")
for d in mv.index[-5:]:
    print(f"  {d.date()} MOVE {mv[d]:7.2f} ret {100*mret[d]:+6.2f}%  q97 {100*q97[d]:5.2f}%  "
          f"spike {bool(spike[d])}  fall {bool(fall[d])}  crush {bool(crush[d])}")
print(f"  full-sample q97 = {100*q97_full:.2f}%")
tlt_lo = rolling_on_valid(px["TLT"], lambda x: x.rolling(252).min())
at_low = (px["TLT"] <= tlt_lo * 1.01)
print(f"  TLT within 1% of 252 low on 09-25: {bool(at_low.iloc[-1])}")

rows = []
for tk in ("TLT", "IEF"):
    for h in (1, 2, 3, 5):
        ret = vehicle_ret(px, [(tk, 1.0)], h)
        valid = ret.notna()
        base = ret[valid]

        def ep(mask, lbl, gap=5):
            d = idx[mask.values & valid.values]
            e = declusters(d, gap, idx)
            v = ret.loc[e].values
            r = summarize(v, lbl)
            if r["n"]:
                w = int((v > 0).sum())
                r["rec"] = f"{w}-{len(v)-w}"
                r["sign_p"] = round(sign_test(w, len(v)), 4)
                r["edge_pp"] = r["mean_pct"] - 100 * base.mean()
            return r, e, v

        r1, e1, v1 = ep(crush_px, f"{tk} h{h} CRUSH ex-ante")
        r2, _, _ = ep(crush_full_px, f"{tk} h{h} CRUSH full-q")
        r3, _, _ = ep(fall_px & ~crush_px, f"{tk} h{h} other >=5% falls")
        r4, _, _ = ep(spike_px, f"{tk} h{h} SPIKE day itself")
        r5 = summarize(base.values, f"{tk} h{h} all days")
        rows += [r1, r2, r3, r4, r5]
show(rows, "B. pattern vs controls (episodes, decluster 5td)")

# C. era + TLT-at-low split for the primary (TLT, h=3 and h=5)
for h in (3, 5):
    ret = vehicle_ret(px, [("TLT", 1.0)], h)
    d = idx[crush_px.values & ret.notna().values]
    e = declusters(d, 5, idx)
    v = ret.loc[e].values
    show(era_split(e, v), f"C. TLT h={h} crush era split")
    lo = at_low.reindex(e).values.astype(bool)
    show([summarize(v[lo], "TLT within 1% of 252 low"),
          summarize(v[~lo], "TLT not at low")], f"C. TLT h={h} crush by TLT-at-low (today = at low)")
    print("  episodes:", ", ".join(f"{x.date()}:{100*y:+.2f}" for x, y in zip(e, v)))

# D. filter vs reanchor: parent = spike day, child = crush day (window 5)
for h in (1, 3, 5):
    ret = vehicle_ret(px, [("TLT", 1.0)], h)
    sp_e = declusters(idx[spike_px.values], 5, idx)
    cr_e = declusters(idx[crush_px.values], 5, idx)
    par = pd.Series(idx.isin(sp_e), index=idx)
    chi = pd.Series(idx.isin(cr_e), index=idx)
    fr = filter_vs_reanchor(ret, par, chi, idx, window_td=6, label=f"TLT h={h} spike -> crush")
    if fr["n_matched"]:
        rn = reanchor_null(ret, [a for a, _, _ in fr["pairs"]], fr["shifts"], idx,
                           ret.reindex([b for _, b, _ in fr["pairs"]]).mean())
        print(f"  reanchor_null p = {rn['p']:.3f} (null mean {rn['null_mean_pct']:+.3f}%)")

# E. battery on the primary
battery(px, crush_px, [("TLT", 1.0)], 3, "C10 TLT long after MOVE crush", cost_bps=3,
        min_gap=5, event_kinds=("nfp",),
        variants={"fall<=-4%": crush_mask(spike.fillna(False), (mret <= -0.04).fillna(False)).reindex(idx, fill_value=False),
                  "fall<=-7%": crush_mask(spike.fillna(False), (mret <= -0.07).fillna(False)).reindex(idx, fill_value=False),
                  "spike q95": crush_mask((mret >= mret.expanding(252).quantile(0.95).shift(1)).fillna(False), fall.fillna(False)).reindex(idx, fill_value=False),
                  "spike q99": crush_mask((mret >= mret.expanding(252).quantile(0.99).shift(1)).fillna(False), fall.fillna(False)).reindex(idx, fill_value=False),
                  "win 3": crush_mask(spike.fillna(False), fall.fillna(False), 3).reindex(idx, fill_value=False),
                  "win 10": crush_mask(spike.fillna(False), fall.fillna(False), 10).reindex(idx, fill_value=False)})
