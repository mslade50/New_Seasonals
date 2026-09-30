"""d11 round 1: long equity vol into the midterm October.

Anchor: signal = September QE-3 close, entry = QE-2 close (lag 1), exit +5 / +10 sessions.
2026: signal 09-25, entry 09-28 close, QE = 09-30.
Objects: ^VIX % change (not tradeable, the proxy), SPY return (short SPY = the alternative
expression), SVXY 2018-03+ and its SPY residual (r_SVXY - 1.48 r_SPY; long vol = short it),
synthetic -0.5x SVXY 2011-10..2018-02 (daily returns x0.5) for the residual pre-break.
Competitors: Sept anchor in ALL years (plain seasonal), non-midterm Septembers, and the
same ME-2 window in every other month (month-of-year ladder), plus offset placebo QE-8..QE+4.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

px = close_panel(["^VIX", "SPY", "SVXY"])
px = px[px["SPY"].notna()]
idx = px.index
BREAK = pd.Timestamp("2018-02-28")
r_sv = px["SVXY"].pct_change(fill_method=None)
r_sv_adj = r_sv.where(idx >= BREAK, 0.5 * r_sv)
S = (1 + r_sv_adj.fillna(0)).cumprod()
S[idx <= px["SVXY"].first_valid_index()] = np.nan
px["S05"] = S
MID = {2002, 2006, 2010, 2014, 2018, 2022}
PRES = {2000, 2004, 2008, 2012, 2016, 2020, 2024}
BETA = 1.48

print(f"LIVE: last bar {idx[-1].date()}  VIX {px['^VIX'].iloc[-1]:.2f}  SVXY {px['SVXY'].iloc[-1]:.2f}")
sep26 = idx[(idx.year == 2026) & (idx.month == 9)]
print(f"  Sept 2026 sessions in cache end {sep26[-1].date()} (QE 09-30 not yet printed); "
      f"signal QE-3 = 2026-09-25 is the last bar -> gate is live by calendar, 2026 is a midterm.")


def seg(col, p0, p1):
    s = px[col].values
    if p0 < 0 or p1 >= len(s) or np.isnan(s[p0]) or np.isnan(s[p1]):
        return np.nan
    return s[p1] / s[p0] - 1.0


def window(y, m, off, h):
    mdays = idx[(idx.year == y) & (idx.month == m)]
    if len(mdays) == 0 or mdays[-1] == idx[-1]:
        return None
    qe = idx.get_loc(mdays[-1])
    ent = qe + off
    ex = ent + h
    if ex >= len(idx):
        return None
    vix = seg("^VIX", ent, ex)
    spy = seg("SPY", ent, ex)
    sv = seg("SVXY", ent, ex) if idx[ent] > BREAK else np.nan
    s05 = seg("S05", ent, ex)
    return {"year": y, "month": m, "entry": idx[ent].date(), "vix": vix, "spy": spy,
            "svxy": sv, "svxy_res": sv - BETA * spy if not np.isnan(sv) else np.nan,
            "s05_res": s05 - (BETA if idx[ent] > BREAK else 1.98) * spy if not np.isnan(s05) else np.nan,
            "vix_sig": px["^VIX"].values[ent - 1]}


rows = []
for y in range(2000, 2027):
    for m in range(1, 13):
        for h in (5, 10):
            for off in range(-8, 5):
                w = window(y, m, off, h)
                if w:
                    w["h"], w["off"] = h, off
                    rows.append(w)
D = pd.DataFrame(rows)
D["cycle"] = np.where(D.year.isin(MID), "mid", np.where(D.year.isin(PRES), "pres", "odd"))


def stat(v, lbl, flip=False):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if flip:
        v = -v
    r = summarize(v, lbl)
    if r["n"]:
        w = int((v > 0).sum())
        r["rec"] = f"{w}-{len(v)-w}"
        r["sign_p"] = round(sign_test(w, len(v)), 4)
    return r


for h in (5, 10):
    base = D[(D.h == h) & (D.off == -2)]
    sep = base[base.month == 9]
    print(f"\n######## h={h}, entry QE-2 ########")
    print(sep[["year", "cycle", "entry", "vix_sig", "vix", "spy", "svxy", "svxy_res", "s05_res"]]
          .assign(vix=lambda d: (100 * d.vix).round(2), spy=lambda d: (100 * d.spy).round(2),
                  svxy=lambda d: (100 * d.svxy).round(2), svxy_res=lambda d: (100 * d.svxy_res).round(2),
                  s05_res=lambda d: (100 * d.s05_res).round(2)).to_string(index=False))
    mid, non = sep[sep.cycle == "mid"], sep[sep.cycle != "mid"]
    show([stat(mid.vix, "VIX% Sept MIDTERM"), stat(non.vix, "VIX% Sept non-mid"),
          stat(sep.vix, "VIX% Sept ALL years"),
          stat(base[base.month != 9].vix, "VIX% other months all yrs"),
          stat(base[(base.month != 9) & (base.cycle == "mid")].vix, "VIX% other months MIDTERM"),
          stat(mid.spy, "SHORT SPY Sept MIDTERM", flip=True),
          stat(non.spy, "SHORT SPY Sept non-mid", flip=True),
          stat(base.spy, "SHORT SPY all months all yrs", flip=True),
          stat(mid.s05_res, "short S05 residual Sept MIDTERM (long vol)", flip=True),
          stat(non.s05_res, "short S05 residual Sept non-mid", flip=True),
          stat(base.s05_res, "short S05 residual all months", flip=True)],
         f"h={h} Sept QE-2 cells (VIX: + = vol up; SHORT rows flip sign so + = trade wins)")
    # month-of-year ladder: VIX % and interaction
    lad = []
    for m in range(1, 13):
        b = base[base.month == m]
        lad.append({"month": m, "vix_all": 100 * b.vix.mean(),
                    "vix_mid": 100 * b[b.cycle == "mid"].vix.mean(),
                    "vix_non": 100 * b[b.cycle != "mid"].vix.mean(),
                    "mid_minus_non": 100 * (b[b.cycle == "mid"].vix.mean() - b[b.cycle != "mid"].vix.mean()),
                    "spy_mid": 100 * b[b.cycle == "mid"].spy.mean(),
                    "n_mid": int(b[b.cycle == "mid"].vix.notna().sum())})
    L = pd.DataFrame(lad)
    show(L.to_dict("records"), f"h={h} month-of-year ladder (ME-2 entry), VIX % change")
    for c in ("vix_all", "vix_mid", "mid_minus_non"):
        print(f"  Sept rank on {c}: {int(L[c].rank(ascending=False)[8])} of 12")
    # offset placebo, Sept midterm
    off = D[(D.h == h) & (D.month == 9)]
    ol = []
    for o in range(-8, 5):
        b = off[off.off == o]
        ol.append({"off": o, "vix_mid": 100 * b[b.cycle == "mid"].vix.mean(),
                   "vix_non": 100 * b[b.cycle != "mid"].vix.mean(),
                   "diff": 100 * (b[b.cycle == "mid"].vix.mean() - b[b.cycle != "mid"].vix.mean()),
                   "spy_mid": 100 * b[b.cycle == "mid"].spy.mean(),
                   "s05res_mid": 100 * b[b.cycle == "mid"].s05_res.mean()})
    O = pd.DataFrame(ol)
    show(O.to_dict("records"), f"h={h} Sept offset placebo QE-8..QE+4")
    print(f"  QE-2 rank on vix_mid: {int(O.vix_mid.rank(ascending=False)[6])} of 13; "
          f"on diff: {int(O['diff'].rank(ascending=False)[6])} of 13")
    # low-VIX split (today VIX 14.87)
    lowv = sep[sep.vix_sig < 17]
    show([stat(lowv[lowv.cycle == "mid"].vix, "VIX% Sept mid, VIX<17 at signal"),
          stat(lowv[lowv.cycle != "mid"].vix, "VIX% Sept non-mid, VIX<17"),
          stat(lowv.vix, "VIX% Sept all, VIX<17")], f"h={h} low-VIX split")
    # era
    show([stat(mid[mid.year < 2018].vix, "VIX% mid pre-2018"), stat(mid[mid.year >= 2018].vix, "VIX% mid 2018+"),
          stat(non[non.year < 2018].vix, "VIX% non pre-2018"), stat(non[non.year >= 2018].vix, "VIX% non 2018+")],
         f"h={h} era")
