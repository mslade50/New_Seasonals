"""kA round 1: c1 long TLT / c5 short DX / c6 long GLD, MOC on the FOMC
DECISION close, h=1..5, gated on ^TNX at (or near) its trailing-252 max on
the EVE close. Signs fixed in advance (Hillenbrand 2021 prior: long yields
fall around FOMC announcements -> TLT up, dollar down, gold up).

Alignment: returns are measured on the vehicle's OWN index, anchored on the
decision session D with lag=0 (entry close D, exit close D+h). Gates read the
^TNX close of the last TNX session strictly before D (the eve, known at
publish). Cross-checked against pitch_lab's eve anchor + lag=1 on 2025-12-10.

Controls: all days, local +/-126td, and the mandatory tdom-matched control
(same trading-day-of-month, FOMC windows removed) plus month+tdom matched.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

HS = [1, 2, 3, 4, 5]
VEH = {"c1 TLT long": ("TLT", 1.0, 2.5),
       "c5 DX short": ("DX-Y.NYB", -1.0, 1.5),
       "c6 GLD long": ("GLD", 1.0, 3.0)}
AUX = {"IEF long": ("IEF", 1.0, 2.0), "GC=F long": ("GC=F", 1.0, 1.0),
       "UUP short": ("UUP", -1.0, 4.0)}

pxd = load_prices(["TLT", "IEF", "GLD", "GC=F", "DX-Y.NYB", "UUP", "^TNX", "^IRX"])
tnx = pxd["^TNX"]["Close"].dropna()
irx = pxd["^IRX"]["Close"].dropna()
mx = tnx.rolling(252).max()
chg21 = tnx - tnx.shift(21)
dist = tnx / mx - 1.0
irx_chg = irx - irx.shift(126)

fomc = load_events(["fomc_decision"])["date"]
fomc = fomc[fomc <= pd.Timestamp("2026-09-15")]


def eve_state(d):
    e = tnx.index[tnx.index < d]
    if len(e) == 0:
        return None
    e = e[-1]
    ir = irx.asof(e)
    ic = irx_chg.asof(e)
    if ir < 0.30:
        reg = "zirp"
    elif ic > 0.25:
        reg = "hike"
    elif ic < -0.25:
        reg = "cut"
    else:
        reg = "flat"
    return {"eve": e, "tnx": tnx[e], "mx": mx[e], "dist": dist[e],
            "chg21": chg21[e], "regime": reg}


st = pd.DataFrame({d: eve_state(d) for d in fomc}).T
st = st.dropna(subset=["mx"])
st["at_max"] = st["tnx"].astype(float) >= st["mx"].astype(float) - 1e-9
for k in (0.5, 1.0, 2.0):
    st[f"w{k}"] = st["dist"].astype(float) >= -k / 100
st["thr20"] = st["chg21"].astype(float) >= 0.20
print(f"FOMC decisions with a 252-bar TNX eve: {len(st)}  "
      f"({st.index.min().date()}..{st.index.max().date()})")
print("gate counts:", {c: int(st[c].sum()) for c in ["at_max", "w0.5", "w1.0", "w2.0", "thr20"]})
print("today's eve (2026-09-15):", "tnx", tnx.iloc[-1], "mx", mx.iloc[-1],
      "chg21", round(chg21.iloc[-1], 3))

SETS = {"ALL": st.index, "at_max": st.index[st.at_max], "w0.5": st.index[st["w0.5"]],
        "w1.0": st.index[st["w1.0"]], "w2.0": st.index[st["w2.0"]],
        "thr20": st.index[st.thr20], "NOT_w2.0": st.index[~st["w2.0"]],
        "at_max&thr20": st.index[st.at_max & st.thr20]}


def tdom_of(idx):
    ym = pd.Series(idx.year * 100 + idx.month, index=idx)
    return ym.groupby(ym.values).cumcount().values + 1


def analyse(label, tkr, w, cost):
    s = pxd[tkr]["Close"].dropna()
    idx = s.index
    td = tdom_of(idx)
    mon = idx.month.values
    pos = pd.Series(np.arange(len(idx)), index=idx)
    dpos = np.array([pos[d] for d in st.index if d in pos.index])
    # FOMC window mask: decision day through +5 (and 5 before) removed from ctrl
    excl = np.zeros(len(idx), bool)
    for p in dpos:
        excl[max(0, p - 5):p + 6] = True
    out_rows, ep_tab = [], {}
    for h in HS:
        r = w * (s.shift(-h) / s - 1.0)
        rv = r.values
        ok = ~np.isnan(rv)
        ctl_all = np.nanmean(rv)
        bucket = {j: np.nanmean(rv[(td == j) & ~excl & ok]) for j in np.unique(td)}
        mbucket = {}
        for sname, dates in SETS.items():
            dd = [d for d in dates if d in pos.index and not np.isnan(rv[pos[d]])]
            if not dd:
                out_rows.append({"set": sname, "h": h, "n": 0})
                continue
            pp = np.array([pos[d] for d in dd])
            v = rv[pp]
            ex = np.array([rv[p] - bucket[td[p]] for p in pp])
            exm = []
            for p in pp:
                key = (mon[p], td[p])
                if key not in mbucket:
                    mm = (mon == mon[p]) & (td == td[p]) & ~excl & ok
                    mbucket[key] = np.nanmean(rv[mm]) if mm.sum() >= 8 else np.nan
                exm.append(rv[p] - mbucket[key])
            exm = np.array(exm)
            wn = int((v > 0).sum())
            we = int((ex > 0).sum())
            yrs = pd.DatetimeIndex(dd).year.values
            row = {"set": sname, "h": h, "n": len(v), "mean": 100 * v.mean(),
                   "hit": 100 * wn / len(v), "sign_p": sign_test(wn, len(v)),
                   "t": v.mean() / (v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 2 else np.nan,
                   "tdomX": 100 * ex.mean(), "X_hit": 100 * we / len(v),
                   "X_sign_p": sign_test(we, len(v)),
                   "monX": 100 * np.nanmean(exm),
                   "pre18X": 100 * ex[yrs < 2018].mean() if (yrs < 2018).any() else np.nan,
                   "18+X": 100 * ex[yrs >= 2018].mean() if (yrs >= 2018).any() else np.nan,
                   "midX": 100 * ex[yrs % 4 == 2].mean() if (yrs % 4 == 2).any() else np.nan,
                   "n_mid": int((yrs % 4 == 2).sum()),
                   "worst": 100 * v.min(), "ctl_all": 100 * ctl_all}
            out_rows.append(row)
            if sname in ("at_max", "w1.0"):
                ep_tab.setdefault(sname, {})[h] = pd.Series(v, index=dd)
    df = pd.DataFrame(out_rows)
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].round(3)
    print(f"\n{'=' * 110}\n{label}  ({tkr}, side {w:+.0f}, cost ~{cost} bp RT)\n{'=' * 110}")
    print(df.to_string(index=False))
    for sname in ("at_max",):
        if sname in ep_tab:
            t = pd.DataFrame({f"h{h}": 100 * ser for h, ser in ep_tab[sname].items()})
            t["tnx_eve"] = st.loc[t.index, "tnx"].astype(float).round(3)
            t["chg21bp"] = (100 * st.loc[t.index, "chg21"].astype(float)).round(1)
            t["regime"] = st.loc[t.index, "regime"]
            print(f"\n  {sname} episodes ({label}), returns in %:")
            print(t.round(3).to_string())
    return df


# alignment check on a known date via pitch_lab's eve anchor + lag=1
tl = close_panel(["TLT"]).dropna()
pos_, kept = anchor_positions(tl.index, [pd.Timestamp("2025-12-10")], offset=-1)
p = pos_[0]
print(f"\nalignment 2025-12-10: eve {tl.index[p].date()}  entry(lag=1) "
      f"{tl.index[p + 1].date()}  exit h=3 {tl.index[p + 4].date()}  "
      f"fwd_lag h3 {100 * fwd_lag(tl['TLT'], 3, 1).iloc[p]:+.3f}%  "
      f"lag0-on-D {100 * (tl['TLT'].iloc[p + 4] / tl['TLT'].iloc[p + 1] - 1):+.3f}%")

res = {}
for lbl, (tk, w, c) in VEH.items():
    res[lbl] = analyse(lbl, tk, w, c)
print("\n\n######## AUX vehicles (cost/basis check only) ########")
for lbl, (tk, w, c) in AUX.items():
    d = analyse(lbl, tk, w, c)

# overlap check: min gap between consecutive decisions in td
tlt = pxd["TLT"]["Close"].dropna()
pp = [tlt.index.get_loc(d) for d in st.index if d in tlt.index]
print("\nmin gap between decisions (td):", int(np.diff(pp).min()), "-> no overlap at h<=5")

# battery on the gated primary (eve mask, lag=1) for each vehicle, h=3
for lbl, (tk, w, c) in VEH.items():
    s = pxd[tk]["Close"].dropna()
    pxv = pd.DataFrame({tk: s})
    eves, evw1 = [], []
    for d in SETS["at_max"]:
        if d in s.index:
            eves.append(s.index[s.index.get_loc(d) - 1])
    for d in SETS["w1.0"]:
        if d in s.index:
            evw1.append(s.index[s.index.get_loc(d) - 1])
    mask = pd.Series(s.index.isin(eves), index=s.index)
    var = {"w1.0": pd.Series(s.index.isin(evw1), index=s.index)}
    battery(pxv, mask, [(tk, w)], 3, f"{lbl} at_max h=3", c, variants=var,
            lag=1, event_kinds=("quad_witching",))
