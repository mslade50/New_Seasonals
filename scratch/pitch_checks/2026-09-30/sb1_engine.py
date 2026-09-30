"""sb1: seasonal-board kill engine (TRV / GS 21d Oct windows). Shared by sb1_trv.py / sb1_gs.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

YEARS = range(2000, 2026)
MID = {2002, 2006, 2010, 2014, 2018, 2022}


def earnings_dates(tkr: str) -> pd.DatetimeIndex:
    e = pd.read_parquet(ROOT / "data" / "earnings_calendar.parquet")
    return pd.DatetimeIndex(pd.to_datetime(e.loc[e.ticker == tkr, "date"]).unique()).sort_values()


def anchors(idx: pd.DatetimeIndex, month=9, day=30) -> dict[int, int]:
    out = {}
    for y in YEARS:
        p = int(idx.searchsorted(pd.Timestamp(y, month, day)))
        if p < len(idx):
            out[y] = p
    return out


def windows(tkr: str, px: pd.DataFrame, lag: int, h: int, off: int = 0) -> pd.DataFrame:
    idx = px.index
    c = px[tkr].values
    f = PX[tkr]
    atr = pd.Series(wilder_atr(f["High"], f["Low"], f["Close"]), index=f.index).reindex(idx).values
    rows = []
    for y, p in anchors(idx).items():
        a = p + off
        e0, e1 = a + lag, a + lag + h
        if e1 >= len(idx):
            continue
        r = c[e1] / c[e0] - 1
        atr_frac = atr[a - 1] / c[a - 1]
        rows.append({"year": y, "anchor": idx[a], "entry": idx[e0], "exit": idx[e1], "e0": e0, "e1": e1,
                     "ret": r, "ret_atr": r / atr_frac,
                     "xlf": px["XLF"].values[e1] / px["XLF"].values[e0] - 1,
                     "spy": px["SPY"].values[e1] / px["SPY"].values[e0] - 1})
    return pd.DataFrame(rows).set_index("year")


def drop_best(v: pd.Series, k: int) -> dict:
    s = v.sort_values(ascending=False).iloc[k:]
    return {"n": len(s), "mean": round(100 * s.mean(), 2), "hit": f"{int((s > 0).sum())}/{len(s)}"}


def run(tkr: str, gate_n: int) -> None:
    global PX
    PX = load_prices([tkr, "XLF", "SPY"])
    px = pd.DataFrame({t: PX[t]["Close"] for t in PX}).dropna()
    idx = px.index
    print(f"\n######## {tkr}  (gate: {gate_n}d return pct_rank <= 15) ########")

    # ---- ROUND 1 --------------------------------------------------------
    uncond = fwd_lag(px[tkr], 21, 1).dropna()
    for lag in (1, 2):
        w = windows(tkr, px, lag, 21)
        v = w["ret"]
        wins = int((v > 0).sum())
        loc = local_control(idx[idx.isin(uncond.index)],pd.DatetimeIndex(w["anchor"]))
        loc_r = fwd_lag(px[tkr], 21, lag).reindex(loc).dropna()
        print(f"\n== R1 lag=T+{lag} h=21: all-years ==")
        show([summarize(v.values, f"{tkr} Oct window N={len(v)}"),
              summarize(uncond.values, "CTRL own drift all days"),
              summarize(loc_r.values, "CTRL local +/-126td")])
        print(f"  record {wins}-{len(v)-wins} sign p={sign_test(wins, len(v)):.4f}  "
              f"worst {100*v.min():.2f}% ({v.idxmin()})  mean ATR {w.ret_atr.mean():+.2f}")
        m = w.loc[w.index.isin(MID)]
        print(f"  MIDTERM: " + ", ".join(f"{y}:{100*r:+.1f}%/{a:+.1f}ATR" for y, r, a in zip(m.index, m.ret, m.ret_atr)))
        print(f"    all6 mean {100*m.ret.mean():+.2f}% ({m.ret_atr.mean():+.2f} ATR) hit {int((m.ret>0).sum())}/6 "
              f"sign p={sign_test(int((m.ret>0).sum()), len(m)):.3f}")
        print(f"    drop-best {drop_best(m.ret,1)} ATR {m.ret_atr.sort_values(ascending=False).iloc[1:].mean():+.2f}; "
              f"drop-2 {drop_best(m.ret,2)} ATR {m.ret_atr.sort_values(ascending=False).iloc[2:].mean():+.2f}")
        nm = w.loc[~w.index.isin(MID)]
        print(f"    NON-midterm N={len(nm)} mean {100*nm.ret.mean():+.2f}% hit {int((nm.ret>0).sum())}/{len(nm)}")
        show(era_split(pd.DatetimeIndex(w.anchor), v.values, "2018-01-01") +
             era_split(pd.DatetimeIndex(w.anchor), v.values, "2010-01-01"), "era splits")
        print("  concentration:", cluster_note(pd.DatetimeIndex(w.anchor), v.values))
        print(f"  all-yrs drop-best {drop_best(v,1)} drop-2 {drop_best(v,2)}")
        print(f"  edge vs own drift {100*(v.mean()-uncond.mean()):+.2f}pp = {1e4*(v.mean()-uncond.mean()):.0f} bps vs ~5bps RT")

    # ---- ROUND 2 --------------------------------------------------------
    w = windows(tkr, px, 2, 21)
    # (b) neighbours
    print("\n== R2b neighbours (lag T+2): anchor offset x horizon, mean% / hit / mid mean% ==")
    grid = {}
    for h in (15, 21, 26):
        row = {}
        for off in range(-3, 4):
            ww = windows(tkr, px, 2, h, off)
            mm = ww.loc[ww.index.isin(MID), "ret"]
            row[off] = f"{100*ww.ret.mean():+.2f}/{int((ww.ret>0).sum())}of{len(ww)}/m{100*mm.mean():+.1f}"
        grid[h] = row
    print(pd.DataFrame(grid).T.to_string())
    ud = {h: 100 * fwd_lag(px[tkr], h, 1).mean() for h in (15, 21, 26)}
    print("  own drift by h:", {h: round(x, 2) for h, x in ud.items()})

    # (c) gate attribution
    pr = pct_rank(px[tkr], gate_n)
    w["gate_pr"] = [pr.iloc[int(e0) - 3] for e0 in w.e0]   # bar B = anchor-1, e0 = anchor+2
    g = w.gate_pr <= 15
    print(f"\n== R2c gate attribution ({gate_n}d pct_rank<=15 at board bar) ==")
    show([summarize(w.ret[g].values, f"gate ON  yrs={list(w.index[g])}"),
          summarize(w.ret[~g].values, "gate OFF")])
    for thr in (25, 33, 50):
        gg = w.gate_pr <= thr
        print(f"  <= {thr}: ON n={gg.sum()} mean {100*w.ret[gg].mean():+.2f}%  OFF mean {100*w.ret[~gg].mean():+.2f}%")
    print(f"  corr(gate pctile, window ret) = {np.corrcoef(w.gate_pr, w.ret)[0,1]:+.2f}")

    # (d) earnings split
    ed = earnings_dates(tkr)
    c = px[tkr].values
    rec = []
    for y, r in w.iterrows():
        hit = ed[(ed > idx[int(r.e0)]) & (ed <= idx[int(r.e1)])]
        if len(hit) == 0:
            rec.append({"year": y, "print": None, "pre": r.ret, "react": 0.0, "post": 0.0, "full": r.ret})
            continue
        pe = int(idx.searchsorted(hit[0]))  # pre-market print -> reaction = close(pe-1)->close(pe)
        pre = c[pe - 1] / c[int(r.e0)] - 1
        react = c[pe] / c[pe - 1] - 1
        post = c[int(r.e1)] / c[pe] - 1
        rec.append({"year": y, "print": hit[0].date(), "pre": pre, "react": react, "post": post, "full": r.ret})
    E = pd.DataFrame(rec).set_index("year")
    inwin = E["print"].notna()
    print(f"\n== R2d earnings split ({inwin.sum()}/{len(E)} windows contain a print) ==")
    for lbl, sub in (("all", E), ("print-in", E[inwin]), ("no-print", E[~inwin]), ("midterm", E[E.index.isin(MID)])):
        if len(sub) == 0:
            continue
        print(f"  {lbl:9s} N={len(sub):2d} full {100*sub.full.mean():+.2f}% | pre-print {100*sub.pre.mean():+.2f}% "
              f"(hit {int((sub.pre>0).sum())}/{len(sub)}) | reaction {100*sub.react.mean():+.2f}% "
              f"(hit {int((sub.react>0).sum())}/{int(sub['print'].notna().sum())}) | post {100*sub.post.mean():+.2f}%")
    print("  midterm rows:\n" + (100 * E.loc[E.index.isin(MID), ["pre", "react", "post", "full"]]).round(2).to_string())
    # control: own drift over same average pre-print length
    npre = int(np.nanmedian([int(idx.searchsorted(pd.Timestamp(p))) - 1 - int(w.loc[y, "e0"])
                             for y, p in E["print"].dropna().items()])) if inwin.any() else 21
    print(f"  median pre-print length {npre} td; own drift over {npre}td = {100*fwd_lag(px[tkr], npre, 1).mean():+.2f}%")
    # reaction-day control: all this ticker's print reactions
    allr = []
    for d in ed:
        pe = int(idx.searchsorted(d))
        if 0 < pe < len(idx) and idx[pe] == d:
            allr.append(c[pe] / c[pe - 1] - 1)
    print(f"  all-quarter print reactions N={len(allr)} mean {100*np.mean(allr):+.2f}% hit {100*np.mean(np.array(allr)>0):.0f}%")

    # (e) sector / market
    beta_rows = []
    rets = px.pct_change()
    for y, r in w.iterrows():
        b0 = int(r.e0) - 2
        win = rets.iloc[max(1, b0 - 252):b0]
        bx = np.polyfit(win["XLF"], win[tkr], 1)[0]
        bs = np.polyfit(win["SPY"], win[tkr], 1)[0]
        beta_rows.append({"year": y, "bx": bx, "bs": bs, "res_xlf": r.ret - bx * r.xlf, "res_spy": r.ret - bs * r.spy,
                          "xs_xlf": r.ret - r.xlf})
    B = pd.DataFrame(beta_rows).set_index("year")
    print("\n== R2e sector/market (same windows, T+2, 21d) ==")
    for col, lbl in (("xlf", "XLF raw"), ("spy", "SPY raw")):
        s = w[col]
        print(f"  {lbl}: mean {100*s.mean():+.2f}% hit {int((s>0).sum())}/{len(s)}  midterm mean {100*s[s.index.isin(MID)].mean():+.2f}% "
              f"({int((s[s.index.isin(MID)]>0).sum())}/6)   own drift {100*fwd_lag(px[col.upper()],21,1).mean():+.2f}%")
    for col in ("xs_xlf", "res_xlf", "res_spy"):
        s = B[col]
        m = s[s.index.isin(MID)]
        print(f"  {tkr} {col}: mean {100*s.mean():+.2f}% hit {int((s>0).sum())}/{len(s)} sign p={sign_test(int((s>0).sum()), len(s)):.3f} | "
              f"midterm {100*m.mean():+.2f}% ({int((m>0).sum())}/6), drop-best {100*m.sort_values().iloc[:-1].mean():+.2f}%")
    print(f"  median beta XLF {B.bx.median():.2f}  SPY {B.bs.median():.2f}")
    print("\n  per-year (T+2, %): \n" + (100 * pd.concat([w[["ret", "xlf", "spy"]], B[["res_xlf"]], E[["pre", "react"]]], axis=1)).round(1).assign(gate=w.gate_pr.round(0)).to_string())
