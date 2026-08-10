"""E1 round 1 -- the quarterly Treasury refunding auction concession.

PRE-SPECIFIED BEFORE MEASURING (stated here so the record shows it):

  H1  Treasury runs a QUARTERLY REFUNDING in Feb / May / Aug / Nov, auctioning
      3y / 10y / 30y across roughly calendar days 9-12. Dealers must absorb
      the supply and are documented to build inventory cheaply -- a PRICE
      CONCESSION (yields up, price down) into the auction, unwinding once it
      clears.
  H2  The tradeable form is LONG DURATION (TLT; IEF as the lower-duration
      sibling) entered MOC at trading-day-of-month 6 -- today's executable
      entry, 2026-08-10 -- and held across the auction window.
  H3  ENTRY CONVENTION.  The anchor is a CALENDAR POSITION known months in
      advance, so there is no signal to lag: entry is MOC on the anchor close
      itself (lag=0 from tdom 6), which is identical to lag=1 from the tdom-5
      close.  Stated explicitly because the book's default is lag=1.
  H4  DECISIVE TEST 1 -- the tdom-matched control.  Refunding months vs
      NON-refunding months AT THE SAME trading-day-of-month.  d5b_tdom_control
      showed TLT's own h=3 drift swings from -0.202% (tdom 2) to +0.215%
      (tdom 14) with no event anywhere, and that profile alone killed
      "long TLT into CPI" this morning (+6.7 bps tdom-matched excess).  If
      Feb/May/Aug/Nov look like Jan/Mar/Apr at the same tdom, E1 is the tdom
      profile wearing a label.
  H5  DECISIVE TEST 2 -- the SHAPE.  The mechanism predicts WEAKNESS into
      roughly tdom 6-8 and STRENGTH tdom 9-12.  A positive mean with a flat,
      monotone or wrong-peaked path falsifies the mechanism inside its own
      window even if the mean is positive.
  H6  DECISIVE TEST 3 -- the INSTITUTIONAL era cut, which is a PREDICTION of
      the mechanism, not a fence around a macro episode.  Treasury moved to
      auctioning 3y/10y/30y EVERY month (reopenings in the off-months) in
      2008-2009.  If the concession mechanism is real, the "refunding month"
      LABEL should stop mattering post-2009 because every month now has the
      auction.  Both branches are informative and both are stated in advance.
  H7  PROXY ERROR.  There is NO auction calendar in this repo.  The auction
      dates are proxied by the calendar.  Quantified in section 5.
  H8  CONTAMINATION.  August 2026 has CPI at tdom 8 and PPI at tdom 9, i.e.
      INSIDE the window.  Disentangled in section 6.

Sections 1-8 below.  Nothing here is chosen after looking.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

REFUND_MONTHS = (2, 5, 8, 11)
ENTRY_TDOM = 6          # today, 2026-08-10
H_MAIN = 5              # tdom 6 -> tdom 11, spans the 9th-12th auctions

px = close_panel(["TLT", "IEF"])


def frame(tkr):
    s = px[tkr].dropna()
    idx = s.index
    c = s.values
    ym = pd.Series(idx.year * 100 + idx.month, index=idx)
    tdom = ym.groupby(ym.values).cumcount().values + 1
    d = pd.DataFrame({"close": c, "tdom": tdom,
                      "year": idx.year, "month": idx.month,
                      "dom": [x.day for x in idx]}, index=idx)
    d["refund"] = d["month"].isin(REFUND_MONTHS)
    d["midterm"] = (d["year"] % 4) == 2
    return d


def fwd(d, h):
    c = d["close"].values
    out = np.full(len(c), np.nan)
    out[:len(c) - h] = c[h:] / c[:-h] - 1.0
    return out


def rep(v, lbl):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return {"cell": lbl, "N": len(v)}
    w = int((v > 0).sum())
    return {"cell": lbl, "N": len(v), "mean_pct": round(100 * v.mean(), 4),
            "hit": round(100 * w / len(v), 1),
            "t": round(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), 2),
            "sign_p": round(sign_test(w, len(v)), 4),
            "worst_pct": round(100 * v.min(), 2)}


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return (a.mean() - b.mean()) / se


for TKR in ["TLT", "IEF"]:
    d = frame(TKR)
    print("\n" + "#" * 100)
    print(f"##### {TKR}   inception {d.index[0].date()}   last {d.index[-1].date()}"
          f"   N={len(d)} sessions")
    print("#" * 100)

    # ---------------------------------------------------------------- 1
    print("\n" + "=" * 100)
    print(f"1. THE PRE-SPECIFIED CELL: long {TKR} MOC at tdom {ENTRY_TDOM}, "
          f"h={H_MAIN} sessions (tdom 6 -> 11)")
    print("   refunding months (Feb/May/Aug/Nov) vs the SAME tdom in "
          "non-refunding months")
    print("=" * 100)
    r = fwd(d, H_MAIN)
    m_entry = (d["tdom"].values == ENTRY_TDOM) & ~np.isnan(r)
    cell = r[m_entry & d["refund"].values]
    ctrl_tdom = r[m_entry & ~d["refund"].values]
    ctrl_all = r[~np.isnan(r)]
    rows = [rep(cell, f"*** CELL: refunding months, tdom {ENTRY_TDOM}, h={H_MAIN} ***"),
            rep(ctrl_tdom, f"CTRL-tdom: NON-refunding months, tdom {ENTRY_TDOM}"),
            rep(r[m_entry], f"CTRL: ALL months at tdom {ENTRY_TDOM}"),
            rep(ctrl_all, "CTRL-b: all days, full history")]
    print(pd.DataFrame(rows).to_string(index=False))
    exc = cell - ctrl_tdom.mean()
    w = int((exc > 0).sum())
    print(f"\n  TDOM-MATCHED EXCESS = {100*exc.mean():+.4f}pp "
          f"({100*100*exc.mean():+.1f} bps), record {w}-{len(exc)-w}, "
          f"sign p {sign_test(w, len(exc)):.4f}, welch t {welch(cell, ctrl_tdom):+.2f}, "
          f"bootstrap P(mean<=0) {bootstrap_p_le0(exc):.3f}")
    cost = 2.5 if TKR == "TLT" else 2.0
    print(f"  cost: ~{cost} bps round trip -> excess = "
          f"{100*100*exc.mean()/cost:.1f}x cost (need >=5x)")
    dates = d.index[m_entry & d["refund"].values]
    print(f"  concentration: {cluster_note(dates, cell)}")
    mid = d["midterm"].values[m_entry & d["refund"].values]
    print(f"  MIDTERM excess {100*exc[mid].mean():+.4f}pp (N={int(mid.sum())})  |  "
          f"non-midterm {100*exc[~mid].mean():+.4f}pp (N={int((~mid).sum())})")
    yr = d["year"].values[m_entry & d["refund"].values]
    for lbl, mm in [("pre-2009", yr < 2009), ("2009+", yr >= 2009),
                    ("pre-2018", yr < 2018), ("2018+", yr >= 2018),
                    ("ex-2008", yr != 2008), ("ex-2020", yr != 2020)]:
        if mm.sum() > 2:
            print(f"    {lbl:9s} excess {100*exc[mm].mean():+.4f}pp "
                  f"(N={int(mm.sum())}, raw {100*cell[mm].mean():+.4f}%)")

    # ---------------------------------------------------------------- 2
    print("\n" + "=" * 100)
    print(f"2. DECISIVE TEST 2 -- THE SHAPE.  Mean 1-SESSION return by tdom, "
          f"{TKR}, refunding vs non-refunding")
    print("   mechanism predicts DIFF < 0 for tdom ~4-8 (concession) and "
          "DIFF > 0 for tdom ~9-12 (unwind)")
    print("=" * 100)
    d1 = np.full(len(d), np.nan)
    cc = d["close"].values
    d1[1:] = cc[1:] / cc[:-1] - 1.0
    rows = []
    for j in range(1, 19):
        mj = (d["tdom"].values == j) & ~np.isnan(d1)
        a = d1[mj & d["refund"].values]
        b = d1[mj & ~d["refund"].values]
        if len(a) < 10 or len(b) < 10:
            continue
        rows.append({"tdom": j, "N_ref": len(a), "ref_bps": round(1e4 * a.mean(), 2),
                     "nonref_bps": round(1e4 * b.mean(), 2),
                     "DIFF_bps": round(1e4 * (a.mean() - b.mean()), 2),
                     "t": round(welch(a, b), 2),
                     "predicted": ("CONCESSION(-)" if 4 <= j <= 8 else
                                   ("UNWIND(+)" if 9 <= j <= 12 else ""))})
    shp = pd.DataFrame(rows)
    print(shp.to_string(index=False))
    conc = shp[(shp.tdom >= 4) & (shp.tdom <= 8)]["DIFF_bps"]
    unw = shp[(shp.tdom >= 9) & (shp.tdom <= 12)]["DIFF_bps"]
    print(f"\n  predicted-concession block (tdom 4-8) mean DIFF "
          f"{conc.mean():+.2f} bps/day, sign of days {list(np.sign(conc))}")
    print(f"  predicted-unwind      block (tdom 9-12) mean DIFF "
          f"{unw.mean():+.2f} bps/day, sign of days {list(np.sign(unw))}")
    print("  MECHANISM CONFIRMED only if concession block < 0 AND unwind block > 0.")

    # ---------------------------------------------------------------- 3
    print("\n" + "=" * 100)
    print("3. THE SAME SHAPE TEST, CUMULATIVE: refunding-month path from the "
          "tdom-1 close")
    print("=" * 100)
    rows = []
    for j in range(1, 17):
        # cumulative return from tdom 1 close to tdom j close, per month
        cum_r, cum_n = [], []
        for (y, mo), g in d.groupby(["year", "month"]):
            if len(g) < j:
                continue
            base = g["close"].values[0]
            v = g["close"].values[j - 1] / base - 1.0
            (cum_r if mo in REFUND_MONTHS else cum_n).append(v)
        if len(cum_r) < 10:
            continue
        rows.append({"tdom": j, "refund_cum_pct": round(100 * np.mean(cum_r), 3),
                     "nonref_cum_pct": round(100 * np.mean(cum_n), 3),
                     "DIFF_pp": round(100 * (np.mean(cum_r) - np.mean(cum_n)), 3)})
    print(pd.DataFrame(rows).to_string(index=False))

    # ---------------------------------------------------------------- 4
    print("\n" + "=" * 100)
    print("4. DEFINITION NEIGHBOURS -- entry tdom x horizon grid, "
          "TDOM-MATCHED EXCESS in bps")
    print("   (a real mechanism survives a one-session nudge; one hot cell "
          "surrounded by zeros is definition fragility)")
    print("=" * 100)
    rows = []
    for k in range(3, 11):
        row = {"entry_tdom": k}
        for h in (3, 4, 5, 6, 7, 8):
            rr = fwd(d, h)
            me = (d["tdom"].values == k) & ~np.isnan(rr)
            a = rr[me & d["refund"].values]
            b = rr[me & ~d["refund"].values]
            row[f"h{h}"] = round(1e4 * (a.mean() - b.mean()), 1) if len(a) > 2 else np.nan
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"  cost bar for {TKR}: an edge needs >=5x {cost} bps = "
          f">={5*cost:.1f} bps of EXCESS.")

    # ---------------------------------------------------------------- 5
    print("\n" + "=" * 100)
    print("5. PROXY ERROR -- there is NO auction calendar in this repo.")
    print("   Where do calendar days 9-12 actually land in tdom terms, "
          "refunding months only?")
    print("=" * 100)
    ref = d[d["refund"]]
    for cd in (9, 10, 11, 12):
        t = ref[ref["dom"] == cd]["tdom"]
        if len(t):
            print(f"   calendar day {cd:2d}: tdom min {t.min()} p25 "
                  f"{int(t.quantile(.25))} median {int(t.median())} p75 "
                  f"{int(t.quantile(.75))} max {t.max()}  (N={len(t)} months)")
    print("   -> a FIXED tdom-6 entry sits a variable number of sessions "
          "before the actual auctions.")
    # calendar-day anchored variant: last session on/before dom 8 -> first on/after dom 13
    cal_r, cal_n = [], []
    for (y, mo), g in d.groupby(["year", "month"]):
        pre = g[g["dom"] <= 8]
        post = g[g["dom"] >= 13]
        if len(pre) == 0 or len(post) == 0:
            continue
        v = post["close"].values[0] / pre["close"].values[-1] - 1.0
        (cal_r if mo in REFUND_MONTHS else cal_n).append(v)
    print("\n   CALENDAR-DAY-ANCHORED variant (last close <= dom 8 -> first "
          "close >= dom 13):")
    print(pd.DataFrame([rep(cal_r, "refunding months"),
                        rep(cal_n, "non-refunding months")]).to_string(index=False))
    print(f"   excess {100*(np.mean(cal_r)-np.mean(cal_n)):+.4f}pp, "
          f"welch t {welch(cal_r, cal_n):+.2f}")

    # ---------------------------------------------------------------- 6
    print("\n" + "=" * 100)
    print("6. CONTAMINATION -- is the refunding cell just the CPI/PPI cell?")
    print("=" * 100)
    ev = load_events(["cpi", "ppi"])
    evd = set(pd.DatetimeIndex(ev["date"]).normalize())
    idx = d.index
    pos = pd.Series(range(len(idx)), index=idx)
    rr = fwd(d, H_MAIN)
    me = (d["tdom"].values == ENTRY_TDOM) & ~np.isnan(rr) & d["refund"].values
    ent_pos = np.where(me)[0]
    has_ev = []
    for p in ent_pos:
        win = idx[p + 1: p + H_MAIN + 1]
        has_ev.append(any(x.normalize() in evd for x in win))
    has_ev = np.array(has_ev)
    v = rr[ent_pos]
    b = ctrl_tdom.mean()
    print(pd.DataFrame([
        rep(v[has_ev] - b, f"refunding tdom6, CPI/PPI IN window (N={int(has_ev.sum())}) EXCESS"),
        rep(v[~has_ev] - b, f"refunding tdom6, NO print in window (N={int((~has_ev).sum())}) EXCESS"),
    ]).to_string(index=False))

    # ---------------------------------------------------------------- 7
    print("\n" + "=" * 100)
    print("7. TODAY'S STATE -- does conditioning on a depressed TLT change it?")
    print("   (TLT closed 82.76, 1.03% off its 52w LOW, rank21 23.0, "
          "rank63 18.7, z10 -0.11)")
    print("=" * 100)
    s = px[TKR].dropna()
    r21 = pct_rank(s, 21).reindex(d.index).values
    dist_low = (s / s.rolling(252).min() - 1.0).reindex(d.index).values
    rr = fwd(d, H_MAIN)
    base_m = (d["tdom"].values == ENTRY_TDOM) & ~np.isnan(rr) & d["refund"].values
    for lbl, gate in [("ungated", np.ones(len(d), bool)),
                      ("rank21 < 40", r21 < 40),
                      ("rank21 < 25", r21 < 25),
                      ("within 3% of 52w low", dist_low < 0.03)]:
        mm = base_m & gate
        if mm.sum() < 3:
            print(f"   {lbl:24s} N={int(mm.sum())} -- too few to state")
            continue
        vv = rr[mm]
        print(f"   {lbl:24s} N={int(mm.sum()):3d} raw {100*vv.mean():+.4f}% "
              f"excess {100*(vv.mean()-b):+.4f}pp  hit "
              f"{100*(vv>0).mean():.1f}%  t "
              f"{vv.mean()/(vv.std(ddof=1)/np.sqrt(len(vv))):+.2f}")

print("\n" + "=" * 100)
print("8. GATE ATTRIBUTION -- run WITHOUT the refunding restriction.")
print("   If tdom-6 entries pay the same in every month, the refunding label "
      "adds nothing.")
print("=" * 100)
for TKR in ["TLT", "IEF"]:
    d = frame(TKR)
    rr = fwd(d, H_MAIN)
    for lbl, mm in [("refunding months only", d["refund"].values),
                    ("non-refunding only", ~d["refund"].values),
                    ("EVERY month", np.ones(len(d), bool))]:
        m = (d["tdom"].values == ENTRY_TDOM) & ~np.isnan(rr) & mm
        v = rr[m]
        print(f"  {TKR} tdom{ENTRY_TDOM} h{H_MAIN} {lbl:22s} N={len(v):3d} "
              f"{100*v.mean():+.4f}%  hit {100*(v>0).mean():5.1f}%  "
              f"t {v.mean()/(v.std(ddof=1)/np.sqrt(len(v))):+.2f}")
