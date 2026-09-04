"""C2 round 1 -- whole-complex metals break read as a LONG DOLLAR signal.

Trigger: GLD, SLV and GDX each close <= -2% on the same session.
Trade:   long DX-Y.NYB (UUP as the ETF contrast), MOC the next session, h 1..10.

Adversarial agenda:
  T1 battery on DX (1.5 bps) and UUP (6 bps)
  T2 GATE ATTRIBUTION -- does the dollar-up subset carry the whole thing, and
     how much of it does the bare "DX up >= 0.4%" cell already contain?
  T3 dose response on break depth (monotone or backwards?)
  T4 definition neighbours (NEM for GDX, ATR rung for pct rung, 2-of-3)
  T5 era / midterm splits, decluster 5/10/21, concentration, sign test
  T6 short-dollar side reported honestly
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa: F401,F403

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)

COMPLEX = ["GLD", "SLV", "GDX"]
ALL = COMPLEX + ["NEM", "DX-Y.NYB", "^TNX", "SPY"]


def mkpanel(tickers, start=None):
    p = close_panel(tickers).dropna()
    if start:
        p = p.loc[start:]
    return p


def dret(s: pd.Series) -> pd.Series:
    return s / s.shift(1) - 1.0


def atr_frac(tkr: str, idx: pd.DatetimeIndex) -> pd.Series:
    """Wilder-14 ATR as a fraction of close, aligned to idx."""
    raw = load_prices([tkr])[tkr]
    a = wilder_atr(raw["High"], raw["Low"], raw["Close"], 14) / raw["Close"]
    return a.reindex(idx)


def midterm_split(dates, vals, label=""):
    d = pd.DatetimeIndex(dates)
    m = (d.year % 4) == 2
    return [summarize(np.asarray(vals)[m], f"{label}midterm (yr%4==2)"),
            summarize(np.asarray(vals)[~m], f"{label}non-midterm")]


def decluster_table(px, mask, legs, h, gaps=(5, 10, 21)):
    ret = vehicle_ret(px, legs, h, lag=1)
    valid = ret.notna()
    sig = px.index[mask.reindex(px.index, fill_value=False).values & valid.values]
    rows = [summarize(ret.loc[sig].values, f"day-level N={len(sig)}")]
    for g in gaps:
        e = declusters(sig, g, px.index)
        r = summarize(ret.loc[e].values, f"decluster min_gap={g}")
        w = int((ret.loc[e].values > 0).sum())
        r["sign_p"] = round(sign_test(w, len(e)), 4)
        rows.append(r)
    return rows


def main():
    px = mkpanel(ALL, "2006-05-22")
    print(f"panel {px.index[0].date()} .. {px.index[-1].date()}  n={len(px)}")

    r = {t: dret(px[t]) for t in ALL}
    trig = (r["GLD"] <= -0.02) & (r["SLV"] <= -0.02) & (r["GDX"] <= -0.02)
    print(f"trigger days: {int(trig.sum())}  last: "
          f"{list(px.index[trig.values][-4:])}")

    dx_legs = [("DX-Y.NYB", 1.0)]

    # ---------------- T1: battery, DX at h=5 and h=3 -------------------------
    variants = {
        "each <= -1.5%": (r["GLD"] <= -0.015) & (r["SLV"] <= -0.015) & (r["GDX"] <= -0.015),
        "each <= -2% (base)": trig,
        "each <= -2.5%": (r["GLD"] <= -0.025) & (r["SLV"] <= -0.025) & (r["GDX"] <= -0.025),
        "each <= -3%": (r["GLD"] <= -0.03) & (r["SLV"] <= -0.03) & (r["GDX"] <= -0.03),
    }
    for h in (3, 5):
        battery(px, trig, dx_legs, h, f"C2 LONG DX  metals-complex break", 1.5,
                variants=variants, min_gap=5)

    print("\n### T1b horizon scan, LONG DX, episodes min_gap=5 ###")
    sig = px.index[trig.values]
    show(horizon_scan(px, sig, dx_legs, hs=(1, 2, 3, 4, 5, 6, 7, 8, 10), min_gap=5),
         "LONG DX h=1..10")

    print("\n### T1c SHORT DX (honest wrong-sign check), h=1..10 ###")
    show(horizon_scan(px, sig, [("DX-Y.NYB", -1.0)], hs=(1, 2, 3, 5, 10), min_gap=5),
         "SHORT DX")

    # UUP contrast
    pu = mkpanel(COMPLEX + ["NEM", "UUP", "DX-Y.NYB", "^TNX", "SPY"], "2007-03-01")
    ru = {t: dret(pu[t]) for t in pu.columns}
    trig_u = (ru["GLD"] <= -0.02) & (ru["SLV"] <= -0.02) & (ru["GDX"] <= -0.02)
    battery(pu, trig_u, [("UUP", 1.0)], 5, "C2 LONG UUP (ETF contrast)", 6.0,
            min_gap=5)
    print("\n### T1d horizon scan LONG UUP ###")
    show(horizon_scan(pu, pu.index[trig_u.values], [("UUP", 1.0)],
                      hs=(1, 2, 3, 5, 10), min_gap=5), "LONG UUP")

    # ---------------- T2: GATE ATTRIBUTION ----------------------------------
    print("\n\n########## T2  GATE ATTRIBUTION ##########")
    for h in (3, 5):
        ret = vehicle_ret(px, dx_legs, h, lag=1)
        valid = ret.notna()
        dx_up = r["DX-Y.NYB"] > 0
        sig = px.index[trig.values & valid.values]
        s_up = px.index[(trig & dx_up).values & valid.values]
        s_dn = px.index[(trig & ~dx_up).values & valid.values]
        rows = [summarize(ret.loc[sig].values, f"h={h} all triggers"),
                summarize(ret.loc[s_up].values, f"h={h} trigger & DX UP that day"),
                summarize(ret.loc[s_dn].values, f"h={h} trigger & DX DOWN that day")]
        # bare dollar-momentum cells, same span
        span = (px.index >= sig[0]) & (px.index <= sig[-1])
        for thr in (0.002, 0.004, 0.006):
            m = (r["DX-Y.NYB"] >= thr).values & span & valid.values
            rows.append(summarize(ret[m].values, f"h={h} BARE DX up >= {thr*100:.1f}% (no metals)"))
        rows.append(summarize(ret[span & valid.values].values, f"h={h} CTRL all days in span"))
        show(rows, f"T2 gate attribution, LONG DX h={h} (day level)")

        # incremental: metals clause ON TOP of DX >= 0.4%
        base_m = (r["DX-Y.NYB"] >= 0.004).values & span & valid.values
        both_m = base_m & trig.values
        only_m = base_m & ~trig.values
        b = summarize(ret[both_m].values, "DX>=0.4% AND metals break")
        o = summarize(ret[only_m].values, "DX>=0.4% and NO metals break")
        show([b, o], f"T2b incremental value of the metals clause, h={h}")
        if b.get("n") and o.get("n"):
            print(f"  metals clause incremental = {b['mean_pct'] - o['mean_pct']:+.3f} pp "
                  f"(N_both={b['n']}, N_only={o['n']})")

    # ---------------- T3: dose response --------------------------------------
    print("\n\n########## T3  DOSE RESPONSE ##########")
    ew = (r["GLD"] + r["SLV"] + r["GDX"]) / 3.0
    for h in (3, 5):
        ret = vehicle_ret(px, dx_legs, h, lag=1)
        rows = []
        for thr in (-0.01, -0.015, -0.02, -0.03, -0.04, -0.05):
            m = ew <= thr
            s = px.index[m.values & ret.notna().values]
            e = declusters(s, 5, px.index)
            rr = summarize(ret.loc[e].values, f"EW complex <= {thr*100:.1f}%")
            rr["n_days"] = len(s)
            rows.append(rr)
        show(rows, f"T3 dose on EQUAL-WEIGHT complex depth, LONG DX h={h}")
        rows = []
        for thr in (-0.02, -0.03, -0.04):
            m = (r["GLD"] <= thr) & (r["SLV"] <= thr) & (r["GDX"] <= thr)
            s = px.index[m.values & ret.notna().values]
            if not len(s):
                rows.append({"label": f"each <= {thr*100:.0f}%", "n": 0})
                continue
            e = declusters(s, 5, px.index)
            rr = summarize(ret.loc[e].values, f"each name <= {thr*100:.0f}%")
            rr["n_days"] = len(s)
            rows.append(rr)
        show(rows, f"T3b dose on EACH-NAME rung, LONG DX h={h}")

    # ---------------- T4: definition neighbours ------------------------------
    print("\n\n########## T4  DEFINITION NEIGHBOURS ##########")
    a_gld = atr_frac("GLD", px.index)
    a_slv = atr_frac("SLV", px.index)
    a_gdx = atr_frac("GDX", px.index)
    neigh = {
        "base GLD+SLV+GDX <= -2%": trig,
        "NEM for GDX": (r["GLD"] <= -0.02) & (r["SLV"] <= -0.02) & (r["NEM"] <= -0.02),
        "ATR rung: each <= -1.5 ATR": (r["GLD"] <= -1.5 * a_gld) & (r["SLV"] <= -1.5 * a_slv) & (r["GDX"] <= -1.5 * a_gdx),
        "ATR rung: each <= -1.0 ATR": (r["GLD"] <= -1.0 * a_gld) & (r["SLV"] <= -1.0 * a_slv) & (r["GDX"] <= -1.0 * a_gdx),
        "2 of 3 <= -2%": (((r["GLD"] <= -0.02).astype(int) + (r["SLV"] <= -0.02).astype(int)
                           + (r["GDX"] <= -0.02).astype(int)) >= 2),
        "GLD alone <= -2%": (r["GLD"] <= -0.02),
        "GDX alone <= -2%": (r["GDX"] <= -0.02),
    }
    for h in (3, 5):
        ret = vehicle_ret(px, dx_legs, h, lag=1)
        rows = []
        for lbl, m in neigh.items():
            s = px.index[m.reindex(px.index, fill_value=False).values & ret.notna().values]
            if not len(s):
                rows.append({"label": lbl, "n": 0})
                continue
            e = declusters(s, 5, px.index)
            rr = summarize(ret.loc[e].values, lbl)
            rr["n_days"] = len(s)
            w = int((ret.loc[e].values > 0).sum())
            rr["sign_p"] = round(sign_test(w, len(e)), 4)
            rows.append(rr)
        show(rows, f"T4 definition neighbours, LONG DX h={h} (episodes gap5)")

    # ---------------- T5: splits ---------------------------------------------
    print("\n\n########## T5  ERA / MIDTERM / DECLUSTER ##########")
    for h in (3, 5):
        show(decluster_table(px, trig, dx_legs, h), f"T5 decluster sensitivity h={h}")
        ret = vehicle_ret(px, dx_legs, h, lag=1)
        s = px.index[trig.values & ret.notna().values]
        e = declusters(s, 5, px.index)
        show(era_split(e, ret.loc[e].values), f"T5b era split (episodes) h={h}")
        show(midterm_split(e, ret.loc[e].values), f"T5c midterm split (episodes) h={h}")
        print("  " + cluster_note(e, ret.loc[e].values, k=3))
        # today's configuration only: DX up AND TNX up
        tnx_up = dret(px["^TNX"]) > 0
        cfg = trig & (r["DX-Y.NYB"] > 0) & tnx_up
        sc = px.index[cfg.values & ret.notna().values]
        ec = declusters(sc, 5, px.index)
        rr = summarize(ret.loc[ec].values, "TODAY's cfg: DX up AND TNX up")
        w = int((ret.loc[ec].values > 0).sum())
        show([rr, {"label": "sign_p", "n": len(ec), "mean_pct": sign_test(w, len(ec))}],
             f"T5d today's exact configuration h={h}")


if __name__ == "__main__":
    main()
