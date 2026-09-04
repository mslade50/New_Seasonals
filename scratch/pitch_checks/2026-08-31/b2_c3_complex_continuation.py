"""C3 round 1 -- the whole-complex metals break as a CONTINUATION (short side).

Trigger: GLD, SLV and GDX each close <= -2% on the same session.
Trade:   SHORT the complex (GLD / GDX / SLV separately + EW basket), MOC the
         next session, h 1..10.

Adversarial agenda:
  T1 battery on the SHORT side per vehicle (4 bps GLD/GDX, 6 bps SLV)
  T2 21-day-run conditioner: does a break out of a HOT run continue or revert,
     and is the gradient monotone?  (today GDX 21d +29.79%, r21 97.2)
  T3 dollar-up/down and yields-up/down splits (today is UP and UP)
  T4 era / midterm / decluster 5-10-21 / concentration / sign test
  T5 REFERENCE CLASS: the identical rule on 6 "complex break" families,
     Cochran Q, I-squared, fixed-effect common excess, permutation max-of-N p
  T6 BOOK OVERLAP: what the systematic book did on these trigger days
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa: F401,F403

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)

FAMILIES = {
    "metals   GLD/SLV/GDX": ["GLD", "SLV", "GDX"],
    "energy   XLE/XOP/USO": ["XLE", "XOP", "USO"],
    "semis    SMH/AMAT/NVDA": ["SMH", "AMAT", "NVDA"],
    "banks    XLF/KRE/BAC": ["XLF", "KRE", "BAC"],
    "hombld   XHB/ITB/DHI": ["XHB", "ITB", "DHI"],
    "materls  XME/FCX/NUE": ["XME", "FCX", "NUE"],
}

BASE = ["GLD", "SLV", "GDX", "NEM", "DX-Y.NYB", "^TNX", "SPY"]


def mkpanel(tickers, start=None):
    p = close_panel(tickers).dropna()
    if start:
        p = p.loc[start:]
    return p


def dret(s):
    return s / s.shift(1) - 1.0


def midterm_split(dates, vals, label=""):
    d = pd.DatetimeIndex(dates)
    m = (d.year % 4) == 2
    return [summarize(np.asarray(vals)[m], f"{label}midterm"),
            summarize(np.asarray(vals)[~m], f"{label}non-midterm")]


def main():
    px = mkpanel(BASE, "2006-05-22")
    r = {t: dret(px[t]) for t in BASE}
    trig = (r["GLD"] <= -0.02) & (r["SLV"] <= -0.02) & (r["GDX"] <= -0.02)
    sig_all = px.index[trig.values]
    print(f"panel {px.index[0].date()}..{px.index[-1].date()} n={len(px)}  "
          f"triggers={len(sig_all)}")

    vehicles = {
        "SHORT GLD": ([("GLD", -1.0)], 4.0),
        "SHORT GDX": ([("GDX", -1.0)], 4.0),
        "SHORT SLV": ([("SLV", -1.0)], 6.0),
        "SHORT EW basket": ([("GLD", -1 / 3), ("SLV", -1 / 3), ("GDX", -1 / 3)], 4.7),
    }

    # ---------------- T1 battery + horizon scan ------------------------------
    for name, (legs, cost) in vehicles.items():
        battery(px, trig, legs, 5, f"C3 {name}  (metals complex break)", cost,
                min_gap=5)
        show(horizon_scan(px, sig_all, legs, hs=(1, 2, 3, 4, 5, 6, 7, 8, 10),
                          min_gap=5), f"T1 horizon scan {name}")
        print(f"   (long side is the exact negative of every row above)")

    # ---------------- T2  21-day-run conditioner -----------------------------
    print("\n\n########## T2  21-DAY RUN CONDITIONER ##########")
    r21 = {t: (px[t] / px[t].shift(21) - 1.0) for t in ["GLD", "SLV", "GDX"]}
    ew21 = (r21["GLD"] + r21["SLV"] + r21["GDX"]) / 3.0
    gdx_rank21 = pct_rank(px["GDX"], 21)
    for name, (legs, cost) in [("SHORT GDX", vehicles["SHORT GDX"]),
                               ("SHORT EW basket", vehicles["SHORT EW basket"])]:
        for h in (3, 5):
            ret = vehicle_ret(px, legs, h, lag=1)
            rows = []
            bands = [(-9, 0.0), (0.0, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 9)]
            for lo, hi in bands:
                m = trig & (ew21 > lo) & (ew21 <= hi)
                s = px.index[m.values & ret.notna().values]
                if not len(s):
                    rows.append({"label": f"EW 21d in ({lo:.2f},{hi:.2f}]", "n": 0})
                    continue
                e = declusters(s, 5, px.index)
                rr = summarize(ret.loc[e].values, f"EW 21d in ({lo:.2f},{hi:.2f}]")
                rr["n_days"] = len(s)
                rows.append(rr)
            # GDX 21d rank rungs (today 97.2)
            for lo, hi in [(0, 50), (50, 80), (80, 95), (95, 101)]:
                m = trig & (gdx_rank21 >= lo) & (gdx_rank21 < hi)
                s = px.index[m.values & ret.notna().values]
                if not len(s):
                    rows.append({"label": f"GDX r21 [{lo},{hi})", "n": 0})
                    continue
                e = declusters(s, 5, px.index)
                rr = summarize(ret.loc[e].values, f"GDX r21 [{lo},{hi})")
                rr["n_days"] = len(s)
                w = int((ret.loc[e].values > 0).sum())
                rr["sign_p"] = round(sign_test(w, len(e)), 4)
                rows.append(rr)
            show(rows, f"T2 run conditioner, {name} h={h} (episodes gap5)")

    # ---------------- T3  dollar / yields splits -----------------------------
    print("\n\n########## T3  DOLLAR-UP AND YIELDS-UP SPLITS ##########")
    dx_up = r["DX-Y.NYB"] > 0
    tnx_up = r["^TNX"] > 0
    for name, (legs, cost) in [("SHORT GDX", vehicles["SHORT GDX"]),
                               ("SHORT EW basket", vehicles["SHORT EW basket"])]:
        for h in (3, 5):
            ret = vehicle_ret(px, legs, h, lag=1)
            rows = []
            for lbl, m in [("all triggers", trig),
                           ("DX up", trig & dx_up),
                           ("DX down", trig & ~dx_up),
                           ("TNX up", trig & tnx_up),
                           ("TNX down", trig & ~tnx_up),
                           ("TODAY cfg: DX up AND TNX up", trig & dx_up & tnx_up),
                           ("neither up", trig & ~dx_up & ~tnx_up)]:
                s = px.index[m.values & ret.notna().values]
                if not len(s):
                    rows.append({"label": lbl, "n": 0})
                    continue
                e = declusters(s, 5, px.index)
                rr = summarize(ret.loc[e].values, lbl)
                rr["n_days"] = len(s)
                w = int((ret.loc[e].values > 0).sum())
                rr["sign_p"] = round(sign_test(w, len(e)), 4)
                rows.append(rr)
            show(rows, f"T3 splits, {name} h={h}")

    # ---------------- T4  era / midterm / decluster --------------------------
    print("\n\n########## T4  ERA / MIDTERM / DECLUSTER ##########")
    for name, (legs, cost) in vehicles.items():
        for h in (5,):
            ret = vehicle_ret(px, legs, h, lag=1)
            s = px.index[trig.values & ret.notna().values]
            rows = [summarize(ret.loc[s].values, f"{name} day-level N={len(s)}")]
            for g in (5, 10, 21):
                e = declusters(s, g, px.index)
                rr = summarize(ret.loc[e].values, f"{name} gap={g}")
                w = int((ret.loc[e].values > 0).sum())
                rr["sign_p"] = round(sign_test(w, len(e)), 4)
                rows.append(rr)
            show(rows, f"T4 decluster {name} h={h}")
            e = declusters(s, 5, px.index)
            show(era_split(e, ret.loc[e].values), f"  era split {name}")
            show(midterm_split(e, ret.loc[e].values), f"  midterm split {name}")
            print("  " + cluster_note(e, ret.loc[e].values, k=3))

    # ---------------- T5  REFERENCE CLASS ------------------------------------
    print("\n\n########## T5  REFERENCE CLASS (6 complex-break families) ##########")
    for h in (3, 5):
        res = []
        for fam, members in FAMILIES.items():
            p = mkpanel(members)
            rr = {t: dret(p[t]) for t in members}
            m = None
            for t in members:
                mm = rr[t] <= -0.02
                m = mm if m is None else (m & mm)
            legs = [(t, -1.0 / len(members)) for t in members]
            ret = vehicle_ret(p, legs, h, lag=1)
            valid = ret.notna()
            s = p.index[m.values & valid.values]
            if len(s) < 5:
                print(f"  {fam}: only {len(s)} trigger days, skipped")
                continue
            e = declusters(s, 5, p.index)
            span = (p.index >= s[0]) & (p.index <= s[-1]) & valid.values
            cond = ret.loc[e].values
            ctrl = ret[span].values
            se = np.sqrt(cond.var(ddof=1) / len(cond) + ctrl.var(ddof=1) / len(ctrl))
            res.append({"family": fam, "n_days": len(s), "n_epi": len(e),
                        "cond_pct": 100 * cond.mean(), "ctrl_pct": 100 * ctrl.mean(),
                        "excess_pct": 100 * (cond.mean() - ctrl.mean()),
                        "se_pct": 100 * se,
                        "t": (cond.mean() - ctrl.mean()) / se,
                        "hit": 100 * (cond > 0).mean()})
        show(res, f"T5 reference class, SHORT the complex, h={h}")
        if len(res) >= 2:
            y = np.array([x["excess_pct"] for x in res])
            se = np.array([x["se_pct"] for x in res])
            w = 1.0 / se ** 2
            fe = (w * y).sum() / w.sum()
            fe_se = 1.0 / np.sqrt(w.sum())
            Q = float((w * (y - fe) ** 2).sum())
            df = len(y) - 1
            I2 = max(0.0, (Q - df) / Q) * 100 if Q > 0 else 0.0
            from scipy import stats as _st
            qp = 1 - _st.chi2.cdf(Q, df)
            print(f"  Cochran Q = {Q:.2f} (df {df}, p {qp:.3f})   I^2 = {I2:.1f}%"
                  f"   fixed-effect common excess = {fe:+.3f}% "
                  f"(se {fe_se:.3f}, t {fe/fe_se:+.2f})")
            print(f"  max family excess = {y.max():+.3f}% ({res[int(np.argmax(y))]['family']})")

        # permutation max-of-N
        rng = np.random.default_rng(42)
        B = 2000
        null_max = np.zeros(B)
        fam_data = []
        for fam, members in FAMILIES.items():
            p = mkpanel(members)
            rr = {t: dret(p[t]) for t in members}
            m = None
            for t in members:
                mm = rr[t] <= -0.02
                m = mm if m is None else (m & mm)
            legs = [(t, -1.0 / len(members)) for t in members]
            ret = vehicle_ret(p, legs, h, lag=1)
            valid = ret.notna()
            s = p.index[m.values & valid.values]
            if len(s) < 5:
                continue
            e = declusters(s, 5, p.index)
            span_idx = p.index[(p.index >= s[0]) & (p.index <= s[-1]) & valid.values]
            fam_data.append((fam, len(e), ret.loc[span_idx].values))
        for b in range(B):
            mx = -9e9
            for fam, k, pool in fam_data:
                samp = rng.choice(pool, size=k, replace=False)
                mx = max(mx, samp.mean() - pool.mean())
            null_max[b] = 100 * mx
        obs_max = max(x["excess_pct"] for x in res)
        print(f"  permutation max-of-{len(fam_data)} p = "
              f"{(null_max >= obs_max).mean():.4f}   (observed max {obs_max:+.3f}%, "
              f"null max median {np.median(null_max):+.3f}%, 95th {np.percentile(null_max,95):+.3f}%)")

    # ---------------- T6  BOOK OVERLAP ---------------------------------------
    print("\n\n########## T6  BOOK OVERLAP ##########")
    led = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
    led["Signal Date"] = pd.to_datetime(led["Signal Date"])
    metals_names = ["GLD", "SLV", "GDX", "GDXJ", "NEM", "NUGT", "DUST", "JNUG",
                    "JDST", "AEM", "GOLD", "AU", "KGC", "PAAS", "AGI", "SIL",
                    "IAU", "SLVP", "WPM", "FNV", "RGLD", "HL", "CDE", "EGO",
                    "SSRM", "BTG", "GFI", "HMY", "SBSW", "PLTM", "PPLT", "SIVR"]
    trig_set = set(pd.DatetimeIndex(sig_all).normalize())
    sub = led[led["Ticker"].isin(metals_names)]
    on = sub[sub["Signal Date"].isin(trig_set)]
    print(f"  ledger metals-name trades: {len(sub)};  fired ON a trigger day: {len(on)}")
    if len(on):
        print(on.groupby(["Strategy", "Direction"]).agg(
            n=("R_Multiple", "size"), avgR=("R_Multiple", "mean")).to_string())
        print("\n  by ticker/direction:")
        print(on.groupby(["Ticker", "Direction"]).size().to_string())
    # also: what fired the NEXT session (our entry day)
    pos = pd.Series(range(len(px.index)), index=px.index)
    nxt = set()
    for d in sig_all:
        p = pos.get(d)
        if p is not None and p + 1 < len(px.index):
            nxt.add(px.index[p + 1])
    on2 = sub[sub["Signal Date"].isin(nxt)]
    print(f"\n  fired on the NEXT session (our entry day): {len(on2)}")
    if len(on2):
        print(on2.groupby(["Strategy", "Direction"]).agg(
            n=("R_Multiple", "size"), avgR=("R_Multiple", "mean")).to_string())


if __name__ == "__main__":
    main()
