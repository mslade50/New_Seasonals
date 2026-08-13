"""b2 - C9: does a fast fragility-dial RISE say anything DIRECTIONAL?

VINTAGE DISCLOSURE, up front: data/rd2_fragility.parquet is append-only
point-in-time ONLY since 2026-07-02. Everything earlier is a RECOMPUTE vintage
that drifted up to ~7 points on the 63d dial, and the signal DEFINITIONS are
today's code applied to all history. The script prints the pre/post share and
re-runs on the PIT-only slice.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import pandas as pd, numpy as np

PIT_CUT = pd.Timestamp("2026-07-02")
px = close_panel(["SPY", "XLV", "TLT", "QQQ"])

for src, label in ((Path("data/rd2_fragility.parquet"), "SIZING parquet (PIT>=2026-07-02, recompute before)"),
                   (Path("data/rd2_fragility_ts.parquet"), "rd2_fragility_ts RESEARCH-ONLY raw recompute")):
    if not src.exists():
        print(f"\n### {label}: FILE ABSENT ({src}) ###")
        continue
    f = pd.read_parquet(src)
    f.index = pd.to_datetime(f.index).tz_localize(None).normalize()
    f = f[~f.index.duplicated(keep="last")].sort_index()
    print("\n" + "#" * 78)
    print(f"### DIAL VINTAGE: {label}")
    print(f"### rows {len(f)}  {f.index[0].date()}..{f.index[-1].date()}  "
          f"cols {f.columns.tolist()}")
    if "63d" not in f.columns:
        print("### no 63d column, skipping")
        continue
    ma = f["63d"].rolling(10).mean()
    d21 = ma - ma.shift(21)

    # align onto the price index
    idx = px.index[(px.index >= f.index[0]) & (px.index <= f.index[-1])]
    ma_a = ma.reindex(idx).ffill(limit=3)
    d21_a = d21.reindex(idx).ffill(limit=3)
    spy = px["SPY"]
    near_high = (spy / spy.rolling(252).max()).reindex(idx) >= 0.99

    print(f"  live: ma10_63d {float(ma.iloc[-1]):.2f}  delta21 {float(d21.iloc[-1]):+.2f}  "
          f"SPY {100*(float(spy.iloc[-1]/spy.rolling(252).max().iloc[-1])-1):.2f}% off 52w high")

    gates = {
        "A full cell  delta21>=+30 AND SPY within 1% of 52w high":
            (d21_a >= 30) & near_high,
        "B delta leg ONLY  delta21>=+30":
            (d21_a >= 30),
        "C near-high leg ONLY  SPY within 1% of high":
            near_high.copy(),
        "D LEVEL only  ma10_63d>=70":
            (ma_a >= 70),
        "E LEVEL>=70 AND near-high (no delta)":
            (ma_a >= 70) & near_high,
        "F full cell + level>=70 (today's actual state)":
            (d21_a >= 30) & near_high & (ma_a >= 70),
    }
    for gname, m in gates.items():
        m = m.reindex(px.index, fill_value=False).fillna(False).astype(bool)
        n = int(m.sum())
        pre = int((m & (px.index < PIT_CUT)).sum())
        print(f"\n----- GATE {gname}: N={n} days "
              f"({pre} pre-{PIT_CUT.date()} = {100*pre/max(n,1):.1f}% recompute vintage)")
        if n == 0:
            continue
        rows = []
        for h in (1, 2, 3, 5, 10):
            r = fwd_lag(px["SPY"], h, 1)
            valid = r.dropna().index
            trig = px.index[m.values].intersection(valid)
            epi = declusters(trig, 21, valid)   # 21d delta = maximal overlap
            base = r.loc[valid]
            rr = summarize(r.loc[epi].values, f"h={h} EPISODES(gap21)")
            rr["day_n"] = len(trig)
            rr["day_mean"] = round(100 * r.loc[trig].mean(), 3)
            rr["ctl_all"] = round(100 * base.mean(), 3)
            rr["edge"] = round(rr.get("mean_pct", np.nan) - 100 * base.mean(), 3)
            rows.append(rr)
        show(rows, f"SPY forward, gate={gname[:1]}")
        # h=5 detail on the full cell
        if gname.startswith("A") or gname.startswith("D") or gname.startswith("F"):
            r = fwd_lag(px["SPY"], 5, 1)
            valid = r.dropna().index
            trig = px.index[m.values].intersection(valid)
            epi = declusters(trig, 21, valid)
            v = r.loc[epi].values
            w = int((v > 0).sum())
            print(f"  h=5 episodes N={len(epi)} record {w}-{len(v)-w} "
                  f"sign p={sign_test(w, len(v)):.4f}  "
                  f"boot P(mean<=0)={bootstrap_p_le0(v):.3f}")
            print(f"  {cluster_note(epi, v)}")
            print("  episode dates:", ", ".join(str(x.date()) for x in epi))
            loc = local_control(valid, trig)
            show([summarize(r.loc[loc].values, "CTRL-c local +/-126td ex-trigger"),
                  summarize(r.loc[valid].values, "CTRL-b all days")], "controls h=5")
            show(era_split(epi, v, "2021-01-01"), "episode split pre/post 2021")
            # PIT-only slice
            pit_trig = trig[trig >= PIT_CUT]
            print(f"  PIT-ONLY slice (>= {PIT_CUT.date()}): N={len(pit_trig)} days"
                  + (f", h=5 mean {100*r.loc[pit_trig].mean():+.3f}%"
                     if len(pit_trig) else " -> NOTHING TESTABLE"))
        # defensive expressions on the full cell
        if gname.startswith("A"):
            for legs, lbl in (([("XLV", 1.0), ("SPY", -1.0)], "long XLV / short SPY"),
                              ([("TLT", 1.0), ("SPY", -1.0)], "long TLT / short SPY"),
                              ([("SPY", -1.0)], "outright SHORT SPY")):
                rr = []
                for h in (3, 5, 10):
                    r = vehicle_ret(px, legs, h, 1)
                    valid = r.dropna().index
                    trig = px.index[m.values].intersection(valid)
                    epi = declusters(trig, 21, valid)
                    o = summarize(r.loc[epi].values, f"{lbl} h={h}")
                    o["ctl_all"] = round(100 * r.loc[valid].mean(), 3)
                    o["edge"] = round(o.get("mean_pct", np.nan)
                                      - 100 * r.loc[valid].mean(), 3)
                    rr.append(o)
                show(rr, f"defensive expression: {lbl}")

    # threshold neighbours on the full cell (delta and near-high tolerance)
    print("\n----- DEFINITION NEIGHBOURS (h=5, episodes, gap21) -----")
    r = fwd_lag(px["SPY"], 5, 1)
    valid = r.dropna().index
    base = 100 * r.loc[valid].mean()
    out = []
    for dthr in (20, 25, 30, 35, 40):
        for nh in (0.97, 0.98, 0.99, 1.0):
            m = ((d21_a >= dthr) & ((spy / spy.rolling(252).max()).reindex(idx) >= nh))
            m = m.reindex(px.index, fill_value=False).fillna(False).astype(bool)
            trig = px.index[m.values].intersection(valid)
            if len(trig) == 0:
                out.append({"d21": dthr, "nearhigh": nh, "n_days": 0})
                continue
            epi = declusters(trig, 21, valid)
            o = summarize(r.loc[epi].values, "")
            o.pop("label")
            o["d21"] = dthr
            o["nearhigh"] = nh
            o["n_days"] = len(trig)
            o["edge"] = round(o["mean_pct"] - base, 3)
            out.append(o)
    df = pd.DataFrame(out)
    cols = ["d21", "nearhigh", "n_days", "n", "mean_pct", "hit", "t", "worst_pct", "edge"]
    print(df[[c for c in cols if c in df.columns]].round(3).to_string(index=False))
    print(f"  (CTRL all-days h=5 = {base:+.3f}%)")
    # lookback neighbour: delta over 10 and 42 sessions
    for w in (10, 21, 42):
        dd = ma - ma.shift(w)
        dda = dd.reindex(idx).ffill(limit=3)
        m = ((dda >= 30) & ((spy / spy.rolling(252).max()).reindex(idx) >= 0.99))
        m = m.reindex(px.index, fill_value=False).fillna(False).astype(bool)
        trig = px.index[m.values].intersection(valid)
        if len(trig) == 0:
            print(f"  delta window {w}td: N=0")
            continue
        epi = declusters(trig, w, valid)
        o = summarize(r.loc[epi].values, f"delta window={w}td")
        print(f"  delta window {w:2d}td: days={len(trig)} epi={o['n']} "
              f"mean {o['mean_pct']:+.3f}% hit {o['hit']:.0f}% worst {o['worst_pct']:+.2f}% "
              f"edge {o['mean_pct']-base:+.3f}%")
