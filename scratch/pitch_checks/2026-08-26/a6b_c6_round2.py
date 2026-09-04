"""a6b / C6 round 2. Round 1 survived: h=3 episode mean +0.530% over 62,
edge +0.354pp, era-stable, 8.8x cost. This is the teardown.

Five attacks, in the order the registry says they land:
  A. LIVE-STATE OUT-OF-SAMPLE on the book's own sizing statistic. The
     2026-08-25 registry filter: "ask what the maximum dial this cell has
     ever been observed at". a6 already answered 88.9 -- and that maximum IS
     TODAY. Quantify it ex-today and grade the dial>=50 slice at episode
     level.
  B. REFERENCE-CLASS PERMUTATION. HYG ranks 2 of 14 at h=3. A max-of-14 null
     asks whether a +0.331pp gate value is what the best of fourteen looks
     like anyway.
  C. DEPTH-MATCHED gate attribution. The SPY-off ladder rises monotonically
     with depth (0.5% -> +0.386, 1% -> +0.530, 2% -> +0.839, 3% -> +0.945),
     which is the signature of a dip-depth trade. Compare the gate against a
     depth-matched parent at TODAY's depth (-1.54%).
  D. BULL-TAPE SELECTOR (the EWZ/SMH over-selection test): what share of
     trigger days sit above SPY's 200d against the base rate?
  E. SIGN TEST against SPY's OWN up-rate, not a coin.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (close_panel, declusters, rolling_on_valid, show,  # noqa
                       sign_test, summarize, vehicle_ret)

REF = ["HYG", "LQD", "IEF", "TLT", "XLU", "XLK", "XLP", "XLV", "XLF", "EFA",
       "EEM", "GLD", "QQQ", "IWM"]
ROOT = Path(__file__).resolve().parents[3]


def off_high(px, t, n=252):
    hi = rolling_on_valid(px[t], lambda x: x.rolling(n).max())
    return px[t] / hi - 1.0


def main() -> None:
    tick = sorted(set(REF + ["SPY"]))
    px = close_panel(tick)
    oh = {t: off_high(px, t) for t in tick}
    hyg_hi = oh["HYG"] >= -0.0005
    spy_off = oh["SPY"] <= -0.01
    m = (hyg_hi & spy_off).fillna(False)
    pxs = px.loc[px.index[m.values].min():]
    m = m.loc[pxs.index]
    r3 = vehicle_ret(pxs, [("SPY", 1.0)], 3, 1)
    r10 = vehicle_ret(pxs, [("SPY", 1.0)], 10, 1)
    d = pxs.index[m.values & r3.notna().values]
    ep = declusters(d, 10, pxs.index)

    print("### A. live state vs the cell's observed dial range ###")
    fr = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
    fr.index = pd.to_datetime(fr.index)
    ma = fr["63d"].rolling(10, min_periods=10).mean().dropna()
    dj = pd.DatetimeIndex(d).intersection(ma.index)
    v = ma.loc[dj]
    ex = v.drop(pd.Timestamp("2026-08-25"), errors="ignore")
    print(f"  cell days with a dial: {len(v)} of {len(d)}   "
          f"max={v.max():.1f} on {v.idxmax().date()}")
    print(f"  MAX EX-TODAY = {ex.max():.1f} on {ex.idxmax().date()};  "
          f"today = {ma.iloc[-1]:.1f}  -> today is "
          f"{ma.iloc[-1] - ex.max():+.1f} pts beyond anything the cell has "
          "ever printed")
    print(f"  distribution of the cell's dial: p50={v.median():.1f} "
          f"p90={v.quantile(0.9):.1f} p99={v.quantile(0.99):.1f}")
    for lbl, cut in [(">=85", 85), (">=70", 70), (">=60", 60), (">=50", 50)]:
        print(f"    cell days {lbl}: {int((ex >= cut).sum())} ex-today")
    epd = pd.DatetimeIndex(ep).intersection(ma.index)
    ev = ma.loc[epd]
    for h, r in [(3, r3), (10, r10)]:
        hi = epd[ev >= 50]
        lo = epd[ev < 50]
        show([summarize(r.reindex(hi).values, f"h={h} EPISODES dial>=50 (N={len(hi)})"),
              summarize(r.reindex(lo).values, f"h={h} EPISODES dial<50 (N={len(lo)})")],
             f"A. episode-level dial split h={h}")
        # walk the dial threshold
        row = []
        for c in (30, 40, 50, 60):
            s = summarize(r.reindex(epd[ev >= c]).values, f"h={h} dial>={c}")
            row.append(s)
        show(row, f"A2. dial-threshold walk h={h} (episodes)")

    print("\n### B. reference-class permutation, max-of-14, h=3 ###")
    base_d = pxs.index[spy_off.loc[pxs.index].fillna(False).values
                       & r3.notna().values]
    base_ep = declusters(base_d, 10, pxs.index)
    base_mean = 100 * r3.loc[base_ep].mean()
    gates = {}
    for t in REF:
        mm = (spy_off & (oh[t] >= -0.0005)).fillna(False).loc[pxs.index]
        dd = pxs.index[mm.values & r3.notna().values]
        if len(dd) == 0:
            continue
        e = declusters(dd, 10, pxs.index)
        gates[t] = (len(e), 100 * r3.loc[e].mean() - base_mean)
    g = pd.Series({k: v[1] for k, v in gates.items()}).sort_values(ascending=False)
    print(f"  parent {base_mean:+.3f}%   family gate values:\n{g.round(3).to_string()}")
    print(f"  family mean {g.mean():+.3f}pp sd {g.std():.3f}pp; "
          f"HYG {g['HYG']:+.3f}pp = z {(g['HYG']-g.mean())/g.std():+.2f} "
          "within its own family")
    # permutation: relocate each vehicle's trigger dates at random within the
    # SPY-off day pool, keeping the count, and record max-of-14 gate value
    rng = np.random.default_rng(42)
    pool = base_d
    counts = {t: int(((spy_off & (oh[t] >= -0.0005)).fillna(False)
                      .loc[pxs.index].values & r3.notna().values).sum())
              for t in gates}
    obs_max = g.max()
    hits = 0
    n_perm = 2000
    for _ in range(n_perm):
        mx = -9e9
        for t, c in counts.items():
            if c == 0 or c > len(pool):
                continue
            samp = pd.DatetimeIndex(rng.choice(pool, size=c, replace=False))
            e = declusters(samp.sort_values(), 10, pxs.index)
            val = 100 * r3.loc[e].mean() - base_mean
            mx = max(mx, val)
        if mx >= obs_max:
            hits += 1
    print(f"  P(max-of-{len(counts)} relocated >= observed max {obs_max:+.3f}pp) "
          f"= {hits/n_perm:.3f}   [{n_perm} permutations]")
    # and specifically for HYG's own value
    hits_h = 0
    for _ in range(n_perm):
        samp = pd.DatetimeIndex(rng.choice(pool, size=counts["HYG"], replace=False))
        e = declusters(samp.sort_values(), 10, pxs.index)
        if 100 * r3.loc[e].mean() - base_mean >= g["HYG"]:
            hits_h += 1
    print(f"  P(a RANDOM {counts['HYG']}-day relocation of HYG's own count beats "
          f"{g['HYG']:+.3f}pp) = {hits_h/n_perm:.3f}")

    print("\n### C. depth-matched gate attribution (today SPY -1.54% off) ###")
    out = []
    for lo, hi in [(0.01, 0.02), (0.01, 0.03), (0.012, 0.02), (0.02, 1.0),
                   (0.03, 1.0)]:
        band = (oh["SPY"] <= -lo) & (oh["SPY"] > -hi)
        for lbl, mm in [("parent", band), ("gated", band & hyg_hi)]:
            mm = mm.fillna(False).loc[pxs.index]
            dd = pxs.index[mm.values & r3.notna().values]
            if len(dd) == 0:
                out.append({"band": f"{100*lo:.1f}-{100*hi:.0f}%", "which": lbl,
                            "n_ep": 0}); continue
            e = declusters(dd, 10, pxs.index)
            out.append({"band": f"{100*lo:.1f}-{100*hi:.0f}%", "which": lbl,
                        "n_ep": len(e),
                        "h3_pct": round(100 * r3.loc[e].mean(), 3),
                        "h10_pct": round(100 * r10.reindex(e).mean(), 3)})
    df = pd.DataFrame(out)
    print(df.to_string(index=False))
    for b in df["band"].unique():
        s = df[df.band == b]
        if len(s) == 2 and s["n_ep"].min() > 0:
            gv = s[s.which == "gated"]["h3_pct"].iloc[0] - \
                 s[s.which == "parent"]["h3_pct"].iloc[0]
            print(f"  band {b}: credit gate worth {gv:+.3f}pp at h=3")

    print("\n### D. bull-tape selector test ###")
    sma200 = rolling_on_valid(pxs["SPY"], lambda x: x.rolling(200).mean())
    above = (pxs["SPY"] > sma200)
    print(f"  trigger days above SPY 200d: "
          f"{100*above.reindex(d).mean():.1f}%  (N={len(d)})")
    print(f"  base rate over the same span: "
          f"{100*above.loc[pxs.index].mean():.1f}%")
    print(f"  SPY-off parent days above 200d: "
          f"{100*above.reindex(base_d).mean():.1f}%")

    print("\n### E. sign test against SPY's OWN up-rate ###")
    for h, r in [(3, r3), (10, r10)]:
        base_up = float((r.dropna() > 0).mean())
        w = int((r.loc[ep] > 0).sum())
        print(f"  h={h}: cell {w}-{len(ep)-w} = {100*w/len(ep):.1f}% vs SPY's own "
              f"{100*base_up:.1f}%   sign p vs coin "
              f"{sign_test(w, len(ep)):.4f}   vs OWN rate "
              f"{sign_test(w, len(ep), base_up):.4f}")

    print("\n### F. recent-era slice: does it still work post-2023? ###")
    for h, r in [(3, r3), (10, r10)]:
        for cut in ("2020-01-01", "2023-01-01"):
            sel = ep >= pd.Timestamp(cut)
            allm = 100 * r.loc[pxs.index >= pd.Timestamp(cut)].dropna().mean()
            s = summarize(r.reindex(ep[sel]).values, f"h={h} {cut[:4]}+")
            s["ctl"] = round(allm, 3)
            s["edge"] = round(s["mean_pct"] - allm, 3)
            show([s], "")


if __name__ == "__main__":
    main()
