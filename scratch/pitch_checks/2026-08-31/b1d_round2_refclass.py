"""Round 2 for C1 and C12.

A. C1 REFERENCE CLASS (run BEFORE round 2 per the registry's 2026-08-28 rule):
   the ME-0 overnight excess measured on every US equity ETF the anchor could
   equally have been applied to.  Fixed-effect common excess, Cochran Q,
   I-squared, and where SPY/IWM rank inside their own family.
B. C1 DEFINITION NEIGHBOURS + the drop-December test: the month-of-year scan
   said December carries the cell.  Take it out.
C. C1 NET-OF-COST equity curve in the live era, the only number that decides
   whether this is a trade.
D. C12 with a LOOSENED laggard gate so gate attribution has real N, walked
   across four thresholds in both directions.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pitch_lab import *  # noqa: E402,F403
from pitch_lab import load_prices, summarize, show, sign_test, pct_rank  # noqa: E402

FAMILY = ["SPY", "IWM", "QQQ", "DIA", "XLK", "XLF", "XLE", "XLV", "XLI",
          "XLP", "XLY", "XLU", "XLB", "EFA", "EEM", "IJH", "IJR", "VTI",
          "IWD", "IWF"]
COST_BPS = 5.0


def me_pos(idx):
    ym = pd.Series(idx.year * 100 + idx.month, index=range(len(idx)))
    return sorted(int(p) for p in
                  ym.groupby(ym.values).apply(lambda s: s.index[-1]).values)


def main() -> None:
    px = load_prices(FAMILY)
    print("=" * 78)
    print("A. C1 REFERENCE CLASS -- ME-0 close -> next open, US equity ETFs")
    print("=" * 78)
    rows, eff, var = [], [], []
    for t in FAMILY:
        if t not in px:
            continue
        d = px[t]
        idx = d.index
        on = (d["Open"].shift(-1) / d["Close"] - 1.0)
        valid = on.dropna()
        if len(valid) < 500:
            continue
        a = idx[[p for p in me_pos(idx) if p < len(idx) - 1]]
        v = on.reindex(a).dropna()
        base = valid.mean()
        ex = v.mean() - base
        se = np.sqrt(v.var(ddof=1) / len(v) + valid.var(ddof=1) / len(valid))
        rows.append({"ticker": t, "n": len(v), "me0_bp": round(1e4 * v.mean(), 2),
                     "uncond_bp": round(1e4 * base, 2),
                     "excess_bp": round(1e4 * ex, 2),
                     "t": round(ex / se, 2), "hit": round(100 * (v > 0).mean(), 1),
                     "from": str(idx[0].date())})
        eff.append(ex)
        var.append(se ** 2)
    df = pd.DataFrame(rows).sort_values("excess_bp", ascending=False)
    print(df.to_string(index=False))
    eff, var = np.array(eff), np.array(var)
    w = 1.0 / var
    fe = float((w * eff).sum() / w.sum())
    fe_se = float(np.sqrt(1.0 / w.sum()))
    Q = float((w * (eff - fe) ** 2).sum())
    dfree = len(eff) - 1
    I2 = max(0.0, 100 * (Q - dfree) / Q) if Q > 0 else 0.0
    from scipy.stats import chi2  # noqa: E402
    print(f"\n  fixed-effect COMMON excess = {1e4*fe:+.2f} bps  (t {fe/fe_se:+.2f})")
    print(f"  Cochran Q = {Q:.2f} on {dfree} df, p = "
          f"{1 - chi2.cdf(Q, dfree):.4f};  I-squared = {I2:.1f}%")
    print(f"  SPY ranks {int((df['ticker'] == 'SPY').argmax()) + 1} of {len(df)}; "
          f"IWM ranks {int((df['ticker'] == 'IWM').argmax()) + 1} of {len(df)}")
    print("  -> a HOMOGENEOUS family with a POSITIVE common effect is one number "
          "wearing 20 labels, not 20 independent confirmations.")

    # ------------------------------------------------------------------ B
    print("\n" + "=" * 78)
    print("B. C1 drop-December (and drop Oct/Nov/Dec) -- the month scan's answer")
    print("=" * 78)
    for t in ["SPY", "IWM", "QQQ", "DIA"]:
        d = px[t]
        idx = d.index
        on = (d["Open"].shift(-1) / d["Close"] - 1.0)
        valid = on.dropna()
        base = valid.mean()
        a = idx[[p for p in me_pos(idx) if p < len(idx) - 1]]
        s = on.reindex(a).dropna()
        out = []
        for lbl, m in [("all months", np.ones(len(s), bool)),
                       ("ex-Dec", s.index.month != 12),
                       ("ex-Oct/Nov/Dec", ~s.index.month.isin([10, 11, 12])),
                       ("ex-Dec, 2013+", (s.index.month != 12)
                        & (s.index.year >= 2013)),
                       ("ex-Dec, 2018+", (s.index.month != 12)
                        & (s.index.year >= 2018)),
                       ("Aug only", s.index.month == 8)]:
            r = summarize(s[m].values, lbl)
            if r["n"]:
                r["excess_bp"] = round(1e4 * (r["mean_pct"] / 100 - base), 2)
                r["x_cost"] = round(abs(r["excess_bp"]) / COST_BPS, 2)
            out.append(r)
        show(out, f"{t} ME-0 overnight, month exclusions")

    # ------------------------------------------------------------------ C
    print("\n" + "=" * 78)
    print("C. C1 NET-OF-COST: long MOC ME-0 -> MOO ME+1, 5 bps round trip")
    print("=" * 78)
    for t in ["SPY", "IWM", "QQQ", "DIA"]:
        d = px[t]
        idx = d.index
        on = (d["Open"].shift(-1) / d["Close"] - 1.0)
        a = idx[[p for p in me_pos(idx) if p < len(idx) - 1]]
        s = on.reindex(a).dropna()
        for lo, lbl in [(None, "full history"), (2013, "2013+"),
                        (2018, "2018+"), (2020, "2020+")]:
            v = s if lo is None else s[s.index.year >= lo]
            net = v.values - COST_BPS / 1e4
            w = int((net > 0).sum())
            print(f"  {t:4s} {lbl:13s} N={len(v):3d}  gross "
                  f"{1e4*v.mean():+6.2f} bps  NET {1e4*net.mean():+6.2f} bps  "
                  f"net win {100*w/len(v):5.1f}%  cum NET "
                  f"{100*net.sum():+6.2f}pp  net sign p "
                  f"{sign_test(w, len(v), p=0.5):.3f}")

    # ------------------------------------------------------------------ D
    print("\n" + "=" * 78)
    print("D. C12 laggard gate LOOSENED so gate attribution has real N")
    print("=" * 78)
    C = pd.DataFrame({t: px[t]["Close"] for t in ("SPY", "IWM")}).dropna()
    idx = C.index
    r = C.pct_change(fill_method=None)
    beta = (r["IWM"].rolling(252).cov(r["SPY"]) / r["SPY"].rolling(252).var()).shift(1)
    r5_i, r5_s = pct_rank(C["IWM"], 5), pct_rank(C["SPY"], 5)
    mep = [p for p in me_pos(idx) if p >= 300]
    med = idx[mep]
    for h in (3, 5):
        fi = C["IWM"].shift(-h) / C["IWM"] - 1.0
        fs = C["SPY"].shift(-h) / C["SPY"] - 1.0
        pr = fi - beta * fs
        ok = pr.notna()
        anch = med[ok.reindex(med).fillna(False).values]
        out = [summarize(pr.reindex(anch).values, f"ME-0 ungated (N={len(anch)})")]
        for thr in (20, 30, 40, 50):
            gate = ((r5_i < thr) & (r5_s > r5_i)).shift(1).fillna(False)
            g = anch[gate.reindex(anch).fillna(False).values.astype(bool)]
            rr = summarize(pr.reindex(g).values, f"gate r5_IWM<{thr} & lagging SPY")
            if rr["n"]:
                rr["gate_pp"] = round(rr["mean_pct"]
                                      - 100 * np.nanmean(pr.reindex(anch).values), 3)
                rr["outright_pct"] = round(100 * np.nanmean(fi.reindex(g).values), 3)
                rr["short_leg_pp"] = round(rr["mean_pct"] - rr["outright_pct"], 3)
            out.append(rr)
        # the complement: IWM LEADING into the turn
        gate = ((r5_i > 70) & (r5_i > r5_s)).shift(1).fillna(False)
        g = anch[gate.reindex(anch).fillna(False).values.astype(bool)]
        out.append(summarize(pr.reindex(g).values, "COMPLEMENT: IWM leading r5>70"))
        show(out, f"C12 gate ladder, pair h={h}")


if __name__ == "__main__":
    main()
