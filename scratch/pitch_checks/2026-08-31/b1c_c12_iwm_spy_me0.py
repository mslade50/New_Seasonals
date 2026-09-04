"""C12 -- the small-cap laggard into the month turn.  Long IWM / short SPY
(beta-adjusted), entered MOC on ME-0.

Live state (2026-08-28 close): IWM r5 rank 20.2, z10 -1.07, IWM -3.06% off its
52w high; SPY r5 52.4, -1.10% off its high.

Convention: the laggard GATE is read at the ME-1 close (lag-1, so a MOC order
at the ME-0 close is placeable), entry is the ME-0 CLOSE, exit is the close h
sessions later.  Beta is a POINT-IN-TIME trailing-252d OLS slope of IWM daily
returns on SPY daily returns, known at the entry close.

Kills attempted:
 1. per-leg attribution -- if the short leg subtracts the pair is strictly
    worse than the outright and must be reported that way (registry 2026-08-19)
 2. gate attribution -- run it WITHOUT the laggard gate
 3. placebo ladder on the month-position anchor
 4. midterm split (REQUIRED) and era split
 5. cost: two legs, ~10-12 bps round trip
 6. re-skin check vs the closed DIA/SPY index pair (2026-08-13) and the closed
    ME-3 IWM session (2026-08-26): residual correlation + mask overlap
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pitch_lab import *  # noqa: E402,F403
from pitch_lab import load_prices, summarize, show, sign_test, pct_rank, zscore  # noqa: E402

HS = (1, 2, 3, 5, 10)
COST_2LEG_BPS = 11.0


def month_end_positions(idx):
    ym = pd.Series(idx.year * 100 + idx.month, index=range(len(idx)))
    return sorted(int(p) for p in
                  ym.groupby(ym.values).apply(lambda s: s.index[-1]).values)


def rolling_beta(ry: pd.Series, rx: pd.Series, n: int = 252) -> pd.Series:
    cov = ry.rolling(n).cov(rx)
    var = rx.rolling(n).var()
    return (cov / var)


def main() -> None:
    px = load_prices(["IWM", "SPY", "DIA", "QQQ"])
    idx = px["SPY"].index.intersection(px["IWM"].index)
    C = pd.DataFrame({t: px[t]["Close"].reindex(idx) for t in px}).dropna(how="any")
    idx = C.index
    r = C.pct_change(fill_method=None)
    beta = rolling_beta(r["IWM"], r["SPY"]).shift(1)     # known at the entry close

    me = [p for p in month_end_positions(idx) if p >= 300]
    me_dates = idx[me]

    # ---- live-state reproduction --------------------------------------
    r5_iwm = pct_rank(C["IWM"], 5)
    r5_spy = pct_rank(C["SPY"], 5)
    z10_iwm = zscore(C["IWM"], 10)
    hi_iwm = C["IWM"] / C["IWM"].rolling(252).max() - 1.0
    hi_spy = C["SPY"] / C["SPY"].rolling(252).max() - 1.0
    last = idx[-1]
    print("=" * 78)
    print("C12  long IWM / short SPY, entered MOC at the ME-0 close")
    print("=" * 78)
    print(f"live state at {last.date()}: IWM r5 {r5_iwm[last]:.1f}  z10 "
          f"{z10_iwm[last]:+.2f}  offhigh {100*hi_iwm[last]:+.2f}% | "
          f"SPY r5 {r5_spy[last]:.1f}  offhigh {100*hi_spy[last]:+.2f}%  | "
          f"beta(IWM~SPY,252d) {beta[last]:.3f}")

    # gate read at ME-1 (one session BEFORE the entry close)
    gate_src = pd.Series(False, index=idx)
    lag_gate = (r5_iwm < 30) & (r5_spy > 40) & (z10_iwm < -0.5)
    gate_src.loc[:] = lag_gate.shift(1).fillna(False).values

    # ---- forward returns ------------------------------------------------
    def legs_for(h):
        fwd_i = C["IWM"].shift(-h) / C["IWM"] - 1.0
        fwd_s = C["SPY"].shift(-h) / C["SPY"] - 1.0
        pair = fwd_i - beta * fwd_s
        return fwd_i, fwd_s, pair

    print("\n" + "=" * 78)
    print("1/2. ME-0 anchor: outright vs beta-neutral pair, gated and ungated")
    print("=" * 78)
    for h in HS:
        fi, fs, pr = legs_for(h)
        base_ok = fi.notna() & fs.notna() & beta.notna()
        anch = me_dates[base_ok.reindex(me_dates).fillna(False).values]
        g = anch[gate_src.reindex(anch).fillna(False).values]
        rows = [
            summarize(fi.reindex(anch).values, f"ME-0 IWM outright (N={len(anch)})"),
            summarize(fs.reindex(anch).values, f"ME-0 SPY outright (N={len(anch)})"),
            summarize(pr.reindex(anch).values, f"ME-0 PAIR beta-neutral"),
            summarize(fi[base_ok].values, "CTRL IWM all days"),
            summarize(pr[base_ok].values, "CTRL PAIR all days"),
            summarize(fi.reindex(g).values, f"ME-0 + LAGGARD GATE, IWM (N={len(g)})"),
            summarize(pr.reindex(g).values, f"ME-0 + LAGGARD GATE, PAIR"),
        ]
        show(rows, f"h={h}")
        m_pair_g = np.nanmean(pr.reindex(g).values) if len(g) else np.nan
        m_pair_u = np.nanmean(pr.reindex(anch).values)
        m_out_g = np.nanmean(fi.reindex(g).values) if len(g) else np.nan
        print(f"  gate attribution PAIR: gated {100*m_pair_g:+.3f}% vs ungated "
              f"{100*m_pair_u:+.3f}%  -> gate worth {100*(m_pair_g-m_pair_u):+.3f}pp")
        print(f"  short-leg attribution: outright IWM {100*m_out_g:+.3f}% vs pair "
              f"{100*m_pair_g:+.3f}%  -> short leg worth "
              f"{100*(m_pair_g-m_out_g):+.3f}pp")
        print(f"  cost: pair {100*100*m_pair_g/COST_2LEG_BPS:.2f}x of "
              f"{COST_2LEG_BPS} bps | outright "
              f"{100*100*m_out_g/(COST_2LEG_BPS/2):.2f}x of "
              f"{COST_2LEG_BPS/2:.1f} bps")
        if len(g) > 2:
            v = pr.reindex(g).dropna().values
            w = int((v > 0).sum())
            bp = float((pr[base_ok] > 0).mean())
            print(f"  gated pair record {w}-{len(v)-w}, sign p vs own base "
                  f"{100*bp:.1f}% = {sign_test(w, len(v), p=bp):.4f}")
            print("  gated dates:", ", ".join(f"{d.date()}" for d in g))

    # ---- 3. placebo ladder ---------------------------------------------
    print("\n" + "=" * 78)
    print("3. PLACEBO LADDER on the month-position anchor (h=3 and h=5, pair)")
    print("=" * 78)
    for h in (3, 5):
        fi, fs, pr = legs_for(h)
        base_ok = fi.notna() & fs.notna() & beta.notna()
        base = pr[base_ok].mean()
        rows = []
        for k in range(-5, 4):
            pos = [p + k for p in me if 0 <= p + k < len(idx)]
            d = idx[pos]
            d = d[base_ok.reindex(d).fillna(False).values]
            r_ = summarize(pr.reindex(d).values,
                           f"ME{k:+d}" if k else "ME-0 (TRUE)")
            if r_["n"]:
                r_["excess_bp"] = round(100 * (r_["mean_pct"] - 100 * base), 2)
            rows.append(r_)
        show(rows, f"pair ladder h={h}")
        ex = [(x["label"], x.get("excess_bp", -1e9)) for x in rows]
        order = sorted(ex, key=lambda z: -z[1])
        rank = [i for i, (l, _) in enumerate(order) if "TRUE" in l][0] + 1
        print(f"  TRUE anchor ranks {rank} of {len(order)}: "
              f"{[l for l, _ in order]}")

    # ---- 4. midterm + era ------------------------------------------------
    print("\n" + "=" * 78)
    print("4. MIDTERM + ERA SPLIT (ME-0 pair, ungated -- the gated cell is tiny)")
    print("=" * 78)
    for h in (3, 5):
        fi, fs, pr = legs_for(h)
        base_ok = fi.notna() & fs.notna() & beta.notna()
        anch = me_dates[base_ok.reindex(me_dates).fillna(False).values]
        s = pr.reindex(anch).dropna()
        mt = s.index.year % 4 == 2
        rows = [summarize(s[mt].values, "MIDTERM"),
                summarize(s[~mt].values, "non-midterm"),
                summarize(s[s.index < "2013-01-01"].values, "pre-2013"),
                summarize(s[s.index >= "2013-01-01"].values, "2013+"),
                summarize(s[s.index >= "2018-01-01"].values, "2018+"),
                summarize(s[s.index.month == 8].values, "AUGUST turns"),
                summarize(s[(s.index.month == 8) & mt].values,
                          "AUGUST x MIDTERM (live)")]
        show(rows, f"pair h={h}")

    # ---- 6. re-skin check ------------------------------------------------
    print("\n" + "=" * 78)
    print("6. RE-SKIN CHECK vs the closed DIA/SPY pair (2026-08-13)")
    print("=" * 78)
    beta_d = rolling_beta(r["DIA"], r["SPY"]).shift(1)
    beta_q = rolling_beta(r["QQQ"], r["SPY"]).shift(1)
    for h in (1, 3, 5):
        fi = C["IWM"].shift(-h) / C["IWM"] - 1.0
        fs = C["SPY"].shift(-h) / C["SPY"] - 1.0
        fd = C["DIA"].shift(-h) / C["DIA"] - 1.0
        fq = C["QQQ"].shift(-h) / C["QQQ"] - 1.0
        p_i = fi - beta * fs
        p_d = fd - beta_d * fs
        p_q = fq - beta_q * fs
        j = pd.concat({"iwm": p_i, "dia": p_d, "qqq": p_q}, axis=1).dropna()
        print(f"  h={h}: corr(IWM/SPY resid, DIA/SPY resid) = "
              f"{j['iwm'].corr(j['dia']):+.3f} ; vs QQQ/SPY resid = "
              f"{j['iwm'].corr(j['qqq']):+.3f}")
    # overlap with the closed ME-3 IWM session
    me3 = set(idx[[p - 3 for p in me if p - 3 >= 0]])
    me0 = set(me_dates)
    print(f"  ME-0 vs the closed ME-3 IWM session: mask overlap "
          f"{len(me0 & me3)} of {len(me0)} -- disjoint by construction; the "
          f"ME-3 kill was a SCANNED session in the SAME 16-session grid this "
          f"ladder walks, so C12 pays the same scan charge.")


if __name__ == "__main__":
    main()
