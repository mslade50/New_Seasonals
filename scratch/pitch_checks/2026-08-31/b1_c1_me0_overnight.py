"""C1 round 1 -- the ME-0 CLOSING AUCTION -> ME+1 OPEN overnight return.

Object: on(t) = Open[t+1] / Close[t] - 1, anchored at the LAST trading session
of each calendar month (ME-0).  Trade would be MOC today -> MOO tomorrow.

This is deliberately a DIFFERENT return object from every month-end closure in
data/pitch_negative_registry.md, all five of which measured close-to-close
session or multi-session returns.  Verified by reading them; stated in the
report.

Kill battery specialised to an overnight object:
 1. full history, mean/median/hit/N, sign test vs the instrument's OWN
    unconditional overnight up-rate (NOT 0.5 -- equity overnight drift is
    positive and large)
 2. controls: unconditional overnight, and every OTHER month-position session
 3. placebo ladder ME-5 .. ME+3 with the rank of the true anchor
 4. era split pre/post 2013 and pre/post 2018
 5. midterm split (year % 4 == 2) -- REQUIRED, registry 2026-08-26
 6. month-of-year scan + max-of-12 permutation charge
 7. cost: MOC->MOO round trip ~5 bps
 8. concentration: cluster_note, drop-best-2, year histogram
 9. dividend-basis check: master_prices is ADJUSTED, so an ex-div night carries
    the dividend in the overnight leg.  ^GSPC is a price index and cannot.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pitch_lab import *  # noqa: E402,F403
from pitch_lab import load_prices, summarize, show, sign_test, cluster_note  # noqa: E402

VEH = ["SPY", "IWM", "^GSPC", "QQQ", "DIA"]
COST_BPS = 5.0          # MOC -> MOO round trip, single leg, mid of 4-6
BAR_X = 5.0             # need >= 5x cost


def month_end_positions(idx: pd.DatetimeIndex) -> list[int]:
    """Integer positions of the LAST trading session of each calendar month."""
    ym = pd.Series(idx.year * 100 + idx.month, index=range(len(idx)))
    last = ym.groupby(ym.values).apply(lambda s: s.index[-1])
    return sorted(int(p) for p in last.values)


def overnight(df: pd.DataFrame) -> pd.Series:
    """Open[t+1] / Close[t] - 1, aligned to t.  FRACTION."""
    return (df["Open"].shift(-1) / df["Close"] - 1.0)


def cell(on: pd.Series, idx: pd.DatetimeIndex, positions: list[int],
         label: str) -> dict:
    d = idx[[p for p in positions if 0 <= p < len(idx)]]
    v = on.reindex(d).values
    return summarize(v, label)


def anchored(idx, me_pos, k):
    """Positions of the ME-0 + k session, both-side guarded."""
    return [p + k for p in me_pos if 0 <= p + k < len(idx) - 1]


def main() -> None:
    px = load_prices(VEH)
    print("=" * 78)
    print("C1  ME-0 CLOSE -> ME+1 OPEN overnight return")
    print("=" * 78)

    store = {}
    for t in VEH:
        df = px[t]
        idx = df.index
        on = overnight(df)
        me = month_end_positions(idx)
        # ME-0 anchors must have a next session
        me = [p for p in me if p < len(idx) - 1]
        store[t] = (df, idx, on, me)

        valid = on.dropna()
        base_up = float((valid > 0).mean())
        a = idx[me]
        v = on.reindex(a).dropna().values
        wins = int((v > 0).sum())
        n = len(v)
        p_own = sign_test(wins, n, p=base_up)
        p_coin = sign_test(wins, n, p=0.5)

        print(f"\n----- {t}  ({idx[0].date()} .. {idx[-1].date()}, "
              f"{len(valid)} overnights) -----")
        rows = [summarize(v, f"ME-0 overnight (N={n})"),
                summarize(valid.values, f"CTRL all overnights (N={len(valid)})")]
        # every OTHER month-position session
        other = valid.drop(a, errors="ignore")
        rows.append(summarize(other.values, f"CTRL ex-ME-0 (N={len(other)})"))
        show(rows, f"{t}: conditional vs unconditional overnight")
        excess = 100 * (np.nanmean(v) - valid.mean())
        print(f"  base overnight up-rate {100*base_up:.2f}%  |  ME-0 hit "
              f"{100*wins/n:.2f}%  ({wins}-{n-wins})")
        print(f"  sign p vs OWN base rate = {p_own:.4f}   (vs a coin = "
              f"{p_coin:.4f})")
        print(f"  EXCESS over unconditional overnight = {excess:+.4f} pp "
              f"= {excess*100:+.2f} bps")
        rt = COST_BPS
        print(f"  cost: 1 leg MOC->MOO ~{rt:.0f} bps round trip -> "
              f"|edge|/cost = {abs(excess)*100/rt:.2f}x  (need >= {BAR_X:.0f}x)")
        print(f"  concentration: {cluster_note(a[:n], v)}")
        srt = np.sort(v)
        if n > 2:
            print(f"  drop-best-2 mean = {100*srt[:-2].mean():+.4f}%  "
                  f"drop-worst-2 mean = {100*srt[2:].mean():+.4f}%")

    # ---------------------------------------------------------------- ladder
    print("\n" + "=" * 78)
    print("3. PLACEBO LADDER on the month-position anchor (same overnight object)")
    print("=" * 78)
    for t in VEH:
        df, idx, on, me = store[t]
        valid = on.dropna()
        base = valid.mean()
        rows = []
        for k in range(-5, 4):
            pos = anchored(idx, me, k)
            r = cell(on, idx, pos, f"ME{k:+d}" if k else "ME-0 (TRUE)")
            if r["n"]:
                r["excess_bp"] = round(100 * 100 * (r["mean_pct"] / 100 - base), 2)
            rows.append(r)
        show(rows, f"{t} ladder (excess_bp = bps over unconditional overnight)")
        ex = [(r["label"], r.get("excess_bp", np.nan)) for r in rows]
        order = sorted(ex, key=lambda x: -(x[1] if x[1] == x[1] else -1e9))
        rank = [i for i, (lbl, _) in enumerate(order) if "TRUE" in lbl][0] + 1
        print(f"  TRUE anchor ranks {rank} of {len(order)} by excess.  "
              f"order: {[l for l, _ in order]}")

    # ------------------------------------------------------------ era splits
    print("\n" + "=" * 78)
    print("4. ERA SPLITS (ME-0 overnight)")
    print("=" * 78)
    for t in VEH:
        df, idx, on, me = store[t]
        a = idx[me]
        s = on.reindex(a).dropna()
        valid = on.dropna()
        rows = []
        for lo, hi, lbl in [(None, "2013-01-01", "pre-2013"),
                            ("2013-01-01", None, "2013+"),
                            (None, "2018-01-01", "pre-2018"),
                            ("2018-01-01", None, "2018+"),
                            ("2020-01-01", None, "2020+")]:
            m = pd.Series(True, index=s.index)
            cm = pd.Series(True, index=valid.index)
            if lo:
                m &= s.index >= pd.Timestamp(lo)
                cm &= valid.index >= pd.Timestamp(lo)
            if hi:
                m &= s.index < pd.Timestamp(hi)
                cm &= valid.index < pd.Timestamp(hi)
            r = summarize(s[m.values].values, lbl)
            if r["n"]:
                r["ctrl_pct"] = round(100 * valid[cm.values].mean(), 4)
                r["excess_bp"] = round(100 * (r["mean_pct"] - r["ctrl_pct"]), 2)
            rows.append(r)
        show(rows, f"{t} era split, ME-0 overnight vs same-era unconditional")

    # --------------------------------------------------------- midterm split
    print("\n" + "=" * 78)
    print("5. MIDTERM SPLIT (year %% 4 == 2)  -- REQUIRED, registry 2026-08-26")
    print("=" * 78)
    for t in VEH:
        df, idx, on, me = store[t]
        s = on.reindex(idx[me]).dropna()
        mt = (s.index.year % 4 == 2)
        rows = [summarize(s[mt].values, "MIDTERM years"),
                summarize(s[~mt].values, "non-midterm")]
        # august-in-a-midterm, the live cell
        aug_mt = mt & (s.index.month == 8)
        rows.append(summarize(s[aug_mt].values, "AUGUST x MIDTERM (live cell)"))
        show(rows, f"{t} midterm split")
        if aug_mt.sum():
            print("   live-cell dates:",
                  ", ".join(f"{d.date()}:{100*v:+.3f}%"
                            for d, v in s[aug_mt].items()))

    # ------------------------------------------------------- month-of-year
    print("\n" + "=" * 78)
    print("6. MONTH-OF-YEAR SCAN + max-of-12 permutation charge")
    print("=" * 78)
    rng = np.random.default_rng(42)
    for t in VEH:
        df, idx, on, me = store[t]
        s = on.reindex(idx[me]).dropna()
        valid = on.dropna()
        base = valid.mean()
        rows = []
        ts = {}
        for m in range(1, 13):
            v = s[s.index.month == m].values
            r = summarize(v, pd.Timestamp(2020, m, 1).strftime("%b"))
            if r["n"] > 1:
                r["excess_bp"] = round(100 * (r["mean_pct"] - 100 * base), 2)
                ts[m] = r["t"]
            rows.append(r)
        show(rows, f"{t} ME-0 overnight by month")
        aug_t = ts.get(8, np.nan)
        obs_max = max(abs(x) for x in ts.values())
        # permutation: reassign month labels at random across ME-0 anchors
        vals = s.values
        months = s.index.month.values
        cnt = 0
        NP = 2000
        for _ in range(NP):
            perm = rng.permutation(vals)
            mx = 0.0
            for m in range(1, 13):
                w = perm[months == m]
                if len(w) > 1 and w.std(ddof=1) > 0:
                    mx = max(mx, abs(w.mean() / (w.std(ddof=1) / np.sqrt(len(w)))))
            if mx >= obs_max:
                cnt += 1
        print(f"  August t = {aug_t:+.2f}   best-month |t| = {obs_max:.2f}   "
              f"max-of-12 permutation P(max|t| >= observed) = {cnt/NP:.3f}")

    # --------------------------------------- dividend-basis contamination check
    print("\n" + "=" * 78)
    print("9. DIVIDEND BASIS: adjusted ETFs vs the ^GSPC price index")
    print("=" * 78)
    g = store["^GSPC"][2].dropna()
    for t in ["SPY", "IWM", "QQQ", "DIA"]:
        s = store[t][2].dropna()
        j = pd.concat({"etf": s, "idx": g}, axis=1).dropna()
        d = j["etf"] - j["idx"]
        me = store[t][3]
        a = store[t][1][me]
        dm = d.reindex(a).dropna()
        print(f"  {t:5s} overnight minus ^GSPC overnight: all days "
              f"{100*100*d.mean():+.3f} bps | ME-0 {100*100*dm.mean():+.3f} bps "
              f"(N={len(dm)})  -> ME-0 div loading "
              f"{100*100*(dm.mean()-d.mean()):+.3f} bps")


if __name__ == "__main__":
    main()
