"""C3 round 3 (develop): long UNG after a volume-confirmed thrust day.
Episode table (data-glitch audit incl. NG=F same-day), horizon_scan 1..10,
entry form MOC(t+1) vs close-anchored LIMIT(close_t - k ATR) working on t+1
as WHOLE variants, exits (time vs stop/target, pessimistic grader convention:
stop arms day 2, both-touch books the stop, gapped stop fills at open - 13 bps),
loser paths, live levels (Wilder-14 ATR)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd
from pitch_grammar import wilder_atr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kC_c3_ung_volthrust import build, eps  # noqa: E402

if __name__ == "__main__":
    d, px, r1, vr = build()
    idx = px.index
    O, H, L, C = (d[k].astype(float).values for k in ("Open", "High", "Low", "Close"))
    atr = pd.Series(wilder_atr(H, L, C), index=idx)
    trig = (r1 >= 0.05) & (vr >= 3)
    tdays = idx[trig.reindex(idx, fill_value=False).values]
    ng = load_prices(["NG=F"])["NG=F"]["Close"].astype(float)
    ng1 = ng.pct_change()

    # 1. episode audit table (all trigger days)
    pos = pd.Series(range(len(idx)), index=idx)
    rows = []
    for t in tdays:
        p = pos[t]
        if p + 4 >= len(idx):
            rows.append({"date": t.date(), "r1": 100 * r1[t], "volx": vr[t], "live": True})
            continue
        rows.append({"date": t.date(), "r1": round(100 * r1[t], 2), "volx": round(vr[t], 2),
                     "ng_same_day": round(100 * ng1.get(t, np.nan), 2),
                     "t+1": round(100 * (C[p + 1] / C[p] - 1), 2),
                     "t+1_ng": round(100 * ng1.get(idx[p + 1], np.nan), 2),
                     "h2_lag1": round(100 * (C[p + 3] / C[p + 1] - 1), 2),
                     "h3_lag1": round(100 * (C[p + 4] / C[p + 1] - 1), 2)})
    print(pd.DataFrame(rows).to_string(index=False))

    legs = [("UNG", 1.0)]
    epi = declusters(tdays, 3, idx)
    show(horizon_scan(px, epi, legs, hs=tuple(range(1, 11)), min_gap=3),
         "2. horizon scan 1..10 (declustered 3td, lag=1)")

    # 3. entry forms as whole variants, exit at close t+1+h
    def limit_variant(k, h):
        out, fills = [], 0
        for t in epi:
            p = pos[t]
            if p + 1 + h >= len(idx):
                continue
            lvl = C[p] - k * atr.iloc[p]
            if L[p + 1] <= lvl:
                fill = min(lvl, O[p + 1])
                out.append(C[p + 1 + h] / fill - 1)
                fills += 1
            else:
                out.append(0.0)
        return np.array(out), fills
    rows = []
    for h in (2, 3):
        moc = vehicle_ret(px, legs, h).loc[epi].dropna().values
        r = summarize(moc, f"MOC t+1 close, h={h}")
        r["fills"] = len(moc)
        rows.append(r)
        moo = np.array([C[pos[t] + 1 + h] / O[pos[t] + 1] - 1 for t in epi
                        if pos[t] + 1 + h < len(idx)])
        r = summarize(moo, f"MOO t+1 open, h={h}")
        r["fills"] = len(moo)
        rows.append(r)
        for k in (0.25, 0.5, 1.0):
            v, f = limit_variant(k, h)
            r = summarize(v, f"LIMIT close-{k}ATR on t+1, h={h} (0 if unfilled)")
            r["fills"] = f
            rows.append(r)
    show(rows, "3. entry form, whole variants (per signal; unfilled = 0)")

    # 4. exits on the MOC entry, h=3: stop / target in Wilder ATR (entry-day ATR)
    def bracket(stop_k, tgt_k, h=3):
        out = []
        for t in epi:
            p = pos[t]
            e = p + 1
            if e + h >= len(idx):
                continue
            ent, a = C[e], atr.iloc[e]
            st = ent - stop_k * a if stop_k else -np.inf
            tg = ent + tgt_k * a if tgt_k else np.inf
            res = C[e + h] / ent - 1
            for j in range(e + 1, e + h + 1):
                if stop_k and j >= e + 1 and L[j] <= st:   # arms day 2 = first day after entry
                    fillp = O[j] * (1 - 0.0013) if O[j] < st else st
                    res = fillp / ent - 1
                    break
                if tgt_k and H[j] >= tg:
                    res = (max(tg, O[j]) if O[j] > tg else tg) / ent - 1
                    break
            out.append(res)
        return np.array(out)
    rows = [summarize(bracket(0, 0), "time only h=3")]
    for sk, tk in [(1.0, 0), (1.5, 0), (2.0, 0), (0, 1.5), (0, 2.0), (1.5, 2.0), (2.0, 3.0)]:
        rows.append(summarize(bracket(sk, tk), f"stop {sk} / target {tk} ATR h=3"))
    show(rows, "4. exits on MOC entry (episodes)")

    # 5. loser paths
    ep = episode_paths(px, epi, legs, 3)
    fin = ep[3]
    losers = ep[fin < 0]
    print("\n5. loser paths (cum % from entry close, day 1..3):")
    print((100 * losers).round(2).to_string())
    mae = ep.min(axis=1)
    print(f"   median day-close MAE all episodes {100*mae.median():+.2f}%, losers "
          f"{100*losers.min(axis=1).median():+.2f}%, worst {100*mae.min():+.2f}%; "
          f"episodes with day-1 close <= -3%: {(ep[1] <= -0.03).sum()} of {len(ep)}, "
          f"of which ended negative {((ep[1] <= -0.03) & (fin < 0)).sum()}")

    # 7. h=2 (the horizon-scan peak) robustness, and the t+1-dip attribution
    r2 = vehicle_ret(px, legs, 2)
    e2 = declusters(tdays, 2, idx)
    e2 = pd.DatetimeIndex([d_ for d_ in e2 if pd.notna(r2.get(d_))])  # drop live day
    v2 = r2.loc[e2].values
    show([summarize(bracket(0, 0, 2), "h=2 time only"),
          summarize(bracket(1.5, 0, 2), "h=2 stop 1.5"),
          summarize(bracket(1.0, 0, 2), "h=2 stop 1.0"),
          summarize(bracket(0, 2.0, 2), "h=2 target 2.0"),
          summarize(bracket(1.5, 2.0, 2), "h=2 stop 1.5 / target 2.0")], "7a. h=2 exits")
    ep2 = episode_paths(px, e2, legs, 2)
    print("   h=2 loser paths:\n", (100 * ep2[ep2[2] < 0]).round(2).to_string())
    yy = pd.DatetimeIndex(e2).year
    mo = pd.DatetimeIndex(e2).month
    wd = pd.DatetimeIndex(e2).weekday
    w = int((v2 > 0).sum())
    print(f"\n7. h=2 N={len(v2)} mean {100*v2.mean():+.3f}% record {w}-{len(v2)-w} "
          f"sign p {sign_test(w, len(v2)):.4f} boot {bootstrap_p_le0(v2):.3f}; "
          f"{cluster_note(e2, v2)}")
    best = int(np.argmax(v2))
    show([summarize(v2[yy < 2011], "pre-2011"), summarize(v2[yy >= 2011], "2011+"),
          summarize(v2[yy < 2018], "pre-2018"), summarize(v2[yy >= 2018], "2018+"),
          summarize(np.delete(v2, best), f"ex best episode {e2[best].date()}"),
          summarize(v2[~yy.isin([2009, 2018])], "ex-2009 & 2018"),
          summarize(v2[np.isin(mo, [9, 10])], "Sep-Oct"),
          summarize(v2[~np.isin(mo, [9, 10])], "other months"),
          summarize(v2[np.isin(wd, [0, 1])], "Mon/Tue signal (storage Thu inside hold)"),
          summarize(v2[~np.isin(wd, [0, 1])], "Wed-Fri signal (no storage in hold)")],
         "7b. h=2 splits (episodes)")
    c = px["UNG"]
    d1 = c.shift(-1) / c - 1.0
    novol = (r1 >= 0.05) & (vr < 3)
    for thr in (-0.01, -0.02):
        m = novol & (d1 <= thr)
        em, vm = eps(r2, m, 2, idx)
        mc = trig & (d1 <= thr)
        ec, vc = eps(r2, mc, 2, idx)
        show([summarize(vm, f"NO-vol +5% with t+1 <= {thr:.0%} (N={len(em)})"),
              summarize(vc, f"VOL +5% with t+1 <= {thr:.0%} (N={len(ec)})")],
             f"7c. is it the t+1 dip? h=2 lag1")

    # 6. live levels
    ngd = load_prices(["NG=F"])["NG=F"]
    ng_atr = wilder_atr(ngd["High"].values, ngd["Low"].values, ngd["Close"].values)[-1]
    print(f"\n6. LIVE {idx[-1].date()}: UNG close {C[-1]:.2f}, 1d {100*r1.iloc[-1]:+.2f}%, "
          f"vol {vr.iloc[-1]:.2f}x, Wilder-14 ATR {atr.iloc[-1]:.4f} "
          f"({100*atr.iloc[-1]/C[-1]:.2f}%); NG=F close {ngd['Close'].iloc[-1]:.3f}, "
          f"ATR {ng_atr:.4f}, 1d {100*ng1.iloc[-1]:+.2f}%")
