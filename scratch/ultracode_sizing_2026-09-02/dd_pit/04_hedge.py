"""dd_pit step 4: re-run the dial-armed whole-book beta hedge on the EXTENDED
point-in-time dial (through 2026-09-01) and the rebuilt per-strategy payload.

Shipped spec (plan section 8): arm at dial >= 50, release < 45 (hysteresis on
the lag-1 10d-MA 63d dial), whole-book target, SPY as the hedge proxy, 126d
rolling beta (lag-1, clipped [-1, 2]), ratio 1.0x, 2 bps x |beta| friction per
arm event. Hedge PnL = -armed x beta x SPY return on flat $750k NAV.

Reports, for three dial vintages (PIT extended / current-weights recompute /
live parquet) on the PIT window 2018-01-02 .. 2026-09-01 (plus live on its
full 2016-07-20+ window): episode count, hedged vs unhedged Sharpe / maxDD /
worst-21d overall and on armed (>= 50 zone) days, the realised armed beta
split into SPY-up and SPY-down days, the Aug-2026 episode day by day, the
result with and without the 2026 episodes, an episode bootstrap, and a
futures-margin line. Writes hedge_results.json + hedge_aug2026_daily.csv.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NAV = 750_000.0
FRICTION_BPS = 2.0
BETA_W = 126
ARM_ON, ARM_OFF = 50.0, 45.0
LAST = pd.Timestamp("2026-09-01")
RNG = np.random.default_rng(2026)
pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 300, "display.float_format", "{:,.3f}".format)
OUT: dict = {}

# margin inputs (edit via CLI: --es-im 20000 --mes-im 2000 to override)
ES_IM = float(sys.argv[sys.argv.index("--es-im") + 1]) if "--es-im" in sys.argv else np.nan
MES_IM = float(sys.argv[sys.argv.index("--mes-im") + 1]) if "--mes-im" in sys.argv else np.nan

W = pd.read_parquet(HERE / "strategy_daily_extended.parquet")
W.index = pd.to_datetime(W.index)
book = (W["book"] / NAV).rename("book")
px = pq.read_table(ROOT / "data/master_prices.parquet", columns=["ticker", "date", "Close"],
                   filters=[("ticker", "in", ["SPY", "^VIX", "^GSPC"])]).to_pandas().pivot(index="date", columns="ticker", values="Close")
px.index = pd.to_datetime(px.index)
px = px[px.index <= LAST]
keep = book.index.intersection(px.index[px["SPY"].notna()])
book = book.loc[keep]; pxk = px.loc[keep].ffill()
spy = pxk["SPY"].pct_change(fill_method=None).rename("spy")
vix_lag = pxk["^VIX"].shift(1)
spx = pxk["^GSPC"]

D = pd.read_parquet(HERE / "pit_dial_extended.parquet")
D.index = pd.to_datetime(D.index)
dials = {"pit": D["pit"], "cur": D["cur_recompute"], "live": D["live"]}
dials_lag = {k: v.shift(1).reindex(keep) for k, v in dials.items()}


def rolling_beta(y, x, w):
    return (y.rolling(w).cov(x) / x.rolling(w).var()).shift(1).clip(-1, 2)


def arm_hysteresis(x: pd.Series, on: float, off: float) -> pd.Series:
    armed = np.zeros(len(x), bool); state = False
    for i, v in enumerate(x.values):
        if np.isnan(v):
            armed[i] = state; continue
        if not state and v >= on:
            state = True
        elif state and v < off:
            state = False
        armed[i] = state
    return pd.Series(armed, index=x.index)


def episodes(armed: pd.Series, gap: int = 21):
    idx = np.where(armed.values)[0]
    if len(idx) == 0:
        return []
    eps = []; s = idx[0]; prev = idx[0]
    for i in idx[1:]:
        if i - prev > gap:
            eps.append((s, prev)); s = i
        prev = i
    eps.append((s, prev)); return eps


def sharpe(r):
    return float(r.mean() / r.std() * np.sqrt(252)) if len(r) > 1 and r.std() > 0 else np.nan


def maxdd(r):
    eq = r.cumsum(); return float((eq - eq.cummax()).min())


def worst21(r):
    return float(r.rolling(21).sum().min()) if len(r) >= 21 else np.nan


def updown_beta(g: pd.DataFrame):
    dn = g[g.f < 0]; up = g[g.f > 0]
    bd = float(np.polyfit(dn.f, dn.r, 1)[0]) if len(dn) > 5 else np.nan
    bu = float(np.polyfit(up.f, up.r, 1)[0]) if len(up) > 5 else np.nan
    ba = float(np.polyfit(g.f, g.r, 1)[0]) if len(g) > 5 else np.nan
    return ba, bu, bd


beta = rolling_beta(book, spy, BETA_W)


def evaluate(idx, dial_lag: pd.Series, label: str, drop_episodes_starting_after=None, drop_year=None) -> dict:
    m = pd.DataFrame({"r": book.reindex(idx), "f": spy.reindex(idx), "b": beta.reindex(idx).fillna(0.0), "d": dial_lag.reindex(idx)})
    m = m.dropna(subset=["r", "f"])
    m["a"] = arm_hysteresis(m["d"], ARM_ON, ARM_OFF).astype(float)
    if drop_year is not None:
        m = m[m.index.year != drop_year]
    h = -m.a * m.b * m.f
    ev = ((m.a.diff() > 0) | ((m.a == 1) & m.a.shift(1).isna())).astype(float)
    fr = ev * FRICTION_BPS / 1e4 * m.b.abs()
    h = h - fr
    eps = episodes(m.a > 0)
    rows = []
    for s, e in eps:
        seg = m.iloc[s:e + 1]; hs = h.iloc[s:e + 1]
        rows.append(dict(start=str(seg.index[0].date()), end=str(seg.index[-1].date()), armed_days=int(seg.a.sum()),
                         hedge_usd=float(hs.sum() * NAV), book_usd=float(seg.r.sum() * NAV), hedged_usd=float((seg.r + hs).sum() * NAV),
                         spy_ret_pct=float(((1 + seg.f).prod() - 1) * 100), beta_mean=float(seg.loc[seg.a > 0, "b"].mean()),
                         vix_start=float(vix_lag.reindex(seg.index).iloc[0]), dial_max=float(seg.d.max()),
                         open=bool(seg.index[-1] == m.index[-1] and seg.a.iloc[-1] == 1)))
    if drop_episodes_starting_after is not None:
        cut = pd.Timestamp(drop_episodes_starting_after)
        for s, e in eps:
            if m.index[s] >= cut:
                h.iloc[s:e + 1] = 0.0; m.iloc[s:e + 1, m.columns.get_loc("a")] = 0.0
        rows = [x for x in rows if pd.Timestamp(x["start"]) < cut]
    ep = np.array([x["hedge_usd"] for x in rows]); n = len(ep)
    t = float(ep.mean() / (ep.std(ddof=1) / np.sqrt(n))) if n > 2 and ep.std(ddof=1) > 0 else np.nan
    am = m.a > 0; ru = m.r; rh = m.r + h
    ga = m[am]; gu = m[~am]
    ba, bu, bd = updown_beta(ga) if am.sum() > 20 else (np.nan, np.nan, np.nan)
    ua, uu, ud = updown_beta(gu)
    ha = h[am]
    res = dict(label=label, days=int(len(m)), first=str(m.index[0].date()), last=str(m.index[-1].date()), armed_days=int(am.sum()), armed_share=float(am.mean()),
               n_episodes=n, hedge_total_usd=float(h.sum() * NAV), friction_usd=float(fr.sum() * NAV),
               ep_mean_usd=float(ep.mean()) if n else 0.0, ep_pos=int((ep > 0).sum()), t_clustered=t,
               sharpe_u=sharpe(ru), sharpe_h=sharpe(rh), ann_u_pct=float(ru.mean() * 252 * 100), ann_h_pct=float(rh.mean() * 252 * 100),
               vol_u_pct=float(ru.std() * np.sqrt(252) * 100), vol_h_pct=float(rh.std() * np.sqrt(252) * 100),
               maxdd_u_pct=maxdd(ru) * 100, maxdd_h_pct=maxdd(rh) * 100, worst21_u_pct=worst21(ru) * 100, worst21_h_pct=worst21(rh) * 100,
               worstday_u_pct=float(ru.min() * 100), worstday_h_pct=float(rh.min() * 100),
               armed_sharpe_u=sharpe(ru[am]), armed_sharpe_h=sharpe(rh[am]), armed_maxdd_u_pct=maxdd(ru[am]) * 100 if am.any() else np.nan,
               armed_maxdd_h_pct=maxdd(rh[am]) * 100 if am.any() else np.nan, armed_worst21_u_pct=worst21(ru[am]) * 100, armed_worst21_h_pct=worst21(rh[am]) * 100,
               armed_sd_u_bps=float(ru[am].std() * 1e4) if am.any() else np.nan, armed_sd_h_bps=float(rh[am].std() * 1e4) if am.any() else np.nan,
               armed_book_usd=float(ru[am].sum() * NAV), armed_hedged_usd=float(rh[am].sum() * NAV),
               armed_spy_ann_pct=float(m.f[am].mean() * 252 * 100) if am.any() else np.nan,
               beta_hat_armed=float(m.b[am].mean()) if am.any() else np.nan,
               beta_real_armed=ba, beta_up_armed=bu, beta_down_armed=bd, beta_real_unarmed=ua, beta_up_unarmed=uu, beta_down_unarmed=ud,
               armed_up_days=int((ga.f > 0).sum()), armed_down_days=int((ga.f < 0).sum()),
               hedge_up_days_usd=float(ha[ga.f > 0].sum() * NAV) if am.any() else 0.0, hedge_down_days_usd=float(ha[ga.f < 0].sum() * NAV) if am.any() else 0.0,
               book_up_days_usd=float(ga.r[ga.f > 0].sum() * NAV) if am.any() else 0.0, book_down_days_usd=float(ga.r[ga.f < 0].sum() * NAV) if am.any() else 0.0,
               episodes=rows)
    yrs = {}
    for y, g in m.groupby(m.index.year):
        if g.a.sum() > 0:
            yrs[int(y)] = dict(armed_days=int(g.a.sum()), hedge_usd=float(h[g.index].sum() * NAV), sh_u=sharpe(g.r), sh_h=sharpe(g.r + h[g.index]))
    res["years"] = yrs
    res["loyo_years"] = len(yrs); res["loyo_not_worse"] = int(sum(1 for v in yrs.values() if v["sh_h"] >= v["sh_u"] - 1e-9)); res["loyo_pnl_pos"] = int(sum(1 for v in yrs.values() if v["hedge_usd"] > 0))
    res["_m"] = m; res["_h"] = h
    return res


HEAD = ["label", "first", "last", "armed_days", "n_episodes", "ep_pos", "hedge_total_usd", "t_clustered", "sharpe_u", "sharpe_h", "maxdd_u_pct", "maxdd_h_pct",
        "worst21_u_pct", "worst21_h_pct", "armed_sharpe_u", "armed_sharpe_h", "armed_maxdd_u_pct", "armed_maxdd_h_pct", "armed_worst21_u_pct", "armed_worst21_h_pct",
        "armed_sd_u_bps", "armed_sd_h_bps", "beta_hat_armed", "beta_real_armed", "beta_up_armed", "beta_down_armed", "loyo_years", "loyo_not_worse", "loyo_pnl_pos"]

pit_win = keep[(keep >= "2018-01-02") & (keep <= LAST)]
live_win = keep[(keep >= "2016-07-20") & (keep <= LAST)]
runs = {
    "pit|2018+": evaluate(pit_win, dials_lag["pit"], "pit|2018+"),
    "cur|2018+": evaluate(pit_win, dials_lag["cur"], "cur|2018+"),
    "live|2018+": evaluate(pit_win, dials_lag["live"], "live|2018+"),
    "live|2016-07+": evaluate(live_win, dials_lag["live"], "live|2016-07+"),
    "cur|2016-07+": evaluate(live_win, dials_lag["cur"], "cur|2016-07+"),
    # the study's window, on the extended inputs, to isolate what the extension changed
    "pit|2018..2026-07-02": evaluate(keep[(keep >= "2018-01-02") & (keep <= "2026-07-02")], dials_lag["pit"], "pit|2018..2026-07-02"),
    # sanity: without the 2026 episodes
    "pit|2018+ ex Aug-2026 ep": evaluate(pit_win, dials_lag["pit"], "pit|2018+ ex Aug-2026 ep", drop_episodes_starting_after="2026-07-01"),
    "pit|2018+ ex 2026": evaluate(pit_win, dials_lag["pit"], "pit|2018+ ex 2026", drop_year=2026),
    "cur|2018+ ex Aug-2026 ep": evaluate(pit_win, dials_lag["cur"], "cur|2018+ ex Aug-2026 ep", drop_episodes_starting_after="2026-07-01"),
    "live|2018+ ex Aug-2026 ep": evaluate(pit_win, dials_lag["live"], "live|2018+ ex Aug-2026 ep", drop_episodes_starting_after="2026-07-01"),
    "live|2016-07+ ex Aug-2026 ep": evaluate(live_win, dials_lag["live"], "live|2016-07+ ex Aug-2026 ep", drop_episodes_starting_after="2026-07-01"),
}
T = pd.DataFrame([{k: v for k, v in r.items() if k in HEAD} for r in runs.values()])[HEAD]
print("=== headline: book / SPY / 126d beta / arm 50 rel 45 / 1.0x ===")
print(T.round(2).to_string(index=False))
OUT["headline"] = T.round(4).to_dict("records")

print("\n=== episode tables ===")
OUT["episodes"] = {}
for k in ["pit|2018+", "cur|2018+", "live|2018+", "live|2016-07+"]:
    r = runs[k]
    print(f"\n{k}: {r['n_episodes']} episodes, {r['ep_pos']} positive, total ${r['hedge_total_usd']:,.0f}, t={r['t_clustered']:.2f}")
    E = pd.DataFrame(r["episodes"]); print(E.round(2).to_string(index=False))
    print("  by year:", {y: (v["armed_days"], round(v["hedge_usd"])) for y, v in r["years"].items()})
    OUT["episodes"][k] = r["episodes"]; OUT.setdefault("years", {})[k] = r["years"]

print("\n=== realised beta anatomy on armed vs unarmed days (is the hedge over-hedging rallies / under-hedging selloffs?) ===")
rows = []
for k in ["pit|2018+", "cur|2018+", "live|2018+", "live|2016-07+", "pit|2018+ ex Aug-2026 ep"]:
    r = runs[k]; m = r["_m"]; h = r["_h"]
    for state, g in [("armed", m[m.a > 0]), ("unarmed", m[m.a == 0])]:
        if len(g) < 30:
            continue
        ba, bu, bd = updown_beta(g); hh = -g.b * g.f
        up = g[g.f > 0]; dn = g[g.f < 0]
        rows.append(dict(run=k, state=state, days=len(g), up_days=len(up), down_days=len(dn), beta_hat=float(g.b.mean()), beta_real=ba, beta_up=bu, beta_down=bd,
                         spy_ann_pct=float(g.f.mean() * 252 * 100), spy_up_mean_bps=float(up.f.mean() * 1e4), spy_down_mean_bps=float(dn.f.mean() * 1e4),
                         book_up_usd=float(up.r.sum() * NAV), book_down_usd=float(dn.r.sum() * NAV),
                         hedge_up_usd=float(hh[g.f > 0].sum() * NAV), hedge_down_usd=float(hh[g.f < 0].sum() * NAV),
                         book_up_bps_per_day=float(up.r.mean() * 1e4), book_down_bps_per_day=float(dn.r.mean() * 1e4),
                         hedge_up_bps_per_day=float(hh[g.f > 0].mean() * 1e4), hedge_down_bps_per_day=float(hh[g.f < 0].mean() * 1e4),
                         sd_book_bps=float(g.r.std() * 1e4), sd_hedged_bps=float((g.r + hh).std() * 1e4),
                         corr_book_spy=float(g.r.corr(g.f))))
A = pd.DataFrame(rows); print(A.round(2).to_string(index=False)); OUT["anatomy"] = A.round(4).to_dict("records")
# what the up/down split implies: the hedge ratio a symmetric hedge would need on each side
print("\n  interpretation: beta_up > beta_down means the book participates more in SPY rallies than in selloffs while armed;")
print("  a symmetric beta_hat hedge then GIVES BACK more on up days than it SAVES on down days per unit SPY move.")

print("\n=== Aug-2026 episode day by day (PIT dial arms 2026-07-23; cur 07-27; live 07-30) ===")
m = runs["pit|2018+"]["_m"]; h = runs["pit|2018+"]["_h"]
seg = m.loc["2026-07-15":LAST].copy()
seg["hedge_usd"] = h.loc[seg.index] * NAV; seg["book_usd"] = seg.r * NAV; seg["hedged_usd"] = seg.book_usd + seg.hedge_usd
seg["spy_pct"] = seg.f * 100
for k in ["cur", "live"]:
    mk = runs[f"{k}|2018+"]["_m"]; hk = runs[f"{k}|2018+"]["_h"]
    seg[f"armed_{k}"] = mk.a.reindex(seg.index); seg[f"hedge_{k}_usd"] = hk.reindex(seg.index) * NAV
seg["dial_cur"] = dials_lag["cur"].reindex(seg.index); seg["dial_live"] = dials_lag["live"].reindex(seg.index)
seg = seg.rename(columns={"d": "dial_pit", "a": "armed_pit", "b": "beta_hat"})
cols = ["dial_pit", "dial_cur", "dial_live", "armed_pit", "armed_cur", "armed_live", "beta_hat", "spy_pct", "book_usd", "hedge_usd", "hedged_usd", "hedge_cur_usd", "hedge_live_usd"]
seg = seg[cols]
seg["cum_book"] = seg.book_usd.cumsum(); seg["cum_hedge_pit"] = seg.hedge_usd.cumsum(); seg["cum_hedge_cur"] = seg.hedge_cur_usd.cumsum(); seg["cum_hedge_live"] = seg.hedge_live_usd.cumsum()
print(seg.round(1).to_string())
seg.round(2).to_csv(HERE / "hedge_aug2026_daily.csv")
for k in ["pit", "cur", "live"]:
    a = seg[f"armed_{k}"] > 0
    hk = seg["hedge_usd"] if k == "pit" else seg[f"hedge_{k}_usd"]
    print(f"  {k:4s}: armed days {int(a.sum())}, book on armed days ${seg.book_usd[a].sum():,.0f}, hedge ${hk[a].sum():,.0f}, hedged ${(seg.book_usd[a] + hk[a]).sum():,.0f}, "
          f"SPY over armed days {((1 + seg.spy_pct[a] / 100).prod() - 1) * 100:+.2f}%, beta_hat mean {seg.beta_hat[a].mean():.2f}")
    OUT.setdefault("aug2026", {})[k] = dict(armed_days=int(a.sum()), book_usd=float(seg.book_usd[a].sum()), hedge_usd=float(hk[a].sum()),
                                            spy_pct=float(((1 + seg.spy_pct[a] / 100).prod() - 1) * 100), beta_hat=float(seg.beta_hat[a].mean()))
ga = seg[seg.armed_pit > 0]
print(f"  Aug-2026 armed (PIT) realised beta: all {updown_beta(pd.DataFrame({'r': ga.book_usd / NAV, 'f': ga.spy_pct / 100}))[0]:.2f}, "
      f"up-days {updown_beta(pd.DataFrame({'r': ga.book_usd / NAV, 'f': ga.spy_pct / 100}))[1]:.2f}, down-days {updown_beta(pd.DataFrame({'r': ga.book_usd / NAV, 'f': ga.spy_pct / 100}))[2]:.2f}")

print("\n=== episode bootstrap / drop-best ===")
OUT["bootstrap"] = {}
for k in ["pit|2018+", "cur|2018+", "live|2018+", "live|2016-07+", "pit|2018+ ex Aug-2026 ep"]:
    ep = np.array([x["hedge_usd"] for x in runs[k]["episodes"]])
    if len(ep) < 3:
        continue
    sims = np.array([RNG.choice(ep, len(ep), replace=True).sum() for _ in range(5000)])
    ep2 = np.delete(ep, ep.argmax()); t2 = float(ep2.mean() / (ep2.std(ddof=1) / np.sqrt(len(ep2))))
    best = runs[k]["episodes"][ep.argmax()]
    print(f"{k:26s}: n={len(ep)} total ${ep.sum():,.0f}  P(boot<=0) {(sims <= 0).mean():.3f}  drop-best ${ep.sum() - ep.max():,.0f} t={t2:.2f}  best ${ep.max():,.0f} ({best['start']})")
    OUT["bootstrap"][k] = dict(n=len(ep), total=float(ep.sum()), p_boot_le0=float((sims <= 0).mean()), drop_best_total=float(ep.sum() - ep.max()), drop_best_t=t2, best_usd=float(ep.max()), best_start=best["start"])

print("\n=== 63d-beta sensitivity on the same runs (the study's original primary) ===")
beta63 = rolling_beta(book, spy, 63)
rows = []
for k, (idx, dl) in {"pit|2018+": (pit_win, dials_lag["pit"]), "cur|2018+": (pit_win, dials_lag["cur"]), "live|2018+": (pit_win, dials_lag["live"]), "live|2016-07+": (live_win, dials_lag["live"])}.items():
    saved = beta.copy(); beta = beta63
    r = evaluate(idx, dl, k + "|b63"); beta = saved
    rows.append({kk: r[kk] for kk in HEAD})
print(pd.DataFrame(rows)[HEAD].round(2).to_string(index=False)); OUT["beta63"] = pd.DataFrame(rows)[HEAD].round(4).to_dict("records")

print("\n=== futures margin line ===")
last_beta = float(beta.reindex(keep).dropna().iloc[-1]); spx_last = float(spx.dropna().iloc[-1]); spy_last = float(pxk["SPY"].iloc[-1])
armed_betas = runs["pit|2018+"]["_m"]; ab = armed_betas[armed_betas.a > 0].b
es_notional = 50 * spx_last; mes_notional = 5 * spx_last
for lab, b in [("today (beta_hat 2026-09-01)", last_beta), ("armed-day mean beta", float(ab.mean())), ("armed-day 95th pct beta", float(ab.quantile(0.95))), ("beta clip ceiling 2.0", 2.0)]:
    hedge_notional = b * NAV
    n_es = hedge_notional / es_notional; n_mes = hedge_notional / mes_notional
    line = dict(case=lab, beta=b, hedge_notional=hedge_notional, es_contracts=n_es, mes_contracts=n_mes,
                es_margin=n_es * ES_IM if not np.isnan(ES_IM) else np.nan, mes_margin=n_mes * MES_IM if not np.isnan(MES_IM) else np.nan)
    line["es_margin_pct_nav"] = line["es_margin"] / NAV * 100; line["mes_margin_pct_nav"] = line["mes_margin"] / NAV * 100
    OUT.setdefault("margin", []).append(line)
    print(f"  {lab:28s} beta {b:.2f} -> hedge notional ${hedge_notional:,.0f} = {n_es:.2f} ES / {n_mes:.1f} MES (SPX {spx_last:,.0f}; ES ${es_notional:,.0f}, MES ${mes_notional:,.0f}); "
          f"IM: ES ${line['es_margin']:,.0f} ({line['es_margin_pct_nav']:.1f}% NAV), MES ${line['mes_margin']:,.0f} ({line['mes_margin_pct_nav']:.1f}% NAV)")
print(f"  (ES_IM={ES_IM}, MES_IM={MES_IM} per contract, overnight initial; pass --es-im/--mes-im)")
OUT["margin_inputs"] = dict(ES_IM=ES_IM, MES_IM=MES_IM, spx_last=spx_last, spy_last=spy_last, nav=NAV)

json.dump({k: v for k, v in OUT.items()}, open(HERE / "hedge_results.json", "w"), indent=1,
          default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else (bool(o) if isinstance(o, np.bool_) else str(o)))
print("\nwrote hedge_results.json, hedge_aug2026_daily.csv")
