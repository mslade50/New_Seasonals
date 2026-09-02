"""Within-strategy adds, step 6: merge the step-2..5 result files into
within_strategy_adds_results.json with a summary block (headline cells, the
edge-lens vs variance-lens comparison per strategy, the Kelly-tax arithmetic at the
book's current fraction of Kelly, and the per-strategy rule verdicts)."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parent
marg = json.load(open(OUT / "within_strategy_adds_marginal.json"))
rep = json.load(open(OUT / "within_strategy_adds_replay.json"))
chk = json.load(open(OUT / "within_strategy_adds_checks.json"))
pkg = json.load(open(OUT / "within_strategy_adds_package.json"))

def cell(s, col, bucket):
    return next(r for r in marg[s][col] if r["bucket"] == bucket)

lens = {}
for s in ["Oversold Low Volume", "Overbot Vol Spike", "Weak Close Decent Sznls", "LT Trend ST OS", "52wh Breakout", "3x ETF Overbot Fade", "3x Bear ETF Overbot Fade"]:
    rows = marg[s]["b_open"]
    solo = rows[0]
    lens[s] = {r["bucket"]: dict(N=r["N"], avgR=round(r["avgR"], 3), pnl_per_atr_risk=round(r["pnl_per_risk"], 3), pnl_per_euler_var=round(r["pnl_per_euler"], 1) if r["pnl_per_euler"] else None,
                                 edge_ratio_vs_solo=round(r["avgR"] / solo["avgR"], 2) if solo["avgR"] else None,
                                 variance_efficiency_vs_solo=round(r["pnl_per_euler"] / solo["pnl_per_euler"], 2) if (r["pnl_per_euler"] and solo["pnl_per_euler"]) else None) for r in rows}

# Kelly tax arithmetic: book at ~6% of full Kelly (plan section 1). For a marginal leg with correlation rho to the
# sleeve, the growth-rate cost of its variance term relative to its mean term is ~ (f/f*) * (1 + 2*rho*(n-1)) / 1 in the
# equicorrelated approximation -- second order at f/f* = 0.06 even at rho = 1 and n = 8.
kelly_tax = {f"rho={rho},n={n}": round(0.06 * (1 + 2 * rho * (n - 1)), 2) for rho in (0.0, 0.5, 1.0) for n in (2, 4, 8)}

summary = dict(
    headline=[
        "In 5 of the 7 strategies with enough data (OLV, OVS, WCDS, LT Trend ST OS, 3x ETF Overbot Fade) the marginal add is a HIGHER-edge leg than the solo leg; the only negative marginal cell is 52wh Breakout at >= 6 open legs, and only in the 2010+ era.",
        "Correlation-lens rules (Kelly concurrency factor, variance parity, rho > 0.6 cuts, same-sector cuts, same-ticker cuts) lose PnL per unit ATR risk in every strategy where rho is high (OLV, WCDS) and are neutral where rho ~ 0; edge-lens rules win at equal risk everywhere except 52wh.",
        "OLV's recency ladder is keyed on the wrong variable: it halves 81% of solo legs (0.48R) but also 43-49% of legs entered into 3+ open names (0.93-1.15R). Re-keying to max(ticker-recency rung, sleeve-depth rung) lifts PnL per $ risk 0.790 -> 0.820, +$125k/21y raw, +$16k at equal risk, with exit-basis worst-21d and maxDD both slightly better (15 of 20 years >= current).",
        "OVS's per-strategy 250 bps daily cap is the binding within-strategy control and it lands on OVS's best cell: P1 fills entered with >= 6 OVS legs open average 0.77-0.82R (t 2.6-4.6, LOYO floor 1.9-4.4, episode-bootstrap P < 0.001, 8/10 years positive) but book 4-23 bps per fill vs 54 bps solo. A 1.5x on those fills (~ cap 375 bps on cluster days) adds $92k/23y, PnL/risk 0.298 -> 0.327, worst day -26k -> -35k (2021-01-27), maxDD -52k -> -55k, worst-21d unchanged.",
        "WCDS and LT Trend ST OS: same-day cluster adds are the edge (WCDS same-day adds 0.73R vs solo 0.21R, t 2.7, LOYO 2.1, P 0.02; LT 3+ open 0.63R vs 0.22R, t 2.65, LOYO 2.0, P 0.009). Solo 0.75x / adds 1.25x raises PnL/risk 13% (WCDS) and 9% (LT) at LOWER sleeve maxDD; LT's cluster adds hold their edge at dial 50+ while its solo legs go negative at 65+.",
        "52wh: the plan's 0.5x at >= 5 open is the wrong threshold (the exact-5 cell is 1.71R on 13 legs, 2010+); >= 6 is where the 2010+ cell is 0.08R, but pre-2010 the same cell was 0.92R on 17 legs (episode bootstrap P = 0.57). The >= 6 rule is tail insurance, not edge: it owns 88% of 52wh's worst 21-day MTM window (Feb 2014) and cuts sleeve worst-21d -59.5k -> -33.3k for -$35k/22y raw, +$7.5k at equal risk.",
        "Combined package (OLV depth-OR-ticker ladder, OVS P1 x1.5 at depth >= 6, WCDS and LT solo 0.75 / adds 1.25; 52wh and 3x fades unchanged): 8-strategy sleeve +$249k (+9.9%) over 2005-2026 on +2.5% risk, PnL/risk 0.414 -> 0.444, Sharpe 1.73 -> 1.77, worst-21d and maxDD slightly better, worst day -36k -> -44k, 17 of 22 years better, +$200k ex the best year.",
    ],
    kelly_tax_share_of_marginal_mu=kelly_tax,
    lens_by_strategy=lens,
    verdicts={
        "Oversold Low Volume": "re-key ladder to max(recency rung, depth rung [0.5 solo, 0.7 at 1-2 open, 1.0 at 3+]); keep same-ticker adds at full size (best cell, 0.98R, 3+ deep 1.92R); keep the 50%-NAV ticker cap (8 clips, $2.6k cost); reject any sleeve cap, sector cut, rho cut or late-add cut",
        "Overbot Vol Spike": "raise the OVS per-strategy daily cap 250 -> 375 bps (P2 aggregate cap unchanged); do not cut same-sector or same-ticker adds (both are better legs)",
        "Weak Close Decent Sznls": "solo 0.75x / adds 1.25x (same-day cluster adds are the edge); the 250 cap binds on 9 of 194 days at 1.25x -- keep the cap, accept the trim",
        "LT Trend ST OS": "solo 0.75x / adds 1.25x; same-sector clusters are better legs (2+ same-sector 0.62R, t 3.5), never cut them",
        "52wh Breakout": "do NOT ship 0.5x at >= 5; >= 6 at 0.5x is acceptable as declared tail insurance only (cost -$35k/22y raw, 88% of the Feb-2014 window), never as an edge claim",
        "3x ETF Overbot Fade": "no change; adds directionally better (6-12 open 1.76R, N 19, 3 episodes) but P = 0.23",
        "3x Bear ETF Overbot Fade": "keep the same-day derate (neutral at equal risk: 0.665 vs 0.677 PnL/risk); N 55 decides nothing",
        "3x Leader Gap Fade": "no change (N 24)",
    },
)
RES = dict(summary=summary, marginal=marg, replay=rep, checks=chk, package=pkg)
json.dump(RES, open(OUT / "within_strategy_adds_results.json", "w"), indent=1, default=float)
print("wrote within_strategy_adds_results.json")
print(json.dumps(summary["lens_by_strategy"]["Oversold Low Volume"], indent=0))
print(kelly_tax)
