"""Print selected tier tables from signal_quality_results.json (inspection only)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.load(open(HERE / "signal_quality_results.json"))
want = {
    "Overbot Vol Spike": ["filt_extremity", "rank_2d", "rank_5d", "rank_10d", "rank_21d", "rank_252d", "gap_atr", "book_sig_5td", "n_sig_strat_day", "n_sig_book_day", "open_legs_strat", "atr_pct", "dist200", "vix", "dial", "days_since_last_sig_c"],
    "Oversold Low Volume": ["dial", "dial_pit", "dial_raw21", "spy_hi252_dist", "spy_ret21", "spy_ret10", "breadth_chg21", "breadth200", "vix", "filt_extremity", "rank_21d", "td_to_next_earn_c", "td_since_last_earn_c", "n_sig_strat_day", "open_legs_strat", "book_sig_5td", "atr_pct", "dist200"],
    "LT Trend ST OS": ["pc_pct_lag1", "book_sig_5td", "n_sig_book_day", "spy_hi252_dist", "spy_ret21", "dial", "filt_extremity", "rank_21d", "n_sig_strat_day", "vix"],
    "Weak Close Decent Sznls": ["move1_atr", "sector_breadth200", "gap_atr", "book_sig_5td", "n_sig_book_day", "range_pct", "rank_2d", "dial", "vix", "n_sig_strat_day"],
    "Indices Oversold Bounce": ["vrp", "vix", "gap_atr", "rank_2d", "book_sig_5td", "dial", "n_sig_strat_day"],
    "SPY QQQ MonFri Reversion": ["gap_atr", "rank_2d", "vix", "book_sig_5td", "dial", "dist50"],
    "52wh Breakout": ["spy_rv21", "vix", "open_legs_strat", "rank_252d", "book_sig_5td", "dial", "atr_pct", "vol_ratio"],
    "Monday Dip": ["gap_atr", "rank_2d", "vix", "book_sig_5td"],
    "3x ETF Overbot Fade": ["rank_2d", "vix_pct252", "filt_extremity", "book_sig_5td"],
    "Sector BO": ["breadth_chg21", "vix", "rank_252d"],
    "ATR Extended Gap Up": ["ret_10d", "atr_pct_rank", "dist50"],
    "St OS Sznl": ["log_dollar_vol", "td_to_next_earn_c", "filt_extremity"],
}
sel = sys.argv[1:] or list(want)
for s in sel:
    P = R["per_strategy"].get(s)
    if not P:
        continue
    print(f"\n##### {s}: N {P['n']} avgR {P['avgR']:+.3f} win {P['win']:.2f} episodes {P['n_episodes']}")
    for f in want[s]:
        r = P["features"].get(f)
        if not r:
            print(f"  {f}: (not testable)")
            continue
        ts = " | ".join(f"T{t['tier']} N{t['n']} R{t['avgR']:+.2f} w{t['win']:.2f} sd{t['sdR']:.2f} min{t['worstR']:+.1f}" for t in r["tiers"])
        print(f"  {f:22s} cuts {[round(c, 2) for c in r['cuts']]} rho {r['spearman']:+.2f} t {r['cluster_t']:+.2f} LOYO {r['loyo_agree'] if r['loyo_agree'] is not None else float('nan'):.2f}/{r['loyo_years']} mono {int(r['monotone'])} pass {int(r['passes'])}\n      {ts}")
print("\n##### detail (quintiles) for near-passers")
for k, v in R["detail"].items():
    print(k, " || ".join(f"Q{q['tier']} N{q['n']} R{q['avgR']:+.2f} w{q['win']:.2f}" for q in v["quintiles"]))
    print("   by-year rho:", {str(y['year']): (y['n'], y['rho']) for y in v["by_year"]})
