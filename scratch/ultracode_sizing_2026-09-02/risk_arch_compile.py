"""Merge the four risk-architect result files into risk_arch_results.json with
a headline block (the numbers the plan cites). Run after risk_arch_01..04."""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent


def main() -> None:
    parts = {n: json.load(open(OUT / f"risk_arch_{n}.json", encoding="utf-8"))
             for n in ("theme_budget", "autocorr_dd", "stack_stress", "grm_frontier")}
    tb, ac, ss, gf = (parts[n] for n in ("theme_budget", "autocorr_dd", "stack_stress", "grm_frontier"))
    live, pit = gf["2016-07+"], gf["PIT 2018+"]

    def fr(blk, key):
        r = blk["frontier"][key]
        return dict(grm=r["grm"], growth=r["growth_ann_pct"], vol=r["ann_vol_pct"], dd1y_med=r["dd1y"]["median_dd"], dd1y_p95=r["dd1y"]["p95_dd"],
                    p1y_gt15=r["dd1y"]["p_dd_gt_15"], p1y_gt20=r["dd1y"]["p_dd_gt_20"], dd3y_med=r["dd3y"]["median_dd"], p3y_gt20=r["dd3y"]["p_dd_gt_20"],
                    p3y_gt30=r["dd3y"]["p_dd_gt_30"], hist_maxdd=r["hist_maxdd_pct"], hist_worst_day=r["hist_worst_day_pct"])

    headline = dict(
        themes=dict(
            eff_n_2010=tb["windows"]["2010+"]["eff_n_themes"], eff_n_2016=tb["windows"]["2016-07+"]["eff_n_themes"],
            avg_pair_corr_2010=tb["windows"]["2010+"]["avg_pair_corr"],
            var_share_2010={t: v["var_share"] for t, v in tb["windows"]["2010+"]["themes"].items()},
            cvar5_share_2010={t: v["cvar5_share"] for t, v in tb["windows"]["2010+"]["themes"].items()},
            pnl_share_2010={t: v["pnl_share"] for t, v in tb["windows"]["2010+"]["themes"].items()},
            kelly_norm_2010={t: v["kelly_weight_norm"] for t, v in tb["windows"]["2010+"]["themes"].items()},
            kelly_norm_2016={t: v["kelly_weight_norm"] for t, v in tb["windows"]["2016-07+"]["themes"].items()},
            tail_days_pair_corr=tb["tail_vs_rest_corr_2010+"]["avg_pair_corr_tail"], rest_pair_corr=tb["tail_vs_rest_corr_2010+"]["avg_pair_corr_rest"],
            tail_eff_n=tb["tail_vs_rest_corr_2010+"]["eff_n_tail"],
            dial65_live={t: dict(var_share=v["var_share"], spy_cov_share=v["spy_cov_share"], sharpe=v["sharpe"]) for t, v in tb["by_dial_bucket_live_lag1"]["65+"]["themes"].items()},
            dial65_pit=dict(days=tb["by_dial_bucket_pit_lag1"]["65+"]["days"], beta=tb["by_dial_bucket_pit_lag1"]["65+"]["book_beta"], r2=tb["by_dial_bucket_pit_lag1"]["65+"]["book_r2"],
                            theme_beta=tb["by_dial_bucket_pit_lag1"]["65+"]["theme_beta"], var_share=tb["by_dial_bucket_pit_lag1"]["65+"]["theme_var_share"])),
        pnl_path=dict(acf_2010=ac["2010+"]["acf_1_5"], lb_p20_2010=ac["2010+"]["p20"], vr21_2010=ac["2010+"]["vr21"], vr5_2010=ac["2010+"]["vr5"],
                      streak3_next_day_bps_2010=ac["2010+"]["streaks"]["after_3_down_days"]["next_day_bps"], base_bps_2010=ac["2010+"]["streaks"]["after_3_down_days"]["base_day_bps"],
                      dd_gt5_fwd21_2010=ac["2010+"]["conditional_fwd"]["dd_gt_5pct"], pod_rule_2010=ac["drawdown_rule_replays"]["2010+"]["pod_cut_50_at_5_stop_at_7.5"],
                      gz_0_9_2010=ac["drawdown_rule_replays"]["2010+"]["grossman_zhou_alpha_0.9"]),
        stress=dict(book=ss["stress_multiple_book"], theme={t: dict(p99=v["p99"], p999=v["p999"], max=v["max"]) for t, v in ss["stress_multiple_by_theme"].items()},
                    open_risk_2016=ss["open_risk_bps_2016+"], gross_2016=ss["open_notional_nav_2016+"]["book_gross"], olv_depth=ss["olv_stress_by_depth"],
                    worst5=ss["worst_20_days"][:5], margin={k: v for k, v in ss.items() if k.startswith("margin_req")},
                    class_share_top1pct=ss["class_share_of_gross_top1pct_days_2016+"]),
        hedge=dict(live={k: v for k, v in live.items() if k not in ("frontier", "theory")}, pit={k: v for k, v in pit.items() if k not in ("frontier", "theory")}),
        frontier=dict(
            live_h40={m: fr(live, f"h0.4_unhedged_m{m}") for m in (1.0, 1.25, 1.5, 1.75, 2.0, 3.0)},
            live_h40_hedged={m: fr(live, f"h0.4_hedged_m{m}") for m in (1.0, 1.25, 1.5, 1.75, 2.0, 3.0)},
            pit_h40={m: fr(pit, f"h0.4_unhedged_m{m}") for m in (1.0, 1.25, 1.5, 1.75, 2.0, 3.0)},
            pit_h40_hedged={m: fr(pit, f"h0.4_hedged_m{m}") for m in (1.0, 1.25, 1.5, 1.75, 2.0, 3.0)},
            y2003_h40={m: fr(gf["2003+_unhedged"], f"h0.4_m{m}") for m in (1.0, 1.25, 1.5, 1.75, 2.0, 3.0)},
            y2003_h71={m: fr(gf["2003+_unhedged"], f"h0.71_m{m}") for m in (1.0, 1.25, 1.5, 2.0)},
            theory_live_h40={m: live["theory"][f"h0.4_m{m}"] for m in (1.0, 1.5, 2.0, 3.0)}))
    json.dump(dict(headline=headline, parts=parts), open(OUT / "risk_arch_results.json", "w", encoding="utf-8"), indent=1, default=str)
    print(json.dumps(headline["frontier"]["live_h40"], indent=0)[:1500])
    print(json.dumps(headline["frontier"]["y2003_h40"], indent=0)[:1500])


if __name__ == "__main__":
    main()
