"""Print the m = 1.25 / 1.5 / 2 rows from the part-1 bootstrap and part-2 ruin
tables so the recommendation cites exact numbers."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
g = json.load(open(HERE / "unconstrained_growth_01_growth.json"))
mg = json.load(open(HERE / "unconstrained_growth_02_margin.json"))
for key in ["2016+|h0", "2016+|h0.5", "2003+|h0", "2003+|h0.5"]:
    for tag in ["1y", "3y"]:
        for m in ["1", "1.25", "1.5", "2", "2.5", "3"]:
            q = g["bootstrap"][key][tag][m]
            print(f"{key:11s} {tag} m={m:4s}: medDD {q['maxdd_median']:.1%} p95DD {q['maxdd_p95']:.1%} P>10 {q['p_dd_gt_10']:.1%} P>20 {q['p_dd_gt_20']:.1%} P>30 {q['p_dd_gt_30']:.1%} "
                  f"flatDD med {q['flat_dd_median']:.1%} p95 {q['flat_dd_p95']:.1%} P(flat>20) {q['p_flat_dd_gt_20']:.1%} | g mean {q['growth_mean']:.1%} p05 {q['growth_p05']:.1%} P(end<start) {q['p_end_below_start']:.1%} "
                  f"| rec med {q['recover_days_median']} mean {q['recover_days_mean']:.0f} p95 {q['recover_days_p95']:.0f} unrec {q['p_unrecovered_at_horizon']:.0%} longest-uw med {q['longest_underwater_median']:.0f} p95 {q['longest_underwater_p95']:.0f}")
print()
for w in ["2016+", "2003+"]:
    for tag in ["1y", "3y"]:
        for sc in ["pm_15", "pm_15_conc30", "pm_25_conc30"]:
            row = mg["ruin_bootstrap"][w][tag][sc]
            print(f"{w} {tag} {sc:13s}: " + "  ".join(f"m{m}: flat {row[m]['p_ruin_flat']:.1%} cush {row[m]['p_ruin_flat_15pct_cushion']:.1%} comp {row[m]['p_ruin_comp']:.1%}" for m in ["1", "1.25", "1.5", "2", "2.5", "3"]))
print()
for w in ["2003+", "2016+"]:
    for h in ["0", "0.25", "0.5"]:
        a = g["analytic"][w]["haircut"][h]
        print(f"{w} h={h}: curve", {k: (None if v is None else round(v, 3)) for k, v in a["curve"].items()})
print(g["analytic"]["2016+"]["base"], g["analytic"]["2003+"]["base"])
