"""Tighten the 08-30 drafts toward the 280 cap and re-flag `long`."""
import json
from pathlib import Path

p = Path("content/queue/2026-08-30.json")
q = json.loads(p.read_text(encoding="utf-8"))
new = {
    "x20260830-1": "my IEF idea from wednesday exits at monday's close, -0.41% so far. monday is also the month's final session, bonds' best calendar cell: IEF up on 180 of 289 finals since 2000, +0.11% vs +0.01% baseline. the two cuts that describe monday: august 14-10, +0.045%. within 3% of the 52-week low, 27-19, +0.036%.",
    "x20260830-2": "VIX closed 14.43 friday, 5th percentile of its 52-week range. the 84 august mondays entered from the bottom third of that range: VIX up 52-32, median +1.58%. middle third, 6-15 up. since 2018 the floor cell is 19-17, so, mostly an old habit.",
    "x20260830-3": "the gold miners' second flush. up 25%+ in 21 sessions, a -3% day, within 10 sessions of an identical one. friday was one. five priors, three runs (2009, 2016, 2020): 4-1 over five sessions, +0.9%, the loser -8.4%. the first-flush version is 6-1 at +3.9%. the second pays a quarter of the first and its loser outweighs its best winner. passed on the first one thursday, leaving this one alone too.",
    "x20260830-5": "how not to validate a sizing rule: run the backtest with it on and admire the improvement. it was fit on that exact history, of course it improves. ours have to survive leave-one-year-out and clustering by episode rather than by day. anything that only works because of one year or one cluster fails there. a few of ours have.",
    "x20260830-6": "for the bitcoin-is-stretched threads: +19.6% in ten sessions, z10 +2.27, down 3.63% friday. stretched alone is fine, 170 of 298 kept going next day, +0.77% vs +0.18% baseline. stretched and down on the day, friday's arm: 15 cases, 8-7, mean -0.24%. ether's same arm is 12-5 up.",
}
for d in q["drafts"]:
    if d["id"] in new:
        d["text"] = new[d["id"]]
    d["long"] = len(d["text"]) > 280
    print(d["id"], len(d["text"]), d["long"])
p.write_text(json.dumps(q, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
