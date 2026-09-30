"""Final text pass on tonight's queue: trim, set long flags, print counts."""
from __future__ import annotations

import json
from pathlib import Path

P = Path(__file__).resolve().parents[3] / "content" / "queue" / "2026-09-25.json"
d = json.loads(P.read_text(encoding="utf-8"))

new = {
    "x20260925-1": (
        "long XLU monday MOO, out 10/12 close, no stop. utilities -8.5% in a month "
        "with the S&P up. 9 priors since 2000, bought the next open: 7-2 two weeks "
        "on, +3.0% vs -0.1% baseline, 4-0 since 2018. a day late, from the second "
        "open it's 8-1. two of the nine are 62% of the gain. grade C."
    ),
    "x20260925-4": (
        "of the 218 large caps i track, 19% have a one-month return in the top half "
        "of their own year. the S&P is 0.6% off its high. share under 20% with the "
        "index within 1% of a high has happened twice since 2001: june 2013 and "
        "christmas eve 2024. two priors is trivia. and it's today's list, so "
        "survivors only."
    ),
    "x20260925-5": (
        "the small-cap short from 9/18 ended at today's close, +0.75% to the short, "
        "about +0.6R. the grader books it monday and grades meaner than i do. it was "
        "drafted and left in the drawer, same as the dollar long that paid this week. "
        "two drawer winners in a row. the machine has started looking at me funny."
    ),
}
for x in d["drafts"]:
    if x["id"] in new:
        x["text"] = new[x["id"]]
    x["long"] = len(x["text"]) > 280
    print(x["id"], len(x["text"]), "long" if x["long"] else "")
P.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
