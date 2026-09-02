"""Mine strategy_config.py git history: first appearance of each live strategy name,
and the union of every strategy name that ever lived in the file (trials count for
the deflated Sharpe ratio)."""
from __future__ import annotations

import json
import re
import subprocess
from collections import OrderedDict
from pathlib import Path

ROOT = Path(r"C:/Users/McKinley Slade/dev/New_Seasonals")
OUT = ROOT / "scratch/ultracode_sizing_2026-09-02"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace").stdout


commits = git("log", "--format=%H %ad", "--date=short", "--reverse", "--", "strategy_config.py").strip().splitlines()
name_re = re.compile(r"""['"]name['"]\s*:\s*['"]([^'"]+)['"]""")
first_seen: "OrderedDict[str, str]" = OrderedDict()
last_seen: dict[str, str] = {}
per_commit: list[dict] = []
for line in commits:
    sha, date = line.split()
    src = git("show", f"{sha}:strategy_config.py")
    names = sorted(set(name_re.findall(src)))
    per_commit.append({"sha": sha[:8], "date": date, "n_strats": len(names), "names": names})
    for n in names:
        first_seen.setdefault(n, date)
        last_seen[n] = date

live = per_commit[-1]["names"]
retired = sorted(n for n in first_seen if n not in live)
summary = {
    "n_commits": len(commits),
    "first_commit": per_commit[0]["date"],
    "last_commit": per_commit[-1]["date"],
    "live_names": live,
    "retired_names": retired,
    "n_ever": len(first_seen),
    "first_seen": dict(first_seen),
    "last_seen": last_seen,
    "strats_per_commit": [(c["date"], c["n_strats"]) for c in per_commit],
}
print(json.dumps({k: v for k, v in summary.items() if k != "strats_per_commit"}, indent=1))
print("strats per commit (date, n):", summary["strats_per_commit"][::5])
(OUT / "estimation_haircut_git_history.json").write_text(json.dumps(summary, indent=1))
