"""Per live strategy: the date its config dict LAST changed materially in
strategy_config.py (the honest out-of-sample start), plus the number of
distinct dict versions it went through (variants tried -> trials count)."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(r"C:/Users/McKinley Slade/dev/New_Seasonals")
OUT = ROOT / "scratch/ultracode_sizing_2026-09-02"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace").stdout


commits = git("log", "--format=%H %ad", "--date=short", "--reverse", "--", "strategy_config.py").strip().splitlines()
name_re = re.compile(r"""['"]name['"]\s*:\s*['"]([^'"]+)['"]""")


def blocks(src: str) -> dict[str, str]:
    """Slice the file at each 'name': occurrence; the block for a strategy is the
    text from its name key to the next name key (universe lists included, which
    is right: a universe change IS a rule change)."""
    hits = list(name_re.finditer(src))
    out = {}
    for i, m in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(src)
        txt = src[m.start():end]
        txt = re.sub(r"#.*", "", txt)            # strip comments
        txt = re.sub(r"\s+", "", txt)            # whitespace-insensitive
        out[m.group(1)] = hashlib.md5(txt.encode()).hexdigest()
    return out


hist: dict[str, list[tuple[str, str]]] = {}
prev: dict[str, str] = {}
for line in commits:
    sha, date = line.split()
    src = git("show", f"{sha}:strategy_config.py")
    b = blocks(src)
    for n, h in b.items():
        if prev.get(n) != h:
            hist.setdefault(n, []).append((date, sha[:8]))
        prev[n] = h
live = set(prev)
res = {}
for n in sorted(live):
    ch = hist[n]
    res[n] = {"first_seen": ch[0][0], "last_change": ch[-1][0], "n_versions": len(ch),
              "changes_2026_05_onward": [d for d, _ in ch if d >= "2026-05-01"],
              "all_change_dates": [d for d, _ in ch]}
    print(f"{n:28s} first {ch[0][0]}  last {ch[-1][0]}  versions {len(ch):3d}  since-May: {res[n]['changes_2026_05_onward']}")
(OUT / "estimation_haircut_freeze_dates.json").write_text(json.dumps(res, indent=1))
