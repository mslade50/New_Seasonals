import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
s = json.loads((ROOT / "data/pitch_state.json").read_text(encoding="utf-8"))
out = {k: s[k] for k in ["asof", "session", "calendar", "risk", "book", "earnings", "seasonality", "scoreboard", "pipeline", "warnings"]}
out["tape_extremes"] = s["tape"]["extremes"]
out["tape_breadth"] = s["tape"]["breadth"]
out["research_docs"] = s["research"]["docs"]
out["history"] = {k: s["history"][k] for k in ["recent_fingerprints", "recent_ideas", "lifetime_pitched", "blocked_since"]}
print(json.dumps(out, indent=1, default=str))
