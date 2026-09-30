from pathlib import Path

DAY = Path(__file__).resolve().parent
smap = DAY / "00_surface_map.md"
text = smap.read_text(encoding="utf-8")
placeholder = (
    "## 4. Watchlist verdicts (64 active, 0 expired)\n\n"
    "(Section written by the watchlist agent from today's computed values; W28 is\n"
    "adjudicated inside candidate S1.)\n"
)
verdicts = (DAY / "00b_watchlist_verdicts.md").read_text(encoding="utf-8").rstrip() + "\n"
assert placeholder in text, "placeholder not found"
smap.write_text(text.replace(placeholder, verdicts), encoding="utf-8")
print("merged", len(verdicts.splitlines()), "lines")
