from pathlib import Path

HERE = Path(__file__).resolve().parent
m = HERE / "00_surface_map.md"
text = m.read_text(encoding="utf-8")
results = (HERE / "_results.md").read_text(encoding="utf-8")
verdicts = (HERE / "01_watchlist_verdicts.md").read_text(encoding="utf-8")
if "## Results" not in text:
    text = text.rstrip("\n") + "\n\n" + results.rstrip("\n") + "\n\n" + verdicts.rstrip("\n") + "\n"
    m.write_text(text, encoding="utf-8")
print(len(text))
