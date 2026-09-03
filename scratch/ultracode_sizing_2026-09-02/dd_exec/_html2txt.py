import re, html, sys
from pathlib import Path

for src in sys.argv[1:]:
    t = Path(src).read_text(encoding="utf-8")
    t = re.sub(r"<style.*?</style>", "", t, flags=re.S)
    t = re.sub(r"<script.*?</script>", "", t, flags=re.S)
    t = re.sub(r"<(br|/p|/h\d|/li|/tr|/div)>", "\n", t)
    t = re.sub(r"</t[dh]>", " | ", t)
    t = re.sub(r"<[^>]+>", "", t)
    t = html.unescape(t)
    t = re.sub(r"\n\s*\n+", "\n", t)
    out = Path(__file__).parent / (Path(src).stem + ".txt")
    out.write_text(t, encoding="utf-8")
    print(out, len(t))
