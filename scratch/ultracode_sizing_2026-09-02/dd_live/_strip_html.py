import html
import re
import sys
from pathlib import Path

t = Path(sys.argv[1]).read_text(encoding="utf-8")
t = re.sub(r"<style.*?</style>", "", t, flags=re.S)
t = re.sub(r"<script.*?</script>", "", t, flags=re.S)
t = re.sub(r"<[^>]+>", " ", t)
t = html.unescape(t)
t = re.sub(r"[ \t]+", " ", t)
t = re.sub(r"\n\s*\n+", "\n", t)
needle = sys.argv[2] if len(sys.argv) > 2 else None
if needle:
    i = t.find(needle)
    print(t[max(0, i - 3500): i + 3000])
else:
    print(t)
