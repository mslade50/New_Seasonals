from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

from scripts import automation_supervisor as supervisor

ROOT = Path(__file__).resolve().parents[1]

TRACKED_TEXT_SUFFIXES = {
    ".bat",
    ".cjs",
    ".cmd",
    ".css",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".mjs",
    ".ps1",
    ".psm1",
    ".py",
    ".sh",
    ".svg",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
TRACKED_TEXT_NAMES = {"Dockerfile", "pytest.ini"}

# Components used to construct emoji sequences even when no pictograph is
# present as a literal character.
EMOJI_COMPONENTS = {
    0x200D,  # zero-width joiner
    0x20E3,  # combining enclosing keycap
    0xFE0F,  # variation selector-16
}

# Unicode emoji-data Extended_Pictographic coverage, represented
# conservatively by the assigned symbol blocks/ranges used by runtime text.
# Separate component ranges below catch flags, skin tones, and tag sequences.
PICTOGRAPHIC_RANGES = (
    (0x00A9, 0x00A9),
    (0x00AE, 0x00AE),
    (0x203C, 0x203C),
    (0x2049, 0x2049),
    (0x2122, 0x2122),
    (0x2139, 0x2139),
    (0x2194, 0x2199),
    (0x21A9, 0x21AA),
    (0x231A, 0x231B),
    (0x2328, 0x2328),
    (0x2388, 0x2388),
    (0x23CF, 0x23CF),
    (0x23E9, 0x23F3),
    (0x23F8, 0x23FA),
    (0x24C2, 0x24C2),
    (0x25AA, 0x25AB),
    (0x25B6, 0x25B6),
    (0x25C0, 0x25C0),
    (0x25FB, 0x25FE),
    (0x2600, 0x27BF),
    (0x2934, 0x2935),
    (0x2B05, 0x2B07),
    (0x2B1B, 0x2B1C),
    (0x2B50, 0x2B50),
    (0x2B55, 0x2B55),
    (0x3030, 0x3030),
    (0x303D, 0x303D),
    (0x3297, 0x3297),
    (0x3299, 0x3299),
    (0x1F000, 0x1FFFF),
)

EMOJI_SEQUENCE_RANGES = (
    (0x1F1E6, 0x1F1FF),  # regional-indicator flags
    (0x1F3FB, 0x1F3FF),  # skin-tone modifiers
    (0xE0020, 0xE007F),  # emoji tag sequences
)

CONSOLE_METHODS = {
    "print",
    "line",
    "debug",
    "info",
    "warning",
    "error",
    "exception",
    "critical",
    "log",
}

NON_ASCII = re.compile(r"[^\x00-\x7f]")

# Slack/Streamlit can render ASCII colon tokens as emoji. Underscored names
# cover the prior ``chart_with_upwards_trend`` failure mode and most CLDR
# aliases; the explicit set covers common one-word aliases in inline text.
COMMON_EMOJI_SHORTCODES = {
    "+1",
    "-1",
    "100",
    "bell",
    "bulb",
    "checkmark",
    "eyes",
    "fire",
    "heart",
    "information",
    "rocket",
    "smile",
    "star",
    "tada",
    "warning",
    "x",
    "zap",
}

UNATTENDED_TRANSITIVE_ENTRYPOINTS = {
    ROOT / "abs_return_dispersion.py",
    ROOT / "daily_pitch.py",
}


def _is_emoji_component(character: str) -> bool:
    codepoint = ord(character)
    if codepoint in EMOJI_COMPONENTS:
        return True
    return any(
        start <= codepoint <= end
        for start, end in (*PICTOGRAPHIC_RANGES, *EMOJI_SEQUENCE_RANGES)
    )


def _tracked_files() -> list[Path]:
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={ROOT.as_posix()}",
            "ls-files",
            "-z",
        ],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    return [ROOT / raw.decode("utf-8") for raw in result.stdout.split(b"\0") if raw]


def _tracked_utf8_text() -> dict[Path, str]:
    files: dict[Path, str] = {}
    for path in _tracked_files():
        if (
            path.suffix.lower() not in TRACKED_TEXT_SUFFIXES
            and path.name not in TRACKED_TEXT_NAMES
        ):
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue
        try:
            files[path] = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            # A tracked binary that happens not to contain NUL is not runtime
            # source. Executable text is required to be UTF-8 elsewhere.
            continue
    return files


def _format_hit(path: Path, line: int, codepoint: int, source: str) -> str:
    return f"{path.relative_to(ROOT)}:{line}:U+{codepoint:04X} ({source})"


def _raw_emoji_hits(files: dict[Path, str]) -> list[str]:
    hits: list[str] = []
    for path, text in files.items():
        for line_number, line in enumerate(text.splitlines(), start=1):
            for match in NON_ASCII.finditer(line):
                character = match.group()
                if _is_emoji_component(character):
                    hits.append(
                        _format_hit(path, line_number, ord(character), "literal")
                    )
    return hits


def _decoded_codepoints(value: str) -> list[int]:
    """Decode actual UTF-16 surrogate pairs left in parsed string values."""

    codepoints: list[int] = []
    index = 0
    while index < len(value):
        codepoint = ord(value[index])
        if 0xD800 <= codepoint <= 0xDBFF and index + 1 < len(value):
            low = ord(value[index + 1])
            if 0xDC00 <= low <= 0xDFFF:
                codepoints.append(
                    0x10000 + ((codepoint - 0xD800) << 10) + (low - 0xDC00)
                )
                index += 2
                continue
        codepoints.append(codepoint)
        index += 1
    return codepoints


def _python_decoded_literal_hits(files: dict[Path, str]) -> list[str]:
    hits: list[str] = []
    for path, text in files.items():
        if path.suffix.lower() != ".py":
            continue
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            for codepoint in _decoded_codepoints(node.value):
                if codepoint > 0x7F and _is_emoji_component(chr(codepoint)):
                    hits.append(
                        _format_hit(
                            path,
                            getattr(node, "lineno", 0),
                            codepoint,
                            "decoded Python literal",
                        )
                    )
    return hits


def _escaped_codepoints(text: str) -> list[tuple[int, int]]:
    """Return (line, codepoint) for JS/JSON/CSS-style Unicode escapes."""

    found: list[tuple[int, int]] = []
    occupied: set[int] = set()

    for pattern in (r"\\u\{([0-9A-Fa-f]{1,6})\}", r"\\U([0-9A-Fa-f]{8})"):
        for match in re.finditer(pattern, text):
            found.append(
                (text.count("\n", 0, match.start()) + 1, int(match.group(1), 16))
            )
            occupied.update(range(match.start(), match.end()))

    four_digit = list(re.finditer(r"\\u([0-9A-Fa-f]{4})", text))
    index = 0
    while index < len(four_digit):
        match = four_digit[index]
        if any(position in occupied for position in range(match.start(), match.end())):
            index += 1
            continue
        codepoint = int(match.group(1), 16)
        if 0xD800 <= codepoint <= 0xDBFF and index + 1 < len(four_digit):
            following = four_digit[index + 1]
            low = int(following.group(1), 16)
            if following.start() == match.end() and 0xDC00 <= low <= 0xDFFF:
                combined = 0x10000 + ((codepoint - 0xD800) << 10) + (low - 0xDC00)
                found.append((text.count("\n", 0, match.start()) + 1, combined))
                index += 2
                continue
        found.append((text.count("\n", 0, match.start()) + 1, codepoint))
        index += 1

    for match in re.finditer(r"&#(?:x([0-9A-Fa-f]+)|([0-9]+));", text):
        codepoint = int(match.group(1), 16) if match.group(1) else int(match.group(2))
        found.append((text.count("\n", 0, match.start()) + 1, codepoint))

    # CSS escapes are a backslash plus one to six hex digits and optional
    # terminating whitespace (for example ``content: \\1F4CA ``).
    for match in re.finditer(r"\\([0-9A-Fa-f]{1,6})(?:[ \t\r\n\f])?", text):
        found.append(
            (text.count("\n", 0, match.start()) + 1, int(match.group(1), 16))
        )
    return found


def _emoji_shortcode_hits(files: dict[Path, str]) -> list[str]:
    hits: list[str] = []
    # Double-colon GitHub workflow annotations such as ``::warning::`` are
    # control syntax, not rendered emoji shortcodes.
    token = re.compile(
        r"(?<![:A-Za-z0-9]):([A-Za-z0-9_+-]{1,64}):(?!:)"
    )
    exact_quoted = re.compile(r"(['\"]):([A-Za-z0-9_+-]{1,64}):\1")
    for path, text in files.items():
        exact_positions = {match.start(2) - 1 for match in exact_quoted.finditer(text)}
        for match in token.finditer(text):
            name = match.group(1).lower()
            if (
                match.start() not in exact_positions
                and "_" not in name
                and name not in COMMON_EMOJI_SHORTCODES
            ):
                continue
            line = text.count("\n", 0, match.start()) + 1
            hits.append(f"{path.relative_to(ROOT)}:{line}:shortcode :{name}:")
    return hits


def _web_decoded_escape_hits(files: dict[Path, str]) -> list[str]:
    hits: list[str] = []
    web_suffixes = {".css", ".html", ".js", ".json", ".jsx", ".mjs", ".ts", ".tsx"}
    for path, text in files.items():
        if path.suffix.lower() not in web_suffixes:
            continue
        for line, codepoint in _escaped_codepoints(text):
            try:
                is_emoji = _is_emoji_component(chr(codepoint))
            except ValueError:
                is_emoji = False
            if is_emoji:
                hits.append(_format_hit(path, line, codepoint, "decoded web escape"))
    return hits


def _catalog_python_entrypoints() -> set[Path]:
    paths = {
        ROOT / "scripts" / "automation_supervisor.py",
        ROOT / "scripts" / "repo_health_check.py",
        *UNATTENDED_TRANSITIVE_ENTRYPOINTS,
    }
    for pipeline in supervisor.CATALOG.values():
        for job in pipeline.jobs:
            for command in job.commands:
                argv = list(command.argv)
                for value in argv:
                    if value.endswith(".py"):
                        paths.add(ROOT / value)
                if "-m" in argv:
                    module_index = argv.index("-m") + 1
                    if module_index < len(argv):
                        module = ROOT / (argv[module_index].replace(".", "/") + ".py")
                        if module.is_file():
                            paths.add(module)
    return {path for path in paths if path.is_file()}


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _non_ascii_console_hits() -> list[str]:
    hits: list[str] = []
    for path in sorted(_catalog_python_entrypoints()):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if (
                not isinstance(node, ast.Call)
                or _call_name(node) not in CONSOLE_METHODS
            ):
                continue
            for argument in (*node.args, *[item.value for item in node.keywords]):
                for literal in ast.walk(argument):
                    if not isinstance(literal, ast.Constant) or not isinstance(
                        literal.value, str
                    ):
                        continue
                    for match in NON_ASCII.finditer(literal.value):
                        hits.append(
                            _format_hit(
                                path,
                                getattr(literal, "lineno", getattr(node, "lineno", 0)),
                                ord(match.group()),
                                f"{_call_name(node)} output",
                            )
                        )
    return hits


def test_tracked_runtime_text_has_no_literal_or_escaped_emoji_components():
    files = _tracked_utf8_text()
    hits = [
        *_raw_emoji_hits(files),
        *_python_decoded_literal_hits(files),
        *_web_decoded_escape_hits(files),
        *_emoji_shortcode_hits(files),
    ]

    assert not hits, "Emoji components remain in tracked runtime text:\n" + "\n".join(
        sorted(set(hits))
    )


def test_guard_decodes_python_and_web_unicode_escape_forms():
    python_path = ROOT / "synthetic_python_escape.py"
    python_source = r'PAGE_ICON = "\U0001F4CA"'
    assert "U+1F4CA" in "\n".join(
        _python_decoded_literal_hits({python_path: python_source})
    )

    web_path = ROOT / "synthetic_web_escape.js"
    web_source = r'const icon = "\uD83D\uDCCA";'
    assert "U+1F4CA" in "\n".join(_web_decoded_escape_hits({web_path: web_source}))

    surrogate_source = 'PAGE_ICON = "' + "\\uD83D\\uDCCA" + '"'
    assert "U+1F4CA" in "\n".join(
        _python_decoded_literal_hits({python_path: surrogate_source})
    )

    css_path = ROOT / "synthetic_escape.css"
    css_source = "content: " + "\\" + "1F4CA "
    assert "U+1F4CA" in "\n".join(
        _web_decoded_escape_hits({css_path: css_source})
    )


def test_guard_detects_rendered_ascii_emoji_shortcodes():
    path = ROOT / "synthetic_shortcode.py"
    source = 'page_icon=":' + "chart_with_upwards_trend" + ':"'
    hits = _emoji_shortcode_hits({path: source})
    assert len(hits) == 1
    assert "chart_with_upwards_trend" in hits[0]


def test_unattended_production_console_and_log_literals_are_ascii_only():
    hits = _non_ascii_console_hits()
    boundary_files = (
        ROOT / "scripts" / "run_local_automation.ps1",
        ROOT / "scripts" / "run_daily_pitch.bat",
    )
    for path in boundary_files:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8-sig").splitlines(), start=1
        ):
            for match in NON_ASCII.finditer(line):
                hits.append(
                    _format_hit(path, line_number, ord(match.group()), "task boundary")
                )

    assert not hits, "Non-ASCII unattended output remains:\n" + "\n".join(
        sorted(set(hits))
    )
