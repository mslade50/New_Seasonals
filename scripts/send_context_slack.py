"""Post the evening Market Context brief to Slack as Block Kit.

Self-contained by design. The denali report card's sender is the model for the
mechanics (one colored attachment, mrkdwn conversion, webhook-first delivery,
429/5xx-only retry, --dry-run) but nothing is imported across repos: the two
products share a channel, not a codebase.

    python scripts/send_context_slack.py                    # today's brief
    python scripts/send_context_slack.py --file data/context_briefs/2026-08-10.md
    python scripts/send_context_slack.py --dry-run          # write .slack.json
    python scripts/send_context_slack.py --webhook-url https://hooks.slack.com/...

DATING. The brief is named for the RUN date (the evening it was written), not
for the session it previews. A Sunday run reads Friday's tape and previews
Monday, and all three of those dates are different; the run date is the only
one that is unique per brief, so it names the file, the cell-map folder and
the delivery check.

THREE GATES, all of which read the disk rather than the payload:

  1. the brief is dated for the run date (no re-posting yesterday's)
  2. scratch/context_checks/<run date>/00_cell_map.md exists (stage B ran, so
     the nuggets were chosen from the sweep instead of from recall)
  3. the tag budget holds: at most two [anecdote] nuggets, never as headline

On a successful post it advances data/context_flag_state.json (the novelty
baseline the next sweep diffs against) and appends one record per nugget to
data/context_journal.jsonl. A dry run advances nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRIEF_DIR = ROOT / "data" / "context_briefs"
CELL_MAP_DIR = ROOT / "scratch" / "context_checks"
FLAG_STATE_PATH = ROOT / "data" / "context_flag_state.json"
JOURNAL_PATH = ROOT / "data" / "context_journal.jsonl"
ENV_PATH = ROOT / ".env"

SLACK_POST_URL = "https://slack.com/api/chat.postMessage"

# One standing color, deliberately not on the report card's severity scale
# (red / amber / grey / green). This product has no severity: it is the same
# kind of message every evening, and a purple bar in the same channel reads as
# "the other thing" at a glance.
BRAND_COLOR = "#6e40c9"
STALE_COLOR = "#d29922"

TAGS = ("solid", "suggestive", "anecdote")
MAX_ANECDOTES = 2
WORD_BUDGET = (250, 400)

# '1. **SPY — down 80bp after a 52w high** [solid]'
ITEM_HEAD = re.compile(r"^\s*\d+\.\s+\*\*(.+?)\*\*\s*(?:\[(\w+)\])?\s*$")
INLINE_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
INLINE_BOLD = re.compile(r"\*\*(.+?)\*\*")
INLINE_ITALIC = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")

LANE_SECTIONS = ("Tomorrow's tape", "Today in context", "Friday in context")

# Advisory verbs, banned as imperatives (spec section 2), matched only where a
# sentence or bullet starts. Two tiers, because half the list has an ordinary
# noun sense that a macro brief uses constantly:
#
#   HARD  no reading of "Fade the move" or "Trim into the print" is context
#   SOFT  "Cut cycles have been kind to duration" and "Sell-side consensus"
#         are legitimate sentences, so these are flagged for the author rather
#         than used to block a post
_SENTENCE_START = r"(?:^|(?<=[.!?]))[\s\-*•]*"
BANNED_IMPERATIVE = re.compile(
    _SENTENCE_START + r"(Trim|Reduce|Consider|Recommend|Hedge|Fade|Avoid)\b")
SOFT_IMPERATIVE = re.compile(
    _SENTENCE_START + r"(Cut|Watch|Exit|Buy|Sell|Add|Position|Short|Long)\b")
SOFT_ADVISORY = re.compile(r"\b(should|ought to|worth (?:a )?(?:look|adding))\b",
                           re.IGNORECASE)

# "Every nugget carries N" (spec section 7). Counted several legal ways: an
# explicit n=, an "x of y" record, or a counted noun. Spelling the number out
# in words does not count, on purpose.
QUOTES_N = re.compile(
    r"\bn\s*=\s*\d+"
    r"|\b\d+\s+of\s+\d+\b"
    r"|\b\d[\d,]*\s+(?:[a-z-]+\s+){0,2}(?:sessions?|instances?|obs|"
    r"observations?|episodes?|occurrences?|streaks?|prints?|reports?|"
    r"thrusts?|years?|days?|cases?|mondays?|tuesdays?|wednesdays?|"
    r"thursdays?|fridays?)\b",
    re.IGNORECASE)


# ---------- markdown -> Slack mrkdwn ----------

def to_mrkdwn(text: str) -> str:
    """CommonMark inline syntax -> Slack mrkdwn. Bold is parked on a sentinel
    before the italic pass so a single '*' left by former '**' is not re-read
    as italic."""
    s = INLINE_LINK.sub(lambda m: f"<{m.group(2)}|{m.group(1)}>", text)
    s = INLINE_BOLD.sub(lambda m: f"\x00{m.group(1)}\x00", s)
    s = INLINE_ITALIC.sub(lambda m: f"_{m.group(1)}_", s)
    return s.replace("\x00", "*")


# ---------- parse ----------

def parse_brief(text: str) -> dict:
    """-> {title, headline, sections:{name:[lines]}, footer, items:{section:[...]}}"""
    title = ""
    headline = ""
    footer_lines: list[str] = []
    sections: dict[str, list[str]] = {}
    current: str | None = None
    in_footer = False

    for raw in text.splitlines():
        line = raw.rstrip()
        if line.startswith("# ") and not title:
            title = line[2:].strip()
            continue
        if line.strip() == "---":
            in_footer = True
            current = None
            continue
        if in_footer:
            if line.strip():
                footer_lines.append(line.strip())
            continue
        if line.startswith("## "):
            current = line[3:].strip()
            sections.setdefault(current, [])
            continue
        if line.lower().startswith("**headline:**"):
            headline = line.split("**", 2)[-1].lstrip(":").strip()
            continue
        if current is not None:
            sections[current].append(line)

    items = {name: parse_items(lines) for name, lines in sections.items()}
    return {"title": title, "headline": headline, "sections": sections,
            "items": items, "footer": " ".join(footer_lines).strip("* ")}


def parse_items(lines: list[str]) -> list[dict]:
    """-> [{title, tag, body}] from '1. **Title** [tag]' + indented body."""
    out: list[dict] = []
    head: dict | None = None
    body: list[str] = []

    def flush() -> None:
        if head is not None:
            out.append({**head, "body": " ".join(
                b.strip() for b in body if b.strip())})

    for line in lines:
        m = ITEM_HEAD.match(line)
        if m:
            flush()
            head = {"title": m.group(1).strip(), "tag": (m.group(2) or "").lower()}
            body = []
        elif head is not None:
            body.append(line)
    flush()
    return out


def load_sidecar(path: Path) -> dict:
    sidecar = path.with_suffix(".json")
    if not sidecar.exists():
        return {}
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


# ---------- gates ----------

def lint_brief(brief: dict, sidecar: dict, run_date: str) -> tuple[list, list]:
    hard: list[str] = []
    soft: list[str] = []

    lane_items = [it for name in LANE_SECTIONS for it in brief["items"].get(name, [])]
    quiet = bool(sidecar.get("quiet"))

    if not brief.get("title"):
        hard.append("no '# Market Context - ...' title line")
    if not brief.get("headline") and not quiet:
        hard.append("no '**Headline:**' line")

    if not quiet:
        if not 4 <= len(lane_items) <= 8:
            hard.append(f"{len(lane_items)} nuggets; the brief ships 4 to 8 "
                        f"(or a QUIET TAPE note with quiet:true in the .json)")
        for item in lane_items:
            if item["tag"] not in TAGS:
                hard.append(f"nugget '{item['title'][:50]}' has tag "
                            f"{item['tag'] or '(none)'}; must be one of {TAGS}")
        anecdotes = [i for i in lane_items if i["tag"] == "anecdote"]
        if len(anecdotes) > MAX_ANECDOTES:
            hard.append(f"{len(anecdotes)} [anecdote] nuggets; the budget is "
                        f"{MAX_ANECDOTES}")
        if lane_items and lane_items[0]["tag"] == "anecdote":
            hard.append("an [anecdote] may never be the headline nugget")
        for item in lane_items:
            if not QUOTES_N.search(item["body"]):
                soft.append(f"nugget '{item['title'][:40]}' quotes no N")

    body_text = " ".join(
        " ".join(lines) for name, lines in brief["sections"].items()
        if name != "Calendar")
    # PROSE, not markup. The numbered head carries the subject, the cell name
    # and the tag on every nugget, which is roughly 70 words of scaffolding in
    # a six-nugget brief; counting it would push a correctly-sized brief over
    # budget and get sentences cut that Scott actually reads.
    prose_parts = [brief.get("headline", "")]
    prose_parts += [i["body"] for i in lane_items]
    prose_parts += [" ".join(lines) for name, lines in brief["sections"].items()
                    if name not in LANE_SECTIONS and name != "Calendar"]
    words = len(" ".join(prose_parts).split())
    if not quiet and not WORD_BUDGET[0] <= words <= WORD_BUDGET[1]:
        soft.append(f"body is {words} words of prose; the budget is "
                    f"{WORD_BUDGET[0]}-{WORD_BUDGET[1]}")

    whole = brief.get("headline", "") + "\n" + body_text
    for m in BANNED_IMPERATIVE.finditer(whole):
        hard.append(f"advisory verb '{m.group(1)}' used as an imperative; this "
                    f"product carries context, never instructions")
    for m in SOFT_IMPERATIVE.finditer(whole):
        soft.append(f"'{m.group(1)}' opens a sentence; fine as a noun ('cut "
                    f"cycles', 'sell-side'), an instruction if it is a verb")
    if SOFT_ADVISORY.search(whole):
        soft.append("reads as advice ('should' / 'worth a look'); state the "
                    "statistic and stop")
    # Prose only. The nugget HEAD is '**<subject> - <cell>** [tag]' and its
    # separator is a spec'd em dash, so checking the whole document would flag
    # the required format as a style violation on every brief.
    prose = brief.get("headline", "") + "\n" + "\n".join(
        i["body"] for i in lane_items)
    if "—" in prose or "–" in prose:
        soft.append("em/en dash in the prose (house style: neither)")

    if not brief.get("footer"):
        soft.append("no footer; the conventions + cells-scanned line is standing")
    elif "scanned" not in brief["footer"].lower():
        soft.append("footer does not quote the cells-scanned count")

    if sidecar:
        if sidecar.get("asof") is None:
            soft.append("sidecar .json has no asof")
        n_json = len(sidecar.get("nuggets") or [])
        if not quiet and n_json != len(lane_items):
            hard.append(f"sidecar lists {n_json} nuggets, the markdown has "
                        f"{len(lane_items)}; they are the same brief")
    else:
        hard.append("no sidecar .json next to the brief (the sender reads "
                    "`quiet` and the journal records from it)")

    cell_map = CELL_MAP_DIR / run_date / "00_cell_map.md"
    if not cell_map.exists():
        try:
            shown = cell_map.relative_to(ROOT)
        except ValueError:
            shown = cell_map          # a test tree, or a relocated checkout
        hard.append(f"no cell map at {shown}; stage B did not run, so nothing "
                    f"publishes")
    return hard, soft


# ---------- Block Kit ----------

def _section(text: str) -> dict:
    return {"type": "section", "text": {"type": "mrkdwn", "text": text[:2900]}}


def _context(text: str) -> dict:
    return {"type": "context", "elements": [{"type": "mrkdwn", "text": text[:2900]}]}


TAG_MARKER = {"solid": "[GREEN]", "suggestive": "[BLUE]", "anecdote": "[NEUTRAL]"}


def build_blocks(brief: dict, sidecar: dict) -> tuple[list[dict], str]:
    blocks: list[dict] = []
    quiet = bool(sidecar.get("quiet"))
    stale = bool(sidecar.get("prices_stale"))

    title = brief.get("title") or "Market Context"
    blocks.append({"type": "header",
                   "text": {"type": "plain_text", "text": title[:150],
                            "emoji": False}})

    if stale:
        last = sidecar.get("last_bar") or "unknown"
        blocks.append(_section(
            f"[WARN] *PRICES STALE (last bar {last})* - today-in-context "
            f"suppressed; scheduled-tape items only."))

    headline = (brief.get("headline") or "").strip()
    if headline:
        marker = "[ACTIVE]" if not quiet else "[QUIET]"
        blocks.append(_section(f"{marker} *{to_mrkdwn(headline)}*"))

    for name in LANE_SECTIONS:
        items = brief["items"].get(name)
        if not items:
            continue
        blocks.append({"type": "divider"})
        blocks.append(_section(f"*{name}*"))
        for i, item in enumerate(items, start=1):
            tag = item["tag"]
            dot = TAG_MARKER.get(tag, "")
            head = f"*{i}. {to_mrkdwn(item['title'])}*"
            if tag:
                head += f"  {dot} _{tag}_"
            body = to_mrkdwn(item["body"]) if item["body"] else ""
            blocks.append(_section(f"{head}\n{body}" if body else head))

    # Any section the lanes did not claim (QUIET TAPE note, ad hoc blocks).
    for name, lines in brief["sections"].items():
        if name in LANE_SECTIONS or name == "Calendar":
            continue
        body = "\n".join(f"• {to_mrkdwn(l[2:].strip())}" if l.lstrip().startswith("- ")
                         else to_mrkdwn(l.strip())
                         for l in lines if l.strip())
        if body:
            blocks.append(_section(f"*{name}*\n{body}"))

    calendar = [l[2:].strip() for l in brief["sections"].get("Calendar", [])
                if l.lstrip().startswith("- ")]
    if calendar:
        body = "\n".join(f"• {to_mrkdwn(c)}" for c in calendar)
        blocks.append(_section(f"*Calendar*\n{body}"))

    footer = (brief.get("footer") or "").strip()
    if footer:
        blocks.append(_context(to_mrkdwn(footer)))

    return blocks, (STALE_COLOR if stale else BRAND_COLOR)


# ---------- post ----------

def post_via_webhook(url: str, text: str, blocks: list[dict], color: str,
                     retries: int = 2) -> None:
    """Webhooks take the chat.postMessage shape minus auth and channel, and
    answer with the literal body 'ok'. 429 and 5xx are transient and retried;
    every other 4xx is a config error that will not fix itself."""
    payload = {"text": text,
               "attachments": [{"color": color, "fallback": text, "blocks": blocks}]}
    data = json.dumps(payload).encode("utf-8")
    last_err: str | None = None
    for attempt in range(1, retries + 2):
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={"Content-Type": "application/json; charset=utf-8"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode("utf-8").strip()
            if resp.status == 200 and body == "ok":
                return
            raise SystemExit(f"Webhook post rejected: HTTP {resp.status} body={body!r}")
        except urllib.error.HTTPError as err:
            detail = ""
            try:
                detail = err.read().decode("utf-8").strip()
            except Exception:  # noqa: BLE001
                pass
            if err.code == 429 or err.code >= 500:
                last_err = f"HTTP {err.code} {detail!r}"
                print(f"  webhook attempt {attempt}/{retries + 1} failed: {last_err}")
                if attempt <= retries:
                    try:
                        delay = float(err.headers.get("Retry-After") or 0)
                    except (TypeError, ValueError):
                        delay = 0.0
                    time.sleep(min(max(delay, 2.0 * attempt), 30.0))
                continue
            raise SystemExit(f"Webhook post rejected: HTTP {err.code} {detail!r} "
                             f"(check SLACK_WEBHOOK_URL is current)")
        except urllib.error.URLError as err:
            last_err = f"{type(err).__name__}: {err}"
            print(f"  webhook attempt {attempt}/{retries + 1} failed: {last_err}")
    raise SystemExit(f"Webhook post FAILED after {retries + 1} attempts: {last_err}")


def post_via_token(token: str, channel: str, text: str, blocks: list[dict],
                   color: str, retries: int = 2) -> None:
    payload = {"channel": channel, "text": text,
               "attachments": [{"color": color, "fallback": text, "blocks": blocks}]}
    data = json.dumps(payload).encode("utf-8")
    last_err: str | None = None
    for attempt in range(1, retries + 2):
        req = urllib.request.Request(
            SLACK_POST_URL, data=data, method="POST",
            headers={"Authorization": f"Bearer {token}",
                     "Content-Type": "application/json; charset=utf-8"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            if body.get("ok"):
                return
            raise SystemExit(f"Slack post rejected: {body.get('error')} "
                             f"(channel={channel}; is the bot invited?)")
        except urllib.error.URLError as err:
            last_err = f"{type(err).__name__}: {err}"
            print(f"  post attempt {attempt}/{retries + 1} failed: {last_err}")
    raise SystemExit(f"Slack post FAILED after {retries + 1} attempts: {last_err}")


# ---------- after publish ----------

def advance_flag_state(sidecar: dict, run_date: str) -> int:
    """Move the novelty baseline forward for exactly what published.

    Deliberately here and not in the engine: the engine runs before anything
    is chosen, so it cannot know what was published, and advancing on a run
    that never posted would block tomorrow's brief from saying something it
    never said."""
    state = {"version": 1, "flags": {}}
    if FLAG_STATE_PATH.exists():
        try:
            loaded = json.loads(FLAG_STATE_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded.get("flags"), dict):
                state = loaded
        except (OSError, json.JSONDecodeError):
            pass   # a corrupt baseline is rebuilt, never allowed to block a post
    flags = state.setdefault("flags", {})
    for nugget in sidecar.get("nuggets") or []:
        fp = nugget.get("fingerprint")
        if not fp:
            continue
        prior = flags.get(fp) or {}
        flags[fp] = {"last_published": run_date,
                     "last_headline_number": nugget.get("mean_pct"),
                     "count": int(prior.get("count", 0)) + 1,
                     "cell": nugget.get("cell")}
    state["updated"] = datetime.now().astimezone().isoformat(timespec="seconds")
    FLAG_STATE_PATH.write_text(json.dumps(state, indent=1), encoding="utf-8")
    return len(sidecar.get("nuggets") or [])


def append_journal(sidecar: dict, run_date: str, brief_path: Path) -> int:
    """The audit trail of what was claimed. Nothing replays it (no scoreboard,
    McKinley 2026-08-09); it exists so a claim can be traced back."""
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    model = os.environ.get("CONTEXT_MODEL", "")
    effort = os.environ.get("CONTEXT_EFFORT", "")
    rows = []
    for i, nugget in enumerate(sidecar.get("nuggets") or []):
        rows.append(json.dumps({
            "ts": stamp, "run_date": run_date, "asof": sidecar.get("asof"),
            "brief": brief_path.name, "kind": "nugget", "rank": i + 1,
            "model": model, "effort": effort,
            **{k: nugget.get(k) for k in
               ("lane", "trigger_id", "fingerprint", "subject", "cell", "tag",
                "n", "mean_pct", "hit", "t", "sign_p", "drill_script")},
        }, ensure_ascii=False))
    if not rows and sidecar.get("quiet"):
        rows.append(json.dumps({"ts": stamp, "run_date": run_date,
                                "asof": sidecar.get("asof"),
                                "brief": brief_path.name, "kind": "quiet",
                                "model": model, "effort": effort},
                               ensure_ascii=False))
    with JOURNAL_PATH.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(rows) + "\n")
    return len(rows)


# ---------- main ----------

def latest_brief() -> Path:
    briefs = sorted(BRIEF_DIR.glob("[0-9]" * 4 + "-[0-9][0-9]-[0-9][0-9].md"))
    if not briefs:
        raise SystemExit(f"No brief found in {BRIEF_DIR}")
    return briefs[-1]


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, help="brief .md (default: latest)")
    ap.add_argument("--run-date", default=None,
                    help="override the run date the gates check (YYYY-MM-DD)")
    ap.add_argument("--channel", help="channel id (bot-token mode)")
    ap.add_argument("--webhook-url", help="overrides SLACK_WEBHOOK_URL")
    ap.add_argument("--dry-run", action="store_true",
                    help="write the payload beside the brief, post nothing, "
                         "advance nothing")
    ap.add_argument("--no-state", action="store_true",
                    help="post but do not advance the novelty baseline or the "
                         "journal (re-posting an already-journaled brief)")
    ap.add_argument("--skip-freshness", action="store_true",
                    help="allow posting a brief that is not dated today")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(ENV_PATH)
    except ImportError:
        pass

    brief_path = args.file or latest_brief()
    if not brief_path.exists():
        raise SystemExit(f"Brief not found: {brief_path}")
    run_date = args.run_date or brief_path.stem
    today = datetime.now().astimezone().strftime("%Y-%m-%d")

    if not args.skip_freshness and run_date != today:
        raise SystemExit(
            f"Brief {brief_path.name} is dated {run_date}, today is {today}. "
            f"A stale brief posted late reads as tonight's; pass "
            f"--skip-freshness only when you mean it.")

    brief = parse_brief(brief_path.read_text(encoding="utf-8"))
    sidecar = load_sidecar(brief_path)
    blocks, color = build_blocks(brief, sidecar)
    subject = brief.get("title") or f"Market Context {run_date}"

    hard, soft = lint_brief(brief, sidecar, run_date)
    for w in soft:
        print(f"  QA warn: {w}")
    for h in hard:
        print(f"  QA HARD: {h}")

    if args.dry_run:
        out = brief_path.with_suffix(".slack.json")
        out.write_text(json.dumps(
            {"text": subject,
             "attachments": [{"color": color, "fallback": subject,
                              "blocks": blocks}]},
            indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Dry run: wrote {out.name} ({len(blocks)} blocks, color={color})")
        print(f"  fallback/subject: {subject}")
        if hard:
            raise SystemExit(f"Dry run: {len(hard)} hard issue(s) - fix before posting.")
        return 0

    if hard:
        raise SystemExit(f"{len(hard)} hard issue(s); nothing posted. Fix the "
                         f"brief, not the gate.")

    webhook = args.webhook_url or os.environ.get("SLACK_WEBHOOK_URL")
    if webhook:
        print(f"Posting '{subject}' from {brief_path.name} via incoming webhook...")
        post_via_webhook(webhook, subject, blocks, color)
        via = "webhook"
    else:
        token = os.environ.get("SLACK_BOT_TOKEN")
        channel = (args.channel or os.environ.get("SLACK_CHANNEL_ID")
                   or os.environ.get("CONTEXT_SLACK_CHANNEL"))
        if not token:
            raise SystemExit("Missing SLACK_WEBHOOK_URL or SLACK_BOT_TOKEN in .env")
        if not channel:
            raise SystemExit("Missing SLACK_CHANNEL_ID in .env (or --channel)")
        print(f"Posting '{subject}' from {brief_path.name} to {channel}...")
        post_via_token(token, channel, subject, blocks, color)
        via = f"chat.postMessage/{channel}"

    ts = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"  POSTED ok {ts} | via={via} | brief={brief_path.name}")

    if args.no_state:
        print("  --no-state: novelty baseline and journal left untouched")
        return 0
    advanced = advance_flag_state(sidecar, run_date)
    journaled = append_journal(sidecar, run_date, brief_path)
    print(f"  novelty baseline advanced for {advanced} nugget(s); "
          f"{journaled} journal record(s) appended")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
