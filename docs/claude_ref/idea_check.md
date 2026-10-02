# Idea Check

McKinley types one trade idea into a private-site tab. A local poller runs the `/idea-check` skill headlessly and the verdict shows in the tab. It reviews one idea: tweaks or a plain "bad trade". It never generates ideas.

## Flow

1. Tab posts the text to the Pages Function, which appends it to `idea_check/queue.json` (R2, oldest first, newest 20).
2. `IdeaCheckPoller` (Task Scheduler, every minute) runs `scripts/run_idea_check_poller.bat`, which runs `idea_check_poller.py --once`.
3. For each new request: write `scratch/idea_checks/<id>/request.json`, upload a `running` result, run `scripts/invoke_idea_check_agent.ps1` (`claude -p "/idea-check <id>"`, 1200 s timeout), read `verdict.json`, validate it, upload `done` or `error`.
4. The tab polls `idea_check/results/<id>.json`.

## R2 keys

- `idea_check/queue.json`: `{"requests":[{"id","text","submitted_at"}]}`. Written only by the Pages Function. The poller only reads it.
- `idea_check/results/<id>.json`: `{id, status running|done|error, verdict KILL|SURVIVES|NEAR-MISS|NEEDS-INFO|null, headline, numbers[], tweaks[], body_md, started_at, finished_at, error}`. Written only by the poller.
- id format `^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{6}$`. Text is 1 to 2000 chars.

## Files

Local half: `idea_check_poller.py`, `scripts/run_idea_check_poller.bat`, `scripts/invoke_idea_check_agent.ps1`, `scripts/register_idea_check_task.ps1`, `.claude/skills/idea-check/SKILL.md`, `tests/test_idea_check_poller.py`.
Site half: `functions/idea-check.js`, `site/idea.html`, `site/assets/idea.js`, `tests/test_idea_check_site.py`, `tests/js/test_idea_tab.js`.
State: `scratch/idea_checks/_state/` (`processed.json`, `poller.lock`, downloaded `queue.json`, agent logs). Per request: `scratch/idea_checks/<id>/`.

## Poller rules

- Skips bad ids, text over 2000 chars, requests older than 24 h, ids already in `processed.json`. Cap 20 reviews per calendar day.
- Lock file with pid and time, stale after 30 minutes, refreshed before each review. A locked pass exits 0 quietly.
- Failures (timeout, non-zero exit, missing or invalid verdict) upload `error`. The id is marked processed either way.
- The idea text never goes on a command line. The agent gets the id and reads the file.
- The book must not import the poller, and the poller must not import `pitch_lab` or `pitch_grammar` (the skill's check scripts do).

## Skill hard rules

Writes only inside `scratch/idea_checks/<id>/`. Never runs `daily_pitch.py`, anything under trading_ibkr, or touches the pitch journal, scoreboard, watchlist, negative registry, Sheets, email or R2. Never places or stages an order. At most 6 check scripts. Risk in ATR terms only.

## Register and knobs

Registration is manual: `powershell -File scripts\register_idea_check_task.ps1`. It writes a hidden-window VBS launcher to `%LOCALAPPDATA%\IdeaCheck` and runs it through `wscript.exe //B`, so no console flashes each minute. The task runs from this checkout, not the pinned runtime worktree.
Knobs (env, read by the bat and the poller): `IDEA_CHECK_MODEL` (default `opus`), `IDEA_CHECK_EFFORT` (default `high`).
Dry run: `python idea_check_poller.py --dry-run`. Log: `scripts\logs\idea_check_last_run.log`.
