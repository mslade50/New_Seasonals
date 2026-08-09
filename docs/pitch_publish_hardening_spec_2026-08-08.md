# Daily Pitch publish hardening, spec for implementation (2026-08-08)

## Why

The pitch pipeline's proof-of-work enforcement is asymmetric. A stand-down
(NO TRADES morning) must prove on disk that the survey happened:
`validate_stand_down` checks that `checks_dir` exists, holds real `.py`
scripts and the stage-B1 `00_surface_map.md`, and carries floors on
candidates, axes, asset classes and named kills. A morning that ships three
ideas proves nothing: `validate_payload` never touches the filesystem on the
ideas path, and `evidence.script` is accepted as any non-empty string. The
lazy failure mode is shipping three recall-generated ideas without a survey,
and that is exactly the unguarded path. This spec closes that gap plus three
smaller ones.

Four changes, in priority order. Change 1 is the point; 2 through 4 are
ratchets. Note for calibration: the agent's observed failure mode so far has
been over-killing (2026-08-07: 24 candidates, 50 scripts, stand-down), not
under-checking, so 3 and 4 guard against drift rather than an observed
incident. Implement all four.

## Files in scope (aligned sites)

- `pitch_grammar.py` (validation logic; all new checks live here)
- `daily_pitch.py` (publisher wiring, CLI flag, email footer for the lint)
- `.claude/skills/daily-pitch/SKILL.md` (stage C round 3, stage D schema
  block, and the Publish section must describe the new requirements)
- `docs/daily_pitch.md` (runbook, one paragraph on the new publish gates)
- `tests/test_pitch_grammar.py`, `tests/test_daily_pitch.py`

Out of scope, do not touch: `pitch_lab.py`, `pitch_journal.py`,
`scripts/grade_pitch_journal.py`, `scripts/check_pitch_delivered.py`, the
stand-down floors, `pitch_moo.py` (OneDrive). The systematic book must
still never import any pitch module. Never loosen an existing rule to make
a new one fit.

---

## Change 1: the ideas path must prove the survey happened

### Requirement

A publish that carries ideas (not a stand-down) fails validation unless the
day's checks directory exists and shows a real survey:

- the directory `scratch/pitch_checks/<asof>/` exists (where `<asof>` is
  `payload.asof`, verbatim)
- it contains `00_surface_map.md` (reuse the existing `SURFACE_MAP_NAME`
  constant)
- it contains at least one `*.py` check script

Error messages should follow the existing voice, e.g. "ideas were published
without a surface map: stage B1 was skipped or its map was not written; the
morning is not surveyed and nothing may publish".

### Exemption

A directed-only publish (every idea carries a non-empty `directed_by`; the
existing `directed_ideas` helper identifies them) is EXEMPT from this check.
Rationale, mirroring the existing directed-ideas comment block: survey
enforcement exists to constrain the agent, not to block the human filter.
McKinley directing a single idea ad hoc must not require a full morning
sweep. A mixed publish (any non-directed idea present) gets the full check.

### Implementation notes

- Put the disk checks in a new function in `pitch_grammar.py`, e.g.
  `validate_survey_evidence(payload, checks_root=None) -> list[str]`, called
  from `validate_payload`. Keep `validate_idea` pure (no filesystem IO), the
  same layering the stand-down path already uses.
- `checks_root` defaults to `ROOT / "scratch" / "pitch_checks"` and is
  injectable so tests run against `tmp_path` and never depend on the repo's
  real scratch state. Thread the parameter through `validate_payload`.
- `daily_pitch.py` gains an optional `--checks-root` CLI flag (default None,
  meaning the real location) so dev and dry runs can point at a fixture
  directory. `--validate-only` enforces the check exactly like a real
  publish; iterating on validate-only is precisely when it should bind.
- Do not double-enforce on the stand-down path: a stand-down carries its own
  explicit `checks_dir` and `validate_stand_down` already checks it.

### Tests (test_pitch_grammar.py)

- valid payload + populated day dir passes
- missing dir fails; dir without `00_surface_map.md` fails; dir with the map
  but zero `.py` files fails, each with a distinct error naming what is
  missing
- a directed-only publish passes with no day dir at all
- a mixed publish (one directed, two composed) still requires the dir
- existing fixture payloads in both test files will need a populated
  `tmp_path` checks dir via the injectable root; update fixtures once in a
  shared helper, not per-test

---

## Change 2: evidence.script must exist and be from this morning

### Requirement

For every idea (directed included; `directed_by` relaxes nothing about
evidence), `evidence.script` must:

- be non-empty (already enforced)
- resolve to a file that exists on disk (relative paths resolve against the
  repo root, same convention as `validate_stand_down`)
- live under the day's checks directory, `scratch/pitch_checks/<asof>/`
  (subdirectories beneath it are fine). This is the machine-checkable proxy
  for the "computed fresh this morning" hard requirement: a path pointing at
  yesterday's folder, or anywhere else in the repo, fails.

### Implementation notes

- Lives in the same `validate_survey_evidence` pass (it needs the same
  `checks_root` injection and asof). The containment check should compare
  resolved paths, not string prefixes, so `..` tricks and case differences
  on Windows do not slip through (use `Path.resolve()` on both sides and
  `is_relative_to`, or `relative_to` in a try/except).
- Directed-only publishes: change 1's exemption does NOT extend here. A
  directed idea still needs a real check script written that morning, which
  means its publisher run needs the day folder to exist with that one script
  in it. That is intended and is cheap for the directed flow.

### Tests

- a script path that does not exist fails with an error naming the path
- a script that exists but sits in a previous day's folder fails with an
  error saying evidence must be fresh
- a script inside a subdirectory of the day folder passes
- an absolute path inside the day folder passes; an absolute path outside
  fails
- a directed idea with a missing script still fails (directed relaxes
  nothing)

---

## Change 3: lint kill reasons against the illegal-kill doctrine

### Requirement

The small-N doctrine (SKILL.md stage C) makes "insufficient N", "not
statistically significant" and "t below 2" illegal as standalone kill
reasons. Nothing enforces this today. Add a WARN-ONLY lint; it must never
block a publish, because the pattern match is heuristic and a false positive
must not stop the morning.

New function in `pitch_grammar.py`:

```
lint_kill_reasons(killed: list[dict]) -> list[str]
```

A kill reason is flagged when it matches a sample-size-only pattern AND does
not also name a substantive kill. Suggested pattern sets (case-insensitive,
tune wording freely, keep both lists as module constants so tests can pin
them):

- sample-size patterns: `insufficient n`, `sample (size )?(is )?(too )?small`,
  `small sample`, `low n`, `n (is )?too (small|low)`, `only \d+ (obs|episode|
  event|sample)`, `not (statistically )?significant`, `t[- ]?stat(istic)?
  .{0,20}(below|under|<) ?2`, `t < ?2`
- substantive-kill markers (presence of any suppresses the flag):
  `mechanism`, `gate`, `filter`, `definition`, `fragil`, `era`, `sign
  (flip|instab)`, `cost`, `cluster`, `concentrat`, `regime`, `drift`,
  `artifact`

### Surfacing

- `daily_pitch.py` prints each flagged kill loudly on every run mode
  (validate-only, dry-run, real publish), prefixed like
  `KILL-LINT: 'Silver catch-up' was killed on sample size alone; the
  doctrine requires a substantive kill (see SKILL.md stage C)`.
- The email's killed-ideas footer marks flagged entries visibly (a short
  bracketed tag on the line is enough) so a doctrine violation is in front
  of McKinley the same morning.
- Applies to the `killed` list on BOTH idea publishes and stand-downs.

### Tests

- "killed: N=6, insufficient sample" flags
- "killed: t=1.4, not significant" flags
- "N=6 but the gate removes one observation from six, nothing attributable
  to the gate" does not flag (substantive marker present)
- "definition fragility: exists at rank>80, gone at rank>75" does not flag
- lint never appends to the validation errors list (publish proceeds);
  email rendering shows the tag (test in test_daily_pitch.py)

---

## Change 4: every composed idea must carry its round-3 development script

### Requirement

Stage C round 3 (horizon scan, entry form, exits, loser paths) is currently
prose-mandatory. Make it structural: `evidence` gains a `dev_script` field,
required for every NON-directed idea, validated with the same rules as
change 2 (exists, under the day folder). Directed ideas may omit it (the
human asked for the trade in a stated form; round 3's job of shaping the
trade is already done), but when present it is validated the same way.

`dev_script` may equal `script` only if that one file demonstrably contains
the development section; do not enforce content, just allow the same path.

### Aligned edits

- SKILL.md stage D schema block adds `"dev_script": "scratch/pitch_checks/
  2026-08-06/gld_slv_nfp_dev.py"` to the evidence example, and stage C round
  3 states the field is required for composed ideas.
- `daily_pitch.py` email evidence block renders the dev script path next to
  the existing script path.
- Journal records already carry the full idea dict, so the field flows
  through with zero grader/journal changes. Verify with the existing
  round-trip test rather than new plumbing.

### Tests

- a composed idea without `dev_script` fails validation with an error
  quoting the round-3 requirement
- a directed idea without it passes
- a `dev_script` in yesterday's folder fails (freshness, same as change 2)
- email renders the path (test_daily_pitch.py)

---

## Cross-cutting constraints

- House prose style in every user-facing string and doc edit: no em dashes,
  no "it's not X, it's Y" constructions.
- All new constants (pattern lists, any floor) live in `pitch_grammar.py`
  with a frozen-values test, matching `test_product_rules_frozen` /
  `test_stand_down_floors_frozen` style.
- Windows paths: all containment checks must be correct on a case-insensitive
  filesystem (resolve before comparing).
- No new dependencies. No network. Nothing writes outside the repo.
- The full existing suites must pass:
  `python -m pytest tests/test_pitch_grammar.py tests/test_daily_pitch.py
  tests/test_pitch_lab.py tests/test_pitch_delivery_check.py
  tests/test_pitch_grader.py -q`
  Fixture churn from changes 1 and 2 is expected (payload fixtures need a
  populated tmp checks dir); do it once in shared fixtures.

## Acceptance walkthrough

1. `python daily_pitch.py --ideas <valid payload> --validate-only` with no
   `scratch/pitch_checks/<asof>/` directory: fails, error names the missing
   surface map.
2. Same payload, directory populated with map + scripts + per-idea dev
   scripts, `evidence.script` paths pointing into it: silent.
3. Same payload with one `evidence.script` pointing at yesterday's folder:
   fails on freshness.
4. A killed entry reading "N too small": validate-only still passes but
   prints a KILL-LINT line, and the rendered email tags the kill.
5. A directed-only payload with only its own check script in the day folder:
   passes without a surface map.
6. Full test suite green.
