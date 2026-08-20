---
name: "run-fundamental-sleeve"
description: "Run and maintain the New_Seasonals fundamental-equity research sleeve: daily broad-universe coverage, bounded source refreshes, archetype-specific triage and underwrites, thesis and trigger monitoring, concise Fundamentals inbox updates, founder/product research maps, and research-only portfolio handoffs. Use for scheduled or ad hoc fundamental sleeve research, coverage or freshness checks, candidate ranking, company deep dives, QUICK REVIEW decisions, research-state controls, or changes to the sleeve workflow and outputs. Never use it to allocate capital, stage or place orders, deploy the private site, upload research, or modify broker or portfolio state."
---

# Run Fundamental Sleeve

Operate the sleeve as a research operating system, not a factor screen. Search broadly, reject aggressively, underwrite narrowly, preserve evidence, and surface only decisions.

## Establish authority and scope

1. Read the repository `AGENTS.md` before acting.
2. Invoke the Public Equity Investing router and select the narrowest owning workflow before substantive investment work. Use `idea-generation` for the broad funnel, then route deep work to the relevant earnings, valuation, thesis-tracker, event, or portfolio-risk workflow.
3. Keep every run research-only unless McKinley explicitly authorizes a different action later. Never infer authority to allocate capital, stage or place orders, send messages, upload, deploy, or change broker/portfolio state.
4. Use `$build-private-site` separately for any private-site build, deployment, live-site repair, stale-tab investigation, or production verification. Never use local `data/` or `dist/` as production evidence.
5. Preserve unrelated work. Follow the repository workspace-hygiene rules before editing a long-lived or dirty worktree, and place disposable browser profiles, screenshots, downloads, and logs under `artifacts/`.

## Apply the mandate

- Maintain a broad long-only research universe while limiting the reader-facing inbox to zero through three names.
- Treat ten positions and 30% of account NAV as hard future portfolio limits, not deployment targets. Keep cash valid.
- Separate company quality, security attractiveness, timing, and portfolio fit.
- Require a specific expectations gap, valuation support, tolerable downside, and observable proof path before surfacing a decision.
- Treat founder leadership and product familiarity as research lenses only. Never let either promote a security.
- Treat price trend as a timing and error-control governor, never as proof of mispricing.
- Prefer no review over a weak review. Never turn a screen score into a recommendation.

## Load references progressively

- Read [daily-runbook.md](references/daily-runbook.md) for every complete scheduled or ad hoc research run.
- Read [decision-gates-and-state.md](references/decision-gates-and-state.md) before creating or changing a candidate, underwrite, trigger, user-facing decision, thesis status, or portfolio handoff.
- Read [archetype-frameworks.md](references/archetype-frameworks.md) only for the business models and idea archetypes selected for the current deep-work names.
- Read [qa-and-safety.md](references/qa-and-safety.md) before finalizing changed data, reports, HTML, automation behavior, or a user-facing conclusion.

## Run the research loop

1. Start with `python scripts/run_fundamental_sleeve.py --as-of YYYY-MM-DD`. It is dry-run-only by default and prints the exact coverage, state, and refresh plan without fetching or writing. Use `--execute` only after the plan is consistent with the requested scope. The orchestrator has no upload, deployment, messaging, broker, order, or allocation capability.
2. Load user research controls, prior decisions, open triggers, manual portfolio context when available, and the last successful run manifest from the canonical paths in the daily runbook.
3. Check source coverage and freshness before fetching. Refresh only missing, stale, or event-invalidated inputs in bounded, resumable batches.
4. Maintain cheap baseline coverage across every currently eligible company, including all specialist lanes. Distinguish broad baseline coverage from expensive deep underwriting.
5. Rebuild applicable business-model and idea-archetype hypotheses. Use screen output only to prioritize research.
6. Process work in this order: held-name exceptions, fired kill/proof triggers, review-ready changes, user `DEEPEN` requests, imminent catalysts, active underwrites, new hypotheses, routine coverage rotation.
7. Select at most three deep-work names. Choose the next research step by decision impact and expected information value, not by raw score alone.
8. Gather primary evidence, normalize the relevant economics, quantify what the current price appears to imply, build archetype-appropriate valuation cases, and run an independent opposing-case check.
9. Propose a state transition with evidence IDs. Apply the decision gates before writing the authoritative decision state.
10. Update additive underwrites, thesis evidence, triggers, current research views, and reader-facing reports only when evidence or state changed.
11. Run relevant tests and inspect changed HTML locally with a headless browser. Record the exact report digest with `scripts/record_fundamental_visual_qa.py`. Keep production untouched.

## Enforce output discipline

Return no action when no company clears every decision gate. Name at most three companies still under investigation only when that context is useful.

For a cleared review, provide one compact `QUICK REVIEW` with:

- the security thesis and exact variant wedge;
- what the price appears to imply and what is already priced in;
- current-price and source as-of timestamps;
- archetype-appropriate bear, base, and bull valuation ranges;
- the earnings or estimate path required for the stock to work;
- why now and the next observable proof trigger;
- the downside mechanism, financing or dilution risk, and kill condition;
- trend/timing and portfolio overlap;
- material contradictory evidence and remaining gaps;
- the exact research decision requested from McKinley.

Keep controls reversible and research-only. Treat `DEEPEN` as one bounded diligence pass, `WATCH` as waiting for a recorded trigger, `PASS` as suppression until a genuine reopening condition fires, and `CLEAR` as removal of the override.

## Judge completion

Call the run complete only after source health, state transitions, evidence links, output caps, tests, and visual QA pass. Report stale or unavailable sources plainly. Do not manufacture activity when nothing changed.
