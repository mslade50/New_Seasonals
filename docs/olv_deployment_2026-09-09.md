# OLV deployment — September 9, 2026

Owner authorization: deploy the approved OLV source changes. Source, pinned
runtime and private-site deployment are complete. The live inventory-dependent
cap and volume-stop handoff remain unverified and are not certified by this release.

## Scope

The model uses the submitted-limit target, pending entries reserve per-stock
capacity, and OLV daily caps floor shares and recompute PnL. The installed
scanner retains the owner's explicit optional-overlay fallback when inventory
or pending-order inputs cannot be verified. No live inventory seed or continuous
fill attestation has been invented. No broker collector/exit candidate is installed,
no working order is adjusted, and the existing PA exit route remains unchanged.
The volume-stop handoff and early-close deadline defect remain open.

## Source and runtime

- [PR #33](https://github.com/mslade50/New_Seasonals/pull/33) merged as
  `74902be33075d74337678fac5f0c67fe9cc9e4f2`.
- Its final [CI](https://github.com/mslade50/New_Seasonals/actions/runs/34365419155)
  passed Linux and Windows checks. The local full suite passed 2,010 tests.
- Runtime v9 was fast-forwarded under the existing supervisor lock to
  `d0ead389a0ecf92f444cf2ee457cfe371a902563`, tag
  `automation-runtime-2026-09-09.4`, at 14:46:11 UTC.
  ValidateOnly and 138 installed-runtime tests passed (one skipped).
  The prior marker is preserved in
  `New_Seasonals-automation-runtime-v9/.local/runtime_promotions/olv_20260909T144611Z/`.
- Pre-install validation caught the unused `.3` tag pointing to the wrong
  checkout. It was never installed and was preserved rather than rewritten.
- Deployment inspection then found the site table/chart independently
  reconstructed the retired fill-based target. The initial
  [cloud build](https://github.com/mslade50/New_Seasonals/actions/runs/34365754263)
  was cancelled before publication.
- [PR #34](https://github.com/mslade50/New_Seasonals/pull/34) carries the actual
  engine target through the ledger into the open-position table and charts.
  New OLV chart keys prevent reuse of old target annotations without deleting
  existing images. The regression reproduced 102.00 versus 104.50 and passes
  after correction. The subsequent full local suite passed 2,011 tests,
  55 skipped and two expected failures; focused chart/site checks passed.
- Final runtime candidate:
  `b17cd79df9c9f2de55b2fca38f1e92d02789b91c`,
  `automation-runtime-2026-09-09.6`. The intermediate `.5` candidate was not
  installed. Final runtime/site files have been compared for equality.

## Final verification

PR #34 merged as `e9e568f12d7b1088a7910aecba1de184aa5f2e6f`.
Its final [CI](https://github.com/mslade50/New_Seasonals/actions/runs/34366722235)
passed Linux (1,966 tests, 102 skipped, plus browser/Worker contracts) and Windows.
Runtime v9 was promoted to `b17cd79df9c9f2de55b2fca38f1e92d02789b91c`,
tag `automation-runtime-2026-09-09.6`, at 14:58:32 UTC under the supervisor
lock. ValidateOnly, 142 installed tests (one skipped), and workspace hygiene
passed. Its predecessor marker is retained in
`New_Seasonals-automation-runtime-v9/.local/runtime_promotions/olv_20260909T145832Z/`.

[Final cloud build](https://github.com/mslade50/New_Seasonals/actions/runs/34367148325)
rebuilt the canonical R2 ledger successfully. Independent read-only R2
verification confirms source `e9e568f`, producer `gha:34367148325`, 4,711
rows, and all 316 OLV targets equal the submitted limit plus 2.5 ATR.
Seventy-six OLV rows have gap-improved modeled fills; those retain their
submitted target. OLV quantities are whole shares. The object was published
at 15:16:48 UTC. Evidence is saved in the isolated worktree's
`artifacts/olv-review/cloud-ledger-verification.json`.

All cloud stages passed, including chart generation, seasonal ideas, R2 bundle
provenance, final site assembly, and the freshness gate. All 316 corrected OLV
chart objects are present in their new R2 namespace.

Cloudflare production deployment `a0b8560e-197f-4cbc-aa09-d888bbe46166`
reports source `e9e568f` on main. Authenticated live checks passed:

- Portfolio rendered the 4,711-row cloud ledger, metrics/charts/trade log,
  producer `gha:34367148325`, and full source SHA `e9e568f12d7b1088a7910aecba1de184aa5f2e6f`.
  Its RTX open-position target is 212.03, replacing the old 211.77 fill-based
  display. The newly keyed RTX OLV chart loaded and rendered its target line.
- Seasonal loaded September 8 inputs, the manual sizer, and the valid
  no-qualifying-setups state.
- Execution loaded online broker data (book 15 seconds old at inspection),
  working orders, and the existing Close/Add/Re-add controls. No action was
  submitted. The pre-existing expected-exit verification-unavailable warning
  remains visible.

The published header's September 3 date is the latest modeled signal date,
not the build timestamp or an assertion that a September 9 scan ran.
The site build timestamp is `2026-09-09T15:23:50Z`.

No daily scanner, broker runner, trade submission, email, or scheduler
cadence change was performed during deployment. Source rollback uses a forward
revert and matching runtime marker/tag, followed by a cloud rebuild; old
markers, Git history, and chart objects remain available.
