# Preserve private-site repairs across scheduled refreshes

The enabled local scheduler and GitHub fallback pinned all child workflows to
the automation runtime tag. A scheduled private-site refresh could therefore
replace newer frontend repairs with the older runtime's site source. The
September 14 private_site_pm run 34900795889 deployed b5b8fa824 even after newer
site commits were available on main.

Only deploy_site.yml now dispatches on main. Producer workflows retain the
runtime's immutable ref, and existing token adoption, receipts and completion
checks are unchanged. The private-site workflow still builds in GitHub Actions
from authoritative R2 inputs and requires its freshness/provenance gate.

Both scheduler hosts must receive the repair: a clean local runtime and the
GitHub fallback's AUTOMATION_RUNTIME_REF. The new immutable runtime release tag
is automation-runtime-2026-09-14.2. The prior runtime and task generation remain
available for rollback; no tasks or historical state are deleted.

Verification includes rejecting the old site dispatch behavior, proving site
dispatch on main and producer dispatch on the configured pin, and token
adoption without duplicate workflow submissions. Operational deployment evidence
is retained in the execution task's release handoff.
