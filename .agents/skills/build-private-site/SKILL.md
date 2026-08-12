---
name: build-private-site
description: Build, rebuild, repair, publish, deploy, or verify the New_Seasonals private Cloudflare Pages site. Use for every private-site or live-site request in this repository, including stale Portfolio or Seasonal tabs, deployment checks, and changes to site/, scripts/build_site.py, scripts/validate_site_freshness.py, .github/workflows/deploy_site.yml, functions/, wrangler.toml, or production data payloads. Enforces cloud-only production builds from Cloudflare R2 through GitHub Actions; never use local data/ or dist/ as a production source.
---

# Build Private Site

Treat GitHub Actions as the only production build environment and Cloudflare R2 as the authoritative data source. Local files may be inspected for development, but they must never be used to build or deploy production.

## Non-negotiable rules

- Never run `python scripts/build_site.py` as a production rebuild.
- Never deploy the local `dist/` directory with Wrangler or any other tool.
- Never use local `data/`, `dist/`, cached JSON, parquet files, or generated assets as evidence that production is fresh.
- Never bypass or weaken `scripts/validate_site_freshness.py` to make a deployment pass.
- Never fall back to a local build when GitHub or Cloudflare credentials are missing. Report the credential blocker and give the exact cloud action needed.
- Do not delete, clean, overwrite, or regenerate local build artifacts unless the user separately authorizes that operation.

## Production workflow

1. Confirm the intended code exists on `origin/main`. If code changes are required, implement and verify them, then push or merge them before rebuilding production.
2. Dispatch `.github/workflows/deploy_site.yml` on `main`:

   ```powershell
   gh workflow run deploy_site.yml --repo mslade50/New_Seasonals --ref main
   ```

3. Find the new run and monitor it through completion:

   ```powershell
   gh run list --repo mslade50/New_Seasonals --workflow deploy_site.yml --limit 5
   gh run watch <run-id> --repo mslade50/New_Seasonals --exit-status
   ```

4. Verify these required stages succeeded in the cloud run:

   - Checkout repo
   - Pull caches from R2
   - Build full-history trade ledger
   - Build daily seasonal ideas
   - Build site
   - Refuse stale or incomplete site data
   - Deploy to Cloudflare Pages

5. Confirm the newest production Cloudflare deployment was created from the intended `origin/main` commit:

   ```powershell
   npx wrangler pages deployment list --project-name=seasonals-mslade --environment=production --json
   ```

6. If a signed-in browser session is available, inspect the live Portfolio, Seasonal, and Execution tabs. Otherwise, distinguish cloud workflow/deployment verification from authenticated UI verification and say what remains unverified.

## Failure handling

- If `gh auth status -h github.com` fails or the token cannot write GitHub Actions, stop the deployment attempt. Ask the user to authenticate with `gh auth login -h github.com` or manually run **Actions > Deploy Private Site > Run workflow**. Do not substitute a local deployment.
- If the R2 download, data generation, or freshness gate fails, inspect the cloud job logs, fix the responsible generator or workflow code, push the fix, and rerun this workflow.
- If the workflow succeeds but a live tab is stale or broken, compare the deployed commit, generated payload contract, frontend expectations, and authenticated browser behavior. Do not infer live state from local `dist/`.
- If the production deployment commit is not the intended `origin/main` SHA, do not call the task complete; dispatch the correct cloud build and verify again.

## Completion standard

Report success only when the GitHub Actions build completed, the freshness gate passed, and Cloudflare shows the new production deployment from the intended commit. State separately whether authenticated live-tab QA was completed.
