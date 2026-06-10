# Private Site — One-Time Cloudflare Setup

The private analytics site (portfolio / ideas / signals / risk) is a static
site built nightly by `.github/workflows/deploy_site.yml` and deployed to
Cloudflare Pages, locked behind Cloudflare Access (email one-time-code login).
Total cost: $0. This doc covers the one-time setup; after that everything is
automatic.

> STATUS 2026-06-09: all steps below are DONE. Project `seasonals-mslade`
> created (the name `seasonals-private` was taken globally), Access app
> "Seasonals private site" live on the `denaliassetmanagement` Zero Trust org
> (allow: mckinleyslade@gmail.com, pj.pierre@denalifunds.com,
> scott@denalifunds.com; One-time PIN; 730h sessions), first deploy verified
> behind the login wall, and `CLOUDFLARE_API_TOKEN` / `CLOUDFLARE_ACCOUNT_ID`
> repo secrets set. The doc is kept for re-setup reference. The API token is
> stored locally at `HKCU\Environment\CLOUDFLARE_API_TOKEN_NS`
> ("seasonals_access" token, perms: Pages Edit + Access Apps/Orgs Edit).

## What you'll end up with

- `https://seasonals-mslade.pages.dev` — the site, on any device
- A Cloudflare Access login wall: enter your email, get a 6-digit code, in.
  Sessions last 30 days by default so you won't log in often per device.
- Nightly refresh at 22:35 UTC (6:35 PM ET) on trading days, after the
  post-close scan finishes. Manual refresh anytime via the workflow's
  `workflow_dispatch` button (or `gh workflow run deploy_site.yml`).

## Step 1 — Create the Pages project (~2 min)

You already have a Cloudflare account (R2 lives there).

1. Cloudflare dashboard -> **Workers & Pages** -> **Create** -> **Pages** ->
   **Upload assets** (direct upload, NOT git integration — GHA does the deploys).
2. Project name: `seasonals-mslade` (must match `--project-name` in
   `deploy_site.yml`; if you pick something else, edit the workflow).
3. Upload any placeholder file to finish creation (e.g. an empty index.html).
   The first real deploy will replace it.

Alternatively from a terminal: `npx wrangler pages project create seasonals-mslade`.

## Step 2 — Lock it behind Cloudflare Access (~3 min)

1. Dashboard -> **Workers & Pages** -> `seasonals-mslade` -> **Settings** ->
   **General** -> **Access policy** -> **Enable**. This auto-creates a Zero
   Trust application covering `*.seasonals-mslade.pages.dev` (preview URLs).
2. Go to **Zero Trust** -> **Access** -> **Applications** -> edit the
   auto-created app:
   - Add the production domain too: `seasonals-mslade.pages.dev` (the
     auto-created entry only covers the `*.` wildcard subdomains).
   - Policy: Allow -> Include -> **Emails** -> `mckinleyslade@gmail.com`,
     `pj.pierre@denalifunds.com`, `scott@denalifunds.com`.
   - Identity provider: One-time PIN (default, no IdP setup needed).
   - Session duration: 30 days (or whatever you prefer).
3. Test later by opening the site in a private browser window — you should
   hit the Access login page, not the site.

## Step 3 — Create the API token (~2 min)

1. Cloudflare dashboard -> My Profile -> **API Tokens** -> **Create Token** ->
   **Custom token**.
2. Permissions: **Account -> Cloudflare Pages -> Edit**. Scope to your account.
3. Copy the token.

Your Account ID is on the right sidebar of any zone page, or under
Workers & Pages -> overview.

## Step 4 — Add GitHub secrets (~1 min)

Repo -> Settings -> Secrets and variables -> Actions:

| Secret | Value |
|---|---|
| `CLOUDFLARE_API_TOKEN` | the token from Step 3 |
| `CLOUDFLARE_ACCOUNT_ID` | your account id |

All other secrets the workflow uses (`R2_*`, `GCP_JSON`) already exist.

## Step 5 — First deploy

Actions -> **Deploy Private Site** -> Run workflow. Takes ~10-15 min
(ledger rebuild dominates). Then open `https://seasonals-mslade.pages.dev`.

## Architecture notes

- **Build pipeline** (all in the workflow): R2 caches ->
  `scripts/build_trade_ledger.py` (full-history ledger) ->
  `daily_seasonal_ideas.py` (best effort) -> `scripts/build_risk_json.py`
  (best effort) -> `scripts/build_site.py` (JSON payloads + static assets ->
  `dist/`) -> wrangler deploy.
- **Client-side recompute**: the browser gets the full trade ledger
  (`trades.json`) plus per-strategy daily MTM series on the flat $750k basis
  (`strategy_daily.json`). Strategy/tier/date filters recompute equity, Sharpe,
  drawdown, heatmaps, tables exactly; direction/ticker filters fall back to
  realized-PnL-at-exit curves (badge appears).
- **Local dev**: `python scripts/build_site.py --no-signals` then
  `python -m http.server 8123 --directory dist` and open
  `http://localhost:8123`. `--no-mtm` skips the slow payloads when iterating
  on frontend only (pair with a previously built `dist/data/`).
- **Payload sizes**: trades 0.6 MB, strategy_daily 0.7 MB raw — Cloudflare
  gzips on the wire. `site/_headers` sets `Cache-Control: no-store` on
  `/data/*` so you always see the latest build.
- The site is read-only by construction: order staging / IBKR stays local.
