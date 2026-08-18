from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_private_site_deploy_has_two_isolated_r2_boundaries():
    workflow = (ROOT / ".github/workflows/deploy_site.yml").read_text(encoding="utf-8")
    assert "stage_private_site_cloud_build.py --source source --dest" in workflow
    assert "pull --phase generator" in workflow
    assert "publish-generated" in workflow
    assert "pull --phase assembler" in workflow
    assert '${{ github.run_id }}-${{ github.run_attempt }}' in workflow
    assert "build_site.py --production" in workflow
    assert "--require-r2-provenance" in workflow
    assert "workingDirectory: assembler" in workflow
    assert "pull_scan_caches.py --set site" not in workflow


def test_repo_wrangler_config_cannot_select_local_dist():
    config = (ROOT / "wrangler.toml").read_text(encoding="utf-8")
    assert not any(
        line.strip().startswith("pages_build_output_dir")
        for line in config.splitlines()
    )
    assert 'name = "seasonals-mslade-local-disabled"' in config


def test_cloudflare_git_bypass_is_locked_and_checked():
    deploy = (ROOT / ".github/workflows/deploy_site.yml").read_text(encoding="utf-8")
    seed = (ROOT / ".github/workflows/seed_site_r2.yml").read_text(encoding="utf-8")
    assert "cloudflare_pages_guard.py check" in deploy
    assert "cloudflare_pages_guard.py disable" in seed
