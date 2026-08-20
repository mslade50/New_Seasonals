from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_broker_deploy_is_main_only_and_worker_defaults_are_disarmed():
    workflow = (ROOT / ".github/workflows/deploy_broker.yml").read_text(encoding="utf-8")
    config = (ROOT / "execution-broker/wrangler.toml").read_text(encoding="utf-8")

    guard = 'test "$GITHUB_REF" = "refs/heads/main"'
    assert guard in workflow
    assert workflow.index(guard) < workflow.index("wrangler deploy")
    assert '[vars]' in config
    assert 'EXEC_LIVE_ENABLED = "0"' in config
    assert 'EXEC_LIVE_TYPES = "NONE"' in config
    assert 'EXEC_LIVE_ACCOUNTS = "NONE"' in config
