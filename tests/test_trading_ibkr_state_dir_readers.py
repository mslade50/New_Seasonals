"""New_Seasonals readers of trading_ibkr state/secrets follow runtime_paths.

trading_ibkr resolves state via TRADING_IBKR_STATE_DIR and secrets via
TRADING_IBKR_SECRETS_DIR, each falling back to the code dir when unset. Every
reader here must be inert with the vars unset (today's OneDrive paths) and
follow them when set. Nothing here connects, uploads, or registers tasks.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import daily_pitch as dp  # noqa: E402
from scripts import automation_supervisor as sup  # noqa: E402
from scripts import expected_exit_task as eet  # noqa: E402

_spec = importlib.util.spec_from_file_location("publish_sleeve_runtime_status", ROOT / "scripts/publish_sleeve_runtime_status.py")
pub = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pub)

POWERSHELL = Path(r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe")
CMD = Path(os.environ.get("COMSPEC", r"C:\Windows\System32\cmd.exe"))
STATE_VAR = "TRADING_IBKR_STATE_DIR"
SECRETS_VAR = "TRADING_IBKR_SECRETS_DIR"


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")
    return path


def _child_env(**values: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in (STATE_VAR, SECRETS_VAR)}
    env.update(values)
    return env


# --- daily_pitch.credentials_path -------------------------------------------

@pytest.fixture
def pitch_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("GCP_CREDENTIALS_FILE", raising=False)
    monkeypatch.delenv(SECRETS_VAR, raising=False)
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    monkeypatch.setattr(dp, "ROOT", tmp_path / "repo")
    return tmp_path


def test_pitch_credentials_unset_is_onedrive(pitch_home: Path) -> None:
    onedrive = _touch(pitch_home / "home/OneDrive/trading_ibkr/credentials.json")
    assert dp.credentials_path() == onedrive


def test_pitch_credentials_follow_secrets_dir(pitch_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _touch(pitch_home / "home/OneDrive/trading_ibkr/credentials.json")
    moved = _touch(pitch_home / "secrets/credentials.json")
    monkeypatch.setenv(SECRETS_VAR, str(pitch_home / "secrets"))
    assert dp.credentials_path() == moved


def test_pitch_credentials_set_but_missing_never_falls_back_to_onedrive(
        pitch_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _touch(pitch_home / "home/OneDrive/trading_ibkr/credentials.json")
    monkeypatch.setenv(SECRETS_VAR, str(pitch_home / "empty"))
    assert dp.credentials_path() is None


# --- automation_supervisor.resolve_external_secret_paths (Local v9) -----------

def test_supervisor_secrets_default_unset_is_onedrive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    gcp, execution = sup.resolve_external_secret_paths(
        config_root=tmp_path, gcp_json_path=None, exec_env_path=None, base_env={})
    base = (tmp_path / "home/OneDrive/trading_ibkr").resolve()
    assert (gcp, execution) == (base / "credentials.json", base / "exec_agent.env")


def test_supervisor_secrets_default_follows_secrets_dir(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("", encoding="utf-8")
    gcp, execution = sup.resolve_external_secret_paths(
        config_root=tmp_path, gcp_json_path=None, exec_env_path=None,
        base_env={SECRETS_VAR: str(tmp_path / "secrets")})
    assert gcp == (tmp_path / "secrets/credentials.json").resolve()
    assert execution == (tmp_path / "secrets/exec_agent.env").resolve()


def test_supervisor_dotenv_keys_still_win(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        f"LOCAL_AUTOMATION_GCP_JSON_PATH={tmp_path / 'a.json'}\n"
        f"LOCAL_AUTOMATION_EXEC_ENV_PATH={tmp_path / 'b.env'}\n", encoding="utf-8")
    gcp, execution = sup.resolve_external_secret_paths(
        config_root=tmp_path, gcp_json_path=None, exec_env_path=None,
        base_env={SECRETS_VAR: str(tmp_path / "secrets")})
    assert (gcp, execution) == ((tmp_path / "a.json").resolve(), (tmp_path / "b.env").resolve())


# --- publish_sleeve_runtime_status --state-dir --------------------------------

def test_sleeve_state_dir_precedence(tmp_path: Path) -> None:
    code = tmp_path / "code"
    assert pub.resolve_state_dir(code, None, {}) == code
    assert pub.resolve_state_dir(code, None, {STATE_VAR: ""}) == code
    assert pub.resolve_state_dir(code, None, {STATE_VAR: str(tmp_path / "st")}) == tmp_path / "st"
    assert pub.resolve_state_dir(code, tmp_path / "cli", {STATE_VAR: str(tmp_path / "st")}) == tmp_path / "cli"


def _fake_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pub.sys, "platform", "win32")
    monkeypatch.setattr(pub.subprocess, "CREATE_NO_WINDOW", 0, raising=False)
    monkeypatch.setattr(pub.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="[]", stderr=""))


def test_sleeve_flags_read_from_state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_inventory(monkeypatch)
    code, state = tmp_path / "code", tmp_path / "state"
    _touch(code / "event_moo_enabled.flag")  # stale copy next to the code: must be ignored
    _touch(state / "legend_ema_enabled.flag")
    payload = pub.collect(code, tmp_path / "runs", state)
    assert payload["event_enabled"] is False
    assert payload["legend_enabled"] is True
    unset = pub.collect(code, tmp_path / "runs")
    assert unset["event_enabled"] is True and unset["legend_enabled"] is False


def test_sleeve_missing_state_dir_refuses_to_publish(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_inventory(monkeypatch)
    (tmp_path / "code").mkdir()
    with pytest.raises(RuntimeError, match="state directory"):
        pub.collect(tmp_path / "code", tmp_path / "runs", tmp_path / "absent")


# --- Expected Exit Monitor -ExecEnv -----------------------------------------------

def test_expected_exit_task_exec_env_default() -> None:
    assert eet.default_exec_env({"USERPROFILE": r"C:\Users\X"}) == str(
        Path(r"C:\Users\X") / "OneDrive" / "trading_ibkr" / "exec_agent.env")
    assert eet.default_exec_env({"USERPROFILE": r"C:\Users\X", SECRETS_VAR: r"C:\trading_state\secrets"}) == str(
        Path(r"C:\trading_state\secrets") / "exec_agent.env")


def _ps_exec_env_default(env: dict[str, str]) -> str:
    script = ROOT / "scripts" / "run_expected_exit_monitor.ps1"
    command = (
        f"$ast = [System.Management.Automation.Language.Parser]::ParseFile('{script}', [ref]$null, [ref]$null); "
        "$p = $ast.ParamBlock.Parameters | Where-Object { $_.Name.VariablePath.UserPath -eq 'ExecEnv' }; "
        "Invoke-Expression $p.DefaultValue.Extent.Text"
    )
    result = subprocess.run([str(POWERSHELL), "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command],
                            capture_output=True, text=True, env=env, check=True, timeout=60)
    return result.stdout.strip()


@pytest.mark.skipif(not POWERSHELL.is_file(), reason="requires Windows PowerShell")
def test_expected_exit_ps1_exec_env_default(tmp_path: Path) -> None:
    unset = _ps_exec_env_default(_child_env(USERPROFILE=r"C:\Users\X"))
    assert unset == r"C:\Users\X\OneDrive\trading_ibkr\exec_agent.env"
    moved = _ps_exec_env_default(_child_env(USERPROFILE=r"C:\Users\X", **{SECRETS_VAR: r"C:\trading_state\secrets"}))
    assert moved == r"C:\trading_state\secrets\exec_agent.env"


# --- scripts/run_radar_sync.bat flag path ---------------------------------------

RADAR_BAT = ROOT / "scripts" / "run_radar_sync.bat"


def test_radar_bat_is_crlf_and_reads_flag_from_state() -> None:
    raw = RADAR_BAT.read_bytes()
    assert raw.count(b"\n") == raw.count(b"\r\n")
    text = raw.decode("utf-8")
    assert r'if exist "%IBKR_STATE%\radar_trail_enabled.flag"' in text
    assert r'"%IBKR%\radar_trail_sync.py"' in text  # code path is Phase 3, unchanged


@pytest.mark.skipif(sys.platform != "win32" or not CMD.is_file(), reason="requires cmd.exe")
def test_radar_bat_flag_dir_resolution(tmp_path: Path) -> None:
    keep = ("set IBKR=", 'set "IBKR_STATE=', "if defined TRADING_IBKR_STATE_DIR")
    lines = [line for line in RADAR_BAT.read_text(encoding="utf-8").splitlines() if line.startswith(keep)]
    assert len(lines) == 3
    probe = tmp_path / "probe.bat"
    probe.write_bytes(("@echo off\r\n" + "\r\n".join(lines) + "\r\necho %IBKR_STATE%\r\n").encode())

    def run(env: dict[str, str]) -> str:
        return subprocess.run([str(CMD), "/c", str(probe)], capture_output=True, text=True,
                              env=env, check=True, timeout=30).stdout.strip()

    assert run(_child_env(USERPROFILE=r"C:\Users\X")) == r"C:\Users\X\OneDrive\trading_ibkr"
    assert run(_child_env(USERPROFILE=r"C:\Users\X", **{STATE_VAR: r"C:\trading_state\trading_ibkr"})) \
        == r"C:\trading_state\trading_ibkr"
