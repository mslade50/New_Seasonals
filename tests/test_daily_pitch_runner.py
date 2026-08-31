from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_batch_disables_internal_wait_ceiling_and_uses_outer_watchdog():
    text = (ROOT / "scripts" / "run_daily_pitch.bat").read_text(
        encoding="utf-8")
    assert "set CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=0" in text
    assert 'set "AGENT_TIMEOUT_SECONDS=6300"' in text
    assert "invoke_daily_pitch_agent.ps1" in text
    assert "--require-r2" in text


def test_batch_preserves_agent_exit_code_while_always_running_postflight():
    lines = (ROOT / "scripts" / "run_daily_pitch.bat").read_text(
        encoding="utf-8").splitlines()
    claude_rc = next(i for i, line in enumerate(lines)
                     if "set CLAUDE_RC=%ERRORLEVEL%" in line)
    postflight = next(i for i, line in enumerate(lines)
                      if "check_pitch_delivered.py" in line
                      and not line.lstrip().startswith("REM"))
    delivery_rc = next(i for i, line in enumerate(lines)
                       if "set DELIVERY_RC=%ERRORLEVEL%" in line)
    preserve = next(i for i, line in enumerate(lines)
                    if 'if not "%CLAUDE_RC%"=="0" set RC=%CLAUDE_RC%' in line)
    assert claude_rc < postflight < delivery_rc < preserve


def test_launcher_timeout_fits_inside_task_scheduler_cap():
    text = (ROOT / "scripts" / "invoke_daily_pitch_agent.ps1").read_text(
        encoding="utf-8")
    assert "[int]$TimeoutSeconds = 6300" in text
    assert "$process.WaitForExit($TimeoutSeconds * 1000)" in text
    assert "taskkill.exe" in text
    assert "@('/PID', $process.Id, '/T', '/F')" in text
    assert "$killProcess.WaitForExit(10000)" in text
    assert "$process.WaitForExit(10000)" in text
    assert "$process.WaitForExit()" not in text
    assert "exit $exitCode" in text


def test_launcher_replays_agent_logs_as_utf8():
    text = (ROOT / "scripts" / "invoke_daily_pitch_agent.ps1").read_text(
        encoding="utf-8")

    assert "Get-Content -LiteralPath $stdoutPath -Encoding UTF8" in text
    assert "Get-Content -LiteralPath $stderrPath -Encoding UTF8" in text
