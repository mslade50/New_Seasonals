'''Exercise the scheduled launcher's actual step 6 with an isolated broker stub.'''
import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
POWERSHELL = Path(r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe")
pytestmark = pytest.mark.skipif(not POWERSHELL.is_file(), reason="Windows PowerShell required")


def report(*failures, retryable=True, orders=0):
    return dict(ok=not failures, failures=list(failures), recovery_retryable=retryable,
                orders_placed=orders)


def run_step(tmp_path, rows, *, now="2026-10-09T08:12:00", sleep_jump=0, existing=False):
    source = ROOT / "scripts/run_open_breakout_daily.ps1"
    if not source.exists():  # Reproduce the installed one-shot launcher before the fix.
        source = ROOT / "artifacts/open_breakout_runs/daily_launch.ps1"
    text = source.read_text(encoding="utf-8-sig")
    step = text.split("    # 6. Read-only preflight", 1)[1].split("\n", 1)[1].split("    # 7. Launch", 1)[0]
    (tmp_path / "scripts").mkdir()
    helper = ROOT / "scripts/open_breakout_startup_preflight.ps1"
    if helper.exists():
        (tmp_path / "scripts" / helper.name).write_bytes(helper.read_bytes())
    runs = tmp_path / "artifacts/open_breakout_runs"
    runs.mkdir(parents=True)
    prior = runs / "preflight-2026-10-09-live.json"
    if existing:
        prior.write_text("retained evidence", encoding="utf-8")
    (tmp_path / "rows.json").write_text(json.dumps(rows), encoding="utf-8")
    quote = lambda value: str(value).replace("'", "''")
    script = r'''
$ErrorActionPreference = 'Stop'
$Repo = '__REPO__'
$Python = 'BROKER_STUB_ONLY'
$RunsRel = 'artifacts/open_breakout_runs'
$Runs = Join-Path $Repo $RunsRel
$LiveConfig = 'config.json'
$Session = '2026-10-09'
$PreflightClient = 927485
$account = 'TEST_ACCOUNT'
$DryRun = $false
$LateCutoff = New-TimeSpan -Hours 9 -Minutes 20
$script:Now = [datetime]'__NOW__'
$script:Rows = Get-Content (Join-Path $Repo 'rows.json') -Raw | ConvertFrom-Json
$script:Calls = New-Object Collections.ArrayList
$script:Delays = New-Object Collections.ArrayList
function Get-NyNow { return $script:Now }
function Write-Log([string]$Message) { }
function Stop-Launch([int]$Code, [string]$Reason) { throw "STOP:$Code $Reason" }
function Start-Sleep([int]$Seconds) {
    [void]$script:Delays.Add($Seconds)
    $script:Now = $script:Now.AddSeconds($Seconds + __SLEEP_JUMP__)
}
function Add-ChildOutput([string]$Tag, [string]$Text) { }
function Invoke-Child([string]$Tag, [string]$File, [string[]]$ArgList, [int]$TimeoutSec,
                      [hashtable]$ChildEnv=@{}, [switch]$QuietStdout) {
    if ($File -ne 'BROKER_STUB_ONLY' -or $ArgList[2] -ne 'preflight') { throw 'unexpected command' }
    $row = $script:Rows[$script:Calls.Count]
    $outIndex = [array]::IndexOf($ArgList, '--out') + 1
    $outPath = Join-Path $Repo $ArgList[$outIndex]
    [void]$script:Calls.Add(@{args=$ArgList; path=$outPath; timeout=$TimeoutSec; ack=$ChildEnv.OPEN_BREAKOUT_LIVE_ACK})
    if ($null -ne $row.report) { $row.report | ConvertTo-Json -Depth 20 | Set-Content $outPath }
    if ($null -ne $row.raw) { $row.raw | Set-Content $outPath }
    if ($row.seconds) { $script:Now = $script:Now.AddSeconds($row.seconds) }
    return [pscustomobject]@{Code=$row.code; TimedOut=($row.timed_out -eq $true); Stdout='stub'}
}
$status = 'OK'
try {
__STEP__
} catch { $status = $_.Exception.Message }
@{status=$status; calls=@($script:Calls.ToArray()); delays=@($script:Delays.ToArray());
  files=@(Get-ChildItem $Runs -File | ForEach-Object { $_.Name })} | ConvertTo-Json -Depth 20 -Compress
'''
    script = (script.replace("__REPO__", quote(tmp_path)).replace("__NOW__", now)
              .replace("__SLEEP_JUMP__", str(sleep_jump)).replace("__STEP__", step))
    harness = tmp_path / "harness.ps1"
    harness.write_text(script, encoding="utf-8-sig")
    result = subprocess.run([str(POWERSHELL), "-NoProfile", "-NonInteractive", "-ExecutionPolicy",
                             "Bypass", "-File", str(harness)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    decoded = json.loads(result.stdout)
    if existing:
        assert prior.read_text(encoding="utf-8") == "retained evidence"
    return decoded


def row(value, **kwargs):
    return dict(report=value, code=0, **kwargs)


def test_changed_recovery_proof_retries_fresh_child_and_preserves_reports(tmp_path):
    result = run_step(tmp_path, [row(report("RECOVERY_EVIDENCE_CHANGED")), row(report())], existing=True)
    assert result["status"] == "OK", result
    assert len(result["calls"]) == 2
    assert result["delays"] == [2]
    assert len({c["path"] for c in result["calls"]}) == 2
    assert all(c["ack"] == "LIVE 2026-10-09 TEST_ACCOUNT" for c in result["calls"])


@pytest.mark.parametrize("failure", ["RECOVERY_INCOMPLETE:timeout", "STREAM_STALE:ES", "STREAM_MISSING:NQ", "TRANSPORT_UNHEALTHY"])
def test_retryable_transport_failures_recover(tmp_path, failure):
    result = run_step(tmp_path, [row(report(failure)), row(report())])
    assert result["status"] == "OK"
    assert len(result["calls"]) == 2


@pytest.mark.parametrize("value", [
    report("RECOVERY_EVIDENCE_CHANGED", retryable=False),
    report("RECOVERY_EVIDENCE_CHANGED", "OWN_WORKING_ORDERS"),
    report("MARGIN_EXCEEDS_LIMIT"), report("MARGIN:MNQ:BUY:-4344.66"), report("RECOVERY_EVIDENCE_CHANGED_OTHER"),
    report("ACCOUNT_MISMATCH"), report("STREAM_STALE:ES", orders=1),
    dict(ok=False, failures=[], recovery_retryable=True, orders_placed=0),
])
def test_business_and_ambiguous_failures_stop_without_retry(tmp_path, value):
    result = run_step(tmp_path, [row(value)])
    assert result["status"].startswith("STOP:5"), result
    assert len(result["calls"]) == 1
    assert result["delays"] == []


def test_retry_budget_is_three_attempts(tmp_path):
    result = run_step(tmp_path, [row(report("RECOVERY_EVIDENCE_CHANGED"))] * 3)
    assert result["status"].startswith("STOP:5"), result
    assert len(result["calls"]) == 3
    assert result["delays"] == [2, 5]


@pytest.mark.parametrize("entry", [dict(code=1, report=report()), dict(code=0, timed_out=True, report=report()),
                                   dict(code=0), dict(code=0, raw="invalid json")])
def test_child_or_report_failure_never_authorizes_launch(tmp_path, entry):
    result = run_step(tmp_path, [entry])
    assert result["status"].startswith("STOP:5"), result
    assert len(result["calls"]) == 1


def test_does_not_start_preflight_at_cutoff(tmp_path):
    result = run_step(tmp_path, [], now="2026-10-09T09:20:00")
    assert result["status"].startswith("STOP:7"), result
    assert result["calls"] == []


def test_retry_delay_cannot_cross_cutoff(tmp_path):
    result = run_step(tmp_path, [row(report("RECOVERY_EVIDENCE_CHANGED"))], now="2026-10-09T09:19:59")
    assert result["status"].startswith("STOP:7"), result
    assert len(result["calls"]) == 1
    assert result["calls"][0]["timeout"] == 1
    assert result["delays"] == []


def test_clock_jump_or_date_rollover_does_not_retry(tmp_path):
    result = run_step(tmp_path, [row(report("RECOVERY_EVIDENCE_CHANGED"))], sleep_jump=86400)
    assert result["status"].startswith("STOP:7"), result
    assert len(result["calls"]) == 1


def test_success_after_cutoff_cannot_authorize_launch(tmp_path):
    result = run_step(tmp_path, [row(report(), seconds=3)], now="2026-10-09T09:19:59")
    assert result["status"].startswith("STOP:7"), result
    assert len(result["calls"]) == 1


def test_inconsistent_success_report_is_rejected(tmp_path):
    value = report("RECOVERY_EVIDENCE_CHANGED")
    value["ok"] = True
    result = run_step(tmp_path, [row(value)])
    assert result["status"].startswith("STOP:5"), result
