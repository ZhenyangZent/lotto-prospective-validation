param(
    [Parameter(Mandatory = $true)]
    [string]$Gitleaks
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$config = Join-Path $root ".gitleaks.toml"
$historyReport = Join-Path $PSScriptRoot "gitleaks_history_output.json"
$worktreeReport = Join-Path $PSScriptRoot "gitleaks_worktree_output.json"
$negativeReport = Join-Path $PSScriptRoot "gitleaks_negative_tests.txt"

function Invoke-Gitleaks([string[]]$Arguments) {
    & $Gitleaks @Arguments
    return $LASTEXITCODE
}

Push-Location $root
try {
    $history = Invoke-Gitleaks @(
        "git", "--no-banner", "--config", $config, "--redact=100",
        "--report-format", "json", "--report-path", $historyReport, "."
    )
    if ($history -ne 0) { throw "repository history scan failed with exit code $history" }

    $worktree = Invoke-Gitleaks @(
        "dir", "--no-banner", "--config", $config, "--redact=100",
        "--report-format", "json", "--report-path", $worktreeReport, "."
    )
    if ($worktree -ne 0) { throw "repository worktree scan failed with exit code $worktree" }
} finally {
    Pop-Location
}

$temporary = Join-Path ([IO.Path]::GetTempPath()) ("prospective-gitleaks-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $temporary | Out-Null
try {
    $synthetic = ("A7zQ" + "9LmN" + "4RxT" + "8KvP" + "2WdY" + "6HsC")
    $hex39 = ("a1b2c3d4e5f60718293a4b5c6d7e8f901234567")
    $validOid = "c86591159f4d7cbca1f8765f7b06dea80b2f571a"
    $cases = @(
        [ordered]@{ name = "exact_public_commit_line_is_allowed"; path = "prospective_validation_v2/remote_verification.json"; content = "  `"github_api_commit_sha`": `"$validOid`",`n"; expected = 0 },
        [ordered]@{ name = "secret_in_same_file_is_detected"; path = "prospective_validation_v2/remote_verification.json"; content = "api_key = `"$synthetic`"`n"; expected = 1 },
        [ordered]@{ name = "secret_in_other_file_is_detected"; path = "security/neighbor.txt"; content = "api_key = `"$synthetic`"`n"; expected = 1 },
        [ordered]@{ name = "non_40_hex_is_not_allowed"; path = "prospective_validation_v2/remote_verification.json"; content = "  `"github_api_commit_sha`": `"$hex39`",`n"; expected = 1 },
        [ordered]@{ name = "other_high_entropy_field_is_not_allowed"; path = "prospective_validation_v2/remote_verification.json"; content = "  `"github_api_session_key`": `"$synthetic`",`n"; expected = 1 }
    )
    $lines = @()
    foreach ($case in $cases) {
        $caseRoot = Join-Path $temporary $case.name
        New-Item -ItemType Directory -Path $caseRoot | Out-Null
        $target = Join-Path $caseRoot $case.path
        New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
        [IO.File]::WriteAllText($target, $case.content, (New-Object Text.UTF8Encoding($false)))
        $caseReport = Join-Path $caseRoot "redacted.json"
        Push-Location $caseRoot
        try {
            $exitCode = Invoke-Gitleaks @(
                "dir", "--no-banner", "--config", $config, "--redact=100",
                "--report-format", "json", "--report-path", $caseReport, "."
            )
        } finally {
            Pop-Location
        }
        $passed = if ($case.expected -eq 0) { $exitCode -eq 0 } else { $exitCode -ne 0 }
        $lines += ("{0}: {1} (exit={2}, expected={3})" -f $case.name, $(if ($passed) { "PASS" } else { "FAIL" }), $exitCode, $case.expected)
        if (-not $passed) { throw "negative-test expectation failed: $($case.name)" }
    }
    $lines += "ALL_GITLEAKS_SCOPE_TESTS_PASSED"
    [IO.File]::WriteAllLines($negativeReport, $lines, (New-Object Text.UTF8Encoding($false)))
} finally {
    if (Test-Path -LiteralPath $temporary) {
        Remove-Item -LiteralPath $temporary -Recurse -Force
    }
}

[pscustomobject]@{
    history_exit_code = $history
    worktree_exit_code = $worktree
    negative_tests = "ALL_GITLEAKS_SCOPE_TESTS_PASSED"
} | ConvertTo-Json

exit 0
