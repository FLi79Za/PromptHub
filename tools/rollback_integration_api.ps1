param(
    [string]$ProjectPath = 'A:\AI\Prompt library\Plib',
    [string]$OriginalBackupPath = 'A:\AI\Prompt library\Plib_Backups\Plib_PreIntegrationAPI_20260801_082317'
)

$ErrorActionPreference = 'Stop'
$project = (Resolve-Path -LiteralPath $ProjectPath).Path.TrimEnd('\')
$originalBackup = (Resolve-Path -LiteralPath $OriginalBackupPath).Path.TrimEnd('\')
if ($project -ne 'A:\AI\Prompt library\Plib') {
    throw "Refusing unexpected project path: $project"
}
if ($originalBackup -ne 'A:\AI\Prompt library\Plib_Backups\Plib_PreIntegrationAPI_20260801_082317') {
    throw "Refusing unexpected original backup path: $originalBackup"
}

$knownPorts = @(8080, 5000, 5173, 3000)
$listeners = Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object {
    $_.LocalPort -in $knownPorts -and $_.LocalAddress -in @('127.0.0.1', '::1', '0.0.0.0', '::')
}
foreach ($listener in $listeners) {
    $process = Get-Process -Id $listener.OwningProcess -ErrorAction SilentlyContinue
    if ($process -and $process.ProcessName -match 'python|flask') {
        throw "PromptHub may be active on port $($listener.LocalPort) (PID $($listener.OwningProcess)). Stop it before rollback."
    }
}

Write-Host "PromptHub Integration API rollback"
Write-Host "Restore source: $originalBackup"
Write-Host "Restore destination: $project"
Write-Host "A safety backup of the current changed state will be created first."
$confirmation = Read-Host "Type RESTORE to continue"
if ($confirmation -cne 'RESTORE') {
    Write-Host "Rollback cancelled."
    exit 1
}

$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$backupRoot = Split-Path -Parent $originalBackup
$safetyBackup = Join-Path $backupRoot "Plib_PreRollbackSafety_$timestamp"
$logPath = Join-Path $backupRoot "rollback_integration_api_$timestamp.log"
New-Item -ItemType Directory -Path $safetyBackup -Force | Out-Null
Start-Transcript -LiteralPath $logPath -Force | Out-Null

try {
    $excluded = @('.git', 'env', '.venv', 'venv', '__pycache__', '.pytest_cache', 'node_modules', 'build', 'dist', 'temp')
    $currentFiles = Get-ChildItem -LiteralPath $project -Recurse -Force -File | Where-Object {
        $relative = $_.FullName.Substring($project.Length).TrimStart('\')
        $parts = $relative -split '[\\/]'
        -not ($parts | Where-Object { $_ -in $excluded }) -and $_.Name -notin @('prompts.db', 'prompts.db-wal', 'prompts.db-shm')
    }
    foreach ($file in $currentFiles) {
        $relative = $file.FullName.Substring($project.Length).TrimStart('\')
        $destination = Join-Path $safetyBackup $relative
        New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
        Copy-Item -LiteralPath $file.FullName -Destination $destination -Force
    }

    $python = Join-Path $project 'env\Scripts\python.exe'
    $sqliteBackup = Join-Path $project 'tools\sqlite_backup.py'
    & $python $sqliteBackup (Join-Path $project 'prompts.db') (Join-Path $safetyBackup 'prompts.db')
    if ($LASTEXITCODE -ne 0) {
        throw "Safety database backup failed with exit code $LASTEXITCODE"
    }
    $safetyChecksumLines = foreach ($file in Get-ChildItem -LiteralPath $safetyBackup -Recurse -Force -File | Sort-Object FullName) {
        $relative = $file.FullName.Substring($safetyBackup.Length).TrimStart('\')
        $hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        "$hash  $relative"
    }
    $safetyChecksumLines | Set-Content -LiteralPath (Join-Path $safetyBackup 'SHA256SUMS.txt') -Encoding UTF8
    @(
        '# PromptHub Pre-Rollback Safety Backup'
        ''
        "- Original project: $project"
        "- Safety backup: $safetyBackup"
        "- Created: $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ssK')"
        '- Database method: Python sqlite3 online backup API'
        '- Database integrity check: ok'
        '- Excluded: .git, virtual environments, caches, dependency/build output, and temp'
    ) | Set-Content -LiteralPath (Join-Path $safetyBackup 'SAFETY_BACKUP_MANIFEST.md') -Encoding UTF8

    $integrationAddedFiles = @(
        'integration_api.py',
        'integration_config.py',
        'prompt_service.py',
        'ROLLBACK_INTEGRATION_API.md',
        'CHANGELOG_INTEGRATION_API.md',
        'docs\INTEGRATION_API_V1.md',
        'tests\__init__.py',
        'tests\test_integration_api.py',
        'tools\manage_integration_api.py',
        'tools\migrate_integration_api.py',
        'tools\rollback_integration_api.ps1',
        'tools\sqlite_backup.py'
    )
    foreach ($relative in $integrationAddedFiles) {
        $target = Join-Path $project $relative
        if (Test-Path -LiteralPath $target -PathType Leaf) {
            Remove-Item -LiteralPath $target -Force
        }
    }

    $restoreFiles = Get-ChildItem -LiteralPath $originalBackup -Recurse -Force -File | Where-Object {
        $_.Name -notin @('BACKUP_MANIFEST.md', 'SHA256SUMS.txt', 'prompts.db', 'prompts.db-wal', 'prompts.db-shm')
    }
    foreach ($file in $restoreFiles) {
        $relative = $file.FullName.Substring($originalBackup.Length).TrimStart('\')
        $destination = Join-Path $project $relative
        New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
        Copy-Item -LiteralPath $file.FullName -Destination $destination -Force
    }

    foreach ($sidecarName in @('prompts.db-wal', 'prompts.db-shm')) {
        $sidecar = Join-Path $project $sidecarName
        if (Test-Path -LiteralPath $sidecar) {
            Remove-Item -LiteralPath $sidecar -Force
        }
    }
    Copy-Item -LiteralPath (Join-Path $originalBackup 'prompts.db') -Destination (Join-Path $project 'prompts.db') -Force

    $failures = @()
    Get-Content -LiteralPath (Join-Path $originalBackup 'SHA256SUMS.txt') | ForEach-Object {
        if ($_ -match '^([0-9a-f]{64})  (.+)$') {
            $expected = $Matches[1]
            $relative = $Matches[2]
            if ($relative -in @('prompts.db-wal', 'prompts.db-shm')) {
                return
            }
            $restored = Join-Path $project $relative
            if (-not (Test-Path -LiteralPath $restored)) {
                $failures += "$relative (missing)"
            } else {
                $actual = (Get-FileHash -LiteralPath $restored -Algorithm SHA256).Hash.ToLowerInvariant()
                if ($actual -ne $expected) {
                    $failures += "$relative (checksum mismatch)"
                }
            }
        }
    }
    if ($failures.Count -gt 0) {
        throw "Rollback verification failed: $($failures -join ', ')"
    }

    Write-Host "Rollback completed and checksums verified."
    Write-Host "Safety backup: $safetyBackup"
    Write-Host "Rollback log: $logPath"
} finally {
    Stop-Transcript | Out-Null
}
