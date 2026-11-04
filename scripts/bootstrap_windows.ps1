$ErrorActionPreference = 'Stop'
$TaskRoot = Split-Path -Parent $PSScriptRoot
$TaskTarget = 'windows-x86_64'
$TaskRuntime = Join-Path $TaskRoot '.runtime'
$TaskPrefix = Join-Path $TaskRuntime $TaskTarget
$TaskPython = Join-Path $TaskPrefix 'python\python.exe'

try {
    if (-not [Environment]::Is64BitOperatingSystem) { throw 'This project requires 64-bit Windows.' }
    $TaskLock = Get-Content -LiteralPath (Join-Path $TaskRoot 'configs\environment_lock.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $TaskSpec = $TaskLock.platforms.'windows-x86_64'.runtime
    $TaskOrigin = Join-Path $TaskPrefix 'runtime_archive.sha256'
    $TaskValidPython = (Test-Path -LiteralPath $TaskPython) -and (Test-Path -LiteralPath $TaskOrigin)
    if ($TaskValidPython) { $TaskValidPython = (Get-Content -LiteralPath $TaskOrigin -Raw).Trim() -eq $TaskSpec.sha256 }
    if (-not $TaskValidPython) {
        Write-Host 'First start: downloading the locked Python runtime from its official source...'
        $TaskCache = Join-Path $TaskRuntime 'downloads'
        New-Item -ItemType Directory -Force -Path $TaskCache | Out-Null
        $TaskArchive = Join-Path $TaskCache $TaskSpec.filename
        if ((-not (Test-Path -LiteralPath $TaskArchive)) -or ((Get-FileHash -LiteralPath $TaskArchive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $TaskSpec.sha256)) {
            [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
            $TaskPartial = $TaskArchive + '.part.' + $PID
            try {
                Invoke-WebRequest -Uri $TaskSpec.url -OutFile $TaskPartial -UseBasicParsing -TimeoutSec 120
                if ((Get-FileHash -LiteralPath $TaskPartial -Algorithm SHA256).Hash.ToLowerInvariant() -ne $TaskSpec.sha256) { throw 'Python archive checksum mismatch.' }
                Move-Item -LiteralPath $TaskPartial -Destination $TaskArchive -Force
            } finally {
                if (Test-Path -LiteralPath $TaskPartial) { Remove-Item -LiteralPath $TaskPartial }
            }
        }
        $TaskStaging = Join-Path $TaskRuntime ($TaskTarget + '.unpack.' + [Guid]::NewGuid().ToString('N'))
        New-Item -ItemType Directory -Path $TaskStaging | Out-Null
        & tar.exe -xzf $TaskArchive -C $TaskStaging
        if ($LASTEXITCODE -ne 0) { throw 'Python archive extraction failed.' }
        if (Test-Path -LiteralPath $TaskPrefix) {
            Move-Item -LiteralPath $TaskPrefix -Destination ($TaskPrefix + '.incomplete.' + [Guid]::NewGuid().ToString('N'))
        }
        Move-Item -LiteralPath $TaskStaging -Destination $TaskPrefix
        [IO.File]::WriteAllText($TaskOrigin, $TaskSpec.sha256 + [Environment]::NewLine)
    }
    & $TaskPython -I -B -X utf8 (Join-Path $TaskRoot 'scripts\bootstrap.py') --platform $TaskTarget
    if ($LASTEXITCODE -ne 0) { throw 'Project dependency setup failed. Check the preceding message and retry.' }
    exit 0
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 2
}
