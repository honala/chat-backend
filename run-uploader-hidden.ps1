# Honala photos uploader autostart script.
# Runs under Scheduled Task with hidden window and writes boot/crash logs.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$logPath = Join-Path $PSScriptRoot "uploader-autostart.log"
function Write-Log([string]$msg) {
  $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  Add-Content -LiteralPath $logPath -Value "[$ts] $msg"
}

$env:HONALA_UPLOAD_TOKEN_USER  = "4829572"
$env:HONALA_UPLOAD_TOKEN_ADMIN = "xxp071018@"

try {
  Write-Log "autostart begin"

  $candidates = @()
  $p3 = Get-Command python.exe -ErrorAction SilentlyContinue
  if ($p3) { $candidates += $p3.Source }
  $candidates += @(
    "C:\Users\XXPxx\AppData\Local\Programs\Python\Python312\python.exe",
    "C:\Users\XXPxx\AppData\Local\Programs\Python\Python311\python.exe",
    "C:\Users\XXPxx\AppData\Local\Programs\Python\Python310\python.exe"
  )

  $pythonExe = $null
  foreach ($c in $candidates) {
    if ($c -and (Test-Path -LiteralPath $c)) { $pythonExe = $c; break }
  }
  if (-not $pythonExe) {
    throw "python.exe not found in PATH/common install locations"
  }

  Write-Log "using python: $pythonExe"
  $p = Start-Process -FilePath $pythonExe `
    -ArgumentList @("-m","uvicorn","server:app","--host","127.0.0.1","--port","5055") `
    -WorkingDirectory $PSScriptRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $logPath `
    -RedirectStandardError $logPath `
    -PassThru

  Write-Log "uvicorn started pid=$($p.Id)"
  Wait-Process -Id $p.Id
  $p.Refresh()
  Write-Log "uvicorn exited code=$($p.ExitCode)"
  exit $p.ExitCode
}
catch {
  Write-Log ("autostart failed: " + $_.Exception.Message)
  exit 1
}
