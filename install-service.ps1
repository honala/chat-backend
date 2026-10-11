$ErrorActionPreference = "Stop"

$serviceName = "HonalaPhotosUploader"
$displayName = "Honala Photos Uploader"
$taskName = "Honala Photos Uploader"
$root = "C:\Users\XXPxx\Desktop\nginx-1.29.5\photos-uploader"
$nssm = "C:\Users\XXPxx\AppData\Local\Microsoft\WinGet\Packages\NSSM.NSSM_Microsoft.Winget.Source_8wekyb3d8bbwe\nssm-2.24-101-g897c7ad\win64\nssm.exe"
$pythonCandidates = @(
  "C:\Users\XXPxx\AppData\Local\Programs\Python\Python312\python.exe",
  "C:\Users\XXPxx\AppData\Local\Programs\Python\Python311\python.exe",
  "C:\Users\XXPxx\AppData\Local\Programs\Python\Python310\python.exe"
)
$uploadToken = "4829572"
$adminToken = "xxp071018@"
$stdoutLog = Join-Path $root "service-stdout.log"
$stderrLog = Join-Path $root "service-stderr.log"

if (-not (Test-Path -LiteralPath $nssm)) {
  throw "nssm.exe not found: $nssm"
}

$pythonExe = $null
foreach ($candidate in $pythonCandidates) {
  if (Test-Path -LiteralPath $candidate) {
    $pythonExe = $candidate
    break
  }
}
if (-not $pythonExe) {
  $py = Get-Command python.exe -ErrorAction SilentlyContinue
  if ($py -and (Test-Path -LiteralPath $py.Source)) {
    $pythonExe = $py.Source
  }
}
if (-not $pythonExe) {
  throw "python.exe not found"
}

Write-Host "Using python: $pythonExe"

$existing = Get-Service -Name $serviceName -ErrorAction SilentlyContinue
if ($existing) {
  Write-Host "Stopping existing service $serviceName"
  Stop-Service -Name $serviceName -Force -ErrorAction SilentlyContinue
  & $nssm remove $serviceName confirm
}

$portProc = Get-NetTCPConnection -LocalPort 5055 -ErrorAction SilentlyContinue |
  Select-Object -ExpandProperty OwningProcess -Unique
if ($portProc) {
  foreach ($procId in $portProc) {
    if ($procId -and $procId -ne 0) {
      Write-Host "Stopping process on port 5055: PID $procId"
      Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
    }
  }
  Start-Sleep -Seconds 1
}

Write-Host "Installing service $serviceName"
& $nssm install $serviceName $pythonExe "-m uvicorn server:app --host 127.0.0.1 --port 5055"
& $nssm set $serviceName DisplayName $displayName
& $nssm set $serviceName Description "Honala photos uploader FastAPI service"
& $nssm set $serviceName AppDirectory $root
& $nssm set $serviceName Start SERVICE_AUTO_START
& $nssm set $serviceName AppStdout $stdoutLog
& $nssm set $serviceName AppStderr $stderrLog
& $nssm set $serviceName AppRotateFiles 1
& $nssm set $serviceName AppRotateOnline 1
& $nssm set $serviceName AppExit Default Restart
& $nssm set $serviceName AppEnvironmentExtra "HONALA_UPLOAD_TOKEN_USER=$uploadToken" "HONALA_UPLOAD_TOKEN_ADMIN=$adminToken"

Write-Host "Starting service $serviceName"
Start-Service -Name $serviceName
Start-Sleep -Seconds 2

$service = Get-Service -Name $serviceName -ErrorAction Stop
if ($service.Status -ne "Running") {
  throw "service did not reach Running state"
}

if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
  Write-Host "Deleting old scheduled task: $taskName"
  Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}

Write-Host ""
Write-Host "Done."
Write-Host "Service: $serviceName"
Write-Host "Status : $($service.Status)"
