param(
  [string]$TaskName = "Honala Photos Uploader"
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$runner = Join-Path $root "run-uploader-hidden.ps1"
$user = "$env:USERDOMAIN\$env:USERNAME"

if (-not (Test-Path -LiteralPath $runner)) {
  throw "Not found: $runner"
}

# Drop any previous registration (e.g. old cmd.exe / start-uploader.cmd task)
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

$arg = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$runner`""
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg

$trigger = New-ScheduledTaskTrigger -AtLogOn -User $user
$trigger.Delay = "PT20S"

$settings = New-ScheduledTaskSettingsSet `
  -ExecutionTimeLimit 0 `
  -StartWhenAvailable `
  -RestartCount 3 `
  -RestartInterval (New-TimeSpan -Minutes 1) `
  -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries

$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force

Start-ScheduledTask -TaskName $TaskName
Write-Host "Task '$TaskName' registered and started (hidden window, ~20s delay after logon)."
Write-Host "If port 5055 is already in use, stop the old uvicorn (Task Manager or the open cmd window) once."
