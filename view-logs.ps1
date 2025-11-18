# View Backend Logs
# This script displays the latest backend logs

param(
    [int]$Lines = 50,
    [switch]$Follow
)

$logFile = "logs\backend.log"

if (-not (Test-Path $logFile)) {
    Write-Host "Log file not found: $logFile" -ForegroundColor Red
    Write-Host "Make sure the backend is running and has created the log file." -ForegroundColor Yellow
    exit 1
}

if ($Follow) {
    Write-Host "Following log file: $logFile (Ctrl+C to stop)" -ForegroundColor Green
    Get-Content $logFile -Wait -Tail $Lines
} else {
    Write-Host "Displaying last $Lines lines from: $logFile" -ForegroundColor Green
    Get-Content $logFile -Tail $Lines
}
