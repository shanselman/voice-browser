# Voice Browser Setup (Windows, Playwright backend)
#
# Usage:
#   .\setup.ps1
#   .\setup.ps1 -SkipDoctor
#   .\setup.ps1 -FullDoctor

param(
    [switch]$SkipDoctor,
    [switch]$FullDoctor
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param(
        [int]$Index,
        [int]$Total,
        [string]$Message
    )
    Write-Host "[$Index/$Total] $Message" -ForegroundColor Yellow
}

function Ensure-Command {
    param(
        [string]$Name,
        [string]$InstallHint
    )
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        Write-Host "  ERROR: Missing command '$Name'." -ForegroundColor Red
        if ($InstallHint) {
            Write-Host "  Hint: $InstallHint" -ForegroundColor Yellow
        }
        exit 1
    }
}

function Get-PythonCommand {
    if (Get-Command python -ErrorAction SilentlyContinue) { return "python" }
    if (Get-Command py -ErrorAction SilentlyContinue) { return "py" }
    Write-Host "  ERROR: Python not found. Install from https://python.org" -ForegroundColor Red
    exit 1
}

function Find-EdgeExecutable {
    $cmd = Get-Command msedge -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $candidates = @(
        "C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    )
    foreach ($path in $candidates) {
        if (Test-Path $path) { return $path }
    }
    return $null
}

function Has-NvidiaGpu {
    try {
        $controllers = Get-CimInstance Win32_VideoController -ErrorAction Stop
        foreach ($controller in $controllers) {
            if ($controller.Name -match "NVIDIA") {
                return $true
            }
        }
    } catch {
    }
    return $false
}

$PythonCmd = Get-PythonCommand
$Root = $PSScriptRoot
$DoctorScript = Join-Path $Root "doctor.ps1"
$Requirements = Join-Path $Root "requirements.txt"

Write-Host ""
Write-Host "=== Voice Browser Setup ===" -ForegroundColor Cyan
Write-Host ""

Write-Step -Index 1 -Total 6 -Message "Checking Python"
& $PythonCmd --version | Out-Null

Write-Step -Index 2 -Total 6 -Message "Checking GitHub Copilot CLI"
Ensure-Command -Name "copilot" -InstallHint "Install GitHub Copilot CLI and sign in first."
copilot --version | Out-Null
Write-Host "  OK: Copilot CLI detected." -ForegroundColor Green

Write-Step -Index 3 -Total 6 -Message "Checking Microsoft Edge availability"
$edgePath = Find-EdgeExecutable
if (-not $edgePath) {
    Write-Host "  WARNING: Edge executable not found. You can still run Chromium/Chrome mode." -ForegroundColor Yellow
} else {
    Write-Host "  OK: Edge found at $edgePath" -ForegroundColor Green
}

Write-Step -Index 4 -Total 6 -Message "Installing Python dependencies"
& $PythonCmd -m pip install -r $Requirements
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: pip install failed." -ForegroundColor Red
    exit 1
}
Write-Host "  OK: Python dependencies installed." -ForegroundColor Green
Write-Host "  Installing optional local STT packages (faster-whisper, numpy)..." -ForegroundColor Cyan
& $PythonCmd -m pip install --disable-pip-version-check faster-whisper numpy
if ($LASTEXITCODE -ne 0) {
    Write-Host "  WARNING: Optional local STT install failed. Google STT will still work." -ForegroundColor Yellow
} else {
    Write-Host "  OK: Optional local STT packages installed." -ForegroundColor Green
}

Write-Step -Index 5 -Total 6 -Message "Verifying Copilot authentication"
$authCheck = @'
import asyncio
from copilot import CopilotClient

async def main():
    client = CopilotClient()
    await client.start()
    status = await client.get_auth_status()
    print(f"Authenticated={status.isAuthenticated}; User={status.login}")
    await client.stop()
    if not status.isAuthenticated:
        raise SystemExit(2)

asyncio.run(main())
'@
$authCheck | & $PythonCmd -
if ($LASTEXITCODE -eq 2) {
    Write-Host "  ERROR: Copilot is not authenticated." -ForegroundColor Red
    Write-Host "  Open a terminal and complete Copilot sign-in first, then rerun setup." -ForegroundColor Yellow
    exit 1
}
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: Copilot auth check failed." -ForegroundColor Red
    exit 1
}
Write-Host "  OK: Copilot authentication verified." -ForegroundColor Green

Write-Step -Index 6 -Total 6 -Message "Running health checks"
if ($SkipDoctor) {
    Write-Host "  Skipped health checks (--SkipDoctor)." -ForegroundColor DarkGray
} elseif (Test-Path $DoctorScript) {
    if ($FullDoctor) {
        Write-Host "  Running full doctor checks..." -ForegroundColor Cyan
        & $DoctorScript
    } else {
        Write-Host "  Running quick doctor checks..." -ForegroundColor Cyan
        & $DoctorScript -Quick
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ERROR: doctor checks failed." -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Green
Write-Host "Start with: .\start.ps1" -ForegroundColor White
if (Has-NvidiaGpu) {
    Write-Host "Local STT (GPU/CPU auto): .\start.ps1 -SttBackend faster-whisper -LocalSttDevice auto" -ForegroundColor White
} else {
    Write-Host "Local STT (laptop-safe CPU): .\start.ps1 -SttBackend faster-whisper -LocalSttDevice cpu -LocalSttComputeType int8" -ForegroundColor White
}
Write-Host ""
