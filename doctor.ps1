# Voice Browser diagnostics (Windows)

param(
    [switch]$Quick
)

$ErrorActionPreference = "Continue"
$script:FailCount = 0

function Invoke-Check {
    param(
        [string]$Name,
        [scriptblock]$Script
    )
    Write-Host "Checking: $Name" -ForegroundColor Yellow
    try {
        & $Script
        if ($LASTEXITCODE -ne 0) {
            throw "Exit code $LASTEXITCODE"
        }
        Write-Host "  OK" -ForegroundColor Green
    } catch {
        $script:FailCount += 1
        Write-Host "  FAIL: $($_.Exception.Message)" -ForegroundColor Red
    }
}

function Get-PythonCommand {
    if (Get-Command python -ErrorAction SilentlyContinue) { return "python" }
    if (Get-Command py -ErrorAction SilentlyContinue) { return "py" }
    return $null
}

$PythonCmd = Get-PythonCommand
if (-not $PythonCmd) {
    Write-Host "FAIL: Python not found." -ForegroundColor Red
    exit 1
}

Invoke-Check "Python runtime" {
    & $PythonCmd --version | Out-Null
}

Invoke-Check "Python packages (speech + copilot sdk + playwright)" {
    @'
import speech_recognition
import pyttsx3
import pyaudio
import copilot
import playwright
print("imports ok")
'@ | & $PythonCmd - | Out-Null
}

Invoke-Check "Microphone availability" {
    @'
import speech_recognition as sr
names = sr.Microphone.list_microphone_names()
if not names:
    raise SystemExit(2)
print(len(names))
'@ | & $PythonCmd - | Out-Null
}

Invoke-Check "Copilot auth status" {
    @'
import asyncio
from copilot import CopilotClient

async def main():
    client = CopilotClient()
    await client.start()
    status = await client.get_auth_status()
    await client.stop()
    if not status.isAuthenticated:
        raise SystemExit(3)
    print(status.login)

asyncio.run(main())
'@ | & $PythonCmd - | Out-Null
}

Invoke-Check "Edge executable (recommended)" {
    $edgePath = $null
    if (Get-Command msedge -ErrorAction SilentlyContinue) {
        $edgePath = (Get-Command msedge).Source
    } elseif (Test-Path "C:\Program Files\Microsoft\Edge\Application\msedge.exe") {
        $edgePath = "C:\Program Files\Microsoft\Edge\Application\msedge.exe"
    } elseif (Test-Path "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe") {
        $edgePath = "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    }
    if (-not $edgePath) {
        throw "Edge executable not found."
    }
    Write-Host "  Edge: $edgePath" -ForegroundColor DarkGray
}

if (-not $Quick) {
    Invoke-Check "Playwright open/get/close flow" {
        @'
from playwright.sync_api import sync_playwright

p = sync_playwright().start()
browser = p.chromium.launch(channel="msedge", headless=False)
ctx = browser.new_context(no_viewport=True)
page = ctx.new_page()
page.goto("https://example.com", wait_until="domcontentloaded", timeout=30000)
title = page.title()
ctx.close()
browser.close()
p.stop()
if not title:
    raise SystemExit(5)
print(title)
'@ | & $PythonCmd - | Out-Null
    }
}

if ($script:FailCount -gt 0) {
    Write-Host ""
    Write-Host "Doctor found $script:FailCount issue(s)." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Doctor checks passed." -ForegroundColor Green
exit 0
