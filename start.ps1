# Voice Browser launcher (Windows)
#
# Usage:
#   .\start.ps1
#   .\start.ps1 https://example.com
#   .\start.ps1 -SkipDoctor
#   .\start.ps1 -UseChrome
#   .\start.ps1 -BrowserExecutablePath "C:\Path\To\browser.exe"
#   .\start.ps1 -MicrophoneIndex 1
#   .\start.ps1 -ListAllMics
#   .\start.ps1 -TtsBackend windows
#   .\start.ps1 -EdgeProfile "Profile 1"
#   .\start.ps1 -EdgeUserDataDir "C:\Users\you\AppData\Local\Microsoft\Edge\User Data"
#   .\start.ps1 -Force
#   .\start.ps1 -NoMiniUi
#   .\start.ps1 -NoTts
#   .\start.ps1 -DisableBargeIn
#   .\start.ps1 -DisableEchoGuard
#   .\start.ps1 -SttBackend faster-whisper -LocalSttModel base.en
#   .\start.ps1 -PhraseLimit 40 -PauseThreshold 1.6
#   .\start.ps1 -SttDebug

param(
    [switch]$SkipDoctor,
    [switch]$UseChrome,
    [Parameter(Position = 0)]
    [string]$Url,
    [string]$BrowserExecutablePath,
    [string]$EdgeProfile,
    [string]$EdgeUserDataDir,
    [switch]$Force,
    [switch]$NoMiniUi,
    [switch]$NoTts,
    [switch]$DisableBargeIn,
    [switch]$DisableEchoGuard,
    [int]$MicrophoneIndex = -1,
    [switch]$ListAllMics,
    [ValidateSet("auto", "google", "faster-whisper")]
    [string]$SttBackend = "",
    [string]$LocalSttModel = "",
    [string]$LocalSttDevice = "",
    [string]$LocalSttComputeType = "",
    [switch]$SttDebug,
    [int]$ListenTimeout = -1,
    [int]$PhraseLimit = -1,
    [double]$PauseThreshold = -1,
    [double]$NonSpeakingDuration = -1,
    [double]$PhraseThreshold = -1,
    [ValidateSet("windows", "pyttsx3", "auto")]
    [string]$TtsBackend = ""
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$DoctorScript = Join-Path $Root "doctor.ps1"
$MainScript = Join-Path $Root "voice-browser.py"

function Get-PythonCommand {
    if (Get-Command python -ErrorAction SilentlyContinue) { return "python" }
    if (Get-Command py -ErrorAction SilentlyContinue) { return "py" }
    Write-Host "ERROR: Python not found." -ForegroundColor Red
    exit 1
}

$PythonCmd = Get-PythonCommand

if (-not (Test-Path $MainScript)) {
    Write-Host "ERROR: voice-browser.py not found in $Root" -ForegroundColor Red
    exit 1
}

if (-not $SkipDoctor -and (Test-Path $DoctorScript)) {
    Write-Host "Running quick health checks..." -ForegroundColor Yellow
    & $DoctorScript -Quick
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: health checks failed. Fix issues or run with -SkipDoctor." -ForegroundColor Red
        exit 1
    }
}

if ($UseChrome) {
    $env:VOICE_BROWSER_BROWSER = "chromium"
} elseif (-not $env:VOICE_BROWSER_BROWSER) {
    $env:VOICE_BROWSER_BROWSER = "edge"
}

if ($BrowserExecutablePath) {
    $env:VOICE_BROWSER_EXECUTABLE_PATH = $BrowserExecutablePath
}

if ($EdgeProfile) {
    $env:VOICE_BROWSER_EDGE_PROFILE = $EdgeProfile
}

if ($EdgeUserDataDir) {
    $env:VOICE_BROWSER_EDGE_USER_DATA_DIR = $EdgeUserDataDir
}

if ($Force) {
    $env:VOICE_BROWSER_FORCE_EDGE_PROFILE = "1"
}

if ($NoMiniUi) {
    $env:VOICE_BROWSER_MINI_UI = "0"
}

if ($NoTts) {
    $env:VOICE_BROWSER_TTS_ENABLED = "0"
}

if ($DisableBargeIn) {
    $env:VOICE_BROWSER_BARGE_IN = "0"
}

if ($DisableEchoGuard) {
    $env:VOICE_BROWSER_ECHO_GUARD = "0"
}

if ($MicrophoneIndex -ge 0) {
    $env:VOICE_BROWSER_MIC_INDEX = "$MicrophoneIndex"
}

if ($ListAllMics) {
    $env:VOICE_BROWSER_LIST_ALL_MICS = "1"
}

if ($ListenTimeout -gt 0) {
    $env:VOICE_BROWSER_LISTEN_TIMEOUT = "$ListenTimeout"
}

if ($PhraseLimit -gt 0) {
    $env:VOICE_BROWSER_PHRASE_LIMIT = "$PhraseLimit"
}

if ($PauseThreshold -gt 0) {
    $env:VOICE_BROWSER_PAUSE_THRESHOLD = "$PauseThreshold"
}

if ($NonSpeakingDuration -gt 0) {
    $env:VOICE_BROWSER_NON_SPEAKING_DURATION = "$NonSpeakingDuration"
}

if ($PhraseThreshold -gt 0) {
    $env:VOICE_BROWSER_PHRASE_THRESHOLD = "$PhraseThreshold"
}

if ($TtsBackend) {
    if ($TtsBackend -eq "auto") {
        Remove-Item Env:VOICE_BROWSER_TTS_BACKEND -ErrorAction SilentlyContinue
    } else {
        $env:VOICE_BROWSER_TTS_BACKEND = $TtsBackend
    }
}

if ($SttBackend) {
    $env:VOICE_BROWSER_STT_BACKEND = $SttBackend
}

if ($LocalSttModel) {
    $env:VOICE_BROWSER_LOCAL_STT_MODEL = $LocalSttModel
}

if ($LocalSttDevice) {
    $env:VOICE_BROWSER_LOCAL_STT_DEVICE = $LocalSttDevice
}

if ($LocalSttComputeType) {
    $env:VOICE_BROWSER_LOCAL_STT_COMPUTE_TYPE = $LocalSttComputeType
}

if ($SttDebug) {
    $env:VOICE_BROWSER_STT_DEBUG = "1"
}

if ($Url) {
    & $PythonCmd $MainScript $Url
} else {
    & $PythonCmd $MainScript
}
