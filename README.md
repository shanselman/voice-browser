# Voice Browser 🎤🌐

A voice-controlled web browser for people with limited mobility. Speak naturally to browse the web — no keyboard or mouse needed.

Powered by [GitHub Copilot SDK](https://github.com/github/copilot-sdk) for natural language understanding and direct [Playwright](https://playwright.dev/python/) automation.

## How It Works

```
Your voice ──→ Speech Recognition (Google or local Whisper) ──→ GitHub Copilot (LLM) ──→ Playwright (headed browser)
                                            │                        │
                                            ▼                        ▼
Your ears  ◀── Text-to-Speech    ◀── Spoken Summary     ◀── Accessibility Tree
```

The key difference from a traditional screen reader: **there are no memorized commands**. GitHub Copilot understands your intent in plain English and translates it into browser actions. Say things however feels natural.

## Quick Start

### Prerequisites
- Python 3.10+
- A microphone
- [GitHub Copilot subscription](https://github.com/features/copilot) (for the LLM brain)
- [GitHub Copilot CLI](https://githubnext.com/projects/copilot-cli/) installed (`copilot` in your PATH)

### Fresh machine setup (copy/paste)
```powershell
git clone https://github.com/shanselman/voice-browser.git
cd voice-browser
.\setup.ps1
.\start.ps1
```

### Setup
```powershell
cd voice-browser
.\setup.ps1
```

### Run
```powershell
.\start.ps1
.\start.ps1 bing.com
```

Optional local STT install:
```powershell
pip install faster-whisper numpy
```

### Local STT hardware profiles (high-end GPU vs regular laptop)

If you have a strong NVIDIA GPU, use CUDA for the fastest local transcription:
```powershell
# 1) Install NVIDIA driver + CUDA 12 runtime + cuDNN 9, then verify:
python -c "import ctranslate2 as ct; print(ct.get_cuda_device_count())"

# 2) Verify faster-whisper can initialize on GPU:
python -c "from faster_whisper import WhisperModel; WhisperModel('base.en', device='cuda', compute_type='float16'); print('gpu-ok')"

# 3) Run Voice Browser with GPU STT:
.\start.ps1 -SttBackend faster-whisper -LocalSttDevice cuda -LocalSttComputeType float16
```

If you are on a regular laptop (or want max compatibility), use CPU `int8` mode:
```powershell
.\start.ps1 -SttBackend faster-whisper -LocalSttDevice cpu -LocalSttComputeType int8
```

If you want one command that "just works" across many machines, use:
```powershell
.\start.ps1 -SttBackend faster-whisper -LocalSttDevice auto
```
This tries GPU first and automatically falls back to CPU when GPU runtime libraries are missing.

### Quick validation checks
```powershell
.\doctor.ps1 -Quick
python -m py_compile .\voice-browser.py .\mini_ui_host.py
```

Note: Voice Browser always runs in headed mode (visible browser window).

### Browser choice (Edge default)
```powershell
# Edge is default:
.\start.ps1

# Switch to Chrome/Chromium:
.\start.ps1 -UseChrome

# Optional explicit browser path (Edge/Chrome/Chromium):
.\start.ps1 -BrowserExecutablePath "C:\Path\To\browser.exe"

# Choose a specific microphone by index:
.\start.ps1 -MicrophoneIndex 1

# Use a specific Edge profile:
.\start.ps1 -EdgeProfile "Profile 1"
# Optional explicit Edge user data directory:
.\start.ps1 -EdgeUserDataDir "C:\Users\you\AppData\Local\Microsoft\Edge\User Data"
# Force-close running Edge first so the selected profile can be opened:
.\start.ps1 -Force

# Disable mini control UI:
.\start.ps1 -NoMiniUi

# Disable all spoken output (recommended if speakers cause echo):
.\start.ps1 -NoTts

# Disable barge-in (wait for TTS to finish before listening):
.\start.ps1 -DisableBargeIn

# Disable speaker-echo guard (normally keep this ON):
.\start.ps1 -DisableEchoGuard

# Prefer local STT (faster-whisper) for lower latency:
.\start.ps1 -SttBackend faster-whisper -LocalSttModel base.en

# Auto device (tries GPU, falls back to CPU automatically if needed):
.\start.ps1 -SttBackend faster-whisper -LocalSttDevice auto

# Laptop-safe CPU mode:
.\start.ps1 -SttBackend faster-whisper -LocalSttDevice cpu -LocalSttComputeType int8

# Slow, longer speech:
.\start.ps1 -PhraseLimit 40 -PauseThreshold 1.6

# STT diagnostics (prints exception type/trace for local STT failures):
.\start.ps1 -SttBackend faster-whisper -SttDebug
```

### Mini control UI
Voice Browser supports a small always-on-top control window for long sessions:
- Live status (listening/thinking/executing)
- Type-and-send prompt box
- Stop Voice button
- Set Mic selector
- Quit button
- Live runtime log panel below the buttons (mirrors console events)

If you ever hit UI issues, run with `-NoMiniUi`; the voice loop still works from the console.

## Onboarding Process (Windows)

1. Run `.\setup.ps1`.
2. The script validates Python, Copilot CLI, Edge availability, and Python packages.
3. It verifies Copilot authentication through the SDK.
4. It runs `doctor.ps1 -Quick` health checks (mic, dependencies, auth).
5. Optional: run full browser-flow diagnostics with `.\setup.ps1 -FullDoctor` or `.\doctor.ps1`.
6. Start day-to-day usage with `.\start.ps1`.

Manual commands:

```powershell
.\doctor.ps1          # Full diagnostics
.\doctor.ps1 -Quick   # Fast diagnostics
.\start.ps1 -SkipDoctor
.\start.ps1 -UseChrome
.\setup.ps1 -FullDoctor
```

### Existing Edge profile
By default, Voice Browser launches headed Edge using your existing profile (`Default`) for speed and familiar session state (bookmarks/cookies/extensions).

If your main Edge profile is locked (for example, Edge is already running and holding a profile lock), Voice Browser automatically falls back to an isolated profile and tells you.

When fallback happens, Voice Browser now first tries to seed that fallback from your selected Edge profile (bookmarks/preferences), so sessions stay more familiar even when the live profile is locked.

If profile access is critical for work logins, run with `-Force` to close running Edge processes before launch so Voice Browser can take the main profile lock.

### Why so many Edge launch flags?
Playwright injects a long list of Chromium flags automatically. They are mostly automation/runtime safety defaults (stable rendering, disabled prompts/background behaviors, deterministic control channel), not custom app arguments.

The key flags to care about here are:
- `--profile-directory=...` + `--user-data-dir=...` (which profile data is used)
- `--start-maximized` (start visible and large)
- `--remote-debugging-pipe` (automation control channel)

## Just Talk Naturally

There are no rigid commands to memorize. The AI understands intent. Here are examples of things you can say:

### Navigation
- "Go to reddit.com"
- "Take me to the New York Times"
- "Open bing in a new tab"
- "Go back to the previous page"
- "Refresh this"

### Interacting with Elements
- "Click the about link"
- "Click the third link"
- "Press the sign in button"
- "Fill in the search box with weather forecast"
- "Check the remember me checkbox"
- "Click the link that says think about"

### Reading & Understanding
- "What's on this page?"
- "Read me the main content"
- "What links are available?"
- "Where am I?"
- "What does the heading say?"

### Tabs & Windows
- "Open a new tab"
- "Switch to the second tab"  
- "Close this tab"
- "How many tabs do I have?"

### Display
- "Make it bigger" / "Zoom in"
- "Make it smaller" / "Zoom out"
- "Maximize the window"
- "Switch to dark mode"
- "Full screen"

### Scrolling
- "Scroll down"
- "Scroll way down" / "Go to the bottom"
- "Page up"

### Utilities
- "Take a screenshot"
- "Search for accessible restaurants near me"
- "Find the word 'pricing' on this page"

### Session
- "Help" / "What can I do?"
- "Quit" / "Goodbye"

## What It Can Do Today

- Natural language browsing and navigation.
- Page interaction with refs (click, fill, type, press, scroll).
- Tab workflows (new/switch/close/list).
- Display controls (zoom, maximize, fullscreen).
- Read content summaries and list interactive items.
- Clarification questions when your intent is ambiguous.
- Confirmation prompts for destructive actions (close tab/browser).
- Better follow-up click grounding using current page elements (e.g., “click first link”, “click send button”).
- Better fuzzy click matching for natural phrases (e.g., “click the link that says…”, minor speech-to-text typos).
- Commands run against the currently focused/front browser page by default.
- Faster turn-taking with non-blocking speech output and barge-in interruption.
- Automatic runtime recovery after browser worker timeouts or planner-session drops.
- Fast local command path for common intents (scroll/back/tabs/zoom/click) before LLM fallback.
- Optional local STT backend (`faster-whisper`) for lower-latency speech recognition.
- Outlook-specific "first email from <name>" click path with bounded re-query retries for dynamic lists.

## Current Limitations

- Local STT accuracy/latency depends on your chosen Whisper model and hardware.
- No secure password handling workflow yet.
- CAPTCHAs and complex anti-bot flows remain difficult.
- No wake-word support yet.
- Windows-first scripts are provided in this phase.

## If It Seems Not Listening

1. Run `.\doctor.ps1 -Quick` first.
2. Start with `.\start.ps1 -ListAllMics` to print full microphone indexes.
3. Then run `.\start.ps1 -MicrophoneIndex <n>` with the mic you want.
4. If you see repeated “Could not understand speech,” speak slower/closer and reduce background noise.
5. If you hear overlapping voices, close extra terminals; only one Voice Browser instance is allowed now.
6. Use the mini UI "Set Mic" button to switch microphone without restarting.
7. If it hears itself from speakers, leave echo guard enabled (default), lower speaker volume, or use headphones.
8. To fully disable spoken responses, run with `.\start.ps1 -NoTts`.
9. For better latency/reliability, run local STT: `.\start.ps1 -SttBackend faster-whisper`.
10. If GPU runtime libraries are missing, local STT now auto-falls back to CPU int8.
11. If local STT has repeated backend errors, Voice Browser auto-switches to Google STT for that session.
12. For long, slower thoughts, increase capture windows: `.\start.ps1 -PhraseLimit 40 -PauseThreshold 1.6`.
13. To diagnose local STT failures, run with `-SttDebug`.

## If TTS Is Not Audible

1. Run this quick test in PowerShell:
   ```powershell
   python -c "import pyttsx3; e=pyttsx3.init(); e.say('Voice browser text to speech test'); e.runAndWait()"
   ```
2. In Voice Browser, TTS now falls back to Windows `System.Speech` if `pyttsx3` fails.
3. If you prefer non-interruptible speech, run with `.\start.ps1 -DisableBargeIn`.
4. Check Windows output audio device/volume and retry with `.\start.ps1 -SkipDoctor`.

## Configuration

Environment variables for customization:

| Variable | Default | Description |
|----------|---------|-------------|
| `VOICE_BROWSER_MODEL` | `claude-sonnet-4.5` | Which Copilot model to use |
| `VOICE_BROWSER_STT_BACKEND` | `auto` | `auto`, `google`, or `faster-whisper` |
| `VOICE_BROWSER_LOCAL_STT_MODEL` | `base.en` | Local faster-whisper model name/path |
| `VOICE_BROWSER_LOCAL_STT_DEVICE` | `auto` | Local STT device (`auto`, `cpu`, `cuda`) |
| `VOICE_BROWSER_LOCAL_STT_COMPUTE_TYPE` | `auto` | faster-whisper compute type (`auto`, `int8`, `float16`, etc.) |
| `VOICE_BROWSER_STT_LANGUAGE` | `en` | Preferred STT language code |
| `VOICE_BROWSER_STT_DEBUG` | `0` | Set to `1` to print detailed local STT exception traces |
| `VOICE_BROWSER_TTS_ENABLED` | `1` | Set to `0` to disable spoken responses entirely |
| `VOICE_BROWSER_TTS_BACKEND` | `pyttsx3` | `windows`, `pyttsx3`, or `auto` |
| `VOICE_BROWSER_TTS_RATE` | `180` | Speech rate (words per minute) |
| `VOICE_BROWSER_SPEAK_STARTUP` | `0` | Set to `1` to speak startup status prompts |
| `VOICE_BROWSER_SPEAK_READY` | `0` | Set to `1` to speak a “Ready” prompt |
| `VOICE_BROWSER_LISTEN_TIMEOUT` | `10` | Seconds of silence before re-listening |
| `VOICE_BROWSER_PHRASE_LIMIT` | `15` | Max seconds for a single phrase |
| `VOICE_BROWSER_PAUSE_THRESHOLD` | `1.2` | Seconds of trailing silence before phrase capture ends |
| `VOICE_BROWSER_NON_SPEAKING_DURATION` | `0.5` | Internal silence padding retained around phrases |
| `VOICE_BROWSER_PHRASE_THRESHOLD` | `0.3` | Minimum speaking audio before phrase is considered valid |
| `VOICE_BROWSER_MAX_ELEMENTS` | `40` | Max interactive elements sent to planner context |
| `VOICE_BROWSER_MAX_HISTORY` | `6` | Number of recent turns retained for follow-up intents |
| `VOICE_BROWSER_LLM_TIMEOUT` | `30` | Copilot planner request timeout in seconds |
| `VOICE_BROWSER_BARGE_IN` | `1` | Set to `0` to wait for speech output to finish before listening |
| `VOICE_BROWSER_ECHO_GUARD` | `1` | Set to `0` to disable self-echo filtering while speaking |
| `VOICE_BROWSER_ECHO_GUARD_SECONDS` | `4` | Time window for self-echo filtering after speech finishes |
| `VOICE_BROWSER_MINI_UI` | `1` | Set to `0` to disable the always-on-top mini control UI |
| `VOICE_BROWSER_SNAPSHOT_STALE_SECONDS` | `8` | Element snapshot freshness threshold before pre-plan refresh |
| `VOICE_BROWSER_BROWSER` | `edge` | Browser engine (`edge` or `chromium`) |
| `VOICE_BROWSER_EXECUTABLE_PATH` | empty | Optional explicit browser executable path |
| `VOICE_BROWSER_EDGE_PROFILE` | `Default` | Edge profile directory name |
| `VOICE_BROWSER_EDGE_USER_DATA_DIR` | empty | Optional explicit Edge user data directory |
| `VOICE_BROWSER_EDGE_COPY_PROFILE_FALLBACK` | `1` | When direct profile launch is locked, seed fallback profile from selected Edge profile |
| `VOICE_BROWSER_FORCE_EDGE_PROFILE` | `0` | Set to `1` to close running Edge processes before launching profile |
| `VOICE_BROWSER_MIC_INDEX` | empty | Optional microphone device index (printed on startup) |
| `VOICE_BROWSER_LIST_ALL_MICS` | `0` | Set to `1` to print all microphone devices on startup |

## Architecture

- **voice-browser.py** — Main loop: listen → ask Copilot → execute → speak
- **GitHub Copilot SDK** — LLM brain that understands any natural language and maps it to browser commands
- **Playwright (Python)** — persistent headed browser control (Edge default)
- **SpeechRecognition** — Voice capture + Google STT path
- **faster-whisper** *(optional)* — Local Whisper STT path for lower latency/offline use
- **pyttsx3** — Offline text-to-speech (Windows SAPI / macOS NSSpeech / Linux espeak)

## Why This Approach?

Traditional accessibility tools require learning specific commands or key combinations. Voice Browser takes a different approach: the AI model understands *intent*. You don't need to know that "zoom in" is `Ctrl+=` or that "maximize" means `set viewport 1920 1080`. Just say what you want.

The LLM also handles:
- **Speech-to-text errors** — "hanselman dot com" → `hanselman.com`
- **Ambiguity** — "click the first one" after listing links
- **Compound requests** — "open a new tab and search for cat videos"
- **Context** — knows what page you're on and what elements are available

## Future Ideas

- **Streaming STT**: Add incremental microphone decoding for faster partial commands
- **Wake word**: "Hey browser" hotword so it's not always listening
- **Streaming TTS**: Start speaking before the full response is ready
- **Vision mode**: Send screenshots to the LLM for visual understanding
- **Custom personas**: Adjust verbosity, personality, reading speed per user

## License

MIT
