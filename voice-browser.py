"""
Voice Browser.

Architecture:
  Microphone -> Speech Recognition -> GitHub Copilot SDK planner -> Playwright
  Speaker <- Text-to-Speech <- Structured execution results
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from difflib import SequenceMatcher
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import pyttsx3
import speech_recognition as sr
from copilot import CopilotClient, MessageOptions, SessionConfig
from copilot.generated.session_events import SessionEventType
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

try:
    import numpy as np
except Exception:
    np = None

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

MODEL = os.environ.get("VOICE_BROWSER_MODEL", "claude-sonnet-4.5")
TTS_RATE = int(os.environ.get("VOICE_BROWSER_TTS_RATE", "180"))
TTS_BACKEND = os.environ.get(
    "VOICE_BROWSER_TTS_BACKEND",
    "pyttsx3",
).strip().lower()
STT_BACKEND = os.environ.get("VOICE_BROWSER_STT_BACKEND", "auto").strip().lower()
STT_DEBUG = os.environ.get("VOICE_BROWSER_STT_DEBUG", "0").strip() == "1"
LOCAL_STT_MODEL = os.environ.get("VOICE_BROWSER_LOCAL_STT_MODEL", "base.en").strip() or "base.en"
LOCAL_STT_DEVICE = os.environ.get("VOICE_BROWSER_LOCAL_STT_DEVICE", "auto").strip() or "auto"
LOCAL_STT_COMPUTE_TYPE = os.environ.get("VOICE_BROWSER_LOCAL_STT_COMPUTE_TYPE", "auto").strip() or "auto"
STT_LANGUAGE = os.environ.get("VOICE_BROWSER_STT_LANGUAGE", "en").strip() or "en"
SPEAK_STARTUP_STATUS = os.environ.get("VOICE_BROWSER_SPEAK_STARTUP", "0").strip() == "1"
SPEAK_READY_MESSAGE = os.environ.get("VOICE_BROWSER_SPEAK_READY", "0").strip() == "1"
LISTEN_TIMEOUT = int(os.environ.get("VOICE_BROWSER_LISTEN_TIMEOUT", "10"))
PHRASE_TIME_LIMIT = int(os.environ.get("VOICE_BROWSER_PHRASE_LIMIT", "15"))
PAUSE_THRESHOLD = float(os.environ.get("VOICE_BROWSER_PAUSE_THRESHOLD", "1.2"))
NON_SPEAKING_DURATION = float(os.environ.get("VOICE_BROWSER_NON_SPEAKING_DURATION", "0.5"))
PHRASE_THRESHOLD = float(os.environ.get("VOICE_BROWSER_PHRASE_THRESHOLD", "0.3"))
MAX_CONTEXT_ELEMENTS = int(os.environ.get("VOICE_BROWSER_MAX_ELEMENTS", "40"))
MAX_HISTORY_ITEMS = int(os.environ.get("VOICE_BROWSER_MAX_HISTORY", "6"))
LLM_TIMEOUT_SECONDS = int(os.environ.get("VOICE_BROWSER_LLM_TIMEOUT", "30"))
SNAPSHOT_STALE_SECONDS = int(os.environ.get("VOICE_BROWSER_SNAPSHOT_STALE_SECONDS", "8"))
BARGE_IN_ENABLED = os.environ.get("VOICE_BROWSER_BARGE_IN", "1").strip() != "0"
MINI_UI_ENABLED = os.environ.get("VOICE_BROWSER_MINI_UI", "1").strip() != "0"
ECHO_GUARD_ENABLED = os.environ.get("VOICE_BROWSER_ECHO_GUARD", "1").strip() != "0"
ECHO_GUARD_SECONDS = float(os.environ.get("VOICE_BROWSER_ECHO_GUARD_SECONDS", "4"))
BROWSER_ENGINE = os.environ.get("VOICE_BROWSER_BROWSER", "edge").strip().lower()
CUSTOM_BROWSER_EXECUTABLE = os.environ.get("VOICE_BROWSER_EXECUTABLE_PATH", "").strip()
EDGE_PROFILE_NAME = os.environ.get("VOICE_BROWSER_EDGE_PROFILE", "Default").strip() or "Default"
EDGE_USER_DATA_DIR = os.environ.get("VOICE_BROWSER_EDGE_USER_DATA_DIR", "").strip()
EDGE_COPY_PROFILE_FALLBACK = os.environ.get("VOICE_BROWSER_EDGE_COPY_PROFILE_FALLBACK", "1").strip() != "0"
FORCE_EDGE_PROFILE = os.environ.get("VOICE_BROWSER_FORCE_EDGE_PROFILE", "0").strip() == "1"
MIC_INDEX_ENV = os.environ.get("VOICE_BROWSER_MIC_INDEX", "").strip()
LIST_ALL_MICROPHONES = os.environ.get("VOICE_BROWSER_LIST_ALL_MICS", "0").strip() == "1"

ALLOWED_ACTIONS = {
    "open",
    "back",
    "forward",
    "reload",
    "click",
    "fill",
    "type_text",
    "press",
    "scroll",
    "tab_new",
    "tab_switch",
    "tab_close",
    "tab_list",
    "set_viewport",
    "maximize_window",
    "zoom_in",
    "zoom_out",
    "zoom_reset",
    "fullscreen_toggle",
    "read_main",
    "list_actions",
    "find_on_page",
    "set_media",
    "search_web",
    "screenshot",
    "snapshot",
    "close_browser",
}

ACTION_ALIASES = {
    "new_tab": "tab_new",
    "switch_tab": "tab_switch",
    "close_tab": "tab_close",
    "type": "type_text",
    "maximize": "maximize_window",
    "fullscreen": "fullscreen_toggle",
    "read_page": "read_main",
    "list_interactive": "list_actions",
    "search": "search_web",
    "close": "close_browser",
}

DESTRUCTIVE_ACTIONS = {"tab_close", "close_browser"}

YES_WORDS = {"yes", "yeah", "yep", "sure", "confirm", "do it", "ok", "okay", "please do"}
NO_WORDS = {
    "no",
    "nope",
    "cancel",
    "stop",
    "dont",
    "don't",
    "do not",
    "never mind",
    "not",
    "not ok",
    "not okay",
}
ORDINAL_WORDS = {
    "first": 0,
    "1st": 0,
    "one": 0,
    "second": 1,
    "2nd": 1,
    "two": 1,
    "third": 2,
    "3rd": 2,
    "three": 2,
    "fourth": 3,
    "4th": 3,
    "four": 3,
    "fifth": 4,
    "5th": 4,
    "five": 4,
}
ROLE_HINTS = {
    "link": "link",
    "button": "button",
    "checkbox": "checkbox",
    "radio": "radio",
    "textbox": "textbox",
    "input": "textbox",
    "field": "textbox",
}
TARGET_STOP_WORDS = {
    "click",
    "press",
    "tap",
    "select",
    "open",
    "on",
    "the",
    "a",
    "an",
    "this",
    "that",
    "please",
    "link",
    "button",
    "thing",
    "item",
    "one",
    "says",
    "say",
    "saying",
    "reads",
    "read",
    "called",
    "named",
    "text",
}


@dataclass
class PlannerResult:
    actions: List[Dict[str, Any]] = field(default_factory=list)
    spoken_response: str = ""
    needs_clarification: bool = False
    clarification_question: str = ""
    confirmation_prompt: str = ""
    quit: bool = False


@dataclass
class BrowserState:
    title: str = ""
    url: str = ""
    tabs: str = ""
    snapshot: str = ""
    elements: List[Dict[str, str]] = field(default_factory=list)
    recent_history: List[Dict[str, str]] = field(default_factory=list)
    pending_confirmation: Optional[PlannerResult] = None
    snapshot_updated_at: float = 0.0


class BrowserCommandError(RuntimeError):
    pass


class MiniControlUI:
    def __init__(self, mic_names: List[str], selected_mic_index: Optional[int]) -> None:
        self._mic_names = mic_names
        self._selected_mic_index = selected_mic_index
        self._ctx = mp.get_context("spawn")
        self._command_queue: "mp.Queue[Any]" = self._ctx.Queue(maxsize=200)
        self._status_queue: "mp.Queue[Any]" = self._ctx.Queue(maxsize=300)
        self._log_queue: "mp.Queue[Any]" = self._ctx.Queue(maxsize=1000)
        self._process: Optional[mp.Process] = None
        self._run_ui = None

    def start(self) -> None:
        if not MINI_UI_ENABLED:
            return
        if self._process is not None and self._process.is_alive():
            return
        if self._run_ui is None:
            try:
                from mini_ui_host import run_ui

                self._run_ui = run_ui
            except Exception as exc:
                log_line(f"WARN: Mini UI unavailable ({exc}).")
                return
        try:
            self._process = self._ctx.Process(
                target=self._run_ui,
                args=(
                    self._mic_display_items(),
                    self._selected_mic_index,
                    self._command_queue,
                    self._status_queue,
                    self._log_queue,
                ),
                daemon=True,
            )
            self._process.start()
        except Exception as exc:
            self._process = None
            log_line(f"WARN: Mini UI failed to start ({exc}).")
            return

    def stop(self) -> None:
        if self._process is None:
            return
        try:
            self._status_queue.put_nowait({"type": "shutdown"})
        except Exception:
            pass
        if self._process.is_alive():
            self._process.join(timeout=3.0)
        if self._process.is_alive():
            try:
                self._process.terminate()
            except Exception:
                pass
        self._process = None

    def poll_event(self) -> Optional[Dict[str, Any]]:
        try:
            event = self._command_queue.get_nowait()
            if isinstance(event, dict):
                return event
            return None
        except queue.Empty:
            return None

    def set_status(self, status: str) -> None:
        if not MINI_UI_ENABLED:
            return
        try:
            self._status_queue.put_nowait({"type": "status", "value": status})
        except queue.Full:
            pass

    def add_log(self, line: str) -> None:
        if not MINI_UI_ENABLED:
            return
        try:
            self._log_queue.put_nowait({"type": "log", "value": line})
        except queue.Full:
            pass

    def _mic_display_items(self) -> List[str]:
        return [f"[{idx}] {name}" for idx, name in enumerate(self._mic_names)]


_ui_logger: Optional[MiniControlUI] = None
_local_stt_lock = threading.Lock()
_local_stt_model: Optional[Any] = None
_local_stt_model_key: Optional[Tuple[str, str, str]] = None
_local_stt_active_device = ""
_local_stt_active_compute_type = ""
_effective_stt_backend = "google"
_local_stt_backend_error_streak = 0


def log_line(message: str) -> None:
    print(message)
    if _ui_logger is not None:
        _ui_logger.add_log(message)


def log_stt_debug(exc: Exception) -> None:
    if not STT_DEBUG:
        return
    log_line(f"DEBUG: STT exception type={exc.__class__.__name__}")
    trace = traceback.format_exc().strip()
    if trace:
        for line in trace.splitlines():
            log_line(f"DEBUG: {line}")


def _resolve_stt_backend() -> str:
    choice = STT_BACKEND.lower()
    if choice not in {"auto", "google", "faster-whisper"}:
        return "google"
    if choice == "google":
        return "google"
    if choice == "faster-whisper":
        return "faster-whisper"
    if WhisperModel is None or np is None:
        return "google"
    return "faster-whisper"


def _stt_candidate_configs() -> List[Tuple[str, str]]:
    requested_device = (LOCAL_STT_DEVICE or "auto").strip().lower()
    requested_compute = (LOCAL_STT_COMPUTE_TYPE or "auto").strip()
    candidates: List[Tuple[str, str]] = []

    def add(device: str, compute: str) -> None:
        entry = (device.strip().lower(), compute.strip() or "auto")
        if entry not in candidates:
            candidates.append(entry)

    if requested_device == "cpu":
        add("cpu", requested_compute)
        add("cpu", "int8")
        return candidates
    if requested_device == "cuda":
        add("cuda", requested_compute)
        add("cuda", "float16")
        add("cpu", "int8")
        return candidates

    add("auto", requested_compute)
    add("cuda", "float16")
    add("cpu", "int8")
    return candidates


def _reset_local_stt_model() -> None:
    global _local_stt_model, _local_stt_model_key, _local_stt_active_device, _local_stt_active_compute_type
    _local_stt_model = None
    _local_stt_model_key = None
    _local_stt_active_device = ""
    _local_stt_active_compute_type = ""


def _build_local_stt_model(device: str, compute_type: str) -> Any:
    if WhisperModel is None or np is None:
        raise RuntimeError("faster-whisper dependencies are not installed.")
    return WhisperModel(
        LOCAL_STT_MODEL,
        device=device,
        compute_type=compute_type,
    )


def _get_local_stt_model(force_reload: bool = False) -> Any:
    global _local_stt_model, _local_stt_model_key, _local_stt_active_device, _local_stt_active_compute_type
    with _local_stt_lock:
        if _local_stt_model is not None and not force_reload:
            return _local_stt_model
        errors: List[str] = []
        for device, compute_type in _stt_candidate_configs():
            key = (LOCAL_STT_MODEL, device, compute_type)
            if _local_stt_model is not None and _local_stt_model_key == key and not force_reload:
                return _local_stt_model
            try:
                model = _build_local_stt_model(device, compute_type)
                _local_stt_model = model
                _local_stt_model_key = key
                _local_stt_active_device = device
                _local_stt_active_compute_type = compute_type
                return model
            except Exception as exc:
                errors.append(f"{device}/{compute_type}: {exc}")
        _reset_local_stt_model()
        raise RuntimeError("Could not initialize local STT. " + " | ".join(errors))


def _switch_local_stt_to_cpu() -> bool:
    global _local_stt_model, _local_stt_model_key, _local_stt_active_device, _local_stt_active_compute_type
    with _local_stt_lock:
        try:
            model = _build_local_stt_model("cpu", "int8")
            _local_stt_model = model
            _local_stt_model_key = (LOCAL_STT_MODEL, "cpu", "int8")
            _local_stt_active_device = "cpu"
            _local_stt_active_compute_type = "int8"
            return True
        except Exception:
            _reset_local_stt_model()
            return False


def _format_stt_error(exc: Exception) -> str:
    details = str(exc).strip()
    if details:
        return details
    return exc.__class__.__name__


def _transcribe_local(audio: sr.AudioData) -> str:
    model = _get_local_stt_model()
    pcm_bytes = audio.get_raw_data(convert_rate=16000, convert_width=2)
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    segments, _info = model.transcribe(
        samples,
        language=STT_LANGUAGE if STT_LANGUAGE else None,
        beam_size=1,
        best_of=1,
        temperature=0.0,
        condition_on_previous_text=False,
        vad_filter=True,
    )
    text = " ".join(seg.text.strip() for seg in segments if seg.text and seg.text.strip())
    return text.strip()


def transcribe_audio(recognizer: sr.Recognizer, audio: sr.AudioData) -> str:
    global _effective_stt_backend, _local_stt_backend_error_streak
    if _effective_stt_backend == "faster-whisper":
        try:
            text = _transcribe_local(audio)
            if text:
                _local_stt_backend_error_streak = 0
                return text
            raise sr.UnknownValueError()
        except sr.UnknownValueError:
            # Empty/noisy audio is not a backend failure; let caller handle it consistently.
            raise
        except Exception as exc:
            error_text = str(exc).lower()
            log_stt_debug(exc)
            if any(marker in error_text for marker in ("cublas", "cuda", "cudnn", "onnxruntime")):
                if _switch_local_stt_to_cpu():
                    log_line("WARN: GPU STT unavailable; switched local STT to CPU (int8).")
                    try:
                        text = _transcribe_local(audio)
                        if text:
                            _local_stt_backend_error_streak = 0
                            return text
                        raise sr.UnknownValueError()
                    except sr.UnknownValueError:
                        raise
                    except Exception as retry_exc:
                        log_stt_debug(retry_exc)
                        log_line(
                            f"WARN: local STT retry failed ({_format_stt_error(retry_exc)}); falling back to Google STT."
                        )
                else:
                    log_line("WARN: GPU STT unavailable and CPU local STT init failed; falling back to Google STT.")
            else:
                log_line(f"WARN: local STT failed ({_format_stt_error(exc)}); falling back to Google STT.")
            _local_stt_backend_error_streak += 1
            if _local_stt_backend_error_streak >= 3:
                _effective_stt_backend = "google"
                log_line("WARN: local STT repeatedly failed; using Google STT for the rest of this session.")
    return recognizer.recognize_google(audio).strip()


def find_edge_executable() -> str:
    edge_from_path = shutil.which("msedge")
    if edge_from_path:
        return edge_from_path

    candidates = [
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return ""


def resolve_browser_executable() -> str:
    if CUSTOM_BROWSER_EXECUTABLE:
        return CUSTOM_BROWSER_EXECUTABLE

    if BROWSER_ENGINE in {"edge", "msedge"}:
        edge_path = find_edge_executable()
        if edge_path:
            return edge_path
        log_line("WARN: Microsoft Edge requested but not found; using default Chromium.")
    return ""


BROWSER_EXECUTABLE = resolve_browser_executable()


def compact_playwright_error(exc: Exception) -> str:
    text = str(exc).replace("\r", " ").replace("\n", " ")
    for marker in ("Browser logs:", "Call log:"):
        if marker in text:
            text = text.split(marker, 1)[0]
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 220:
        text = text[:217] + "..."
    return text


def close_running_edge_processes() -> Tuple[int, int]:
    if os.name != "nt":
        return 0, 0

    command = """
$procs = @(Get-Process msedge -ErrorAction SilentlyContinue)
$killed = 0
if ($procs.Count -gt 0) {
  foreach ($proc in $procs) {
    try {
      Stop-Process -Id $proc.Id -Force -ErrorAction Stop
      $killed += 1
    } catch {
    }
  }
  Start-Sleep -Milliseconds 400
}
$remaining = @(Get-Process msedge -ErrorAction SilentlyContinue).Count
Write-Output "$killed|$remaining"
""".strip()
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception as exc:
        log_line(f"WARN: Unable to close running Edge processes ({exc}).")
        return 0, 0

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        log_line(f"WARN: Edge force-close failed ({detail}).")
        return 0, 0

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return 0, 0
    try:
        killed_str, remaining_str = lines[-1].split("|", 1)
        return int(killed_str), int(remaining_str)
    except ValueError:
        return 0, 0


class BrowserRuntime:
    def __init__(self) -> None:
        self._playwright = None
        self._browser = None
        self._context = None
        self._current_page = None
        self.profile_note = ""

    def _edge_user_data_dir(self) -> str:
        if EDGE_USER_DATA_DIR:
            return EDGE_USER_DATA_DIR
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        if local_app_data:
            return os.path.join(local_app_data, "Microsoft", "Edge", "User Data")
        return str(Path.home() / ".voice-browser" / "edge-user-data")

    def _fallback_profile_dir(self) -> str:
        fallback_dir = Path.home() / ".voice-browser" / "profiles" / "edge-fallback"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        return str(fallback_dir)

    def _seed_fallback_from_edge_profile(
        self,
        source_user_data_dir: str,
        profile_name: str,
        fallback_user_data_dir: str,
    ) -> bool:
        source_root = Path(source_user_data_dir)
        source_profile = source_root / profile_name
        if not source_profile.exists():
            return False

        fallback_root = Path(fallback_user_data_dir)
        fallback_root.mkdir(parents=True, exist_ok=True)
        fallback_profile = fallback_root / profile_name

        ignored_dirs = {
            "Cache",
            "Code Cache",
            "GPUCache",
            "GrShaderCache",
            "ShaderCache",
            "DawnCache",
            "Crashpad",
            "Service Worker",
        }
        ignored_files = {"LOCK", "lockfile", "SingletonCookie", "SingletonLock", "SingletonSocket"}

        def _ignore(_path: str, names: List[str]) -> List[str]:
            return [n for n in names if n in ignored_dirs or n in ignored_files or n.startswith("Singleton")]

        try:
            local_state = source_root / "Local State"
            if local_state.exists():
                shutil.copy2(local_state, fallback_root / "Local State")

            if not fallback_profile.exists():
                shutil.copytree(source_profile, fallback_profile, ignore=_ignore)
            else:
                for filename in ("Bookmarks", "Preferences"):
                    src_file = source_profile / filename
                    if src_file.exists():
                        shutil.copy2(src_file, fallback_profile / filename)
            return True
        except Exception as exc:
            log_line(f"WARN: Unable to copy Edge profile to fallback ({exc}).")
            return False

    def _launch(self) -> None:
        if self._context is not None:
            return

        self._playwright = sync_playwright().start()
        # Always headed for accessibility and direct visual feedback.
        launch_kwargs: Dict[str, Any] = {"headless": False}
        if BROWSER_EXECUTABLE:
            launch_kwargs["executable_path"] = BROWSER_EXECUTABLE

        if BROWSER_ENGINE in {"edge", "msedge"}:
            launch_kwargs["channel"] = "msedge"
            user_data_dir = self._edge_user_data_dir()
            args = [f"--profile-directory={EDGE_PROFILE_NAME}", "--start-maximized"]
            force_note = ""
            if FORCE_EDGE_PROFILE:
                killed_count, remaining_count = close_running_edge_processes()
                if remaining_count == 0:
                    force_note = f"Force mode closed {killed_count} Edge process(es). "
                else:
                    force_note = (
                        f"Force mode closed {killed_count} Edge process(es), "
                        f"{remaining_count} still running. "
                    )
            try:
                self._context = self._playwright.chromium.launch_persistent_context(
                    user_data_dir=user_data_dir,
                    no_viewport=True,
                    args=args,
                    **launch_kwargs,
                )
                self.profile_note = f"{force_note}Using Edge profile '{EDGE_PROFILE_NAME}'."
            except PlaywrightError as exc:
                primary_reason = compact_playwright_error(exc)
                fallback_dir = self._fallback_profile_dir()
                copied = False
                fallback_args = ["--start-maximized"]
                if EDGE_COPY_PROFILE_FALLBACK:
                    copied = self._seed_fallback_from_edge_profile(
                        user_data_dir,
                        EDGE_PROFILE_NAME,
                        fallback_dir,
                    )
                    if copied:
                        fallback_args.insert(0, f"--profile-directory={EDGE_PROFILE_NAME}")

                try:
                    self._context = self._playwright.chromium.launch_persistent_context(
                        user_data_dir=fallback_dir,
                        no_viewport=True,
                        args=fallback_args,
                        **launch_kwargs,
                    )
                except PlaywrightError as fallback_exc:
                    fallback_reason = compact_playwright_error(fallback_exc)
                    raise RuntimeError(
                        f"Edge profile launch failed ({primary_reason}); fallback failed ({fallback_reason})."
                    ) from fallback_exc

                if copied:
                    self.profile_note = (
                        f"{force_note}"
                        f"Edge profile unavailable ({primary_reason}). "
                        f"Using managed copy of '{EDGE_PROFILE_NAME}'."
                    )
                else:
                    self.profile_note = (
                        f"{force_note}"
                        f"Edge profile unavailable ({primary_reason}). Using isolated fallback profile."
                    )
        else:
            if BROWSER_ENGINE == "chrome":
                launch_kwargs["channel"] = "chrome"
            self._browser = self._playwright.chromium.launch(
                args=["--start-maximized"],
                **launch_kwargs,
            )
            self._context = self._browser.new_context(no_viewport=True)
            self.profile_note = "Using isolated browser context."

    def _ensure_page(self):
        self._launch()
        if self._context.pages:
            for candidate in reversed(self._context.pages):
                if candidate.is_closed():
                    continue
                try:
                    if candidate.evaluate("() => document.hasFocus()"):
                        self._current_page = candidate
                        break
                except Exception:
                    continue
        if self._current_page is None or self._current_page.is_closed():
            if self._context.pages:
                self._current_page = self._context.pages[-1]
            else:
                self._current_page = self._context.new_page()
        return self._current_page

    def _target_selector(self, target: str) -> str:
        if target.startswith("@"):
            ref = target[1:]
            return f'[data-vb-ref="{ref}"]'
        return target

    def _readable_page_title(self, page) -> str:
        try:
            title = page.title().strip()
            return title or "(untitled)"
        except Exception:
            return "(untitled)"

    def _snapshot_interactive(self) -> str:
        page = self._ensure_page()
        items = page.evaluate(
            """() => {
                const visible = (el) => {
                  if (!(el instanceof HTMLElement)) return false;
                  const style = window.getComputedStyle(el);
                  if (style.display === 'none' || style.visibility === 'hidden') return false;
                  const rect = el.getBoundingClientRect();
                  return rect.width > 0 && rect.height > 0;
                };
                const roleFor = (el) => {
                  const explicit = (el.getAttribute('role') || '').trim();
                  if (explicit) return explicit;
                  const tag = el.tagName.toLowerCase();
                  if (tag === 'a') return 'link';
                  if (tag === 'button') return 'button';
                  if (tag === 'input') {
                    const t = (el.getAttribute('type') || '').toLowerCase();
                    if (t === 'checkbox') return 'checkbox';
                    if (t === 'radio') return 'radio';
                    if (t === 'submit' || t === 'button') return 'button';
                    return 'textbox';
                  }
                  if (tag === 'textarea') return 'textbox';
                  if (tag === 'select') return 'combobox';
                  return 'generic';
                };
                const isInteractive = (el) => {
                  const tag = el.tagName.toLowerCase();
                  if (['a','button','input','textarea','select'].includes(tag)) return true;
                  if ((el.getAttribute('role') || '').trim()) return true;
                  if (el.hasAttribute('onclick')) return true;
                  if (el.hasAttribute('tabindex') && el.getAttribute('tabindex') !== '-1') return true;
                  return false;
                };
                const nameFor = (el) => {
                  const parts = [
                    el.getAttribute('aria-label') || '',
                    el.innerText || '',
                    el.getAttribute('placeholder') || '',
                    el.getAttribute('title') || '',
                    el.getAttribute('value') || ''
                  ].map(x => x.trim()).filter(Boolean);
                  return parts.length ? parts[0] : '';
                };

                document.querySelectorAll('[data-vb-ref]').forEach(el => el.removeAttribute('data-vb-ref'));
                const candidates = Array.from(document.querySelectorAll('a,button,input,textarea,select,[role],[tabindex],[onclick]'));
                const out = [];
                let idx = 1;
                for (const el of candidates) {
                  if (!visible(el) || !isInteractive(el)) continue;
                  const ref = `e${idx++}`;
                  el.setAttribute('data-vb-ref', ref);
                  out.push({
                    ref,
                    role: roleFor(el),
                    name: nameFor(el)
                  });
                  if (out.length >= 200) break;
                }
                return out;
            }"""
        )
        lines: List[str] = []
        for item in items:
            role = (item.get("role") or "generic").strip()
            name = (item.get("name") or role).replace('"', "'").strip()
            ref = item.get("ref")
            lines.append(f'- {role} "{name}" [ref={ref}]')
        return "\n".join(lines)

    def _click_fast(self, page, selector: str) -> None:
        locator = page.locator(selector).first
        try:
            locator.scroll_into_view_if_needed(timeout=1200)
        except Exception:
            pass
        try:
            locator.click(timeout=1800)
            return
        except Exception:
            pass
        try:
            locator.click(timeout=1800, force=True)
            return
        except Exception:
            pass
        handle = locator.element_handle(timeout=1200)
        if handle is None:
            raise RuntimeError("Target element was not found.")
        page.evaluate("(el) => { try { el.click(); return true; } catch { return false; } }", handle)

    def _click_outlook_sender(self, page, sender: str) -> bool:
        needle = sender.strip().lower()
        if not needle:
            return False
        for _attempt in range(4):
            try:
                clicked = page.evaluate(
                    """(senderNeedle) => {
                        const needle = (senderNeedle || "").toLowerCase();
                        const textFor = (el) => ((el.innerText || el.textContent || el.getAttribute('aria-label') || '') + '').toLowerCase();
                        const containerSelectors = [
                          'div[role="row"]',
                          'div[role="option"]',
                          'div[data-convid]',
                          'div[aria-selected]',
                          'li[role="option"]'
                        ];
                        for (const selector of containerSelectors) {
                          const items = Array.from(document.querySelectorAll(selector));
                          for (const item of items) {
                            const text = textFor(item);
                            if (!text || !text.includes(needle)) continue;
                            item.scrollIntoView({ block: 'center', inline: 'nearest' });
                            const target = item.querySelector('a,button,[role="link"],[role="button"]') || item;
                            try {
                              target.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, cancelable: true }));
                              target.dispatchEvent(new MouseEvent('mouseup', { bubbles: true, cancelable: true }));
                              target.click();
                              return true;
                            } catch {
                            }
                          }
                        }
                        return false;
                    }""",
                    needle,
                )
                if clicked:
                    return True
            except Exception:
                pass
            try:
                page.mouse.wheel(0, 520)
            except Exception:
                pass
            try:
                page.wait_for_timeout(200)
            except Exception:
                pass
        return False

    def _tab_list(self) -> str:
        self._launch()
        if not self._context.pages:
            return "No tabs open."
        lines = []
        for idx, page in enumerate(self._context.pages, start=1):
            marker = "→ " if page == self._current_page else "  "
            lines.append(f"{marker}[{idx}] {self._readable_page_title(page)} - {page.url}")
        return "\n".join(lines)

    def _maximize_window(self) -> None:
        page = self._ensure_page()
        try:
            cdp = self._context.new_cdp_session(page)
            win = cdp.send("Browser.getWindowForTarget")
            cdp.send("Browser.setWindowBounds", {"windowId": win["windowId"], "bounds": {"windowState": "maximized"}})
            return
        except Exception:
            pass
        try:
            page.set_viewport_size({"width": 1920, "height": 1080})
        except Exception:
            pass

    def _close(self) -> None:
        try:
            if self._context is not None:
                try:
                    self._context.close()
                except Exception:
                    pass
        finally:
            self._context = None
            self._current_page = None
            if self._browser is not None:
                try:
                    self._browser.close()
                except Exception:
                    pass
                self._browser = None
            if self._playwright is not None:
                try:
                    self._playwright.stop()
                except Exception:
                    pass
                self._playwright = None

    def execute(self, args: List[str]) -> str:
        if not args:
            return ""

        command = args[0]
        page = None
        if command != "close":
            page = self._ensure_page()
            try:
                page.bring_to_front()
            except Exception:
                pass

        if command == "open":
            url = args[1]
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            self._current_page = page
            return f"✓ {self._readable_page_title(page)}\n  {page.url}"
        if command == "back":
            page.go_back(wait_until="domcontentloaded")
            return "✓ Back"
        if command == "forward":
            page.go_forward(wait_until="domcontentloaded")
            return "✓ Forward"
        if command == "reload":
            page.reload(wait_until="domcontentloaded")
            return "✓ Reloaded"
        if command == "click":
            target = args[1]
            if target.lower().startswith("outlook_sender:"):
                sender = target.split(":", 1)[1].strip()
                if self._click_outlook_sender(page, sender):
                    return "✓ Clicked"
                raise RuntimeError(f"Could not find an Outlook email from '{sender}'.")
            selector = self._target_selector(target)
            try:
                self._click_fast(page, selector)
                return "✓ Clicked"
            except Exception as first_exc:
                if target.startswith("@"):
                    raise RuntimeError("Target item changed on the page; try again.") from first_exc
                if looks_like_selector(target):
                    raise first_exc
                try:
                    snapshot = self._snapshot_interactive()
                    elements = parse_snapshot_elements(snapshot)
                    state = BrowserState(elements=elements, snapshot=snapshot, snapshot_updated_at=time.time())
                    guessed_ref = resolve_ref_from_text(target, state)
                    if guessed_ref:
                        self._click_fast(page, self._target_selector(guessed_ref))
                        return "✓ Clicked"
                except Exception:
                    pass
                raise RuntimeError(f"Could not find clickable item matching '{target}'.") from first_exc
        if command == "fill":
            page.locator(self._target_selector(args[1])).first.fill(args[2], timeout=3500)
            return "✓ Filled"
        if command == "type":
            page.locator(self._target_selector(args[1])).first.type(args[2], timeout=3500)
            return "✓ Typed"
        if command == "press":
            page.keyboard.press(args[1])
            return "✓ Key pressed"
        if command == "scroll":
            direction = args[1].lower()
            pixels = int(args[2]) if len(args) > 2 else 500
            dx, dy = 0, 0
            if direction == "down":
                dy = pixels
            elif direction == "up":
                dy = -pixels
            elif direction == "right":
                dx = pixels
            elif direction == "left":
                dx = -pixels
            page.mouse.wheel(dx, dy)
            return "✓ Scrolled"
        if command == "state":
            return json.dumps(
                {
                    "title": self._readable_page_title(page),
                    "url": page.url,
                    "tabs": self._tab_list(),
                },
                ensure_ascii=True,
            )
        if command == "get":
            field = args[1]
            if field == "title":
                return self._readable_page_title(page)
            if field == "url":
                return page.url
            if field == "text":
                selector = self._target_selector(args[2])
                return page.locator(selector).first.inner_text(timeout=10000)
            return ""
        if command == "snapshot":
            return self._snapshot_interactive()
        if command == "tab":
            if len(args) == 1:
                return self._tab_list()
            sub = args[1]
            if sub == "new":
                new_page = self._context.new_page()
                self._current_page = new_page
                if len(args) > 2 and args[2]:
                    new_page.goto(args[2], wait_until="domcontentloaded", timeout=30000)
                return "✓ New tab"
            if sub == "close":
                if len(args) > 2:
                    idx = max(1, int(args[2])) - 1
                    target = self._context.pages[idx] if idx < len(self._context.pages) else self._current_page
                else:
                    target = self._current_page
                if target is not None:
                    target.close()
                if self._context.pages:
                    self._current_page = self._context.pages[-1]
                else:
                    self._current_page = self._context.new_page()
                return "✓ Tab closed"
            idx = max(1, int(sub)) - 1
            if idx < len(self._context.pages):
                self._current_page = self._context.pages[idx]
                self._current_page.bring_to_front()
                return f"✓ Tab {idx + 1}"
            raise RuntimeError("Tab index out of range.")
        if command == "set":
            if args[1] == "viewport":
                width = int(args[2])
                height = int(args[3])
                page.set_viewport_size({"width": width, "height": height})
                return "✓ Viewport set"
            if args[1] == "media":
                mode = args[2]
                page.emulate_media(color_scheme=mode)
                return "✓ Media set"
        if command == "maximize":
            self._maximize_window()
            return "✓ Maximized"
        if command == "eval":
            expression = " ".join(args[1:])
            result = page.evaluate(expression)
            return "" if result is None else str(result)
        if command == "screenshot":
            full = "--full" in args
            path = ""
            filtered = [a for a in args[1:] if a != "--full"]
            if filtered:
                path = filtered[0]
            if not path:
                screenshot_dir = Path.home() / ".voice-browser" / "screenshots"
                screenshot_dir.mkdir(parents=True, exist_ok=True)
                path = str(screenshot_dir / f"screenshot-{int(time.time() * 1000)}.png")
            page.screenshot(path=path, full_page=full)
            return f"✓ Screenshot saved to {path}"
        if command == "close":
            self._close()
            return "✓ Browser closed"

        raise RuntimeError(f"Unsupported browser command: {command}")


BROWSER = BrowserRuntime()
_BROWSER_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="browser-runtime")
_browser_runtime_lock = threading.Lock()

_tts_lock = threading.Lock()
_tts_engine: Optional[Any] = None
_tts_current_process: Optional[subprocess.Popen] = None
_tts_queue: "queue.Queue[Tuple[Optional[str], Optional[threading.Event]]]" = queue.Queue()
_tts_stop_event = threading.Event()
_tts_shutdown_event = threading.Event()
_tts_is_speaking = threading.Event()
_tts_worker_thread: Optional[threading.Thread] = None
_tts_last_text: str = ""
_tts_last_started_at: float = 0.0
_tts_last_ended_at: float = 0.0


def _init_tts_engine() -> None:
    global _tts_engine
    if TTS_BACKEND in {"windows", "powershell", "system"}:
        _tts_engine = None
        return
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", TTS_RATE)
        _tts_engine = engine
    except Exception as exc:
        _tts_engine = None
        log_line(f"WARN: pyttsx3 init failed ({exc}). Using PowerShell speech fallback.")


def _speak_windows_fallback(text: str, stop_event: Optional[threading.Event] = None) -> None:
    global _tts_current_process
    escaped = text.replace("'", "''")
    command = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.Speak('{escaped}')"
    )
    try:
        proc = subprocess.Popen(
            ["powershell", "-NoProfile", "-Command", command],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        with _tts_lock:
            _tts_current_process = proc
        while proc.poll() is None:
            if stop_event is not None and stop_event.is_set():
                try:
                    proc.terminate()
                    proc.wait(timeout=1.0)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                break
            time.sleep(0.05)
    except Exception as exc:
        log_line(f"WARN: fallback speech failed ({exc}).")
    finally:
        with _tts_lock:
            _tts_current_process = None


def _drain_tts_queue() -> None:
    while True:
        try:
            queued_text, done_event = _tts_queue.get_nowait()
            if done_event is not None:
                done_event.set()
            _tts_queue.task_done()
        except queue.Empty:
            break


def _tts_worker_loop() -> None:
    global _tts_engine, _tts_last_text, _tts_last_started_at, _tts_last_ended_at
    while not _tts_shutdown_event.is_set():
        try:
            item = _tts_queue.get(timeout=0.15)
        except queue.Empty:
            continue
        text, done_event = item
        if text is None:
            if done_event is not None:
                done_event.set()
            _tts_queue.task_done()
            continue
        _tts_stop_event.clear()
        _tts_is_speaking.set()
        with _tts_lock:
            _tts_last_text = text
            _tts_last_started_at = time.time()
        try:
            if TTS_BACKEND in {"windows", "powershell", "system"}:
                _speak_windows_fallback(text, _tts_stop_event)
            elif _tts_engine is not None:
                try:
                    _tts_engine.say(text)
                    _tts_engine.runAndWait()
                except Exception as exc:
                    log_line(f"WARN: pyttsx3 speak failed ({exc}). Switching to fallback.")
                    with _tts_lock:
                        _tts_engine = None
                    _speak_windows_fallback(text, _tts_stop_event)
            elif os.name == "nt":
                _speak_windows_fallback(text, _tts_stop_event)
            else:
                log_line("WARN: No working TTS backend on this platform.")
        finally:
            with _tts_lock:
                _tts_last_ended_at = time.time()
            _tts_is_speaking.clear()
            if done_event is not None:
                done_event.set()
            _tts_queue.task_done()


def _start_tts_worker() -> None:
    global _tts_worker_thread
    if _tts_worker_thread is not None and _tts_worker_thread.is_alive():
        return
    _tts_shutdown_event.clear()
    _tts_worker_thread = threading.Thread(target=_tts_worker_loop, name="voice-browser-tts", daemon=True)
    _tts_worker_thread.start()


def stop_speaking(clear_queue: bool = True) -> None:
    _tts_stop_event.set()
    with _tts_lock:
        engine = _tts_engine
        current = _tts_current_process
    if engine is not None:
        try:
            engine.stop()
        except Exception:
            pass
    if current is not None and current.poll() is None:
        try:
            current.terminate()
        except Exception:
            pass
    if clear_queue:
        _drain_tts_queue()


def is_speaking() -> bool:
    return _tts_is_speaking.is_set()


def _normalize_echo_text(text: str) -> str:
    lowered = text.strip().lower()
    lowered = re.sub(r"[^a-z0-9\s]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def is_probable_tts_echo(user_text: str) -> bool:
    if not ECHO_GUARD_ENABLED:
        return False
    normalized_user = _normalize_echo_text(user_text)
    if len(normalized_user) < 4:
        return False
    with _tts_lock:
        spoken = _tts_last_text
        started_at = _tts_last_started_at
        ended_at = _tts_last_ended_at
    if not spoken:
        return False
    now = time.time()
    if is_speaking():
        if now - started_at > ECHO_GUARD_SECONDS * 6:
            return False
    else:
        if ended_at <= 0:
            return False
        if now - ended_at > ECHO_GUARD_SECONDS:
            return False
    normalized_spoken = _normalize_echo_text(spoken)
    if len(normalized_spoken) < 4:
        return False
    if len(normalized_user) >= 6 and normalized_user in normalized_spoken:
        return True
    similarity = SequenceMatcher(None, normalized_user, normalized_spoken).ratio()
    return similarity >= 0.84


def shutdown_tts() -> None:
    _tts_shutdown_event.set()
    stop_speaking(clear_queue=True)
    _tts_queue.put((None, None))
    if _tts_worker_thread is not None:
        _tts_worker_thread.join(timeout=1.5)


_init_tts_engine()
_start_tts_worker()


def speak(text: str, wait: bool = False) -> None:
    log_line(f"  TTS: {text}")
    _start_tts_worker()
    stop_speaking(clear_queue=True)
    if wait:
        done = threading.Event()
        _tts_queue.put((text, done))
        done.wait(timeout=30)
        return
    _tts_queue.put((text, None))


def speak_and_wait(text: str) -> None:
    speak(text, wait=True)


def _reset_browser_runtime(reason: str) -> None:
    global BROWSER, _BROWSER_EXECUTOR
    log_line(f"WARN: Resetting browser runtime ({reason}).")
    with _browser_runtime_lock:
        old_browser = BROWSER
        old_executor = _BROWSER_EXECUTOR
        BROWSER = BrowserRuntime()
        _BROWSER_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="browser-runtime")
    try:
        old_browser._close()
    except Exception:
        pass
    try:
        old_executor.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass


def run_ab(args: List[str]) -> Tuple[str, str, int]:
    log_line(f"  RUN: {' '.join(args)}")
    with _browser_runtime_lock:
        browser = BROWSER
        executor = _BROWSER_EXECUTOR
    future = None
    try:
        future = executor.submit(browser.execute, args)
        output = future.result(timeout=60)
        return output.strip(), "", 0
    except FuturesTimeoutError:
        if future is not None:
            future.cancel()
        _reset_browser_runtime("command timeout")
        return "", "Browser command timed out after 60 seconds. Runtime restarted.", 1
    except Exception as exc:
        if isinstance(exc, PlaywrightError):
            err = compact_playwright_error(exc)
        else:
            err = str(exc)
        lowered = err.lower()
        if args and args[0] == "close" and (
            "browser has been closed" in lowered
            or "target page, context or browser has been closed" in lowered
        ):
            return "✓ Browser closed", "", 0
        if "browser has been closed" in lowered or "target page, context or browser has been closed" in lowered:
            _reset_browser_runtime(err)
        return "", err, 1


def acquire_single_instance_lock() -> Optional[Any]:
    if os.name != "nt":
        return None
    try:
        import ctypes

        handle = ctypes.windll.kernel32.CreateMutexW(None, False, "Global\\VoiceBrowserSingleton")
        if not handle:
            return None
        if ctypes.windll.kernel32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
            ctypes.windll.kernel32.CloseHandle(handle)
            return None
        return handle
    except Exception:
        return None


def release_single_instance_lock(handle: Optional[Any]) -> None:
    if os.name != "nt" or not handle:
        return
    try:
        import ctypes

        ctypes.windll.kernel32.ReleaseMutex(handle)
        ctypes.windll.kernel32.CloseHandle(handle)
    except Exception:
        pass


def run_ab_ok(args: List[str]) -> str:
    out, err, code = run_ab(args)
    if code != 0:
        message = err or f"Browser command failed: {' '.join(args)}"
        log_line(f"  ERROR: {message}")
        raise BrowserCommandError(message)
    return out


def create_microphone() -> Tuple[sr.Microphone, Optional[int], str, List[str]]:
    names = sr.Microphone.list_microphone_names()
    if not names:
        raise RuntimeError("No microphone devices found.")

    selected_index: Optional[int] = None
    if MIC_INDEX_ENV:
        try:
            selected_index = int(MIC_INDEX_ENV)
        except ValueError:
            log_line(f"WARN: Invalid VOICE_BROWSER_MIC_INDEX='{MIC_INDEX_ENV}'. Using system default.")

    if selected_index is not None and (selected_index < 0 or selected_index >= len(names)):
        log_line(
            f"WARN: VOICE_BROWSER_MIC_INDEX {selected_index} is out of range (0-{len(names)-1}). "
            "Using system default."
        )
        selected_index = None

    if selected_index is None:
        selected_name = "System default microphone"
    else:
        selected_name = f"{selected_index}: {names[selected_index]}"

    log_line("Available microphones:")
    display_count = len(names) if LIST_ALL_MICROPHONES else min(15, len(names))
    for idx, name in enumerate(names[:display_count]):
        marker = "*" if selected_index == idx else " "
        log_line(f"  {marker} [{idx}] {name}")
    if display_count < len(names):
        log_line(
            f"  ... {len(names) - display_count} more devices. "
            "Set VOICE_BROWSER_LIST_ALL_MICS=1 to list all."
        )

    mic = sr.Microphone(device_index=selected_index)
    return mic, selected_index, selected_name, names


def parse_snapshot_elements(snapshot: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    pattern = re.compile(r'^\s*-\s+([a-zA-Z0-9_]+)\s+"(.*?)"\s+\[ref=(e\d+)\]')
    for line in snapshot.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        role, name, ref = match.groups()
        items.append({"role": role, "name": name, "ref": f"@{ref}"})
    return items


def refresh_page_state(state: BrowserState, mode: str = "full") -> None:
    refresh_mode = mode if mode in {"basic", "snapshot", "full"} else "full"
    if refresh_mode in {"basic", "full"}:
        try:
            raw = run_ab_ok(["state"])
            payload = json.loads(raw)
            if isinstance(payload, dict):
                state.title = str(payload.get("title", "")).strip()
                state.url = str(payload.get("url", "")).strip()
                state.tabs = str(payload.get("tabs", "")).strip()
        except Exception:
            state.title = run_ab_ok(["get", "title"])
            state.url = run_ab_ok(["get", "url"])
            state.tabs = run_ab_ok(["tab"])
    if refresh_mode in {"snapshot", "full"}:
        state.snapshot = run_ab_ok(["snapshot", "-i", "-c"])
        state.elements = parse_snapshot_elements(state.snapshot)
        state.snapshot_updated_at = time.time()


def should_refresh_elements_for_utterance(user_text: str, state: BrowserState) -> bool:
    lowered = user_text.lower()
    if not state.elements:
        return True
    if any(word in lowered for word in ("click", "press", "tap", "select", "fill", "type")):
        if time.time() - state.snapshot_updated_at > SNAPSHOT_STALE_SECONDS:
            return True
    return False


def push_history(state: BrowserState, user_text: str, response: str) -> None:
    state.recent_history.append({"user": user_text, "assistant": response})
    if len(state.recent_history) > MAX_HISTORY_ITEMS:
        state.recent_history = state.recent_history[-MAX_HISTORY_ITEMS:]


PLANNER_PROMPT = """You are Voice Browser Planner.
Return ONLY a valid JSON object with this exact shape:
{
  "quit": boolean,
  "needs_clarification": boolean,
  "clarification_question": string,
  "confirmation_prompt": string,
  "spoken_response": string,
  "actions": [
    { "type": "open", "url": "https://example.com" }
  ]
}

Rules:
1) Output JSON only. No markdown and no code fences.
2) Use only allowed action types.
3) Keep spoken_response short and friendly.
4) If ambiguous, set needs_clarification=true and ask one concise question.
5) For quit/exit intent, set quit=true.
6) If action can close tab/browser, include confirmation_prompt.
7) Prefer refs from context elements when clicking/filling.
8) Use recent_history for follow-up requests like "click the first one" or "that button".

Allowed action types:
open, back, forward, reload, click, fill, type_text, press, scroll,
tab_new, tab_switch, tab_close, tab_list, set_viewport, maximize_window,
zoom_in, zoom_out, zoom_reset, fullscreen_toggle, read_main, list_actions,
find_on_page, set_media, search_web, screenshot, snapshot, close_browser.
"""


def build_planner_input(user_text: str, state: BrowserState) -> str:
    context = {
        "title": state.title,
        "url": state.url,
        "tabs": state.tabs,
        "elements": state.elements[:MAX_CONTEXT_ELEMENTS],
        "recent_history": state.recent_history[-MAX_HISTORY_ITEMS:],
    }
    return (
        f"{PLANNER_PROMPT}\n\n"
        f"CURRENT_CONTEXT_JSON:\n{json.dumps(context, ensure_ascii=True)}\n\n"
        f'USER_UTTERANCE:\n{json.dumps(user_text, ensure_ascii=True)}\n'
    )


def extract_json_object(text: str) -> Optional[str]:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, end = decoder.raw_decode(text[index:])
            if isinstance(payload, dict):
                return text[index:index + end]
        except json.JSONDecodeError:
            continue
    return None


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return default


def normalize_action_type(action_type: str) -> str:
    lowered = action_type.strip().lower()
    return ACTION_ALIASES.get(lowered, lowered)


def sanitize_action(raw_action: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not isinstance(raw_action, dict):
        return None, "Action item must be an object."
    action_type = normalize_action_type(str(raw_action.get("type", "")))
    if action_type not in ALLOWED_ACTIONS:
        return None, f"Unsupported action type '{action_type}'."

    clean: Dict[str, Any] = {"type": action_type}
    if action_type == "open":
        url = str(raw_action.get("url", "")).strip()
        if not url:
            return None, "Action 'open' requires url."
        clean["url"] = url
    elif action_type in {"click", "fill", "type_text"}:
        target = str(raw_action.get("target", "")).strip()
        if not target:
            return None, f"Action '{action_type}' requires target."
        clean["target"] = target
        if action_type in {"fill", "type_text"}:
            clean["text"] = str(raw_action.get("text", ""))
    elif action_type == "press":
        key = str(raw_action.get("key", "")).strip()
        if not key:
            return None, "Action 'press' requires key."
        clean["key"] = key
    elif action_type == "scroll":
        direction = str(raw_action.get("direction", "down")).strip().lower()
        if direction not in {"up", "down", "left", "right"}:
            direction = "down"
        clean["direction"] = direction
        clean["pixels"] = max(100, min(5000, _to_int(raw_action.get("pixels", 500), 500)))
    elif action_type == "tab_new":
        clean["url"] = str(raw_action.get("url", "")).strip()
    elif action_type == "tab_switch":
        index = _to_int(raw_action.get("index"), 1)
        clean["index"] = max(1, index)
    elif action_type == "tab_close":
        if "index" in raw_action and raw_action.get("index") is not None:
            index = _to_int(raw_action.get("index"), 1)
            clean["index"] = max(1, index)
    elif action_type == "set_viewport":
        width = max(320, min(5000, _to_int(raw_action.get("width"), 1280)))
        height = max(240, min(5000, _to_int(raw_action.get("height"), 900)))
        clean["width"] = width
        clean["height"] = height
    elif action_type == "find_on_page":
        text = str(raw_action.get("text", "")).strip()
        if not text:
            return None, "Action 'find_on_page' requires text."
        clean["text"] = text
    elif action_type == "set_media":
        mode = str(raw_action.get("mode", "")).strip().lower()
        if mode not in {"dark", "light"}:
            return None, "Action 'set_media' mode must be dark or light."
        clean["mode"] = mode
    elif action_type == "search_web":
        query = str(raw_action.get("query", "")).strip()
        if not query:
            return None, "Action 'search_web' requires query."
        clean["query"] = query
    elif action_type == "screenshot":
        clean["path"] = str(raw_action.get("path", "")).strip()
        clean["full"] = _to_bool(raw_action.get("full"), False)

    return clean, None


def parse_planner_response(raw_response: Optional[str]) -> Tuple[PlannerResult, List[str]]:
    result = PlannerResult()
    errors: List[str] = []
    if not raw_response:
        errors.append("No assistant response received.")
        return result, errors

    payload_text = extract_json_object(raw_response)
    if not payload_text:
        errors.append("Assistant response did not contain valid JSON.")
        return result, errors

    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        errors.append(f"JSON parse error: {exc}")
        return result, errors

    if not isinstance(payload, dict):
        errors.append("JSON payload must be an object.")
        return result, errors

    result.quit = bool(payload.get("quit", False))
    result.needs_clarification = bool(payload.get("needs_clarification", False))
    result.clarification_question = str(payload.get("clarification_question", "")).strip()
    result.confirmation_prompt = str(payload.get("confirmation_prompt", "")).strip()
    result.spoken_response = str(payload.get("spoken_response", "")).strip()

    actions_value = payload.get("actions", [])
    if not isinstance(actions_value, list):
        errors.append("Field 'actions' must be an array.")
        return result, errors

    for raw_action in actions_value:
        action, action_error = sanitize_action(raw_action)
        if action_error:
            errors.append(action_error)
            continue
        if action:
            result.actions.append(action)

    return result, errors


def normalize_url(raw_url: str) -> str:
    text = raw_url.strip()
    text = text.replace(" dot ", ".")
    text = text.replace(" slash ", "/")
    text = text.replace(" h t t p s ", " https ")
    text = re.sub(r"\s+", "", text)
    if not re.match(r"^[a-zA-Z]+://", text):
        text = "https://" + text
    return text


def summarize_interactive_elements(state: BrowserState, limit: int = 8) -> str:
    if not state.elements:
        return "I do not see interactive items yet."
    preview = state.elements[:limit]
    text = ", ".join([f'{item["name"]} ({item["ref"]})' for item in preview if item["name"]])
    if not text:
        return "I found interactive elements but could not summarize names."
    return f"I can interact with items such as: {text}."


def normalize_tokens(text: str) -> List[str]:
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9\s]+", " ", lowered)
    return [token for token in lowered.split() if token]


def extract_role_hint(text: str) -> Optional[str]:
    lowered = text.lower()
    for keyword, role in ROLE_HINTS.items():
        if re.search(rf"\b{re.escape(keyword)}\b", lowered):
            return role
    return None


def extract_ordinal_index(text: str) -> Optional[int]:
    tokens = normalize_tokens(text)
    for token in tokens:
        if token in ORDINAL_WORDS:
            return ORDINAL_WORDS[token]
    ordinal_match = re.search(r"\b(\d+)(?:st|nd|rd|th)\b", text.lower())
    if ordinal_match:
        value = int(ordinal_match.group(1))
        if value > 0:
            return value - 1
    return None


def extract_click_phrase(text: str) -> str:
    match = re.search(r"\b(?:click|press|tap|select|open)\b(?:\s+on)?\s+(.+)$", text, re.IGNORECASE)
    phrase = match.group(1) if match else text
    phrase = phrase.strip().strip(".!?")
    phrase = re.sub(r"^(?:the|a|an)\s+", "", phrase, flags=re.IGNORECASE)
    return phrase


def normalize_click_query(phrase: str) -> str:
    cleaned = phrase.strip()
    cleaned = re.sub(r"\b(?:that|which)\s+(?:says?|reads?)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:the|a|an)\s+", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def looks_like_selector(text: str) -> bool:
    value = text.strip()
    if not value:
        return False
    lowered = value.lower()
    if lowered.startswith(("css=", "xpath=", "//", "(//")):
        return True
    if any(ch in value for ch in ("[", "]", "(", ")", ">", ":", "#", "=")):
        return True
    return False


def resolve_ref_from_text(target_text: str, state: BrowserState) -> Optional[str]:
    target = target_text.strip()
    if not target:
        return None

    direct_ref = re.search(r"@?e(\d+)\b", target.lower())
    if direct_ref:
        return f"@e{direct_ref.group(1)}"

    phrase = normalize_click_query(extract_click_phrase(target))
    role_hint = extract_role_hint(phrase)
    ordinal_index = extract_ordinal_index(phrase)

    candidates = [item for item in state.elements if item.get("ref")]
    if role_hint:
        role_candidates = [item for item in candidates if item.get("role", "").lower() == role_hint]
        if role_candidates:
            candidates = role_candidates

    if ordinal_index is not None and candidates:
        bounded = max(0, min(ordinal_index, len(candidates) - 1))
        return candidates[bounded]["ref"]

    phrase_tokens = [tok for tok in normalize_tokens(phrase) if tok not in TARGET_STOP_WORDS and tok not in ORDINAL_WORDS]
    if not phrase_tokens:
        return None
    phrase_text = " ".join(phrase_tokens)

    scored: List[Tuple[int, int, str]] = []
    for idx, item in enumerate(candidates):
        name = (item.get("name") or "").lower()
        if not name:
            continue
        name_tokens = normalize_tokens(name)
        score = 0
        if phrase_text and phrase_text in name:
            score += 25
            if name.startswith(phrase_text):
                score += 6
        phrase_similarity = SequenceMatcher(None, phrase_text, name).ratio()
        if phrase_similarity >= 0.72:
            score += int(phrase_similarity * 18)
        for token in phrase_tokens:
            if token in name:
                score += 8
                continue
            best_token_similarity = 0.0
            for name_token in name_tokens:
                if len(name_token) < 2:
                    continue
                similarity = SequenceMatcher(None, token, name_token).ratio()
                if similarity > best_token_similarity:
                    best_token_similarity = similarity
            if best_token_similarity >= 0.8:
                score += int(best_token_similarity * 7)
        if role_hint and item.get("role", "").lower() == role_hint:
            score += 10
        if score > 0:
            scored.append((score, -idx, item["ref"]))

    if not scored:
        return None
    scored.sort(reverse=True)
    if scored[0][0] < 8:
        return None
    return scored[0][2]


def resolve_action_target(target: str, state: BrowserState) -> str:
    trimmed = target.strip()
    if not trimmed:
        return target
    if re.fullmatch(r"@?e\d+", trimmed.lower()):
        return f"@{trimmed.lstrip('@').lower()}"
    if looks_like_selector(trimmed):
        return trimmed
    resolved = resolve_ref_from_text(trimmed, state)
    return resolved if resolved else trimmed


def build_click_fallback_plan(user_text: str, state: BrowserState) -> Optional[PlannerResult]:
    lowered = user_text.strip().lower()
    if not any(word in lowered for word in ("click", "press", "tap", "select")):
        return None
    target = resolve_ref_from_text(user_text, state)
    if not target:
        return None
    return PlannerResult(actions=[{"type": "click", "target": target}], spoken_response="Clicked.")


def _extract_tab_index_from_text(text: str) -> Optional[int]:
    match = re.search(r"\btab\s+(\d+)\b", text)
    if match:
        value = int(match.group(1))
        return value if value > 0 else None
    any_number = re.search(r"\b(\d+)\b", text)
    if any_number:
        value = int(any_number.group(1))
        return value if value > 0 else None
    ordinal = extract_ordinal_index(text)
    if ordinal is not None:
        return ordinal + 1
    return None


def _extract_url_candidate(text: str) -> Optional[str]:
    match = re.search(
        r"(https?://[^\s]+|www\.[^\s]+|[a-z0-9][a-z0-9-]*(?:\.[a-z0-9-]+)+(?:/[^\s]*)?)",
        text,
        re.IGNORECASE,
    )
    if match:
        candidate = match.group(1).strip().rstrip(".,!?;:")
        if candidate:
            return candidate
    path_match = re.search(r"\b(?:go to|open|navigate to)\s+(.+)$", text, re.IGNORECASE)
    if not path_match:
        return None
    candidate = path_match.group(1).strip().rstrip(".,!?;:")
    candidate = re.sub(r"\s+in\s+(?:a\s+)?new\s+tab.*$", "", candidate, flags=re.IGNORECASE).strip()
    if any(part in candidate.lower() for part in (" dot ", ".", " slash ", "/", "http", "www")):
        return candidate
    return None


def build_fast_path_plan(user_text: str, state: BrowserState) -> Optional[PlannerResult]:
    utterance = user_text.strip()
    lowered = utterance.lower()
    if not lowered:
        return None

    if re.search(r"\b(quit|exit|goodbye|close browser)\b", lowered):
        return PlannerResult(actions=[{"type": "close_browser"}], spoken_response="Closing browser.")
    if re.search(r"\b(go back|back|previous page)\b", lowered):
        return PlannerResult(actions=[{"type": "back"}], spoken_response="Went back.")
    if re.search(r"\b(go forward|forward|next page)\b", lowered):
        return PlannerResult(actions=[{"type": "forward"}], spoken_response="Went forward.")
    if re.search(r"\b(reload|refresh)\b", lowered):
        return PlannerResult(actions=[{"type": "reload"}], spoken_response="Reloaded.")

    if "page down" in lowered:
        return PlannerResult(actions=[{"type": "press", "key": "PageDown"}], spoken_response="Scrolled down.")
    if "page up" in lowered:
        return PlannerResult(actions=[{"type": "press", "key": "PageUp"}], spoken_response="Scrolled up.")
    if re.search(r"\b(go to )?(top|start of page)\b", lowered):
        return PlannerResult(actions=[{"type": "press", "key": "Home"}], spoken_response="Went to top.")
    if re.search(r"\b(go to )?(bottom|end of page)\b", lowered):
        return PlannerResult(actions=[{"type": "press", "key": "End"}], spoken_response="Went to bottom.")
    if "scroll" in lowered:
        direction = "down"
        if " up" in f" {lowered}" or lowered.endswith("up"):
            direction = "up"
        elif "left" in lowered:
            direction = "left"
        elif "right" in lowered:
            direction = "right"
        pixels = 500
        if any(word in lowered for word in ("way", "lot", "far")):
            pixels = 900
        elif any(word in lowered for word in ("little", "bit", "slightly", "small")):
            pixels = 260
        return PlannerResult(
            actions=[{"type": "scroll", "direction": direction, "pixels": pixels}],
            spoken_response=f"Scrolled {direction}.",
        )

    if re.search(r"\b(zoom in|make (it )?bigger|larger)\b", lowered):
        return PlannerResult(actions=[{"type": "zoom_in"}], spoken_response="Zoomed in.")
    if re.search(r"\b(zoom out|make (it )?smaller)\b", lowered):
        return PlannerResult(actions=[{"type": "zoom_out"}], spoken_response="Zoomed out.")
    if re.search(r"\b(reset zoom|normal size|zoom reset)\b", lowered):
        return PlannerResult(actions=[{"type": "zoom_reset"}], spoken_response="Reset zoom.")
    if "maximize" in lowered:
        return PlannerResult(actions=[{"type": "maximize_window"}], spoken_response="Maximized.")
    if re.search(r"\b(fullscreen|full screen)\b", lowered):
        return PlannerResult(actions=[{"type": "fullscreen_toggle"}], spoken_response="Toggled full screen.")

    if re.search(r"\b(list tabs|show tabs|how many tabs|what tabs)\b", lowered):
        return PlannerResult(actions=[{"type": "tab_list"}], spoken_response="Here are your tabs.")
    if re.search(r"\b(close( this)? tab)\b", lowered):
        return PlannerResult(actions=[{"type": "tab_close"}], spoken_response="Closing tab.")
    if re.search(r"\b(new tab|open tab)\b", lowered):
        url_candidate = _extract_url_candidate(utterance)
        if url_candidate:
            return PlannerResult(
                actions=[{"type": "tab_new", "url": normalize_url(url_candidate)}],
                spoken_response="Opened a new tab.",
            )
        return PlannerResult(actions=[{"type": "tab_new"}], spoken_response="Opened a new tab.")
    if ("switch" in lowered and "tab" in lowered) or re.search(r"\btab\s+\d+\b", lowered):
        tab_index = _extract_tab_index_from_text(lowered)
        if tab_index is not None:
            return PlannerResult(
                actions=[{"type": "tab_switch", "index": tab_index}],
                spoken_response=f"Switched to tab {tab_index}.",
            )

    search_match = re.search(r"\bsearch(?: the web)? for\s+(.+)$", utterance, re.IGNORECASE)
    if search_match:
        query = search_match.group(1).strip().rstrip(".!?")
        if query:
            return PlannerResult(actions=[{"type": "search_web", "query": query}], spoken_response=f"Searched for {query}.")

    find_match = re.search(r"\bfind(?: on page)?\s+(.+)$", utterance, re.IGNORECASE)
    if find_match and "find me" not in lowered:
        query = find_match.group(1).strip().rstrip(".!?")
        if query:
            return PlannerResult(actions=[{"type": "find_on_page", "text": query}], spoken_response=f"Searched for {query}.")

    if re.search(r"\b(what can i click|what links|list links|list actions|show links)\b", lowered):
        return PlannerResult(actions=[{"type": "list_actions"}], spoken_response="Here are clickable items.")
    if re.search(r"\b(read page|read this page|read main|read main content)\b", lowered):
        return PlannerResult(actions=[{"type": "read_main"}], spoken_response="")

    url_candidate = _extract_url_candidate(utterance)
    if url_candidate and re.search(r"\b(go to|open|navigate to)\b", lowered):
        normalized = normalize_url(url_candidate)
        if "new tab" in lowered:
            return PlannerResult(actions=[{"type": "tab_new", "url": normalized}], spoken_response="Opened a new tab.")
        return PlannerResult(actions=[{"type": "open", "url": normalized}], spoken_response=f"Opened {normalized}.")

    email_match = re.search(
        r"\b(?:click|open)\s+(?:on\s+)?(?:the\s+)?(?:first|1st)\s+email\s+from\s+(.+)$",
        utterance,
        re.IGNORECASE,
    )
    if email_match:
        sender = email_match.group(1).strip().strip(".!?")
        if sender:
            return PlannerResult(
                actions=[{"type": "click", "target": f"outlook_sender:{sender}"}],
                spoken_response=f"Opening first email from {sender}.",
            )

    if any(word in lowered for word in ("click", "press", "tap", "select")):
        target = resolve_ref_from_text(utterance, state)
        if target is None:
            target = normalize_click_query(extract_click_phrase(utterance))
        if target:
            return PlannerResult(actions=[{"type": "click", "target": target}], spoken_response="Clicked.")

    return None


def execute_action(action: Dict[str, Any], state: BrowserState) -> Tuple[str, str, bool]:
    action_type = action["type"]

    if action_type == "open":
        url = normalize_url(action["url"])
        run_ab_ok(["open", url])
        return f"Opened {url}.", "full", False
    if action_type == "back":
        run_ab_ok(["back"])
        return "Went back.", "full", False
    if action_type == "forward":
        run_ab_ok(["forward"])
        return "Went forward.", "full", False
    if action_type == "reload":
        run_ab_ok(["reload"])
        return "Reloaded the page.", "full", False
    if action_type == "click":
        target = resolve_action_target(action["target"], state)
        run_ab_ok(["click", target])
        return "Clicked the requested item.", "snapshot", False
    if action_type == "fill":
        target = resolve_action_target(action["target"], state)
        run_ab_ok(["fill", target, action["text"]])
        return "Filled the field.", "snapshot", False
    if action_type == "type_text":
        target = resolve_action_target(action["target"], state)
        run_ab_ok(["type", target, action["text"]])
        return "Typed text.", "snapshot", False
    if action_type == "press":
        run_ab_ok(["press", action["key"]])
        return f"Pressed {action['key']}.", "snapshot", False
    if action_type == "scroll":
        run_ab_ok(["scroll", action["direction"], str(action["pixels"])])
        return f"Scrolled {action['direction']}.", "snapshot", False
    if action_type == "tab_new":
        if action.get("url"):
            run_ab_ok(["tab", "new", normalize_url(action["url"])])
        else:
            run_ab_ok(["tab", "new"])
        return "Opened a new tab.", "full", False
    if action_type == "tab_switch":
        run_ab_ok(["tab", str(action["index"])])
        return f"Switched to tab {action['index']}.", "full", False
    if action_type == "tab_close":
        if "index" in action:
            run_ab_ok(["tab", "close", str(action["index"])])
        else:
            run_ab_ok(["tab", "close"])
        return "Closed the tab.", "full", False
    if action_type == "tab_list":
        tabs = run_ab_ok(["tab"])
        return tabs if tabs else "Listed tabs.", "none", False
    if action_type == "set_viewport":
        run_ab_ok(["set", "viewport", str(action["width"]), str(action["height"])])
        return f"Set viewport to {action['width']} by {action['height']}.", "snapshot", False
    if action_type == "maximize_window":
        run_ab_ok(["maximize"])
        return "Maximized the browser window.", "snapshot", False
    if action_type == "zoom_in":
        run_ab_ok(["press", "Control+Equal"])
        return "Zoomed in.", "snapshot", False
    if action_type == "zoom_out":
        run_ab_ok(["press", "Control+Minus"])
        return "Zoomed out.", "snapshot", False
    if action_type == "zoom_reset":
        run_ab_ok(["press", "Control+0"])
        return "Reset zoom.", "snapshot", False
    if action_type == "fullscreen_toggle":
        run_ab_ok(["press", "F11"])
        return "Toggled full screen.", "snapshot", False
    if action_type == "read_main":
        text = run_ab_ok(
            [
                "eval",
                (
                    "(() => { const main = document.querySelector('main'); "
                    "const src = (main && main.innerText && main.innerText.trim()) ? main : document.body; "
                    "return src ? src.innerText : ''; })()"
                ),
            ]
        )
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return "I could not read page content.", "none", False
        return text[:500], "none", False
    if action_type == "list_actions":
        if not state.elements:
            refresh_page_state(state, mode="snapshot")
        return summarize_interactive_elements(state), "none", False
    if action_type == "find_on_page":
        run_ab_ok(["eval", f"window.find({json.dumps(action['text'])})"])
        return f"Searched for '{action['text']}' on the page.", "none", False
    if action_type == "set_media":
        run_ab_ok(["set", "media", action["mode"]])
        return f"Set media mode to {action['mode']}.", "none", False
    if action_type == "search_web":
        query = action["query"]
        search_url = "https://www.google.com/search?q=" + quote_plus(query)
        run_ab_ok(["open", search_url])
        return f"Searched the web for {query}.", "full", False
    if action_type == "screenshot":
        cmd = ["screenshot"]
        if action.get("full"):
            cmd.append("--full")
        if action.get("path"):
            cmd.append(action["path"])
        output = run_ab_ok(cmd)
        return output if output else "Captured a screenshot.", "none", False
    if action_type == "snapshot":
        refresh_page_state(state, mode="snapshot")
        return "Refreshed page elements.", "none", False
    if action_type == "close_browser":
        run_ab_ok(["close"])
        return "Closed the browser.", "none", True

    return f"Skipped unsupported action '{action_type}'.", "none", False


def _merge_refresh_mode(current: str, candidate: str) -> str:
    order = {"none": 0, "basic": 1, "snapshot": 2, "full": 3}
    cur = current if current in order else "none"
    can = candidate if candidate in order else "none"
    return can if order[can] > order[cur] else cur


def execute_actions(actions: List[Dict[str, Any]], state: BrowserState) -> Tuple[List[str], bool, bool]:
    messages: List[str] = []
    should_stop = False
    refresh_mode = "none"
    had_error = False

    for action in actions:
        try:
            message, refresh_after, stop_after = execute_action(action, state)
        except BrowserCommandError as exc:
            messages.append(f"I couldn't complete that action: {exc}")
            had_error = True
            break
        except Exception as exc:
            messages.append(f"I couldn't complete that action: {exc}")
            had_error = True
            break
        if message:
            messages.append(message)
        refresh_mode = _merge_refresh_mode(refresh_mode, refresh_after)
        should_stop = should_stop or stop_after
        if should_stop:
            break

    if refresh_mode != "none" and not should_stop and not had_error:
        try:
            refresh_page_state(state, mode=refresh_mode)
        except BrowserCommandError as exc:
            messages.append(f"I couldn't refresh browser context: {exc}")
            had_error = True
    return messages, should_stop, had_error


def _contains_phrase(text: str, phrase: str) -> bool:
    escaped = re.escape(phrase.strip())
    escaped = escaped.replace(r"\ ", r"\s+")
    return re.search(rf"(?<!\w){escaped}(?!\w)", text) is not None


def classify_yes_no(text: str) -> Optional[bool]:
    lowered = text.strip().lower()
    cleaned = re.sub(r"[^a-z0-9'\s]+", " ", lowered)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return None
    cleaned_no_apostrophe = cleaned.replace("'", "")
    if any(
        _contains_phrase(cleaned, word)
        or _contains_phrase(cleaned_no_apostrophe, word.replace("'", ""))
        for word in NO_WORDS
    ):
        return False
    if any(
        _contains_phrase(cleaned, word)
        or _contains_phrase(cleaned_no_apostrophe, word.replace("'", ""))
        for word in YES_WORDS
    ):
        return True
    return None


def build_confirmation_prompt(plan: PlannerResult) -> str:
    if plan.confirmation_prompt:
        return plan.confirmation_prompt
    destructive = [a["type"] for a in plan.actions if a["type"] in DESTRUCTIVE_ACTIONS]
    if destructive:
        return "This action can close a tab or browser. Do you want me to continue? Please say yes or no."
    return "Please confirm by saying yes or no."


async def create_copilot_session() -> Tuple[CopilotClient, Any]:
    client = CopilotClient()
    await client.start()
    session = await client.create_session(SessionConfig(model=MODEL))
    return client, session


async def ask_copilot(session: Any, prompt: str) -> Optional[str]:
    event = await session.send_and_wait(MessageOptions(prompt=prompt), timeout=LLM_TIMEOUT_SECONDS)
    if event and event.type == SessionEventType.ASSISTANT_MESSAGE:
        return event.data.content
    return None


async def reconnect_copilot_session(current_client: CopilotClient) -> Tuple[CopilotClient, Any]:
    try:
        await current_client.stop()
    except Exception:
        pass
    return await create_copilot_session()


def _calibrate_microphone(recognizer: sr.Recognizer, microphone: sr.Microphone, duration: float) -> None:
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=duration)


def _capture_audio(recognizer: sr.Recognizer, microphone: sr.Microphone) -> sr.AudioData:
    with microphone as source:
        return recognizer.listen(
            source,
            timeout=LISTEN_TIMEOUT,
            phrase_time_limit=PHRASE_TIME_LIMIT,
        )


async def main() -> None:
    global _ui_logger, _effective_stt_backend
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = max(0.3, PAUSE_THRESHOLD)
    recognizer.non_speaking_duration = max(0.1, NON_SPEAKING_DURATION)
    recognizer.phrase_threshold = max(0.1, PHRASE_THRESHOLD)
    microphone, selected_mic_index, selected_mic_name, microphone_names = create_microphone()
    state = BrowserState()
    unknown_streak = 0
    planner_failure_streak = 0
    ui: Optional[MiniControlUI] = None

    if MINI_UI_ENABLED:
        ui = MiniControlUI(microphone_names, selected_mic_index)
        ui.start()
        ui.set_status("Starting...")
        _ui_logger = ui

    _effective_stt_backend = _resolve_stt_backend()
    if _effective_stt_backend == "faster-whisper":
        log_line(f"STT backend: faster-whisper (local, model={LOCAL_STT_MODEL})")
        try:
            await asyncio.to_thread(_get_local_stt_model)
            log_line(
                f"Local STT model ready (device={_local_stt_active_device}, compute={_local_stt_active_compute_type})."
            )
            if STT_DEBUG:
                log_line("STT debug logging: enabled")
        except Exception as exc:
            log_line(f"WARN: local STT unavailable ({exc}); using Google STT.")
            _effective_stt_backend = "google"
    else:
        if STT_BACKEND == "faster-whisper":
            log_line("WARN: VOICE_BROWSER_STT_BACKEND=faster-whisper but dependencies are missing; using Google STT.")
        log_line("STT backend: google")
    log_line(f"TTS backend: {TTS_BACKEND}")
    log_line("Browser mode: headed (always).")
    if SPEAK_STARTUP_STATUS:
        speak_and_wait("Voice Browser starting. Calibrating microphone.")
    if selected_mic_index is None:
        log_line(f"Using microphone: {selected_mic_name}")
    else:
        log_line(f"Using microphone index {selected_mic_index}: {selected_mic_name}")
    await asyncio.to_thread(_calibrate_microphone, recognizer, microphone, 1.5)

    if SPEAK_STARTUP_STATUS:
        speak_and_wait("Connecting to GitHub Copilot.")
    if ui:
        ui.set_status("Connecting to Copilot...")
    client, session = await create_copilot_session()

    initial_url = "https://www.hanselman.com"
    if len(sys.argv) > 1:
        initial_url = normalize_url(sys.argv[1])

    try:
        await asyncio.to_thread(run_ab_ok, ["open", initial_url])
    except BrowserCommandError as exc:
        speak_and_wait(f"Browser launch failed: {exc}")
        if ui:
            ui.set_status("Browser launch failed")
        return
    if BROWSER.profile_note:
        log_line(BROWSER.profile_note)
        profile_note_lower = BROWSER.profile_note.lower()
        if "fallback profile" in profile_note_lower:
            speak("Could not use your active Edge profile. Using a fallback profile.")
    await asyncio.to_thread(refresh_page_state, state, "full")
    if SPEAK_READY_MESSAGE:
        speak_and_wait("Ready.")
    else:
        log_line("Ready.")
    if ui:
        ui.set_status("Ready")

    running = True
    while running:
        log_line("\nListening...")
        try:
            typed_utterance = ""
            if ui:
                event = ui.poll_event()
                while event is not None:
                    event_type = str(event.get("type", ""))
                    if event_type == "stop_speech":
                        stop_speaking(clear_queue=True)
                    elif event_type == "quit":
                        running = False
                        break
                    elif event_type == "set_mic":
                        mic_index = event.get("index")
                        if mic_index is None:
                            microphone = sr.Microphone(device_index=None)
                            selected_mic_index = None
                            selected_mic_name = "System default microphone"
                        elif isinstance(mic_index, int) and 0 <= mic_index < len(microphone_names):
                            microphone = sr.Microphone(device_index=mic_index)
                            selected_mic_index = mic_index
                            selected_mic_name = f"{mic_index}: {microphone_names[mic_index]}"
                        else:
                            speak("That microphone index is not valid.")
                            event = ui.poll_event()
                            continue
                        await asyncio.to_thread(_calibrate_microphone, recognizer, microphone, 0.8)
                        msg = f"Using microphone: {selected_mic_name}"
                        log_line(msg)
                        speak(msg)
                        ui.set_status(msg)
                    elif event_type == "utterance":
                        typed_utterance = str(event.get("text", "")).strip()
                    event = ui.poll_event()

            if not running:
                break

            if typed_utterance:
                stop_speaking(clear_queue=False)
                user_text = typed_utterance
                log_line(f'  Typed: "{user_text}"')
            else:
                if ui:
                    ui.set_status("Listening...")
                if not BARGE_IN_ENABLED:
                    while is_speaking():
                        await asyncio.sleep(0.05)
                was_speaking_before_listen = is_speaking()
                audio = await asyncio.to_thread(_capture_audio, recognizer, microphone)
                user_text = (await asyncio.to_thread(transcribe_audio, recognizer, audio)).strip()
                if BARGE_IN_ENABLED and is_probable_tts_echo(user_text):
                    log_line("  Ignored speaker echo.")
                    if ui:
                        ui.set_status("Ready")
                    continue
                if BARGE_IN_ENABLED and was_speaking_before_listen:
                    stop_speaking(clear_queue=False)
                log_line(f'  Heard: "{user_text}"')

            if not user_text:
                continue
            unknown_streak = 0
            if ui:
                ui.set_status("Thinking...")

            if state.pending_confirmation is not None:
                decision = classify_yes_no(user_text)
                if decision is None:
                    speak("Please say yes or no.")
                    continue
                if decision is False:
                    state.pending_confirmation = None
                    speak("Okay, canceled.")
                    continue

                confirmed_plan = state.pending_confirmation
                state.pending_confirmation = None
                if ui:
                    ui.set_status("Executing...")
                messages, should_stop, had_error = await asyncio.to_thread(execute_actions, confirmed_plan.actions, state)
                if had_error:
                    spoken = messages[0] if messages else "I couldn't complete that action."
                else:
                    spoken = confirmed_plan.spoken_response or (messages[0] if messages else "Done.")
                speak(spoken)
                push_history(state, user_text, spoken)
                if should_stop:
                    running = False
                continue

            if should_refresh_elements_for_utterance(user_text, state):
                try:
                    await asyncio.to_thread(refresh_page_state, state, "snapshot")
                except BrowserCommandError as exc:
                    log_line(f"WARN: Could not refresh elements pre-plan ({exc}).")

            fast_plan = build_fast_path_plan(user_text, state)
            if fast_plan is not None:
                if any(action["type"] in DESTRUCTIVE_ACTIONS for action in fast_plan.actions):
                    state.pending_confirmation = fast_plan
                    speak(build_confirmation_prompt(fast_plan))
                    push_history(state, user_text, "Asked for confirmation")
                    continue
                if ui:
                    ui.set_status("Executing...")
                messages, should_stop, had_error = await asyncio.to_thread(execute_actions, fast_plan.actions, state)
                if had_error:
                    spoken = messages[0] if messages else "I couldn't complete that action."
                else:
                    spoken = fast_plan.spoken_response or (messages[0] if messages else "Done.")
                speak(spoken)
                push_history(state, user_text, spoken)
                if should_stop:
                    running = False
                if ui:
                    ui.set_status("Ready")
                continue

            prompt = build_planner_input(user_text, state)
            try:
                raw_response = await ask_copilot(session, prompt)
            except Exception as exc:
                log_line(f"WARN: Planner call failed ({exc}).")
                raw_response = None

            if raw_response is None:
                planner_failure_streak += 1
                if planner_failure_streak >= 2:
                    speak_and_wait("Planner connection dropped. Reconnecting.")
                    try:
                        client, session = await reconnect_copilot_session(client)
                        planner_failure_streak = 0
                        speak("Reconnected.")
                    except Exception as exc:
                        log_line(f"WARN: Reconnect failed ({exc}).")
                        speak("I could not reconnect yet. Please try again.")
                else:
                    speak("I had trouble with planning. Please try again.")
                continue
            planner_failure_streak = 0

            plan, parse_errors = parse_planner_response(raw_response)

            if parse_errors and not plan.actions and not plan.needs_clarification and not plan.quit:
                fallback_plan = build_click_fallback_plan(user_text, state)
                if fallback_plan is not None:
                    plan = fallback_plan
                    parse_errors = []
                else:
                    speak("I had trouble interpreting that request. Please try again.")
                    log_line("  Planner parse errors: " + "; ".join(parse_errors))
                    push_history(state, user_text, "Parse error")
                    continue

            if plan.needs_clarification:
                question = plan.clarification_question or "Can you clarify what you want me to do?"
                speak(question)
                push_history(state, user_text, question)
                continue

            if plan.quit:
                await asyncio.to_thread(run_ab_ok, ["close"])
                speak_and_wait("Closing the browser. Goodbye.")
                push_history(state, user_text, "Quit")
                running = False
                continue

            if any(action["type"] in DESTRUCTIVE_ACTIONS for action in plan.actions):
                state.pending_confirmation = plan
                speak(build_confirmation_prompt(plan))
                push_history(state, user_text, "Asked for confirmation")
                continue

            if ui:
                ui.set_status("Executing...")
            messages, should_stop, had_error = await asyncio.to_thread(execute_actions, plan.actions, state)
            if had_error:
                spoken = messages[0] if messages else "I couldn't complete that action."
            else:
                spoken = plan.spoken_response or (messages[0] if messages else "Done.")
            speak(spoken)
            push_history(state, user_text, spoken)
            if should_stop:
                running = False
            if ui:
                ui.set_status("Ready")

        except sr.WaitTimeoutError:
            if ui:
                ui.set_status("Ready")
            continue
        except sr.UnknownValueError:
            unknown_streak += 1
            log_line("  Could not understand speech; listening again.")
            if unknown_streak in {3, 6}:
                speak(
                    "I heard audio but could not understand words. "
                    "Try speaking closer, slower, or set VOICE_BROWSER_MIC_INDEX to a different microphone."
                )
            if ui:
                ui.set_status("Ready")
        except sr.RequestError as exc:
            speak(f"Speech recognition error: {exc}")
            if ui:
                ui.set_status("Speech recognition error")
        except KeyboardInterrupt:
            try:
                await asyncio.to_thread(run_ab_ok, ["close"])
            except Exception:
                pass
            speak_and_wait("Interrupted. Closing browser.")
            running = False
        except Exception as exc:
            log_line(f"  Unexpected error: {exc}")
            speak("Something went wrong. Please try again.")
            if ui:
                ui.set_status("Error")

    try:
        await client.stop()
    finally:
        try:
            await asyncio.to_thread(run_ab_ok, ["close"])
        except Exception:
            pass
        with _browser_runtime_lock:
            _BROWSER_EXECUTOR.shutdown(wait=False, cancel_futures=True)
        shutdown_tts()
        if ui:
            ui.stop()
        _ui_logger = None
        log_line("Voice Browser closed.")


if __name__ == "__main__":
    instance_lock = acquire_single_instance_lock()
    if os.name == "nt" and instance_lock is None:
        log_line("Voice Browser is already running. Close the other instance first.")
        sys.exit(1)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        try:
            run_ab_ok(["close"])
        except Exception:
            pass
        log_line("Interrupted. Exiting cleanly.")
    finally:
        release_single_instance_lock(instance_lock)
