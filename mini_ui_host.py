import queue
import re
import time
from typing import Any, List, Optional

import tkinter as tk
from tkinter import ttk


def run_ui(
    mic_items: List[str],
    selected_mic_index: Optional[int],
    command_queue: Any,
    status_queue: Any,
    log_queue: Any,
) -> None:
    root = tk.Tk()
    root.title("Voice Browser")
    root.geometry("420x320")
    root.attributes("-topmost", True)
    root.resizable(True, True)

    tk.Label(root, text="Status:").pack(anchor="w", padx=8, pady=(8, 0))
    status_label = tk.Label(root, text="Starting...", wraplength=340)
    status_label.pack(anchor="w", padx=8)

    tk.Label(root, text="Microphone:").pack(anchor="w", padx=8, pady=(8, 0))
    mic_combo = ttk.Combobox(root, values=mic_items, state="readonly")
    mic_combo.pack(fill="x", padx=8)
    if selected_mic_index is not None and 0 <= selected_mic_index < len(mic_items):
        mic_combo.current(selected_mic_index)
    elif mic_items:
        mic_combo.current(0)

    prompt_entry = tk.Entry(root)
    prompt_entry.pack(fill="x", padx=8, pady=(8, 0))

    button_row = tk.Frame(root)
    button_row.pack(fill="x", padx=8, pady=8)

    def send_text() -> None:
        text = prompt_entry.get().strip()
        if not text:
            return
        command_queue.put({"type": "utterance", "text": text})
        prompt_entry.delete(0, "end")

    def stop_speech() -> None:
        command_queue.put({"type": "stop_speech"})

    def apply_mic() -> None:
        match = re.search(r"\[(\d+)\]", mic_combo.get())
        if not match:
            command_queue.put({"type": "set_mic", "index": None})
            return
        command_queue.put({"type": "set_mic", "index": int(match.group(1))})

    def request_quit() -> None:
        command_queue.put({"type": "quit"})
        try:
            root.destroy()
        except Exception:
            pass

    tk.Button(button_row, text="Send", command=send_text, width=8).pack(side="left")
    tk.Button(button_row, text="Stop Voice", command=stop_speech, width=10).pack(side="left", padx=(6, 0))
    tk.Button(button_row, text="Set Mic", command=apply_mic, width=8).pack(side="left", padx=(6, 0))
    tk.Button(button_row, text="Quit", command=request_quit, width=8).pack(side="right")

    tk.Label(root, text="Log:").pack(anchor="w", padx=8)
    log_frame = tk.Frame(root)
    log_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
    log_text = tk.Text(log_frame, height=8, wrap="word", state="disabled")
    log_scroll = tk.Scrollbar(log_frame, command=log_text.yview)
    log_text.configure(yscrollcommand=log_scroll.set)
    log_text.pack(side="left", fill="both", expand=True)
    log_scroll.pack(side="right", fill="y")

    prompt_entry.bind("<Return>", lambda _evt: send_text())

    def append_log_line(line: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        log_text.configure(state="normal")
        log_text.insert("end", f"[{timestamp}] {line}\n")
        log_text.see("end")
        if int(float(log_text.index("end-1c").split(".")[0])) > 300:
            log_text.delete("1.0", "50.0")
        log_text.configure(state="disabled")

    def poll_queues() -> None:
        while True:
            try:
                payload = status_queue.get_nowait()
            except queue.Empty:
                break
            if not isinstance(payload, dict):
                continue
            event_type = payload.get("type")
            if event_type == "shutdown":
                request_quit()
                return
            if event_type == "status":
                status_label.configure(text=str(payload.get("value", "")))
        while True:
            try:
                payload = log_queue.get_nowait()
            except queue.Empty:
                break
            if not isinstance(payload, dict):
                continue
            if payload.get("type") == "log":
                append_log_line(str(payload.get("value", "")))
        root.after(120, poll_queues)

    root.protocol("WM_DELETE_WINDOW", request_quit)
    poll_queues()
    prompt_entry.focus_set()
    root.mainloop()
