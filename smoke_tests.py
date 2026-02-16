import importlib.util
import os
from pathlib import Path


def load_module():
    os.environ.setdefault("VOICE_BROWSER_TTS_ENABLED", "0")
    target = Path(__file__).with_name("voice-browser.py")
    spec = importlib.util.spec_from_file_location("voice_browser_module", target)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def run():
    mod = load_module()

    state = mod.BrowserState()
    state.focused_element = {"editable": True, "name": "Message body"}
    state.last_pointer_target = {"editable": False, "name": "Send button", "ref": "e9"}
    state.editable_candidates = [{"ref": "e4", "editable": True, "name": "Body"}]

    assert mod.resolve_action_target("this", state) == mod.FOCUSED_TARGET
    assert mod.resolve_action_target("here", state) == mod.FOCUSED_TARGET
    assert mod.resolve_action_target("this", state, "click") == "@e9"

    plan = mod.build_fast_path_plan("enter hello world in this text box", state)
    assert plan and plan.actions and plan.actions[0]["type"] == "type_text"
    assert plan.actions[0]["target"] == mod.FOCUSED_TARGET
    assert plan.actions[0]["text"] == "hello world"

    clarify_plan = mod.build_fast_path_plan("fill out this form", state)
    assert clarify_plan and clarify_plan.needs_clarification

    assert mod.build_text_entry_fallback_plan("click the enter button", state) is None

    assert mod.extract_inline_dictation_text("dictate: this is a test") == "this is a test"
    assert mod.normalize_key_combo("Shift+Ctrl+Right") == "Shift+Control+ArrowRight"

    print("smoke_tests: ok")


if __name__ == "__main__":
    run()
