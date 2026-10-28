"""Behavioral coverage for the opt-in observer browser integration."""

import importlib.util
import os
import sys
import uuid
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "wp2_model_run.py"


def _module(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    monkeypatch.syspath_prepend(str(SCRIPT.parents[1]))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--turns", "1",
                                        "--run-id", "observer-test-" + uuid.uuid4().hex])
    spec = importlib.util.spec_from_file_location("wp2_model_run_observer_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._OBSERVER_URL_PRINTED = False
    module._OBSERVER_OPEN_ATTEMPTED = False
    return module


def _capture(module, monkeypatch):
    lines = []
    monkeypatch.setattr(module, "log", lines.append)
    return lines


def test_no_flag_prints_url_without_opening(monkeypatch):
    module = _module(monkeypatch)
    lines = _capture(module, monkeypatch)
    opened = []
    monkeypatch.setattr(module.webbrowser, "open", lambda *a, **k: opened.append(a))

    module.progress({"event": "session_ready", "port": 6501})

    assert "civserverport=6501" in "\n".join(lines)
    assert not opened


def test_open_observer_uses_actual_port_once(monkeypatch):
    module = _module(monkeypatch)
    lines = _capture(module, monkeypatch)
    monkeypatch.setattr(module, "OPEN_OBSERVER", True)
    monkeypatch.setenv("DISPLAY", ":99")
    opened = []
    monkeypatch.setattr(module.webbrowser, "open",
                        lambda *a, **k: opened.append((a, k)) or True)

    event = {"event": "session_ready", "port": 6517}
    module.progress(event)
    module.progress(event)

    assert len(opened) == 1
    assert opened[0][0][0] == (
        "http://localhost:8080/webclient/?action=observe&"
        "civserverport=6517&civserverhost=unknown&multi=true&type=multiplayer")
    assert sum("observer URL:" in line for line in lines) == 1


def test_open_false_and_exception_are_nonfatal(monkeypatch):
    module = _module(monkeypatch)
    monkeypatch.setattr(module, "OPEN_OBSERVER", True)
    monkeypatch.setenv("DISPLAY", ":99")
    lines = _capture(module, monkeypatch)
    monkeypatch.setattr(module.webbrowser, "open", lambda *a, **k: False)
    module.progress({"event": "session_ready", "port": 6502})
    assert '"status": "open_failed"' in "\n".join(lines)

    module._OBSERVER_URL_PRINTED = False
    module._OBSERVER_OPEN_ATTEMPTED = False
    def fail(*args, **kwargs):
        raise OSError("no browser")
    monkeypatch.setattr(module.webbrowser, "open", fail)
    module.progress({"event": "session_ready", "port": 6503})
    assert "OSError" in "\n".join(lines)


def test_missing_port_and_headless_are_structured_nonfatal(monkeypatch):
    module = _module(monkeypatch)
    module.OPEN_OBSERVER = True
    lines = _capture(module, monkeypatch)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("MIR_SOCKET", raising=False)
    monkeypatch.delenv("BROWSER", raising=False)
    opened = []
    monkeypatch.setattr(module.webbrowser, "open", lambda *a, **k: opened.append(a))
    module.progress({"event": "session_ready"})
    assert '"status": "unavailable"' in "\n".join(lines)
    assert not opened

    module._OBSERVER_URL_PRINTED = False
    module.progress({"event": "session_ready", "port": 6504})
    assert '"status": "headless"' in "\n".join(lines)


def test_help_mentions_open_observer(monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    monkeypatch.syspath_prepend(str(SCRIPT.parents[1]))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--help"])
    spec = importlib.util.spec_from_file_location("wp2_model_run_help_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except SystemExit:
        pass
    assert "--open-observer" in capsys.readouterr().out
