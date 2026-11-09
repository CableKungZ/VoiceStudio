"""KhanomTan engine: registry wiring, request normalization, sidecar framing."""
from __future__ import annotations

import base64
import importlib
import json
import struct
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def _backend_cls():
    return importlib.import_module("engines.khanomtan").KhanomTanBackend


def test_registry_resolves_khanomtan_to_its_class():
    from services.tts_backend import _LAZY_REGISTRY

    mod_path, attr = _LAZY_REGISTRY["khanomtan"]
    assert (mod_path, attr) == ("engines.khanomtan", "KhanomTanBackend")
    cls = getattr(importlib.import_module(mod_path), attr)
    assert cls.id == "khanomtan"


def test_engine_is_not_claimed_to_clone_voices():
    cls = _backend_cls()
    assert cls.supports_cloning is False
    assert cls.PRESET_VOICES == ["Linda", "Bernard", "Kerstin", "Thorsten"]
    assert cls.DEFAULT_VOICE in cls.PRESET_VOICES


def test_generate_falls_back_to_default_voice_and_thai(monkeypatch):
    from engines.khanomtan import constants as kt
    from services.subprocess_backend import SubprocessBackend

    seen: dict = {}

    def fake_generate(self, text, **kw):
        seen.update(kw)
        return "ok"

    monkeypatch.setattr(SubprocessBackend, "generate", fake_generate)
    backend = object.__new__(_backend_cls())

    backend.generate("สวัสดี", voice="Nobody", language="auto")
    assert seen == {"voice": kt.DEFAULT_VOICE, "language": kt.DEFAULT_LANGUAGE}

    backend.generate("สวัสดี", voice="Bernard", language="fr-fr")
    assert seen == {"voice": "Bernard", "language": "fr-fr"}

    # The app normalizes names and two-letter codes; both map to checkpoint ids.
    backend.generate("Hallo", voice="Kerstin", language="German")
    assert seen == {"voice": "Kerstin", "language": "x-de"}
    backend.generate("สวัสดี", voice="Linda", language="th")
    assert seen == {"voice": "Linda", "language": "th-th"}


def test_unavailable_without_its_own_venv(monkeypatch):
    cls = _backend_cls()
    monkeypatch.setattr("engines.khanomtan.backend._own_venv_python", lambda: None)
    ok, reason = cls.is_available()
    assert ok is False
    assert "not installed" in reason


def test_thai_is_word_segmented_and_falls_back_when_pythainlp_missing(monkeypatch):
    sidecar = importlib.import_module("engines.khanomtan.sidecar")
    pytest.importorskip("pythainlp")
    segmented = sidecar._segment_thai("วันนี้อากาศดีมากครับ")
    assert segmented.count(" ") >= 2  # real word boundaries, not one blob

    monkeypatch.setitem(sys.modules, "pythainlp.tokenize", None)  # force ImportError
    assert sidecar._segment_thai("วันนี้อากาศดีมากครับ") == "วันนี้อากาศดีมากครับ"


def test_pcm_conversion_clips_and_reports_length():
    sidecar = importlib.import_module("engines.khanomtan.sidecar")
    b64, sr, n = sidecar._wav_to_pcm_b64([0.0, 1.5, -2.0], 16000)
    assert (sr, n) == (16000, 3)
    samples = struct.unpack("<3h", base64.b64decode(b64))
    assert samples == (0, 32767, -32767)


def test_frame_roundtrip_matches_parent_protocol():
    import io

    sidecar = importlib.import_module("engines.khanomtan.sidecar")
    buf = io.BytesIO()
    sidecar._send(buf, {"op": "ping"})
    buf.seek(0)
    assert sidecar._recv(buf) == {"op": "ping"}
    assert json.loads(buf.getvalue()[4:].decode()) == {"op": "ping"}
