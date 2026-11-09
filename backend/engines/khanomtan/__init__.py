"""KhanomTan TTS engine (VITS / YourTTS Thai model, Apache-2.0).

``services.tts_backend._LAZY_REGISTRY`` resolves ``"khanomtan"`` to
:class:`KhanomTanBackend`. The sidecar (``sidecar.py``) runs in its own venv
and is never imported by the parent process.
"""
from __future__ import annotations

from engines.khanomtan.backend import KhanomTanBackend

__all__ = ["KhanomTanBackend"]
