"""KhanomTan TTSBackend: VITS (YourTTS) Thai TTS in its own sidecar venv.

Subclasses SubprocessBackend like engines/supertonic3. The engine needs
``coqui-tts``, which pins its own torch/transformers stack, so it always runs
from the venv the one-click installer makes (no parent-interpreter fallback).
Apache-2.0 weights; preset speakers only (no cloning).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from services.subprocess_backend import SubprocessBackend
from engines.khanomtan import constants as kt_constants

if TYPE_CHECKING:
    import torch  # noqa: F401

logger = logging.getLogger("omnivoice.khanomtan")

_VENV_ENV_VAR = "OMNIVOICE_KHANOMTAN_DIR"

KHANOMTAN_SIDECAR_SCRIPT: Path = Path(__file__).parent / "sidecar.py"


def _own_venv_python() -> "Path | None":
    """The venv the one-click installer made for this engine, if any."""
    from services.sidecar_install import engine_venv_python

    return engine_venv_python(_VENV_ENV_VAR)


class KhanomTanBackend(SubprocessBackend):
    """KhanomTan TTS: Thai-capable VITS checkpoint, CPU or CUDA via coqui-tts."""

    id = "khanomtan"
    display_name = "KhanomTan TTS (Thai, Apache-2.0, preset voices)"
    supports_voice_design = False
    supports_cloning = False  # preset speakers only; no reference audio path
    PRESET_VOICES = list(kt_constants.VOICE_PRESETS)
    DEFAULT_VOICE = kt_constants.DEFAULT_VOICE
    gpu_compat: tuple[str, ...] = ("cuda", "cpu")
    _DEFAULT_SAMPLE_RATE = kt_constants.SAMPLE_RATE

    # ── SubprocessBackend contract ─────────────────────────────────────

    @classmethod
    def shares_parent_torch(cls) -> bool:
        return False  # always its own venv

    @classmethod
    def venv_python(cls) -> Path:
        # is_available() refuses to start without this venv, so the fallback
        # only matters if someone spawns the backend directly.
        return _own_venv_python() or Path(sys.executable)

    @classmethod
    def sidecar_script(cls) -> Path:
        return KHANOMTAN_SIDECAR_SCRIPT

    # ── availability ───────────────────────────────────────────────────

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        if _own_venv_python() is None:
            return False, "KhanomTan is not installed. Install it from Model Catalogue."
        return True, f"ready (license {kt_constants.LICENSE})"

    # ── TTSBackend protocol ────────────────────────────────────────────

    @property
    def sample_rate(self) -> int:
        return self._DEFAULT_SAMPLE_RATE

    @property
    def supported_languages(self) -> list[str]:
        return list(kt_constants.SUPPORTED_LANGUAGES)

    # ── generate ───────────────────────────────────────────────────────

    def generate(self, text: str, **kw) -> "torch.Tensor":
        """Synthesize one utterance.

        kwargs honored:
          * ``voice``    one of :data:`VOICE_PRESETS`; unknown ids fall back to
                         ``DEFAULT_VOICE`` with a log line.
          * ``language`` one of :data:`SUPPORTED_LANGUAGES`; anything else
                         (including ``auto``/``None``) uses ``th-th``.

        Returns a tensor ``(1, n_samples)`` at :attr:`sample_rate`.
        """
        voice = kw.get("voice") or kt_constants.DEFAULT_VOICE
        if voice not in kt_constants.VOICE_PRESETS:
            logger.info(
                "khanomtan: unknown voice %r, falling back to %r. Valid: %s",
                voice, kt_constants.DEFAULT_VOICE, kt_constants.VOICE_PRESETS,
            )
            voice = kt_constants.DEFAULT_VOICE

        language = self._checkpoint_language(kw.get("language"))
        return super().generate(text, voice=voice, language=language)

    def _checkpoint_language(self, language: object) -> str:
        """Map the app's language (raw or normalized) to a checkpoint id.

        Unmapped or Auto input uses Thai, the engine's purpose. Validation has
        already rejected unsupported names upstream.
        """
        if not isinstance(language, str):
            return kt_constants.DEFAULT_LANGUAGE
        raw = language.strip().lower()
        if raw in kt_constants.LANGUAGE_IDS:
            return kt_constants.LANGUAGE_IDS[raw]
        code = self._normalize_language_code(language)
        return kt_constants.LANGUAGE_IDS.get(code or "", kt_constants.DEFAULT_LANGUAGE)


__all__ = ["KhanomTanBackend", "KHANOMTAN_SIDECAR_SCRIPT"]
