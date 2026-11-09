"""KhanomTan TTS constants.

The Hugging Face repo is pinned to a reviewed commit (T-03-02 pattern from
engines/supertonic3): the sidecar never follows a moving ``main``.
"""
from __future__ import annotations

REPO_ID: str = "wannaphong/khanomtan-tts-v1.1"
PINNED_REVISION_SHA: str = "9bfdb75211eb645cd6bf75e69d483bd06e32f5c6"

#: Native rate of the VITS checkpoint (config.json ``audio.sample_rate``).
SAMPLE_RATE: int = 16000

#: Built-in speakers from speakers.pth. KhanomTan has no zero-shot cloning.
VOICE_PRESETS: tuple[str, ...] = ("Linda", "Bernard", "Kerstin", "Thorsten")
DEFAULT_VOICE: str = "Linda"

#: Language ids from language_ids.json. Thai is the point of this engine.
DEFAULT_LANGUAGE: str = "th-th"

#: Accepted spellings -> checkpoint language id. Both forms are advertised:
#: the app normalizes ``th-th`` to ``th``, but ``x-de`` stays as written.
LANGUAGE_IDS: dict[str, str] = {
    "th": "th-th", "th-th": "th-th",
    "en": "en",
    "fr": "fr-fr", "fr-fr": "fr-fr",
    "pt": "pt-br", "pt-br": "pt-br",
    "de": "x-de", "x-de": "x-de",
    "lb": "x-lb", "x-lb": "x-lb",
}
SUPPORTED_LANGUAGES: tuple[str, ...] = tuple(LANGUAGE_IDS)

LICENSE: str = "Apache-2.0"
