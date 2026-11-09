"""KhanomTan sidecar: runs the VITS checkpoint in its own venv.

Wire protocol matches engines/supertonic3/sidecar.py (length-prefixed JSON
on a private stdout fd; fd 1 is redirected to stderr so library prints
cannot corrupt frames). Ops: ready, ping/pong, synthesize -> audio, shutdown.
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import struct
import sys
import traceback
from pathlib import Path

# Mirrors backend/services/subprocess_backend.py::MAX_FRAME_BYTES.
MAX_FRAME_BYTES: int = 64 * 1024 * 1024

logger = logging.getLogger("khanomtan.sidecar")

# Import as a script (python sidecar.py) so the sibling module resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import constants as kt_constants  # noqa: E402


def _send(stream, obj: dict) -> None:
    body = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    stream.write(struct.pack("!I", len(body)))
    stream.write(body)
    stream.flush()


def _recv(stream):
    header = stream.read(4)
    if len(header) < 4:
        return None  # EOF
    (n,) = struct.unpack("!I", header)
    if n > MAX_FRAME_BYTES:
        raise IOError(f"frame too large: {n}")
    body = bytearray()
    while len(body) < n:
        chunk = stream.read(n - len(body))
        if not chunk:
            raise IOError("short read")
        body.extend(chunk)
    return json.loads(bytes(body).decode("utf-8"))


def _resolve_snapshot() -> Path:
    """Download the pinned revision (cached after the first call)."""
    from huggingface_hub import snapshot_download

    # Immutable revision pin; the weights never float to a newer commit.
    path = snapshot_download(  # nosec B615 -- pinned revision
        repo_id=kt_constants.REPO_ID,
        revision=kt_constants.PINNED_REVISION_SHA,
        allow_patterns=["*.pth", "*.json"],
    )
    return Path(path)


_TTS = None


def _load_tts(stdout):
    """Load the synthesizer once; emit coarse progress frames while it loads."""
    global _TTS
    if _TTS is not None:
        return _TTS
    _send(stdout, {"op": "progress", "stage": "loading_model", "percent": 0})
    snap = _resolve_snapshot()
    _send(stdout, {"op": "progress", "stage": "loading_model", "percent": 50})
    from TTS.utils.synthesizer import Synthesizer  # coqui-tts, own venv

    # config.json names speakers.pth and language_ids.json relatively, and
    # coqui resolves them against the working directory. This process is
    # dedicated to this engine, so entering the snapshot is safe.
    os.chdir(snap)
    _TTS = Synthesizer(
        tts_checkpoint=str(snap / "best_model.pth"),
        tts_config_path=str(snap / "config.json"),
        tts_speakers_file=str(snap / "speakers.pth"),
        use_cuda=_cuda_available(),
    )
    _send(stdout, {"op": "progress", "stage": "loading_model", "percent": 100})
    return _TTS


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _wav_to_pcm_b64(wav, sample_rate: int) -> tuple[str, int, int]:
    import numpy as np

    arr = np.asarray(wav, dtype=np.float32).reshape(-1)
    arr = np.clip(arr, -1.0, 1.0)
    pcm = (arr * 32767.0).astype(np.int16).tobytes()
    return base64.b64encode(pcm).decode("ascii"), int(sample_rate), int(arr.shape[0])


def _segment_thai(text: str) -> str:
    """Space-separate Thai words. The checkpoint reads raw characters and was
    trained on word-segmented text (its own test sentence is segmented), so
    unsegmented Thai degrades badly. Falls back to the raw text if pythainlp
    is unavailable."""
    try:
        from pythainlp.tokenize import word_tokenize
    except ImportError:
        logger.warning("khanomtan: pythainlp missing; sending unsegmented Thai")
        return text
    return " ".join(w for w in word_tokenize(text, engine="newmm") if w.strip())


def _handle_synthesize(msg: dict, stdout) -> None:
    text = msg.get("text")
    if not text or not isinstance(text, str):
        raise ValueError("synthesize: missing or non-string 'text'")

    voice = msg.get("voice") or kt_constants.DEFAULT_VOICE
    language = msg.get("language") or kt_constants.DEFAULT_LANGUAGE
    if voice not in kt_constants.VOICE_PRESETS:
        raise ValueError(f"unknown voice {voice!r}; expected one of {kt_constants.VOICE_PRESETS}")
    if language == "th-th":
        text = _segment_thai(text)

    tts = _load_tts(stdout)
    wav = tts.tts(text=text, speaker_name=voice, language_name=language)
    sr = int(getattr(tts, "output_sample_rate", kt_constants.SAMPLE_RATE))
    pcm_b64, sr, n_samples = _wav_to_pcm_b64(wav, sr)
    _send(stdout, {"op": "audio", "audio_pcm_b64": pcm_b64, "sample_rate": sr, "n_samples": n_samples})


def _run_selftest() -> int:
    """Import coqui-tts and resolve the pinned snapshot, then exit 0."""
    from TTS.utils.synthesizer import Synthesizer  # noqa: F401

    snap = _resolve_snapshot()
    missing = [n for n in ("best_model.pth", "config.json", "speakers.pth", "language_ids.json") if not (snap / n).is_file()]
    if missing:
        print(f"selftest: missing {missing}", file=sys.stderr)
        return 1
    print("selftest ok", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="KhanomTan sidecar")
    parser.add_argument("--selftest", action="store_true", help="Import coqui-tts, resolve the snapshot, exit 0")
    args = parser.parse_args(argv)
    if args.selftest:
        return _run_selftest()

    stdin = sys.stdin.buffer
    # Keep a private frame channel; send library stdout noise to stderr (#1428).
    _frame_fd = os.dup(1)
    os.dup2(2, 1)
    stdout = os.fdopen(_frame_fd, "wb")

    try:
        import TTS  # type: ignore[import-not-found]

        version = getattr(TTS, "__version__", None)
    except ImportError:
        version = None

    _send(stdout, {
        "op": "ready",
        "engine": "khanomtan",
        "sample_rate": kt_constants.SAMPLE_RATE,
        "version": version,
    })

    while True:
        try:
            msg = _recv(stdin)
        except Exception as exc:
            _send(stdout, {"op": "error", "stage": "recv", "message": f"{type(exc).__name__}: {exc}",
                           "traceback": traceback.format_exc()})
            return 1
        if msg is None:
            return 0
        op = msg.get("op") if isinstance(msg, dict) else None
        try:
            if op == "ping":
                _send(stdout, {"op": "pong"})
            elif op == "synthesize":
                _handle_synthesize(msg, stdout)
            elif op == "shutdown":
                return 0
            else:
                _send(stdout, {"op": "error", "stage": "dispatch", "message": f"unknown op: {op!r}"})
        except Exception as exc:
            # Per-op failure is recoverable: report it and stay alive.
            _send(stdout, {"op": "error", "stage": op or "unknown", "message": f"{type(exc).__name__}: {exc}",
                           "traceback": traceback.format_exc()})


if __name__ == "__main__":
    sys.exit(main())
