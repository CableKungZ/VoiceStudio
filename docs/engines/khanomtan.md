# VoiceStudio — KhanomTan TTS Engine

KhanomTan TTS v1.1 (`wannaphong/khanomtan-tts-v1.1`) is a VITS (YourTTS) model
trained with 🐸 Coqui-TTS on Thai corpora (TSync 1 and 2). It supports Thai,
English, French, Portuguese, Danish and Luxembourgish, and runs in its own
sidecar process with its own Python environment.

## When to pick it

- Thai narration and voice-over with the built-in voices.
- A small, CPU-friendly model that also runs on CUDA.

## Limits

- **Preset voices only.** Linda, Bernard, Kerstin and Thorsten. Cloning from a
  reference recording is not supported.
- Native sample rate is 16 kHz.
- The model card's license is Apache-2.0, but the LICENSE file has not been
  reviewed in this repo yet, so commercial use is marked as not cleared in
  `backend/config/model_licenses.json`. Check it before shipping commercially.

## Setup

1. Open **Model Catalogue → KhanomTan** and click **Install**. This creates an
   engine-only venv under VoiceStudio's data directory with `coqui-tts` pinned
   to 0.28.0 and the app's torch 2.8.0 pair (CUDA build when available).
   **Uninstall** removes only that folder.
2. Select the engine in the TTS tab (**Use**) or set
   `OMNIVOICE_TTS_BACKEND=khanomtan`.

The first synthesis downloads the checkpoint (about 0.7 GB), pinned to revision
`9bfdb75211eb645cd6bf75e69d483bd06e32f5c6`, into the Hugging Face cache. See
[downloading-models.md](../downloading-models.md).

## Thai text handling

The checkpoint reads raw characters and was trained on word-segmented Thai, so
the sidecar splits Thai words with `pythainlp` (installed into the engine's own
venv) before synthesis. If `pythainlp` is missing, the raw text is sent and
quality drops.

## Quality (measured, not a promise)

Checked by synthesizing with the Linda speaker and transcribing the audio with
faster-whisper `small`:

- Short phrases are intelligible (for example "สวัสดีครับ").
- Medium and long Thai sentences are only partly intelligible. The model is the
  limit here, not the wrapper.
- English prompts are mostly intelligible.

Listen to your own text before you use it for narration.

## Troubleshooting

- **"KhanomTan is not installed"**: run the install from Model Catalogue.
- The built-in speakers are English, French and German voices that speak Thai
  through the shared `th-th` language ID. Compare speakers on your text before
  choosing a default.
