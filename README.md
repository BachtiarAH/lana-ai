# lana-ai — live mic VAD + STT (100% local)

Mic → **Silero VAD** (speech detection) → **faster-whisper** (speech-to-text).
No API key / token needed. Whisper model downloads once from Hugging Face, then works offline.

## Setup

```
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```

## Run

```
.venv/Scripts/python live_mic.py --model medium --lang id
.venv/Scripts/python live_mic.py --model medium --lang id --offline   # 100% offline, model must be cached
.venv/Scripts/python live_mic.py --model tiny --lang id               # faster, less accurate
.venv/Scripts/python live_mic.py --no-stt                             # VAD only, save wav
```

Each utterance is saved to `utterances/*.wav` and printed as `[text] ...`.

## Files

- `live_mic.py` — mic capture (16 kHz mono, 32 ms chunks) → VAD → wav + transcribe
- `vad_service.py` — reusable Silero VAD wrapper (`process_chunk`, `timestamps_for_file`)
- `stt_service.py` — faster-whisper wrapper (default `medium`, CPU int8)
- `test_vad.py` — VAD smoke test without a microphone: `.venv/Scripts/python test_vad.py`
