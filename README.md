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
.venv/Scripts/python live_mic.py --model medium --lang auto
.venv/Scripts/python live_mic.py --model medium --lang auto --offline   # 100% offline, model must be cached
.venv/Scripts/python live_mic.py --model tiny --lang id                 # faster, less accurate
.venv/Scripts/python live_mic.py --no-stt                               # VAD only, save wav
.venv/Scripts/python live_mic.py --no-llm                               # tanpa local LLM
```

## TUI + local LLM

```
.venv/Scripts/python tui.py --llm-model llama3.1:8b
.venv/Scripts/python tui.py --no-audio   # buka UI tanpa mic (test)
```

Butuh Ollama jalan di `localhost:11434` + 1 model ter-pull (`ollama pull llama3.1:8b`).
Ganti model saat jalan: `/model qwen2.5:7b` · list: `/models` · reset history: `/clear`.

Each utterance is saved to `utterances/*.wav` and printed as `[text] ...`.

## Files

- `live_mic.py` — mic capture (16 kHz mono, 32 ms chunks) → VAD → wav + transcribe + LLM
- `tui.py` — Textual TUI: status mic | transkrip | jawaban LLM
- `config.py` — satu Config untuk headless + TUI
- `llm_service.py` — Ollama wrapper, model swappable by name
- `vad_service.py` — reusable Silero VAD wrapper (`process_chunk`, `timestamps_for_file`)
- `stt_service.py` — faster-whisper wrapper (default `medium`, CPU int8)
- `test_vad.py` — VAD smoke test without a microphone: `.venv/Scripts/python test_vad.py`
