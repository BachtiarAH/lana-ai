"""STT via faster-whisper (local, offline after first download)."""
from pathlib import Path

_model = None
_model_name = None


def get_model(name: str = "medium", device: str = "cpu", compute_type: str = "int8",
              local_only: bool = False):
    global _model, _model_name
    if _model is None or _model_name != name:
        from faster_whisper import WhisperModel

        print(f"[stt] loading faster-whisper '{name}' ({device}/{compute_type}) local_only={local_only} ...")
        # No token needed: Systran models are public. First run downloads once,
        # next runs are 100% offline with local_only=True / HF_HUB_OFFLINE=1.
        _model = WhisperModel(name, device=device, compute_type=compute_type,
                              local_files_only=local_only)
        _model_name = name
        print("[stt] model ready")
    return _model


def transcribe_wav(path: str | Path, model_name="medium", language="id", local_only=False) -> str:
    """Transcribe a 16k mono wav file to text. Returns stripped text."""
    model = get_model(model_name, local_only=local_only)
    segments, _info = model.transcribe(str(path), language=language, beam_size=5, vad_filter=False)
    text = " ".join(s.text.strip() for s in segments).strip()
    return text
