"""Single source of runtime config (headless + TUI share this)."""
from dataclasses import dataclass


@dataclass
class Config:
    sample_rate: int = 16000
    vad_threshold: float = 0.5
    min_silence_ms: int = 300
    speech_pad_ms: int = 30
    min_utterance_s: float = 0.5
    stt_model: str = "medium"
    stt_lang: str | None = None  # None = auto-detect
    llm_host: str = "http://localhost:11434"
    llm_model: str = "llama3.1:8b"
    llm_enabled: bool = True
    history_max: int = 20  # max chat messages (user+assistant) sent to LLM
    device: str | None = None

    @classmethod
    def add_cli_args(cls, p):
        p.add_argument("--device", default=None, help="sounddevice index or name")
        p.add_argument("--threshold", type=float, default=0.5)
        p.add_argument("--model", default="medium", help="tiny/base/small/medium/turbo")
        p.add_argument("--lang", default="auto", help="id/en, atau auto untuk auto-detect")
        p.add_argument("--no-stt", action="store_true", help="hanya simpan wav, tanpa teks")
        p.add_argument("--offline", action="store_true", help="STT 100%% offline (model harus sudah cached)")
        p.add_argument("--llm-model", default="llama3.1:8b", help="nama model Ollama, cth: qwen2.5:7b")
        p.add_argument("--llm-host", default="http://localhost:11434")
        p.add_argument("--no-llm", action="store_true", help="matikan local LLM")
        return p

    @classmethod
    def from_args(cls, args) -> "Config":
        return cls(
            vad_threshold=args.threshold,
            stt_model=args.model,
            stt_lang=None if args.lang == "auto" else args.lang,
            device=args.device,
            llm_host=args.llm_host,
            llm_model=args.llm_model,
            llm_enabled=not args.no_llm,
        )
