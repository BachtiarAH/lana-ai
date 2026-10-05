"""Reusable Silero VAD wrapper for live-mic + file use."""
import torch
from silero_vad import load_silero_vad, VADIterator, get_speech_timestamps


class SileroVADService:
    def __init__(
        self,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_silence_duration_ms: int = 300,
        speech_pad_ms: int = 30,
    ):
        assert sampling_rate in (8000, 16000), "Silero supports only 8000/16000"
        self.sampling_rate = sampling_rate
        self.chunk_samples = 512 if sampling_rate == 16000 else 256
        self.model = load_silero_vad()  # bundled .jit, torch-only, no onnxruntime needed
        self.iterator = VADIterator(
            self.model,
            threshold=threshold,
            sampling_rate=sampling_rate,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )

    def reset(self):
        self.iterator.reset_states()

    @torch.no_grad()
    def process_chunk(self, chunk) -> dict | None:
        """Feed one 32ms chunk (512 samples @16k). Returns {'start':...}, {'end':...}, or None."""
        if not isinstance(chunk, torch.Tensor):
            chunk = torch.tensor(chunk, dtype=torch.float32)
        else:
            chunk = chunk.to(torch.float32)
        chunk = chunk.flatten()
        if len(chunk) != self.chunk_samples:
            raise ValueError(f"chunk must be {self.chunk_samples} samples, got {len(chunk)}")
        return self.iterator(chunk, return_seconds=False)

    @torch.no_grad()
    def speech_prob(self, chunk) -> float:
        if not isinstance(chunk, torch.Tensor):
            chunk = torch.tensor(chunk, dtype=torch.float32)
        return float(self.model(chunk.flatten(), self.sampling_rate).item())

    @torch.no_grad()
    def timestamps_for_file(self, audio: torch.Tensor, **kwargs) -> list:
        """Offline timestamps for a full 1-D float32 waveform."""
        self.model.reset_states()
        return get_speech_timestamps(audio, self.model, sampling_rate=self.sampling_rate, **kwargs)
