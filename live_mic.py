"""Live mic -> Silero VAD -> STT -> local LLM. Works headless AND as TUI backend.

Headless:  .venv/Scripts/python live_mic.py --model medium --lang auto
TUI:       .venv/Scripts/python tui.py
"""
import queue
import threading
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch

from config import Config
from vad_service import SileroVADService

CHUNK = 512  # 32 ms @16k — required by Silero
OUT_DIR = Path(__file__).parent / "utterances"
OUT_DIR.mkdir(exist_ok=True)


def save_wav(path: Path, audio_np: np.ndarray, sr: int = 16000):
    pcm = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


class PrintEvents:
    """Headless fallback: replicate the original print behavior."""

    def state(self, s: str):
        print(f"[{s}]", flush=True)

    def meter(self, level: float):
        pass

    def saved(self, path: Path, dur: float):
        print(f"[saved] {path} ({dur:.2f}s)")

    def transcript(self, path: Path, text: str):
        print(f"[text] {text if text else '(kosong — tidak ada ucapan jelas)'}")

    def answer(self, text: str):
        print(f"[lana] {text}")

    def error(self, msg: str):
        print(f"[error] {msg}")


def run_pipeline(cfg: Config, events=None, stop=None, stt_offline: bool = False,
                 llm=None, history: list | None = None):
    """Blocking loop. TUI passes its own events + stop flag + llm.

    history: shared list of {'role','content'}; capped at cfg.history_max.
    """
    from stt_service import transcribe_wav

    ev = events or PrintEvents()
    stop = stop or threading.Event()
    history = history if history is not None else []
    lock = threading.Lock()

    vad = SileroVADService(
        threshold=cfg.vad_threshold,
        sampling_rate=cfg.sample_rate,
        min_silence_duration_ms=cfg.min_silence_ms,
        speech_pad_ms=cfg.speech_pad_ms,
    )
    q: queue.Queue = queue.Queue()
    speaking = False
    buffer: list[np.ndarray] = []

    def audio_callback(indata, frames, time_info, status):
        if status:
            ev.error(str(status))
        mono = indata[:, 0].copy()
        q.put(mono)

    def handle_utterance(audio_np: np.ndarray, sr: int, path: Path):
        dur = len(audio_np) / sr
        if dur < cfg.min_utterance_s:
            try:
                path.unlink()
            except OSError:
                pass
            return
        ev.saved(path, dur)
        if not cfg.stt_model:  # --no-stt: hanya simpan wav
            return
        # STT
        ev.state("transcribing")
        try:
            text = transcribe_wav(path, model_name=cfg.stt_model,
                                  language=cfg.stt_lang, local_only=stt_offline)
        except Exception as e:
            ev.error(f"stt: {e}")
            ev.state("listening")
            return
        if not text:
            ev.transcript(path, "")
            ev.state("listening")
            return
        ev.transcript(path, text)
        # LLM
        if llm is None or not cfg.llm_enabled:
            ev.state("listening")
            return
        ev.state("thinking")
        with lock:
            history.append({"role": "user", "content": text})
            msgs = history[-cfg.history_max:]
        try:
            reply = llm.chat(msgs)
        except Exception as e:
            ev.error(f"llm: {e}")
            ev.state("listening")
            return
        with lock:
            history.append({"role": "assistant", "content": reply})
            while len(history) > cfg.history_max:
                history.pop(0)
        ev.answer(reply)
        ev.state("listening")

    ev.state(f"listening (stt={cfg.stt_model} llm={llm.model if llm else 'off'})")
    stream = sd.InputStream(
        samplerate=cfg.sample_rate, channels=1, dtype="float32",
        blocksize=CHUNK, callback=audio_callback, device=cfg.device,
    )
    with stream:
        while not stop.is_set():
            try:
                mono = q.get(timeout=0.5)
            except queue.Empty:
                continue
            for i in range(0, len(mono), CHUNK):
                piece = mono[i:i + CHUNK]
                if len(piece) < CHUNK:
                    piece = np.pad(piece, (0, CHUNK - len(piece)))
                ev.meter(float(np.sqrt(np.mean(piece ** 2))))
                try:
                    event = vad.process_chunk(torch.from_numpy(piece))
                except Exception as e:
                    ev.error(f"vad: {e}")
                    continue
                if event and "start" in event:
                    if not speaking:
                        ev.state("speaking")
                        speaking = True
                        buffer = []
                if speaking:
                    buffer.append(piece.copy())
                if event and "end" in event:
                    ev.state("listening")
                    speaking = False
                    if buffer:
                        utt = np.concatenate(buffer)
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                        path = OUT_DIR / f"utt_{ts}.wav"
                        save_wav(path, utt, cfg.sample_rate)
                        threading.Thread(target=handle_utterance,
                                         args=(utt, cfg.sample_rate, path),
                                         daemon=True).start()
                        buffer = []


def main():
    import argparse

    p = argparse.ArgumentParser(description="lana-ai headless: mic -> VAD -> STT -> local LLM")
    Config.add_cli_args(p)
    args = p.parse_args()
    cfg = Config.from_args(args)
    stt_offline = args.offline

    llm = None
    if cfg.llm_enabled:
        from llm_service import LocalLLM

        llm = LocalLLM(host=cfg.llm_host, model=cfg.llm_model)
        # preload STT supaya ucapan pertama tidak delay (download sekali saja)
        if not args.no_stt:
            from stt_service import get_model

            try:
                get_model(cfg.stt_model)
            except Exception as e:
                print(f"[warn] preload stt gagal: {e}")
    if args.no_stt:
        cfg.stt_model = ""
    try:
        run_pipeline(cfg, stt_offline=stt_offline, llm=llm)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
