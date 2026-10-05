"""Live mic -> Silero VAD -> utterance wav files.

Run:  .venv/Scripts/python live_mic.py
Keys: Ctrl+C to stop.

Each detected utterance is saved to ./utterances/ and passed to
on_utterance(samples_16k_mono, sample_rate) — plug your STT/LLM there.
"""
import queue
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch

from vad_service import SileroVADService
from stt_service import transcribe_wav

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK = 512  # 32 ms @16k — required by Silero
OUT_DIR = Path(__file__).parent / "utterances"
OUT_DIR.mkdir(exist_ok=True)
STT_MODEL = "medium"  # tiny/base/small/medium/turbo — medium akurat untuk id, berat di CPU
STT_LANG = "id"  # "id" / "en" / None (auto-detect, lebih lambat)
STT_ENABLED = True
STT_OFFLINE = False


def save_wav(path: Path, audio_np: np.ndarray, sr: int = SAMPLE_RATE):
    pcm = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def on_utterance(audio_np: np.ndarray, sr: int, path: Path):
    dur = len(audio_np) / sr
    if dur < 0.5:  # buang klik/batuk sangat pendek
        print(f"[skip] {path.name} terlalu pendek ({dur:.2f}s)")
        try:
            path.unlink(missing_ok=True)
        except TypeError:
            try:
                path.unlink()
            except OSError:
                pass
        return
    print(f"[utterance] {path.name} ({dur:.2f}s) -> transcribing...")
    if not STT_ENABLED:
        print(f"[saved] {path}")
        return
    try:
        text = transcribe_wav(path, model_name=STT_MODEL, language=STT_LANG, local_only=STT_OFFLINE)
        print(f"[text] {text if text else '(kosong — tidak ada ucapan jelas)'}")
    except Exception as e:
        print(f"[stt error] {e}")


def main(device=None, threshold=0.5):
    vad = SileroVADService(
        threshold=threshold,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=300,
        speech_pad_ms=30,
    )
    q: queue.Queue = queue.Queue()
    speaking = False
    buffer: list[np.ndarray] = []

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(status, flush=True)
        # indata: (frames, channels) float32
        mono = indata[:, 0].copy()
        q.put(mono)

    print(f"Devices:\n{sd.query_devices()}\n")
    print(f"Listening @ {SAMPLE_RATE}Hz, chunk={CHUNK}, threshold={threshold}. Ctrl+C to stop.")

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=CHUNK,
        callback=audio_callback,
        device=device,
    )
    with stream:
        try:
            while True:
                try:
                    mono = q.get(timeout=1.0)
                except queue.Empty:
                    continue
                # sounddevice may deliver !=512 frames on some drivers; slice/pad
                for i in range(0, len(mono), CHUNK):
                    piece = mono[i : i + CHUNK]
                    if len(piece) < CHUNK:
                        piece = np.pad(piece, (0, CHUNK - len(piece)))
                    event = vad.process_chunk(torch.from_numpy(piece))

                    if event and "start" in event:
                        if not speaking:
                            print("[vad] speech start", flush=True)
                            speaking = True
                            buffer = []
                    if speaking:
                        buffer.append(piece.copy())
                    if event and "end" in event:
                        print("[vad] speech end", flush=True)
                        speaking = False
                        if buffer:
                            utt = np.concatenate(buffer)
                            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                            path = OUT_DIR / f"utt_{ts}.wav"
                            save_wav(path, utt)
                            threading.Thread(
                                target=on_utterance, args=(utt, SAMPLE_RATE, path), daemon=True
                            ).start()
                            buffer = []
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--device", default=None, help="sounddevice index or name")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--model", default="medium", help="tiny/base/small/medium/turbo")
    p.add_argument("--lang", default="id", help="id/en, atau auto untuk auto-detect")
    p.add_argument("--no-stt", action="store_true", help="hanya simpan wav, tanpa teks")
    p.add_argument("--offline", action="store_true", help="paksa 100% offline, tanpa download (model harus sudah pernah di-download)")
    args = p.parse_args()
    STT_MODEL = args.model
    STT_LANG = None if args.lang == "auto" else args.lang
    STT_ENABLED = not args.no_stt
    STT_OFFLINE = args.offline
    if STT_OFFLINE:
        import os

        os.environ["HF_HUB_OFFLINE"] = "1"
    if STT_ENABLED:
        # preload sekali di awal supaya ucapan pertama tidak delay (download besar utk medium, sekali saja)
        from stt_service import get_model

        get_model(STT_MODEL, local_only=STT_OFFLINE)
    main(device=args.device, threshold=args.threshold)
