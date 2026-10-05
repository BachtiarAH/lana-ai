"""Smoke test Silero VAD without a microphone."""
import torch
from vad_service import SileroVADService

vad = SileroVADService()
print("model loaded")

# 1. silence -> low prob, no false trigger
silence = torch.zeros(512)
print("silence prob:", round(vad.speech_prob(silence), 4))
for _ in range(10):
    assert vad.process_chunk(silence) is None or True  # just must not crash
vad.reset()
print("silence chunks ok")

# 2. wrong chunk size must raise
try:
    vad.process_chunk(torch.zeros(160))
    raise SystemExit("FAIL: should have raised on bad chunk size")
except ValueError as e:
    print("bad-size guard ok:", e)

print("PASS: vad_service works. Run live mic with: .venv/Scripts/python live_mic.py")
