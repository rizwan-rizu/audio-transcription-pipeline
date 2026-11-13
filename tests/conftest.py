"""Shared test helpers.

Tests never load a real model: `FakeTranscriber` returns canned segments, so the
suite runs in seconds and checks our pipeline logic, not Whisper's accuracy.
"""

import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from app.audio import AudioInfo
from app.config import Settings

CANNED_SEGMENTS = [
    {"id": 0, "start": 0.0, "end": 1.8, "text": " Hello and welcome.",
     "avg_logprob": -0.2, "no_speech_prob": 0.01},
    {"id": 1, "start": 1.9, "end": 3.6, "text": "  This is   a test. ",
     "avg_logprob": -0.3, "no_speech_prob": 0.02},
]


class FakeTranscriber:
    model_name = "fake"

    def __init__(self, fail_times: int = 0, error: Exception = None, block: threading.Event = None):
        self.fail_times = fail_times
        self.error = error or RuntimeError("model crashed")
        self.block = block
        self.calls = 0

    def transcribe(self, path, *, language=None, word_timestamps=False, on_progress=None):
        self.calls += 1
        if self.block is not None:
            self.block.wait(5)
        if self.calls <= self.fail_times:
            raise self.error
        if on_progress:
            on_progress(0.5)
            on_progress(1.0)
        return {"language": "en", "language_probability": 0.98, "duration": 4.0,
                "model": self.model_name, "segments": [dict(s) for s in CANNED_SEGMENTS]}


def fake_probe(path, max_duration_s=None):
    return AudioInfo(duration_s=4.0, sample_rate=16000, channels=1, codec="pcm_s16le", container="wav")


def make_wav(path: Path, seconds: float = 1.0, rate: int = 16000, channels: int = 1) -> Path:
    """Write a short sine-wave WAV (mock audio, no real speech needed)."""
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    tone = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    frames = np.repeat(tone[:, None], channels, axis=1)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(frames.tobytes())
    return path


def wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(data_dir=tmp_path / "data", warm_up=False, retry_base_delay_s=0.01,
                    max_attempts=3, workers=1, max_queue_size=5)
