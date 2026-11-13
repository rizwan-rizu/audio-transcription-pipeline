"""Speech-to-text engine.

The rest of the app only knows about the `Transcriber` interface, so the engine can
be swapped (a hosted API, a GPU server) or faked in tests without touching the
pipeline.
"""

import logging
import threading
from pathlib import Path
from typing import Callable, Optional, Protocol

from .config import Settings

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float], None]


class Transcriber(Protocol):
    model_name: str

    def transcribe(
        self,
        path: Path,
        *,
        language: Optional[str] = None,
        word_timestamps: bool = False,
        on_progress: Optional[ProgressCallback] = None,
    ) -> dict: ...


class WhisperTranscriber:
    """faster-whisper (Whisper on CTranslate2), loaded once and shared."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self.model_name = f"faster-whisper/{settings.model_size}"
        self._model = None
        self._load_lock = threading.Lock()

    def warm_up(self) -> None:
        self._get_model()

    def _get_model(self):
        # Loading takes seconds (and downloads the weights the first time), so do it once.
        with self._load_lock:
            if self._model is None:
                from faster_whisper import WhisperModel

                s = self._settings
                log.info("Loading %s (device=%s, compute=%s)", self.model_name, s.device, s.compute_type)
                self._model = WhisperModel(
                    s.model_size,
                    device=s.device,
                    compute_type=s.compute_type,
                    num_workers=s.workers,  # lets several threads transcribe in parallel
                )
        return self._model

    def transcribe(
        self,
        path: Path,
        *,
        language: Optional[str] = None,
        word_timestamps: bool = False,
        on_progress: Optional[ProgressCallback] = None,
    ) -> dict:
        s = self._settings
        segments, info = self._get_model().transcribe(
            str(path),
            language=language,  # None = auto-detect
            beam_size=s.beam_size,
            vad_filter=s.vad_filter,
            vad_parameters={"min_silence_duration_ms": 500} if s.vad_filter else None,
            word_timestamps=word_timestamps,
            condition_on_previous_text=s.condition_on_previous_text,
        )

        # `segments` is a generator: the audio is decoded window by window as we
        # iterate, so we can report progress on long files as we go.
        results = []
        for seg in segments:
            item = {
                "id": seg.id,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "avg_logprob": seg.avg_logprob,
                "no_speech_prob": seg.no_speech_prob,
            }
            if word_timestamps and seg.words:
                item["words"] = [
                    {"start": w.start, "end": w.end, "word": w.word, "probability": w.probability}
                    for w in seg.words
                ]
            results.append(item)
            if on_progress and info.duration:
                on_progress(min(seg.end / info.duration, 1.0))

        return {
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "model": self.model_name,
            "segments": results,
        }
