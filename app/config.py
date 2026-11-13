"""Runtime settings.

Everything is read from environment variables so the same code runs on a laptop
(small model, CPU) and on a GPU box (bigger model, more workers) without changes.
"""

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # Model
    model_size: str = "base"          # tiny / base / small / medium / large-v3 ...
    device: str = "cpu"               # "cpu", "cuda" or "auto"
    compute_type: str = "int8"        # int8 keeps memory and CPU time low
    beam_size: int = 5
    vad_filter: bool = True           # skip silence: less compute, fewer hallucinations
    condition_on_previous_text: bool = True  # set False if long files get repetition loops

    # Limits
    data_dir: Path = Path("data")
    max_upload_mb: int = 200
    max_duration_s: float = 4 * 3600   # hard cap for any file
    sync_max_duration_s: float = 120   # longer files must go through /v1/jobs

    # Concurrency and retries
    workers: int = 1                   # how many transcriptions run at the same time
    max_queue_size: int = 20           # back-pressure: new jobs beyond this get HTTP 429
    max_attempts: int = 3
    retry_base_delay_s: float = 2.0    # exponential backoff: 2s, 4s, 8s ...

    warm_up: bool = True               # load the model at startup, not on the first request

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model_size=os.getenv("WHISPER_MODEL", cls.model_size),
            device=os.getenv("WHISPER_DEVICE", cls.device),
            compute_type=os.getenv("WHISPER_COMPUTE_TYPE", cls.compute_type),
            beam_size=int(os.getenv("WHISPER_BEAM_SIZE", cls.beam_size)),
            vad_filter=_env_bool("VAD_FILTER", cls.vad_filter),
            condition_on_previous_text=_env_bool(
                "CONDITION_ON_PREVIOUS_TEXT", cls.condition_on_previous_text
            ),
            data_dir=Path(os.getenv("DATA_DIR", str(cls.data_dir))),
            max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", cls.max_upload_mb)),
            max_duration_s=float(os.getenv("MAX_DURATION_S", cls.max_duration_s)),
            sync_max_duration_s=float(os.getenv("SYNC_MAX_DURATION_S", cls.sync_max_duration_s)),
            workers=int(os.getenv("WORKERS", cls.workers)),
            max_queue_size=int(os.getenv("MAX_QUEUE_SIZE", cls.max_queue_size)),
            max_attempts=int(os.getenv("MAX_ATTEMPTS", cls.max_attempts)),
            retry_base_delay_s=float(os.getenv("RETRY_BASE_DELAY_S", cls.retry_base_delay_s)),
            warm_up=_env_bool("WARM_UP", cls.warm_up),
        )
