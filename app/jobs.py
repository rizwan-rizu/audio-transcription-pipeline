"""Background jobs: queueing, workers, retries and crash recovery.

Lifecycle:  queued -> processing -> completed
                         |
                         +-> failed (permanent error, or out of retry attempts)
                         +-> queued again after a backoff (transient error)

Concurrency is bounded in one place: `slots`, a semaphore sized to the number of
workers. Both background jobs and synchronous requests must hold a slot while the
model runs, so the machine is never asked to run more transcriptions than it can
handle. New work beyond `max_queue_size` is rejected (HTTP 429) instead of piling up.

This is an in-process queue, which is right for a single machine. In production the
same states and rules would sit on a real broker (SQS, Redis) with a database
holding the job records, so jobs survive and scale across machines.
"""

import logging
import queue
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import BinaryIO, Callable, Optional

from . import audio
from .config import Settings
from .errors import BusyError, PermanentError
from .postprocess import build_transcript
from .storage import LocalStorage, is_valid_job_id, safe_suffix
from .transcriber import Transcriber

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    filename: str
    audio_path: str
    status: JobStatus = JobStatus.QUEUED
    language: Optional[str] = None
    word_timestamps: bool = False
    duration_s: Optional[float] = None
    attempts: int = 0
    progress: float = 0.0
    error: Optional[str] = None
    retryable: Optional[bool] = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Job":
        data = dict(data)
        data["status"] = JobStatus(data["status"])
        return cls(**data)

    def public(self) -> dict:
        """What API clients see (no internal file paths)."""
        data = self.to_dict()
        data.pop("audio_path")
        return data


class JobManager:
    def __init__(
        self,
        storage: LocalStorage,
        transcriber: Transcriber,
        settings: Settings,
        probe: Callable = audio.probe,
    ):
        self._storage = storage
        self._transcriber = transcriber
        self._settings = settings
        self._probe = probe
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._record_lock = threading.Lock()  # guards read-modify-write of job records
        self._reserve_lock = threading.Lock()  # guards queue-place reservations
        self._reserved = 0  # uploads in progress that already hold a queue place
        self.slots = threading.BoundedSemaphore(settings.workers)
        self._stop = threading.Event()
        self._threads: list = []

    # Lifecycle

    def start(self) -> None:
        for i in range(self._settings.workers):
            thread = threading.Thread(target=self._worker_loop, name=f"worker-{i}", daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout)

    def recover(self) -> int:
        """Run on startup: put back any job a crash or restart left unfinished."""
        recovered = 0
        for job_id in self._storage.list_job_ids():
            data = self._storage.load_job(job_id)
            if not data:
                continue
            job = Job.from_dict(data)
            if job.status not in (JobStatus.QUEUED, JobStatus.PROCESSING):
                continue
            if job.status is JobStatus.PROCESSING and job.attempts >= self._settings.max_attempts:
                # A job that keeps killing the process (for example by running out of
                # memory) must not loop forever across restarts.
                self._update(job_id, status=JobStatus.FAILED, retryable=False,
                             error="Interrupted too many times; giving up.")
                continue
            self._update(job_id, status=JobStatus.QUEUED)
            self._queue.put(job_id)
            recovered += 1
        if recovered:
            log.info("Recovered %d unfinished job(s)", recovered)
        return recovered

    # Public operations

    def create_job(
        self,
        *,
        upload: BinaryIO,
        filename: str,
        language: Optional[str] = None,
        word_timestamps: bool = False,
    ) -> Job:
        # Reserve a place in the queue BEFORE saving the upload. Saving and
        # validating take time, and without a reservation many uploads arriving
        # together would all see a short queue and all get in.
        self._reserve_queue_place()
        job_id = uuid.uuid4().hex
        audio_path = self._storage.audio_path(job_id, safe_suffix(filename))
        try:
            try:
                self._storage.save_upload(upload, audio_path, self._settings.max_upload_bytes)
                # Reject bad input now, while the client is still connected, rather
                # than accepting a job that is bound to fail.
                info = self._probe(audio_path, max_duration_s=self._settings.max_duration_s)
            except BaseException:
                self._storage.delete_job(job_id)
                raise

            job = Job(
                id=job_id,
                filename=filename,
                audio_path=str(audio_path),
                language=language,
                word_timestamps=word_timestamps,
                duration_s=info.duration_s,
            )
            self._storage.save_job(job.to_dict())
            self._queue.put(job_id)
            return job
        finally:
            # On success the queue itself now counts the job; on failure the
            # place is simply given back.
            self._release_queue_place()

    def _reserve_queue_place(self) -> None:
        # The limit applies to new work only. Jobs already accepted (retries,
        # recovered jobs) are never dropped.
        with self._reserve_lock:
            if self._queue.qsize() + self._reserved >= self._settings.max_queue_size:
                raise BusyError("Too many jobs waiting. Try again shortly.")
            self._reserved += 1

    def _release_queue_place(self) -> None:
        with self._reserve_lock:
            self._reserved -= 1

    def get_job(self, job_id: str) -> Optional[Job]:
        if not is_valid_job_id(job_id):
            return None
        data = self._storage.load_job(job_id)
        return Job.from_dict(data) if data else None

    def get_transcript(self, job_id: str) -> Optional[dict]:
        return self._storage.load_transcript(job_id) if is_valid_job_id(job_id) else None

    def retry_job(self, job_id: str) -> Job:
        """Manual retry for a job that ran out of automatic attempts."""
        job = self.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status is not JobStatus.FAILED or not job.retryable:
            raise ValueError("Only failed jobs with a transient error can be retried.")
        job = self._update(job_id, status=JobStatus.QUEUED, attempts=0, error=None, progress=0.0)
        self._queue.put(job_id)
        return job

    def transcribe_now(
        self, path: Path, *, language: Optional[str] = None, word_timestamps: bool = False,
        wait_s: float = 30.0,
    ) -> dict:
        """Synchronous path for short clips. Shares the same slots as the workers."""
        self._probe(path, max_duration_s=self._settings.sync_max_duration_s)
        if not self.slots.acquire(timeout=wait_s):
            raise BusyError("All transcription slots are busy. Try again or use /v1/jobs.")
        try:
            raw = self._transcriber.transcribe(
                path, language=language, word_timestamps=word_timestamps
            )
        finally:
            self.slots.release()
        return build_transcript(raw)

    def queue_depth(self) -> int:
        return self._queue.qsize()

    # Worker

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._process(job_id)
            except Exception:
                # A bug in one job must never kill the worker thread.
                log.exception("Unexpected error while processing job %s", job_id)
            finally:
                self._queue.task_done()

    def _process(self, job_id: str) -> None:
        job = self.get_job(job_id)
        if job is None or job.status is not JobStatus.QUEUED:
            return  # deleted, or already handled
        job = self._update(job_id, status=JobStatus.PROCESSING, attempts=job.attempts + 1,
                           progress=0.0)
        log.info("Job %s: attempt %d/%d", job_id, job.attempts, self._settings.max_attempts)

        try:
            with self.slots:
                raw = self._transcriber.transcribe(
                    Path(job.audio_path),
                    language=job.language,
                    word_timestamps=job.word_timestamps,
                    on_progress=lambda p: self._report_progress(job_id, p),
                )
            self._storage.save_transcript(job_id, build_transcript(raw))
            self._update(job_id, status=JobStatus.COMPLETED, progress=1.0, error=None)
            log.info("Job %s: completed", job_id)

        except PermanentError as exc:
            self._update(job_id, status=JobStatus.FAILED, error=str(exc), retryable=False)
            log.warning("Job %s: failed permanently: %s", job_id, exc)

        except Exception as exc:
            # Unknown errors are treated as transient. Retrying a truly broken job
            # costs a few extra attempts, while failing a recoverable job loses work.
            if job.attempts < self._settings.max_attempts:
                delay = self._settings.retry_base_delay_s * 2 ** (job.attempts - 1)
                self._update(job_id, status=JobStatus.QUEUED, retryable=True,
                             error=f"Attempt {job.attempts} failed: {exc}. Retrying in {delay:.0f}s.")
                timer = threading.Timer(delay, self._queue.put, args=(job_id,))
                timer.daemon = True
                timer.start()
                log.warning("Job %s: attempt %d failed (%s); retrying in %.1fs",
                            job_id, job.attempts, exc, delay)
            else:
                self._update(job_id, status=JobStatus.FAILED, retryable=True,
                             error=f"Failed after {job.attempts} attempts: {exc}")
                log.error("Job %s: out of attempts: %s", job_id, exc)

    def _report_progress(self, job_id: str, progress: float) -> None:
        job = self.get_job(job_id)
        # Only write when progress moves at least 1%, to avoid a disk write per segment.
        if job and progress - job.progress >= 0.01:
            self._update(job_id, progress=round(progress, 3))

    def _update(self, job_id: str, **changes) -> Job:
        with self._record_lock:
            job = Job.from_dict(self._storage.load_job(job_id))
            for key, value in changes.items():
                setattr(job, key, value)
            job.updated_at = _now()
            self._storage.save_job(job.to_dict())
            return job
