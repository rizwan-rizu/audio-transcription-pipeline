"""HTTP API.

Run with:  uvicorn app.api:create_app --factory

Two ways in:
- POST /v1/transcribe  synchronous, for short clips (up to SYNC_MAX_DURATION_S).
- POST /v1/jobs        asynchronous, for anything long. Returns a job ID right away;
                       poll GET /v1/jobs/{id}, then fetch /v1/jobs/{id}/transcript.

Interactive docs (OpenAPI) are served at /docs.
"""

import logging
import re
from contextlib import asynccontextmanager
from typing import Callable, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, PlainTextResponse

from . import audio
from .config import Settings
from .errors import (
    AudioTooLongError,
    BusyError,
    InvalidAudioError,
    TranscriptionError,
    UploadTooLargeError,
)
from .jobs import JobManager, JobStatus
from .postprocess import render
from .storage import LocalStorage, safe_suffix
from .transcriber import Transcriber, WhisperTranscriber

log = logging.getLogger(__name__)

LANGUAGE_CODE = re.compile(r"^[a-z]{2,3}$")
FORMAT_QUERY = Query("json", pattern="^(json|srt|txt)$", description="json, srt or txt")
MEDIA_TYPES = {"srt": "application/x-subrip", "txt": "text/plain"}

# Our errors -> HTTP status codes. Order matters: most specific first.
ERROR_STATUS = [
    (UploadTooLargeError, 413),
    (AudioTooLongError, 413),
    (InvalidAudioError, 400),
    (BusyError, 429),
]


def create_app(
    settings: Optional[Settings] = None,
    transcriber: Optional[Transcriber] = None,
    probe: Callable = audio.probe,
) -> FastAPI:
    settings = settings or Settings.from_env()
    storage = LocalStorage(settings.data_dir)
    transcriber = transcriber or WhisperTranscriber(settings)
    manager = JobManager(storage, transcriber, settings, probe=probe)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.warm_up and hasattr(transcriber, "warm_up"):
            await run_in_threadpool(transcriber.warm_up)
        manager.start()
        manager.recover()
        yield
        manager.stop()

    app = FastAPI(title="Transcription Service", version="0.1.0", lifespan=lifespan)
    app.state.manager = manager

    @app.exception_handler(TranscriptionError)
    async def handle_known_errors(request: Request, exc: TranscriptionError):
        status = next((code for cls, code in ERROR_STATUS if isinstance(exc, cls)), 500)
        headers = {"Retry-After": "10"} if isinstance(exc, BusyError) else None
        return JSONResponse(
            status_code=status,
            content={"error": type(exc).__name__, "detail": str(exc)},
            headers=headers,
        )

    # Routes are plain `def` (not async) on purpose: transcription is blocking CPU
    # work, so FastAPI runs these in its thread pool instead of blocking the event loop.

    @app.get("/health")
    def health():
        return {"status": "ok", "model": transcriber.model_name, "queued": manager.queue_depth()}

    @app.post("/v1/transcribe")
    def transcribe_sync(
        file: UploadFile = File(..., description="Audio file (WAV, MP3, M4A, FLAC, ...)"),
        language: Optional[str] = Form(None, description="ISO code like 'en'; omit to auto-detect"),
        word_timestamps: bool = Form(False),
        format: str = FORMAT_QUERY,
    ):
        _check_language(language)
        tmp_path = storage.temp_path(safe_suffix(file.filename))
        try:
            storage.save_upload(file.file, tmp_path, settings.max_upload_bytes)
            try:
                transcript = manager.transcribe_now(
                    tmp_path, language=language, word_timestamps=word_timestamps
                )
            except AudioTooLongError as exc:
                raise AudioTooLongError(f"{exc} Use POST /v1/jobs for longer files.") from exc
        finally:
            tmp_path.unlink(missing_ok=True)  # synchronous requests keep no audio
        return _respond(transcript, format)

    @app.post("/v1/jobs", status_code=202)
    def create_job(
        response: Response,
        file: UploadFile = File(...),
        language: Optional[str] = Form(None),
        word_timestamps: bool = Form(False),
    ):
        _check_language(language)
        job = manager.create_job(
            upload=file.file,
            filename=file.filename or "audio",
            language=language,
            word_timestamps=word_timestamps,
        )
        response.headers["Location"] = f"/v1/jobs/{job.id}"
        return job.public()

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str):
        return _job_or_404(manager, job_id).public()

    @app.get("/v1/jobs/{job_id}/transcript")
    def get_transcript(job_id: str, format: str = FORMAT_QUERY):
        job = _job_or_404(manager, job_id)
        if job.status is not JobStatus.COMPLETED:
            raise HTTPException(409, f"Transcript not ready; job is {job.status.value}.")
        return _respond(manager.get_transcript(job_id), format)

    @app.post("/v1/jobs/{job_id}/retry", status_code=202)
    def retry_job(job_id: str):
        _job_or_404(manager, job_id)
        try:
            return manager.retry_job(job_id).public()
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    return app


def _check_language(language: Optional[str]) -> None:
    if language is not None and not LANGUAGE_CODE.match(language):
        raise HTTPException(400, "language must be an ISO code such as 'en' or 'ur'.")


def _job_or_404(manager: JobManager, job_id: str):
    job = manager.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found.")
    return job


def _respond(transcript: dict, fmt: str):
    if fmt == "json":
        return transcript
    return PlainTextResponse(render(transcript, fmt), media_type=MEDIA_TYPES[fmt])
