# Transcription Service

A small service that takes an audio file, transcribes it with timestamps per segment, and returns a clean transcript for downstream use (JSON, SRT subtitles or plain text).

It uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper), which runs OpenAI's Whisper models on the CTranslate2 inference engine. No model training is involved. The work here is the pipeline around the model: input validation, long audio, concurrency, failures, and the shape of the output.

```
upload ──► validate (FFmpeg probe) ──► store ──► queue ──► worker ──► faster-whisper ──► post-process ──► store transcript
             │ bad input → 400/413       │ full → 429      │ holds a slot   │ VAD + 30s windows   │ clean text, confidence
             └ before any work is queued                    └ retries + crash recovery              │ flags, JSON / SRT / TXT
```

## Quick start (macOS / Linux)

Requires Python 3.10+. No system FFmpeg needed (PyAV bundles it).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Make a test recording (macOS has text-to-speech built in; this also produces an AIFF file,
# which shows that format handling is not limited to WAV/MP3)
say -o samples/sample.aiff "One day the Hare laughed at the short feet and slow speed of the Tortoise. The Tortoise replied: You may be as fast as the wind, but I will beat you in a race! The Hare thought this idea was impossible and he agreed to the proposal. It was agreed that the Fox should choose the course and decide the end. The day for the race came, and the Tortoise and Hare started together. The Tortoise never stopped for a moment, walking slowly but steadily, right to the end of the course. The Hare ran fast and stopped to lie down for a rest. But he fell fast asleep. Eventually, he woke up and ran as fast as he could. But when he reached the end, he saw the Tortoise there already, sleeping comfortably after her effort."

# Command line
python transcribe.py samples/sample.aiff
python transcribe.py samples/sample.aiff --format srt

# API (the first start downloads the model weights)
uvicorn app.api:create_app --factory --port 8000
# open http://localhost:8000/docs for interactive API docs

# Tests (no model download needed, the model is faked)
pytest
```

## API

| Method | Path | What it does |
|---|---|---|
| `POST` | `/v1/transcribe` | Synchronous. For short clips (≤ 120 s by default). Returns the transcript directly. |
| `POST` | `/v1/jobs` | Asynchronous. Validates and stores the file, queues it, returns `202` with a job ID and a `Location` header. |
| `GET` | `/v1/jobs/{id}` | Job status: `queued`, `processing` (with `progress`), `completed` or `failed` (with `error` and whether it is `retryable`). |
| `GET` | `/v1/jobs/{id}/transcript` | The result. `?format=json` (default), `srt` or `txt`. `409` if not finished yet. |
| `POST` | `/v1/jobs/{id}/retry` | Manually retry a job that failed with a transient error. |
| `GET` | `/health` | Liveness, model name and queue depth. |

Uploads are `multipart/form-data` with `file`, plus optional `language` (ISO code like `en`; omitted = auto-detect) and `word_timestamps` (`true`/`false`).

```bash
curl -F "file=@samples/sample.aiff" http://localhost:8000/v1/transcribe
curl -F "file=@long-meeting.mp3" http://localhost:8000/v1/jobs
curl http://localhost:8000/v1/jobs/<id>
curl "http://localhost:8000/v1/jobs/<id>/transcript?format=srt"
```

Example response (trimmed):

```json
{
  "text": "Hello, this is a short test of the transcription pipeline.",
  "language": "en",
  "language_probability": 0.98,
  "duration": 4.1,
  "model": "faster-whisper/base",
  "segments": [
    {"id": 0, "start": 0.0, "end": 3.8, "text": "Hello, this is a short test of the transcription pipeline.",
     "low_confidence": false, "avg_logprob": -0.21, "no_speech_prob": 0.01}
  ],
  "stats": {"segment_count": 1, "low_confidence_count": 0}
}
```

Errors always come back as `{"error": "<Type>", "detail": "<message>"}`: `400` unreadable audio, `413` file too large or too long, `429` busy (with `Retry-After`).

## Code layout

| File | Responsibility |
|---|---|
| `app/config.py` | All settings, read from environment variables |
| `app/errors.py` | Error types, split into **permanent** (bad input) and everything else (transient) |
| `app/audio.py` | Validates input by probing it with FFmpeg: real audio, non-empty, within limits |
| `app/transcriber.py` | The faster-whisper wrapper behind a small `Transcriber` interface |
| `app/postprocess.py` | Cleans model output, flags low-confidence segments, renders JSON/SRT/TXT |
| `app/storage.py` | Local storage for audio, job records and transcripts (atomic writes) |
| `app/jobs.py` | Queue, workers, bounded concurrency, retries with backoff, crash recovery |
| `app/api.py` | FastAPI routes and error-to-HTTP mapping |
| `transcribe.py` | CLI using the same pipeline |

## Design decisions

**1. faster-whisper, running locally, rather than a hosted API.**
Whisper handles punctuation, casing and many languages out of the box, and returns segment timestamps natively. faster-whisper runs the same models more efficiently and supports int8 on CPU, so it runs on a laptop. I chose local over a hosted API because anyone can run this repo without an API key or cost, audio never leaves the machine, and there is no vendor file-size limit (OpenAI's transcription API, for example, caps uploads at 25 MB and only returns timestamps from its `whisper-1` model). The cost is that I manage compute myself, and CPU is much slower than GPU. The engine sits behind the `Transcriber` interface, so a hosted backend could be added without touching the pipeline.

**2. Format handling: normalise at the boundary, never trust the extension.**
Every file is opened with FFmpeg (through PyAV) and checked for a real audio stream, a non-zero duration and the size and duration limits. A text file renamed to `.mp3` is rejected with a clear `400`, not a stack trace from deep inside the model. Anything FFmpeg can read is accepted (WAV, MP3, M4A, FLAC, OGG, AIFF, video with an audio track), and faster-whisper decodes it to 16 kHz mono, which is what Whisper expects. Adding a new format therefore needs no new code. Stereo is downmixed. For call recordings with one speaker per channel, I would transcribe each channel separately instead.

**3. Long audio.**
Whisper processes audio in 30-second windows, so length is really two problems: model quality and service behaviour.
- *Model:* voice activity detection (Silero VAD, built into faster-whisper) skips silence. That saves compute and reduces Whisper's tendency to invent text during silent stretches. `CONDITION_ON_PREVIOUS_TEXT` can be switched off if a long file falls into a repetition loop.
- *Service:* an HTTP request should not stay open for an hour of audio. The sync endpoint is capped (120 s by default) and tells the client to use `/v1/jobs` instead. Jobs report progress as segments come out of the model, because faster-whisper yields segments lazily rather than all at the end.

**4. Two endpoints instead of one.**
Short clips go to `/v1/transcribe` and get the transcript straight back. Anything long goes to `/v1/jobs`. I thought about a single endpoint that decides for itself based on file length, but then the response would sometimes be a transcript and sometimes a job ID, so every client would need to handle both anyway. Two endpoints keep each response predictable. `/v1/jobs` works for any length, so if a client doesn't know how long its file is, that's the safe choice. Long term I'd move to a single async endpoint with webhooks (see "API" below), with the sync endpoint kept only as a shortcut for very short clips.

**5. Bounded concurrency and back-pressure.**
Transcription is CPU/GPU-bound, so running more of them at once than the hardware can handle just makes every one slower. A single semaphore (`WORKERS`, default 1) bounds model use for both sync requests and background jobs. Oversized uploads are rejected from the `Content-Length` header before the body is read (FastAPI otherwise receives the whole multipart body before a route runs, so a limit checked inside the route would only apply after every byte had arrived). Uploads without a declared length are still caught by a byte count while the file is copied to job storage in 1 MB chunks. A place in the queue is reserved *before* the upload is saved and validated, and given back if that fails, so uploads arriving at the same moment can't all slip past the limit. When the queue is full (`MAX_QUEUE_SIZE`), new work gets `429` with `Retry-After` instead of being accepted and then timing out. Work that has already been accepted is never dropped.

**6. Failures: permanent vs transient.**
Bad input is rejected *before* queueing, while the client is still connected. In the worker:
- **Permanent errors** (corrupt or invalid audio): fail immediately. Retrying gives the same result.
- **Anything else** (model crash, memory, I/O): retried with exponential backoff (2 s, 4 s, …) up to `MAX_ATTEMPTS`. After that the job is `failed` but marked `retryable`, and `POST /v1/jobs/{id}/retry` re-runs it.
- **Crash recovery:** on startup, jobs left `queued` or `processing` are queued again. A job that has already used all its attempts, for example one that keeps running the process out of memory, is failed instead of looping across restarts.

I treat unknown errors as transient on purpose: retrying a broken job wastes a couple of attempts, while giving up on a recoverable job loses work.

**7. Storage.**
Local disk, laid out the way production would be: `audio.*` (→ object storage), `job.json` (→ database row), `transcript.json` (→ database or object storage). Job records are written atomically (temp file plus rename), so a crash never leaves a half-written record behind. Synchronous requests keep no audio. Job IDs are validated before being used in paths, which blocks path traversal.

**8. Output built for downstream use.**
The transcript is cleaned (whitespace, empty segments), timestamps are kept sane (never negative, end ≥ start, within the file), and each segment carries `low_confidence`. That flag reuses the thresholds Whisper applies internally (`avg_logprob < -1.0` or `no_speech_prob > 0.6`), so a downstream step can send doubtful lines for review instead of trusting every line equally. JSON for systems, SRT for subtitles, TXT for search and LLM input.

## Scaling this to production

The code above runs on one machine. The states and rules stay the same at scale; the pieces underneath change.

**Concurrent uploads.** Clients upload straight to object storage using presigned URLs, so API servers never stream audio bytes and stay small and stateless. The API then creates a job and puts a message on a queue (SQS, or Redis/RabbitMQ). A separate pool of GPU workers pulls from the queue and autoscales on queue depth. Per-client rate limits and quotas stop one tenant from starving everyone else. Short clips can go to a priority queue so they aren't stuck behind hour-long files.

**Storage.** Audio in S3 (or GCS) with encryption and a lifecycle rule that deletes originals after N days, because recordings often contain personal data. Job metadata in Postgres. Transcripts as JSONB in Postgres if they are queried, or in S3 if they are large and only fetched whole. Transcript text can be indexed for search.

**Retries and recovery.** The same permanent/transient split. Workers take a lease on a job (the queue's visibility timeout): if a worker dies, the lease expires and another worker picks the job up. Jobs that use up their attempts go to a dead-letter queue with alerting. Processing is idempotent per job ID (one job ID, one transcript), so a duplicate delivery doesn't create duplicate work. Very long files are split at silences and each chunk is checkpointed, so a retry resumes rather than starting again.

**API.** The async job pattern shown here is the core: `POST` returns `202` and a job ID, the client polls or registers a **webhook** to be notified on completion. Add API-key or OAuth auth, an `Idempotency-Key` header so retried uploads don't create duplicate jobs, versioned routes (`/v1`), consistent error bodies, and the OpenAPI spec (already generated at `/docs`) as the contract.

## Limitations and next steps

- In-process queue and local disk: fine for one machine, but jobs can't be spread across servers (see above).
- No speaker diarization. Next step: pyannote or WhisperX.
- Word timestamps (`word_timestamps=true`) are approximate. For subtitle-grade word timing I'd add forced alignment (for example WhisperX's wav2vec2 alignment).
- No accuracy evaluation yet. Next step: a small labelled set covering accents, noise and formats, tracking word error rate (WER) whenever the model or settings change.
- CPU-only by default. `WHISPER_DEVICE=cuda` with `WHISPER_COMPUTE_TYPE=float16` on a GPU machine, and `WHISPER_MODEL=small` or larger for better accuracy.
- No auth on the API.
- FFmpeg parses files from untrusted users. In production I'd restrict accepted formats to the ones actually needed and run decoding in an isolated worker.
- The low-confidence thresholds are Whisper's defaults. They should be tuned on real recordings.

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `WHISPER_MODEL` | `base` | `tiny`, `base`, `small`, `medium`, `large-v3`, … Bigger is more accurate and slower |
| `WHISPER_DEVICE` / `WHISPER_COMPUTE_TYPE` | `cpu` / `int8` | `cuda` / `float16` on a GPU |
| `VAD_FILTER` | `true` | Skip silence |
| `CONDITION_ON_PREVIOUS_TEXT` | `true` | Set `false` if long files produce repetition loops |
| `WORKERS` | `1` | Concurrent transcriptions |
| `MAX_QUEUE_SIZE` | `20` | Beyond this, new jobs get `429` |
| `MAX_ATTEMPTS` / `RETRY_BASE_DELAY_S` | `3` / `2` | Retry policy |
| `MAX_UPLOAD_MB` / `MAX_DURATION_S` | `200` / `14400` | Input limits |
| `SYNC_MAX_DURATION_S` | `120` | Longer files must use `/v1/jobs` |
| `DATA_DIR` | `data` | Where audio, job records and transcripts are stored |

## Testing

`pytest` runs in seconds without downloading a model: a fake transcriber stands in for Whisper, and test audio is generated (sine-wave WAVs, stereo, corrupt bytes, a text file renamed `.mp3`). The tests cover format validation, timestamp clean-up, SRT output, the job lifecycle, retries, giving up after the maximum attempts, crash recovery, back-pressure and the HTTP status codes. Model accuracy is checked separately by running real recordings through the CLI.
