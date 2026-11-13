"""API behaviour with a fake model. Needs fastapi and httpx."""

import dataclasses
import threading

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from app.api import create_app  # noqa: E402
from app.audio import AudioInfo  # noqa: E402
from app.errors import AudioTooLongError, InvalidAudioError  # noqa: E402
from tests.conftest import FakeTranscriber, fake_probe, wait_for  # noqa: E402

AUDIO = {"file": ("clip.wav", b"fake audio bytes", "audio/wav")}


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings, FakeTranscriber(), probe=fake_probe)) as c:
        yield c


def test_sync_transcribe_returns_segments_with_timestamps(client):
    r = client.post("/v1/transcribe", files=AUDIO)
    assert r.status_code == 200
    body = r.json()
    assert body["language"] == "en"
    assert body["segments"][0] == {
        "id": 0, "start": 0.0, "end": 1.8, "text": "Hello and welcome.",
        "low_confidence": False, "avg_logprob": -0.2, "no_speech_prob": 0.01,
    }


def test_sync_transcribe_as_srt(client):
    r = client.post("/v1/transcribe?format=srt", files=AUDIO)
    assert r.status_code == 200
    assert r.text.startswith("1\n00:00:00,000 --> 00:00:01,800\nHello and welcome.")


def test_invalid_audio_returns_400(settings):
    def bad_probe(path, max_duration_s=None):
        raise InvalidAudioError("File has no audio stream.")

    with TestClient(create_app(settings, FakeTranscriber(), probe=bad_probe)) as c:
        r = c.post("/v1/transcribe", files=AUDIO)
    assert r.status_code == 400
    assert r.json()["error"] == "InvalidAudioError"


def test_long_audio_on_sync_endpoint_points_to_jobs(settings):
    def long_probe(path, max_duration_s=None):
        raise AudioTooLongError("Audio is 600s long; the limit is 120s.")

    with TestClient(create_app(settings, FakeTranscriber(), probe=long_probe)) as c:
        r = c.post("/v1/transcribe", files=AUDIO)
    assert r.status_code == 413
    assert "/v1/jobs" in r.json()["detail"]


def test_bad_language_code_returns_400(client):
    r = client.post("/v1/transcribe", files=AUDIO, data={"language": "english!"})
    assert r.status_code == 400


def test_job_flow(client):
    r = client.post("/v1/jobs", files=AUDIO)
    assert r.status_code == 202
    job_id = r.json()["id"]
    assert r.headers["location"] == f"/v1/jobs/{job_id}"
    assert "audio_path" not in r.json()  # internal paths are not exposed

    assert wait_for(lambda: client.get(f"/v1/jobs/{job_id}").json()["status"] == "completed")
    r = client.get(f"/v1/jobs/{job_id}/transcript?format=txt")
    assert r.status_code == 200
    assert r.text == "Hello and welcome. This is a test.\n"


def test_transcript_before_completion_returns_409(settings):
    gate = threading.Event()
    with TestClient(create_app(settings, FakeTranscriber(block=gate), probe=fake_probe)) as c:
        job_id = c.post("/v1/jobs", files=AUDIO).json()["id"]
        r = c.get(f"/v1/jobs/{job_id}/transcript")
        assert r.status_code == 409
        gate.set()


def test_full_queue_returns_429(settings):
    gate = threading.Event()
    small = dataclasses.replace(settings, max_queue_size=1)
    with TestClient(create_app(small, FakeTranscriber(block=gate), probe=fake_probe)) as c:
        c.post("/v1/jobs", files=AUDIO)          # picked up by the worker (blocked)
        assert wait_for(lambda: c.get("/health").json()["queued"] == 0)
        c.post("/v1/jobs", files=AUDIO)          # waits in the queue
        r = c.post("/v1/jobs", files=AUDIO)      # over the limit
        assert r.status_code == 429
        assert r.headers["retry-after"] == "10"
        gate.set()


def test_unknown_job_returns_404(client):
    assert client.get("/v1/jobs/" + "f" * 32).status_code == 404
    assert client.get("/v1/jobs/not-a-job").status_code == 404
