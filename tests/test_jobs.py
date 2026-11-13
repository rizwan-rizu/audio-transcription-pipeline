import io

import pytest

from app.errors import BusyError, InvalidAudioError
from app.jobs import Job, JobManager, JobStatus
from app.storage import LocalStorage
from tests.conftest import FakeTranscriber, fake_probe, wait_for


def make_manager(settings, transcriber=None, probe=fake_probe):
    storage = LocalStorage(settings.data_dir)
    return JobManager(storage, transcriber or FakeTranscriber(), settings, probe=probe), storage


def submit(manager):
    return manager.create_job(upload=io.BytesIO(b"fake audio bytes"), filename="clip.wav")


def status_of(manager, job_id):
    return manager.get_job(job_id).status


def test_job_completes_and_stores_transcript(settings):
    manager, _ = make_manager(settings)
    manager.start()
    job = submit(manager)
    assert wait_for(lambda: status_of(manager, job.id) is JobStatus.COMPLETED)
    done = manager.get_job(job.id)
    assert done.progress == 1.0 and done.attempts == 1
    assert manager.get_transcript(job.id)["text"] == "Hello and welcome. This is a test."
    manager.stop()


def test_permanent_error_is_not_retried(settings):
    transcriber = FakeTranscriber(fail_times=99, error=InvalidAudioError("corrupt"))
    manager, _ = make_manager(settings, transcriber)
    manager.start()
    job = submit(manager)
    assert wait_for(lambda: status_of(manager, job.id) is JobStatus.FAILED)
    failed = manager.get_job(job.id)
    assert failed.attempts == 1 and failed.retryable is False
    assert transcriber.calls == 1
    manager.stop()


def test_transient_error_is_retried_until_success(settings):
    manager, _ = make_manager(settings, FakeTranscriber(fail_times=2))
    manager.start()
    job = submit(manager)
    assert wait_for(lambda: status_of(manager, job.id) is JobStatus.COMPLETED)
    assert manager.get_job(job.id).attempts == 3
    manager.stop()


def test_gives_up_after_max_attempts_then_manual_retry_works(settings):
    manager, _ = make_manager(settings, FakeTranscriber(fail_times=3))
    manager.start()
    job = submit(manager)
    assert wait_for(lambda: status_of(manager, job.id) is JobStatus.FAILED)
    failed = manager.get_job(job.id)
    assert failed.attempts == 3 and failed.retryable is True

    manager.retry_job(job.id)  # 4th call succeeds
    assert wait_for(lambda: status_of(manager, job.id) is JobStatus.COMPLETED)
    manager.stop()


def test_bad_input_is_rejected_before_queueing(settings):
    def bad_probe(path, max_duration_s=None):
        raise InvalidAudioError("no audio stream")

    manager, storage = make_manager(settings, probe=bad_probe)
    with pytest.raises(InvalidAudioError):
        submit(manager)
    assert list(storage.list_job_ids()) == []  # nothing left on disk
    assert manager.queue_depth() == 0


def test_queue_limit_applies_back_pressure(settings):
    manager, storage = make_manager(settings)  # workers not started: queue only fills
    for _ in range(settings.max_queue_size):
        submit(manager)
    with pytest.raises(BusyError):
        submit(manager)
    assert len(list(storage.list_job_ids())) == settings.max_queue_size


def test_recovery_requeues_interrupted_jobs(settings):
    manager, storage = make_manager(settings)
    interrupted = Job(id="a" * 32, filename="x.wav", audio_path="x.wav",
                      status=JobStatus.PROCESSING, attempts=1)
    storage.save_job(interrupted.to_dict())

    assert manager.recover() == 1
    manager.start()
    assert wait_for(lambda: status_of(manager, interrupted.id) is JobStatus.COMPLETED)
    manager.stop()


def test_recovery_gives_up_on_jobs_that_keep_crashing(settings):
    manager, storage = make_manager(settings)
    poison = Job(id="b" * 32, filename="x.wav", audio_path="x.wav",
                 status=JobStatus.PROCESSING, attempts=settings.max_attempts)
    storage.save_job(poison.to_dict())

    assert manager.recover() == 0
    assert status_of(manager, poison.id) is JobStatus.FAILED


def test_unknown_or_malformed_job_id(settings):
    manager, _ = make_manager(settings)
    assert manager.get_job("../../etc/passwd") is None
    assert manager.get_job("f" * 32) is None
