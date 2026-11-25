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


def test_queue_limit_holds_for_uploads_arriving_together(settings):
    # Regression: the limit used to be checked before the upload was saved and the
    # job queued only after, so simultaneous uploads could all get past it.
    import dataclasses
    import threading

    in_probe = threading.Event()
    release = threading.Event()

    def slow_probe(path, max_duration_s=None):
        in_probe.set()
        release.wait(5)
        return fake_probe(path)

    one_slot = dataclasses.replace(settings, max_queue_size=1)
    manager, storage = make_manager(one_slot, probe=slow_probe)

    first = threading.Thread(target=submit, args=(manager,))
    first.start()
    assert in_probe.wait(5)          # first upload is mid-validation, queue still empty
    assert manager.queue_depth() == 0
    with pytest.raises(BusyError):   # second upload must still be refused
        submit(manager)
    release.set()
    first.join(5)
    assert manager.queue_depth() == 1


def test_failed_upload_gives_its_queue_place_back(settings):
    import dataclasses

    def bad_probe(path, max_duration_s=None):
        raise InvalidAudioError("no audio stream")

    one_slot = dataclasses.replace(settings, max_queue_size=1)
    manager, _ = make_manager(one_slot, probe=bad_probe)
    for _ in range(3):  # each failure must release its place, or the 2nd try would get 429
        with pytest.raises(InvalidAudioError):
            submit(manager)
