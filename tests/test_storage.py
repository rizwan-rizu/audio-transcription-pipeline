import io

import pytest

from app.errors import InvalidAudioError, UploadTooLargeError
from app.storage import LocalStorage, is_valid_job_id, safe_suffix


def test_upload_over_limit_is_rejected_and_cleaned_up(tmp_path):
    storage = LocalStorage(tmp_path)
    dest = tmp_path / "out.wav"
    with pytest.raises(UploadTooLargeError):
        storage.save_upload(io.BytesIO(b"x" * 3000), dest, max_bytes=1000)
    assert not dest.exists()
    assert list(tmp_path.glob("*.part")) == []


def test_empty_upload_is_rejected(tmp_path):
    with pytest.raises(InvalidAudioError):
        LocalStorage(tmp_path).save_upload(io.BytesIO(b""), tmp_path / "x.wav", max_bytes=1000)


def test_job_record_round_trip(tmp_path):
    storage = LocalStorage(tmp_path)
    job_id = "a" * 32
    storage.save_job({"id": job_id, "status": "queued"})
    assert storage.load_job(job_id) == {"id": job_id, "status": "queued"}
    assert list(storage.list_job_ids()) == [job_id]
    assert not list(storage.job_dir(job_id).glob("*.tmp"))  # atomic write left nothing behind


def test_job_id_validation_blocks_path_traversal():
    assert is_valid_job_id("0123456789abcdef0123456789abcdef")
    assert not is_valid_job_id("../../etc/passwd")
    assert not is_valid_job_id("")


def test_safe_suffix():
    assert safe_suffix("Meeting.MP3") == ".mp3"
    assert safe_suffix("evil.wav/../../x") == ""
    assert safe_suffix(None) == ""
