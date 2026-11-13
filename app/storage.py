"""Local filesystem storage.

The layout mirrors what production would use, so swapping it out is a contained
change:

    data/jobs/<job_id>/audio.<ext>        -> object storage (S3, GCS)
    data/jobs/<job_id>/job.json           -> database row (Postgres)
    data/jobs/<job_id>/transcript.json    -> database (JSONB) or object storage
    data/tmp/                             -> scratch space for synchronous requests

JSON files are written atomically (write to a temp file, then rename), so a crash
mid-write never leaves a half-written job record behind.
"""

import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import BinaryIO, Iterator, Optional

from .errors import InvalidAudioError, UploadTooLargeError

CHUNK_SIZE = 1024 * 1024  # stream uploads 1 MB at a time instead of loading them into memory
JOB_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
SAFE_SUFFIX = re.compile(r"^\.[a-z0-9]{1,5}$")


def is_valid_job_id(job_id: str) -> bool:
    # Job IDs end up in file paths, so reject anything that isn't one of ours
    # (this also blocks path traversal like "../../etc").
    return bool(JOB_ID_PATTERN.match(job_id))


def safe_suffix(filename: Optional[str]) -> str:
    """Keep the extension for readability only; we never rely on it to detect the format."""
    suffix = Path(filename or "").suffix.lower()
    return suffix if SAFE_SUFFIX.match(suffix) else ""


class LocalStorage:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.jobs_dir = self.root / "jobs"
        self.tmp_dir = self.root / "tmp"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    # Uploads

    def save_upload(self, source: BinaryIO, dest: Path, max_bytes: int) -> int:
        """Stream an upload to disk, enforcing the size limit as bytes arrive."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        partial = dest.with_name(dest.name + ".part")
        written = 0
        try:
            with open(partial, "wb") as out:
                while chunk := source.read(CHUNK_SIZE):
                    written += len(chunk)
                    if written > max_bytes:
                        raise UploadTooLargeError(
                            f"File is larger than the {max_bytes // (1024 * 1024)} MB limit."
                        )
                    out.write(chunk)
            if written == 0:
                raise InvalidAudioError("Uploaded file is empty.")
            os.replace(partial, dest)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        return written

    def temp_path(self, suffix: str = "") -> Path:
        return self.tmp_dir / f"{uuid.uuid4().hex}{suffix}"

    # Jobs

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def audio_path(self, job_id: str, suffix: str = "") -> Path:
        return self.job_dir(job_id) / f"audio{suffix}"

    def save_job(self, job: dict) -> None:
        self._write_json(self.job_dir(job["id"]) / "job.json", job)

    def load_job(self, job_id: str) -> Optional[dict]:
        return self._read_json(self.job_dir(job_id) / "job.json")

    def list_job_ids(self) -> Iterator[str]:
        for path in sorted(self.jobs_dir.iterdir()):
            if path.is_dir() and is_valid_job_id(path.name):
                yield path.name

    def save_transcript(self, job_id: str, transcript: dict) -> None:
        self._write_json(self.job_dir(job_id) / "transcript.json", transcript)

    def load_transcript(self, job_id: str) -> Optional[dict]:
        return self._read_json(self.job_dir(job_id) / "transcript.json")

    def delete_job(self, job_id: str) -> None:
        shutil.rmtree(self.job_dir(job_id), ignore_errors=True)

    # Helpers

    @staticmethod
    def _write_json(path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)  # atomic on the same filesystem

    @staticmethod
    def _read_json(path: Path) -> Optional[dict]:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
