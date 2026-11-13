"""Error types.

The important split is permanent vs transient:
- PermanentError: the input itself is the problem (corrupt file, no audio, too long).
  Retrying will give the same result, so we fail immediately.
- Anything else (model crash, out of memory, disk hiccup) is treated as transient
  and retried with backoff, up to a limit.
"""


class TranscriptionError(Exception):
    """Base class for errors this service raises on purpose."""


class PermanentError(TranscriptionError):
    """Retrying will not help."""


class InvalidAudioError(PermanentError):
    """The file is not readable audio (corrupt, empty, no audio stream)."""


class AudioTooLongError(PermanentError):
    """The audio is longer than the configured limit."""


class UploadTooLargeError(PermanentError):
    """The uploaded file is bigger than the configured size limit."""


class BusyError(TranscriptionError):
    """The service is at capacity. The client should retry later."""
