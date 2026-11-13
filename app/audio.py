"""Audio validation.

We never trust the file extension or the client's MIME type. Instead we open the
file with FFmpeg (through PyAV, which faster-whisper already depends on) and check
that it really contains audio. Any format FFmpeg can read is accepted: WAV, MP3,
M4A, FLAC, OGG, AIFF, and video files with an audio track.

Decoding to 16 kHz mono (what Whisper expects) is done by faster-whisper itself,
so the rest of the pipeline never has to care which format came in.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .errors import AudioTooLongError, InvalidAudioError


@dataclass(frozen=True)
class AudioInfo:
    duration_s: float
    sample_rate: Optional[int]
    channels: Optional[int]
    codec: Optional[str]
    container: Optional[str]


def probe(path: Path, max_duration_s: Optional[float] = None) -> AudioInfo:
    """Check the file is real, non-empty audio within the duration limit."""
    import av  # imported here so the rest of the app can be tested without PyAV

    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise InvalidAudioError("File has no audio stream.")
            stream = container.streams.audio[0]
            duration = _duration_seconds(av, container, stream)
            info = AudioInfo(
                duration_s=round(duration, 3),
                sample_rate=stream.codec_context.sample_rate or None,
                channels=_channel_count(stream),
                codec=stream.codec_context.name,
                container=container.format.name,
            )
    except av.error.FFmpegError as exc:
        raise InvalidAudioError(f"Could not read the file as audio: {exc}") from exc

    if info.duration_s <= 0:
        raise InvalidAudioError("Audio has zero duration.")
    if max_duration_s is not None and info.duration_s > max_duration_s:
        raise AudioTooLongError(
            f"Audio is {info.duration_s:.0f}s long; the limit is {max_duration_s:.0f}s."
        )
    return info


def _duration_seconds(av, container, stream) -> float:
    # Most formats store the duration in the header, which is instant to read.
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        return container.duration / av.time_base
    # Some streams have no duration header, so we decode them to measure it.
    total = 0.0
    for frame in container.decode(stream):
        total += frame.samples / frame.sample_rate
    return total


def _channel_count(stream) -> Optional[int]:
    # The attribute name differs between PyAV versions.
    codec = stream.codec_context
    layout = getattr(codec, "layout", None)
    if layout is not None and getattr(layout, "nb_channels", None):
        return layout.nb_channels
    return getattr(codec, "channels", None)
