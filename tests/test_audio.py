"""Format handling against real files. Needs PyAV (installed with faster-whisper)."""

import pytest

pytest.importorskip("av")

from app.audio import probe  # noqa: E402
from app.errors import AudioTooLongError, InvalidAudioError  # noqa: E402
from tests.conftest import make_wav  # noqa: E402


def test_reads_mono_wav(tmp_path):
    info = probe(make_wav(tmp_path / "a.wav", seconds=1.0))
    assert info.duration_s == pytest.approx(1.0, abs=0.05)
    assert info.sample_rate == 16000 and info.channels == 1


def test_reads_stereo_44k_wav(tmp_path):
    info = probe(make_wav(tmp_path / "b.wav", seconds=2.0, rate=44100, channels=2))
    assert info.channels == 2 and info.sample_rate == 44100
    assert info.duration_s == pytest.approx(2.0, abs=0.05)


def test_extension_is_not_trusted(tmp_path):
    fake = tmp_path / "notes.mp3"
    fake.write_text("this is a text file, not audio")
    with pytest.raises(InvalidAudioError):
        probe(fake)


def test_corrupt_bytes_are_rejected(tmp_path):
    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"RIFF\x00\x00garbage" * 10)
    with pytest.raises(InvalidAudioError):
        probe(junk)


def test_duration_limit(tmp_path):
    with pytest.raises(AudioTooLongError):
        probe(make_wav(tmp_path / "long.wav", seconds=3.0), max_duration_s=2.0)
