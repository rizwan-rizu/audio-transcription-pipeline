"""Command-line transcription.

    python transcribe.py samples/sample.aiff
    python transcribe.py meeting.mp3 --format srt --output meeting.srt
    python transcribe.py interview.m4a --language en --words

Uses the same validation, model and post-processing as the API.
"""

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

from app.audio import probe
from app.config import Settings
from app.errors import PermanentError
from app.postprocess import OUTPUT_FORMATS, build_transcript, render
from app.transcriber import WhisperTranscriber


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Transcribe an audio file with timestamps.")
    parser.add_argument("audio", type=Path, help="path to an audio file")
    parser.add_argument("--model", help="Whisper model size (default from WHISPER_MODEL or 'base')")
    parser.add_argument("--language", help="ISO code like 'en'; omit to auto-detect")
    parser.add_argument("--words", action="store_true", help="include word-level timestamps")
    parser.add_argument("--format", choices=OUTPUT_FORMATS, default="json")
    parser.add_argument("--output", type=Path, help="write to this file instead of stdout")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

    settings = Settings.from_env()
    if args.model:
        settings = dataclasses.replace(settings, model_size=args.model)

    try:
        info = probe(args.audio, max_duration_s=settings.max_duration_s)
    except PermanentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"{info.container}/{info.codec}, {info.duration_s:.1f}s, "
        f"{info.sample_rate} Hz, {info.channels} channel(s)",
        file=sys.stderr,
    )

    def show_progress(p: float) -> None:
        print(f"\rprogress: {p:4.0%}", end="", file=sys.stderr, flush=True)

    raw = WhisperTranscriber(settings).transcribe(
        args.audio, language=args.language, word_timestamps=args.words, on_progress=show_progress
    )
    print(file=sys.stderr)
    transcript = build_transcript(raw)

    output = (
        json.dumps(transcript, indent=2, ensure_ascii=False)
        if args.format == "json"
        else render(transcript, args.format)
    )
    if args.output:
        args.output.write_text(output, encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
