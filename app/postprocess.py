"""Turn raw model output into a clean transcript that downstream systems can use.

- tidies whitespace and drops empty segments
- keeps timestamps sane (never negative, end >= start, within the file)
- flags low-confidence segments so a downstream step (search, summaries, human
  review) can treat them with care instead of trusting every line equally
- renders JSON, SRT subtitles or plain text
"""

from typing import Optional

# The same thresholds Whisper uses internally when deciding a window failed or was
# silence. We use them as a "worth a second look" flag, not to delete text.
LOW_AVG_LOGPROB = -1.0
HIGH_NO_SPEECH_PROB = 0.6

OUTPUT_FORMATS = ("json", "srt", "txt")


def clean_text(text: str) -> str:
    return " ".join(text.split())


def build_transcript(raw: dict) -> dict:
    duration: Optional[float] = raw.get("duration")
    segments = []

    for seg in sorted(raw["segments"], key=lambda s: s["start"]):
        text = clean_text(seg["text"])
        if not text:
            continue

        start = max(0.0, float(seg["start"]))
        end = max(start, float(seg["end"]))
        if duration:
            start, end = min(start, duration), min(end, duration)

        avg_logprob = seg.get("avg_logprob")
        no_speech_prob = seg.get("no_speech_prob")
        low_confidence = (avg_logprob is not None and avg_logprob < LOW_AVG_LOGPROB) or (
            no_speech_prob is not None and no_speech_prob > HIGH_NO_SPEECH_PROB
        )

        item = {
            "id": len(segments),
            "start": round(start, 3),
            "end": round(end, 3),
            "text": text,
            "low_confidence": low_confidence,
            "avg_logprob": _round(avg_logprob),
            "no_speech_prob": _round(no_speech_prob),
        }
        if seg.get("words"):
            item["words"] = [
                {
                    "start": round(w["start"], 3),
                    "end": round(w["end"], 3),
                    "word": w["word"].strip(),
                    "probability": _round(w.get("probability")),
                }
                for w in seg["words"]
            ]
        segments.append(item)

    return {
        "text": " ".join(s["text"] for s in segments),
        "language": raw.get("language"),
        "language_probability": _round(raw.get("language_probability")),
        "duration": _round(duration),
        "model": raw.get("model"),
        "segments": segments,
        "stats": {
            "segment_count": len(segments),
            "low_confidence_count": sum(s["low_confidence"] for s in segments),
        },
    }


def to_srt(transcript: dict) -> str:
    blocks = []
    for i, seg in enumerate(transcript["segments"], start=1):
        blocks.append(f"{i}\n{_srt_time(seg['start'])} --> {_srt_time(seg['end'])}\n{seg['text']}\n")
    return "\n".join(blocks)


def render(transcript: dict, fmt: str) -> str:
    """Text formats only. JSON is returned as a dict by the caller."""
    if fmt == "srt":
        return to_srt(transcript)
    if fmt == "txt":
        return transcript["text"] + "\n"
    raise ValueError(f"Unsupported text format: {fmt}")


def _srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _round(value: Optional[float], digits: int = 3) -> Optional[float]:
    return None if value is None else round(float(value), digits)
