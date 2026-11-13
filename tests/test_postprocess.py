from app.postprocess import build_transcript, render, to_srt


def raw(segments, duration=10.0):
    return {"language": "en", "language_probability": 0.9, "duration": duration,
            "model": "fake", "segments": segments}


def seg(start, end, text, logprob=-0.2, no_speech=0.01):
    return {"start": start, "end": end, "text": text, "avg_logprob": logprob,
            "no_speech_prob": no_speech}


def test_cleans_whitespace_and_drops_empty_segments():
    t = build_transcript(raw([seg(0, 1, "  Hello   world "), seg(1, 2, "   "), seg(2, 3, "Bye")]))
    assert [s["text"] for s in t["segments"]] == ["Hello world", "Bye"]
    assert [s["id"] for s in t["segments"]] == [0, 1]  # renumbered after dropping
    assert t["text"] == "Hello world Bye"


def test_flags_low_confidence_segments():
    t = build_transcript(raw([
        seg(0, 1, "clear", logprob=-0.2),
        seg(1, 2, "mumbled", logprob=-1.4),
        seg(2, 3, "probably silence", no_speech=0.9),
    ]))
    assert [s["low_confidence"] for s in t["segments"]] == [False, True, True]
    assert t["stats"] == {"segment_count": 3, "low_confidence_count": 2}


def test_timestamps_are_sane():
    t = build_transcript(raw([seg(-0.5, 1.0, "a"), seg(3.0, 2.0, "b"), seg(9.0, 12.0, "c")],
                             duration=10.0))
    for s in t["segments"]:
        assert 0 <= s["start"] <= s["end"] <= 10.0


def test_segments_sorted_by_start():
    t = build_transcript(raw([seg(5, 6, "second"), seg(1, 2, "first")]))
    assert [s["text"] for s in t["segments"]] == ["first", "second"]


def test_srt_output():
    t = build_transcript(raw([seg(0, 1.5, "Hello"), seg(61.25, 3725.5, "Long one")],
                             duration=4000))
    assert to_srt(t) == (
        "1\n00:00:00,000 --> 00:00:01,500\nHello\n\n"
        "2\n00:01:01,250 --> 01:02:05,500\nLong one\n"
    )


def test_txt_output():
    t = build_transcript(raw([seg(0, 1, "Hello"), seg(1, 2, "there")]))
    assert render(t, "txt") == "Hello there\n"
