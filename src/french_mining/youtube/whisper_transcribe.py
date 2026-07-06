"""Local Whisper transcription fallback (§9 extended): most YouTube videos
have captions (`transcripts.download_subtitles`), but some don't — a video
you watched with no subtitles otherwise can't feed the card or condensed-
audio pipelines at all. Whisper transcription is the fallback: fully local
(no API cost), just slower than downloading captions.

Uses faster-whisper (CTranslate2-based) rather than the original openai-
whisper package: no PyTorch dependency, and int8 CPU inference is fast
enough for a background pipeline step.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from french_mining.nlp import TranscriptSegment

# "small" balances accuracy and speed/download size (~500MB) for a
# background job; bump to "medium"/"large-v3" if quality matters more than
# turnaround, or down to "base" on constrained hardware.
DEFAULT_MODEL_SIZE = "small"


@lru_cache(maxsize=1)
def load_whisper_model(model_size: str = DEFAULT_MODEL_SIZE):
    """Lazily load (and cache) a faster-whisper model. First call downloads
    the model weights from Hugging Face — requires network, one-time per
    model size, then cached locally.
    """
    from faster_whisper import WhisperModel

    return WhisperModel(model_size, device="cpu", compute_type="int8")


def transcribe_with_whisper(
    audio_path: str | Path,
    language: str = "fr",
    model_size: str = DEFAULT_MODEL_SIZE,
    model=None,
) -> list[TranscriptSegment]:
    """Transcribe an audio file into timestamped segments, in the same shape
    `transcripts.parse_vtt` produces so downstream code (`nlp.parse_transcript`)
    can't tell the difference. `model` can be injected (duck-typed: needs a
    `.transcribe(path, language=...)` method returning `(segments, info)`
    where each segment has `.start`/`.end`/`.text`) for testing without
    downloading real model weights.
    """
    model = model or load_whisper_model(model_size)
    raw_segments, _info = model.transcribe(str(audio_path), language=language)

    segments = []
    for seg in raw_segments:
        text = seg.text.strip()
        if text:
            segments.append(TranscriptSegment(start=seg.start, end=seg.end, text=text))
    return segments
