"""Wires the YouTube-specific pieces into the shared scoring/generation/write
pipeline: clips real source audio for the sentence (§9) and grabs a raw
source frame at the sentence's midpoint for `french_mining.images` to judge
(§10) — this module doesn't decide whether a frame is any good, it just
produces the candidate frame.
"""
from __future__ import annotations

from pathlib import Path

from french_mining.anki.connect import AnkiConnectClient
from french_mining.nlp import TranscriptSegment
from french_mining.scoring import ScoredCandidate
from french_mining.youtube.media import clip_audio, download_audio, extract_frame
from french_mining.youtube.transcripts import download_subtitles, load_transcript
from french_mining.youtube.whisper_transcribe import DEFAULT_MODEL_SIZE, transcribe_with_whisper


def get_transcript(
    video_id: str,
    work_dir: str | Path,
    lang: str = "fr",
    use_whisper_fallback: bool = True,
    whisper_model_size: str = DEFAULT_MODEL_SIZE,
) -> tuple[list[TranscriptSegment], str]:
    """Get timestamped transcript segments for a video: try YouTube's own
    captions first (free, instant), and fall back to local Whisper
    transcription of the downloaded audio when no captions exist at all
    (not every watched video has them). Whisper is slower but has no API
    cost — it just needs the audio, which this downloads (or reuses if
    already present in `work_dir`, per `media.download_audio`).

    Returns `(segments, source)` where `source` is `"captions"` or
    `"whisper"`, so callers can report which path was taken.
    """
    try:
        vtt_path = download_subtitles(video_id, work_dir, lang=lang)
        return load_transcript(vtt_path), "captions"
    except RuntimeError:
        if not use_whisper_fallback:
            raise
        audio_path = download_audio(video_id, work_dir)
        return transcribe_with_whisper(audio_path, language=lang, model_size=whisper_model_size), "whisper"


def _safe_filename_part(lemma: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in lemma) or "word"


def attach_source_media(
    anki_client: AnkiConnectClient,
    scored: ScoredCandidate,
    audio_source_path: str | Path,
    work_dir: str | Path,
) -> dict[str, str]:
    """Clip source audio for one scored candidate's sentence and store it in
    Anki's media folder. Returns `{"SentenceAudio": ...}` — empty string if
    there's no timing to clip from (e.g. the sentence's text couldn't be
    matched back to the transcript).
    """
    candidate = scored.candidate
    if candidate.start_time is None or candidate.end_time is None:
        return {"SentenceAudio": ""}

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    name_part = _safe_filename_part(candidate.target_lemma)

    audio_clip_path = work_dir / f"{name_part}_sentence.mp3"
    clip_audio(audio_source_path, candidate.start_time, candidate.end_time, audio_clip_path)
    audio_filename = anki_client.store_media_file(audio_clip_path.name, str(audio_clip_path))
    return {"SentenceAudio": f"[sound:{audio_filename}]"}


def grab_source_frame(
    scored: ScoredCandidate, video_source_path: str | Path, work_dir: str | Path
) -> Path | None:
    """Extract the raw video frame at the sentence's midpoint, for
    `images.resolve_image_field` to judge (talking-head/text pre-filter,
    then a vision relevance check) — no judgment happens here.

    Returns None if there's no timing to extract from.
    """
    candidate = scored.candidate
    if candidate.start_time is None or candidate.end_time is None:
        return None

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    name_part = _safe_filename_part(candidate.target_lemma)

    frame_path = work_dir / f"{name_part}_frame.jpg"
    midpoint = (candidate.start_time + candidate.end_time) / 2
    extract_frame(video_source_path, midpoint, frame_path)
    return frame_path
