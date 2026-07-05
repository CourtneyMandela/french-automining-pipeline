"""Wires the YouTube-specific pieces into the shared scoring/generation/write
pipeline: clips real source audio for the sentence (§9) and grabs a raw
source frame at the sentence's midpoint for `french_mining.images` to judge
(§10) — this module doesn't decide whether a frame is any good, it just
produces the candidate frame.
"""
from __future__ import annotations

from pathlib import Path

from french_mining.anki.connect import AnkiConnectClient
from french_mining.scoring import ScoredCandidate
from french_mining.youtube.media import clip_audio, extract_frame


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
