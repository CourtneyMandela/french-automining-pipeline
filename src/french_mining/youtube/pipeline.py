"""Wires the YouTube-specific pieces into the shared scoring/generation/write
pipeline: clips real source audio for the sentence and grabs a raw source
frame (§9, §10), using the timing `french_mining.candidates` already carried
through from the timestamped transcript.

Talking-head filtering and the Unsplash fallback (§10) are a separate,
later stage (build-order step 8) — this just grabs the raw frame at the
sentence's midpoint; whether that frame is any good is step 8's job.
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
    video_source_path: str | Path | None,
    work_dir: str | Path,
) -> dict[str, str]:
    """Clip source audio (and grab a source frame, if a video path is given)
    for one scored candidate's sentence, store both in Anki's media folder,
    and return `{"SentenceAudio": ..., "Image": ...}` field overrides —
    empty strings for whichever wasn't produced (e.g. no timing available).
    """
    candidate = scored.candidate
    if candidate.start_time is None or candidate.end_time is None:
        return {"SentenceAudio": "", "Image": ""}

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    name_part = _safe_filename_part(candidate.target_lemma)

    audio_clip_path = work_dir / f"{name_part}_sentence.mp3"
    clip_audio(audio_source_path, candidate.start_time, candidate.end_time, audio_clip_path)
    audio_filename = anki_client.store_media_file(audio_clip_path.name, str(audio_clip_path))
    fields = {"SentenceAudio": f"[sound:{audio_filename}]", "Image": ""}

    if video_source_path is not None:
        frame_path = work_dir / f"{name_part}_frame.jpg"
        midpoint = (candidate.start_time + candidate.end_time) / 2
        extract_frame(video_source_path, midpoint, frame_path)
        image_filename = anki_client.store_media_file(frame_path.name, str(frame_path))
        fields["Image"] = f'<img src="{image_filename}">'

    return fields
