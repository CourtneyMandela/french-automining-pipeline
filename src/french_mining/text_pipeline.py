"""Shared write path for text-only sources (LingQ PDFs and the LingQ API):
both take ranked candidates with no source audio/video, generate card
content, resolve an image (Unsplash tier only — there's no video frame for
text sources), write to Anki, and return the new note IDs.

Extracted so the PDF and LingQ-API scripts can't drift apart. The YouTube
script keeps its own write path since it additionally clips source audio
and grabs source frames per card.
"""
from __future__ import annotations

import anthropic

from french_mining.anki.collocation_note_type import add_card as add_collocation_card
from french_mining.anki.connect import AnkiConnectClient
from french_mining.anki.note_type import add_card
from french_mining.collocation_generation import generate_collocation_card_content
from french_mining.generation import generate_card_content
from french_mining.images import resolve_image_field
from french_mining.scoring import ScoredCandidate
from french_mining.tts import resolve_tts_fields


def _safe_filename_part(lemma: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in lemma) or "word"


def write_ranked_candidates(
    anki_client: AnkiConnectClient,
    anthropic_client: anthropic.Anthropic,
    ranked_to_write: list[ScoredCandidate],
    image_work_dir: str,
    audio_work_dir: str = "tts_audio",
) -> list[int]:
    """Generate content, resolve images (Unsplash-only) and word/second-
    example audio (ElevenLabs TTS, opt-in via `ELEVENLABS_API_KEY`), and
    write each ranked candidate — single words and collocations to their
    respective note types. Returns new note IDs in input order.

    `SentenceAudio` stays blank for text sources (LingQ PDFs/API) — there's
    no source video to clip it from, unlike the YouTube pipeline.
    """
    word_items = [s for s in ranked_to_write if not s.candidate.is_collocation]
    collocation_items = [s for s in ranked_to_write if s.candidate.is_collocation]

    note_ids: list[int] = []

    for fields, scored in zip(generate_card_content(anthropic_client, word_items), word_items):
        fields["Image"] = resolve_image_field(
            anki_client, anthropic_client, scored, fields["TargetWordGloss"], frame_path=None, work_dir=image_work_dir
        )
        fields.update(
            resolve_tts_fields(
                anki_client,
                fields["TargetWordForm"],
                fields["SecondExample"],
                audio_work_dir,
                _safe_filename_part(scored.candidate.target_lemma),
                target_field_name="WordAudio",
            )
        )
        note_ids.append(add_card(anki_client, fields))

    for fields, scored in zip(
        generate_collocation_card_content(anthropic_client, collocation_items), collocation_items
    ):
        fields["Image"] = resolve_image_field(
            anki_client, anthropic_client, scored, fields["TargetChunkGloss"], frame_path=None, work_dir=image_work_dir
        )
        fields.update(
            resolve_tts_fields(
                anki_client,
                fields["TargetChunkForm"],
                fields["SecondExample"],
                audio_work_dir,
                _safe_filename_part(scored.candidate.target_lemma),
                target_field_name="ChunkAudio",
            )
        )
        note_ids.append(add_collocation_card(anki_client, fields))

    return note_ids
