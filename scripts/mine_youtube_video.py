#!/usr/bin/env python3
"""Mine one YouTube video for i+1 sentence-mining candidates -- both single
words and collocations (§7) -- transcript -> spaCy -> vocab state -> i+1
filter -> (optionally) Stage 2 scoring, card generation, source audio/frame
attachment, writing, and queue reordering.

Requires (on your own machine, not this sandboxed session — see README):
  - the French spaCy model (GitHub releases are blocked here)
  - network access to YouTube (also blocked in this sandboxed session)
  - ffmpeg installed (for audio clipping / frame extraction)
  - YOUTUBE_API_KEY (video metadata) and ANTHROPIC_API_KEY (Stage 2+) in .env

    .venv/bin/python scripts/mine_youtube_video.py VIDEO_ID
    .venv/bin/python scripts/mine_youtube_video.py VIDEO_ID --write 20 --with-images
"""
import argparse
import os
from pathlib import Path

from french_mining.anki.collocation_note_type import MODEL_NAME as COLLOCATION_MODEL_NAME
from french_mining.anki.collocation_note_type import add_card as add_collocation_card
from french_mining.anki.connect import AnkiConnectClient
from french_mining.anki.note_type import DEFAULT_DECK_NAME, MODEL_NAME, add_card
from french_mining.anki.vocab_state import VocabularyState
from french_mining.candidates import find_i_plus_1_candidates, select_best_sentences
from french_mining.collocation_generation import generate_collocation_card_content
from french_mining.collocations import find_collocation_candidates, load_collocations
from french_mining.frequency import FrequencyList
from french_mining.generation import generate_card_content
from french_mining.images import resolve_image_field
from french_mining.nlp import parse_transcript
from french_mining.queue_ordering import (
    get_new_backlog_note_ids,
    merge_priority_with_backlog,
    reorder_queue,
)
from french_mining.scoring import build_client, keep_and_rank, score_candidates
from french_mining.youtube.media import download_audio, download_video
from french_mining.youtube.metadata import fetch_video_metadata, format_source
from french_mining.youtube.pipeline import attach_source_media, get_transcript, grab_source_frame
from french_mining.youtube.whisper_transcribe import DEFAULT_MODEL_SIZE as DEFAULT_WHISPER_MODEL_SIZE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_id")
    parser.add_argument("--lang", default="fr")
    parser.add_argument(
        "--write",
        type=int,
        default=0,
        metavar="N",
        help="Generate content, attach source media, and write the top N ranked candidates to Anki.",
    )
    parser.add_argument(
        "--with-images",
        action="store_true",
        help="Also download the video (not just audio) and attach a source frame per card.",
    )
    parser.add_argument(
        "--no-whisper-fallback",
        action="store_true",
        help="Skip videos with no captions instead of transcribing them locally with Whisper.",
    )
    parser.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL_SIZE)
    parser.add_argument("--work-dir", default="youtube_work")
    args = parser.parse_args()

    work_dir = Path(args.work_dir) / args.video_id
    work_dir.mkdir(parents=True, exist_ok=True)

    metadata = fetch_video_metadata(args.video_id)
    print(f"Video: {metadata.title} ({metadata.channel_title})\n")

    # Falls back to local Whisper transcription (no API cost, just slower)
    # when the video has no captions at all.
    segments, transcript_source = get_transcript(
        args.video_id,
        work_dir,
        lang=args.lang,
        use_whisper_fallback=not args.no_whisper_fallback,
        whisper_model_size=args.whisper_model,
    )
    print(f"Transcript source: {transcript_source}")
    sentences = parse_transcript(segments)

    anki_client = AnkiConnectClient()
    vocab_state = VocabularyState.build(
        anki_client,
        model_names=[MODEL_NAME],
        collocation_model_names=[COLLOCATION_MODEL_NAME],
    )

    word_candidates = find_i_plus_1_candidates(sentences, vocab_state)
    collocation_candidates = find_collocation_candidates(sentences, vocab_state, load_collocations())
    all_candidates = word_candidates + collocation_candidates
    for candidate in all_candidates:
        candidate.video_id = args.video_id
        candidate.source = format_source(metadata, candidate.start_time)
    # Single words and collocations share the same backlog and compete for
    # the same daily slots (§7), so they're selected/scored/ranked together.
    best = select_best_sentences(all_candidates)

    print(
        f"{len(sentences)} sentences -> {len(word_candidates)} word + "
        f"{len(collocation_candidates)} collocation i+1 candidates "
        f"-> {len(best)} after local best-sentence selection\n"
    )

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set — skipping Stage 2 API scoring. Stage 1 survivors:\n")
        for c in best:
            kind = "collocation" if c.is_collocation else "word"
            print(f"[{kind}: {c.target_lemma}] {c.sentence_text}")
        return

    active_queue_lemmas = [
        lemma for lemma, knowledge in vocab_state.known_words.items() if knowledge.in_learning
    ]
    anthropic_client = build_client()
    scored = score_candidates(
        anthropic_client,
        best,
        frequency_list=vocab_state.frequency_list or FrequencyList.load(),
        active_queue_lemmas=active_queue_lemmas,
    )
    ranked = keep_and_rank(scored)

    print(f"Stage 2: {len(best)} scored -> {len(ranked)} kept, ranked by priority:\n")
    for s in ranked:
        interference = f" [interferes with {s.interference_risk}]" if s.interference_risk else ""
        kind = "collocation" if s.candidate.is_collocation else "word"
        print(
            f"{s.priority_score:6.2f}  [{kind}: {s.candidate.target_lemma}]{interference} "
            f"{s.candidate.sentence_text}\n         {s.reasoning}"
        )

    if args.write <= 0:
        return

    to_write = ranked[: args.write]
    word_items = [s for s in to_write if not s.candidate.is_collocation]
    collocation_items = [s for s in to_write if s.candidate.is_collocation]

    print(f"\nDownloading source audio{' + video' if args.with_images else ''} for {args.video_id}...")
    audio_path = download_audio(args.video_id, work_dir)
    video_path = download_video(args.video_id, work_dir) if args.with_images else None

    print(
        f"Generating content, source audio, and images for {len(word_items)} word "
        f"card(s) and {len(collocation_items)} collocation card(s)..."
    )

    def attach_media_and_image(fields: dict, scored_candidate, gloss_field: str) -> dict:
        media_fields = attach_source_media(anki_client, scored_candidate, audio_path, work_dir / "clips")
        fields.update({k: v for k, v in media_fields.items() if v})

        frame_path = (
            grab_source_frame(scored_candidate, video_path, work_dir / "frames")
            if video_path is not None
            else None
        )
        fields["Image"] = resolve_image_field(
            anki_client,
            anthropic_client,
            scored_candidate,
            fields[gloss_field],
            frame_path,
            work_dir / "images",
        )
        return fields

    note_ids = []
    for fields, scored_candidate in zip(generate_card_content(anthropic_client, word_items), word_items):
        fields = attach_media_and_image(fields, scored_candidate, "TargetWordGloss")
        note_ids.append(add_card(anki_client, fields))

    for fields, scored_candidate in zip(
        generate_collocation_card_content(anthropic_client, collocation_items), collocation_items
    ):
        fields = attach_media_and_image(fields, scored_candidate, "TargetChunkGloss")
        note_ids.append(add_collocation_card(anki_client, fields))

    print(f"Wrote note IDs: {note_ids}")

    backlog_note_ids = get_new_backlog_note_ids(
        anki_client, [MODEL_NAME, COLLOCATION_MODEL_NAME], DEFAULT_DECK_NAME
    )
    merged_order = merge_priority_with_backlog(note_ids, backlog_note_ids)
    rewritten = reorder_queue(anki_client, merged_order)
    print(f"Reordered {len(rewritten)} still-new card(s) in the queue.")


if __name__ == "__main__":
    main()
