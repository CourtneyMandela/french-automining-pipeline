#!/usr/bin/env python3
"""Mine one YouTube video for i+1 sentence-mining candidates: transcript ->
spaCy -> vocab state -> i+1 filter -> (optionally) Stage 2 scoring, card
generation, source audio/frame attachment, writing, and queue reordering.

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

from french_mining.anki.connect import AnkiConnectClient
from french_mining.anki.note_type import DEFAULT_DECK_NAME, MODEL_NAME, add_card
from french_mining.anki.vocab_state import VocabularyState
from french_mining.candidates import find_i_plus_1_candidates, select_best_sentences
from french_mining.frequency import FrequencyList
from french_mining.generation import generate_card_content
from french_mining.nlp import parse_transcript
from french_mining.queue_ordering import (
    get_new_backlog_note_ids,
    merge_priority_with_backlog,
    reorder_queue,
)
from french_mining.scoring import build_client, keep_and_rank, score_candidates
from french_mining.youtube.media import download_audio, download_video
from french_mining.youtube.metadata import fetch_video_metadata, format_source
from french_mining.youtube.pipeline import attach_source_media
from french_mining.youtube.transcripts import download_subtitles, load_transcript


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
    parser.add_argument("--work-dir", default="youtube_work")
    args = parser.parse_args()

    work_dir = Path(args.work_dir) / args.video_id
    work_dir.mkdir(parents=True, exist_ok=True)

    metadata = fetch_video_metadata(args.video_id)
    print(f"Video: {metadata.title} ({metadata.channel_title})\n")

    vtt_path = download_subtitles(args.video_id, work_dir, lang=args.lang)
    segments = load_transcript(vtt_path)
    sentences = parse_transcript(segments)

    anki_client = AnkiConnectClient()
    vocab_state = VocabularyState.build(anki_client, model_names=[MODEL_NAME])

    candidates = find_i_plus_1_candidates(sentences, vocab_state)
    for candidate in candidates:
        candidate.video_id = args.video_id
        candidate.source = format_source(metadata, candidate.start_time)
    best = select_best_sentences(candidates)

    print(
        f"{len(sentences)} sentences -> {len(candidates)} i+1 candidates "
        f"-> {len(best)} after local best-sentence selection\n"
    )

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set — skipping Stage 2 API scoring. Stage 1 survivors:\n")
        for c in best:
            print(f"[{c.target_lemma}] {c.sentence_text}")
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
        print(
            f"{s.priority_score:6.2f}  [{s.candidate.target_lemma}]{interference} "
            f"{s.candidate.sentence_text}\n         {s.reasoning}"
        )

    if args.write <= 0:
        return

    to_write = ranked[: args.write]
    print(f"\nDownloading source audio{' + video' if args.with_images else ''} for {args.video_id}...")
    audio_path = download_audio(args.video_id, work_dir)
    video_path = download_video(args.video_id, work_dir) if args.with_images else None

    print(f"Generating content and attaching source media for {len(to_write)} card(s)...")
    field_sets = generate_card_content(anthropic_client, to_write)
    note_ids = []
    for fields, scored_candidate in zip(field_sets, to_write):
        media_fields = attach_source_media(
            anki_client, scored_candidate, audio_path, video_path, work_dir / "clips"
        )
        fields.update({k: v for k, v in media_fields.items() if v})
        note_ids.append(add_card(anki_client, fields))

    print(f"Wrote note IDs: {note_ids}")

    backlog_note_ids = get_new_backlog_note_ids(anki_client, MODEL_NAME, DEFAULT_DECK_NAME)
    merged_order = merge_priority_with_backlog(note_ids, backlog_note_ids)
    rewritten = reorder_queue(anki_client, merged_order)
    print(f"Reordered {len(rewritten)} still-new card(s) in the queue.")


if __name__ == "__main__":
    main()
