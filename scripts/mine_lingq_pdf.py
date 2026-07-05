#!/usr/bin/env python3
"""Run the local Stage 1 filter (§6) on a weekly LingQ PDF drop -- both
single words and collocations (§7) -- then Stage 2 API scoring and
(optionally) card generation + writing, if ANTHROPIC_API_KEY is set.

Requires the French spaCy model, which this sandboxed session couldn't
download (GitHub releases are blocked here) — run this on your own machine
after `python -m spacy download fr_core_news_sm`, with Anki + AnkiConnect
running so real vocabulary state can be read.

    .venv/bin/python scripts/mine_lingq_pdf.py path/to/reading.pdf
    .venv/bin/python scripts/mine_lingq_pdf.py path/to/reading.pdf --write 20
"""
import argparse
import os

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
from french_mining.lingq.pdf_extract import extract_text
from french_mining.nlp import parse_text
from french_mining.queue_ordering import (
    get_new_backlog_note_ids,
    merge_priority_with_backlog,
    reorder_queue,
)
from french_mining.scoring import build_client, keep_and_rank, score_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf_path")
    parser.add_argument(
        "--write",
        type=int,
        default=0,
        metavar="N",
        help="Generate full card content and write the top N ranked candidates to Anki (default: 0, dry run only).",
    )
    args = parser.parse_args()

    text = extract_text(args.pdf_path)
    sentences = parse_text(text)

    anki_client = AnkiConnectClient()
    vocab_state = VocabularyState.build(
        anki_client,
        model_names=[MODEL_NAME],
        collocation_model_names=[COLLOCATION_MODEL_NAME],
    )

    word_candidates = find_i_plus_1_candidates(sentences, vocab_state, source=args.pdf_path)
    collocation_candidates = find_collocation_candidates(
        sentences, vocab_state, load_collocations(), source=args.pdf_path
    )
    # Single words and collocations share the same backlog and compete for
    # the same daily slots (§7), so they're selected/scored/ranked together.
    best = select_best_sentences(word_candidates + collocation_candidates)

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

    if args.write > 0:
        to_write = ranked[: args.write]
        word_items = [s for s in to_write if not s.candidate.is_collocation]
        collocation_items = [s for s in to_write if s.candidate.is_collocation]

        print(
            f"\nGenerating content and resolving images for {len(word_items)} word "
            f"card(s) and {len(collocation_items)} collocation card(s)..."
        )
        note_ids = []

        for fields, scored_candidate in zip(generate_card_content(anthropic_client, word_items), word_items):
            # LingQ PDFs have no video frame, so this only ever reaches the
            # Unsplash tier (or no image, for abstract words) — §10 tier 1
            # (source frame) is YouTube-only.
            fields["Image"] = resolve_image_field(
                anki_client,
                anthropic_client,
                scored_candidate,
                fields["TargetWordGloss"],
                frame_path=None,
                work_dir="lingq_work/images",
            )
            note_ids.append(add_card(anki_client, fields))

        for fields, scored_candidate in zip(
            generate_collocation_card_content(anthropic_client, collocation_items), collocation_items
        ):
            fields["Image"] = resolve_image_field(
                anki_client,
                anthropic_client,
                scored_candidate,
                fields["TargetChunkGloss"],
                frame_path=None,
                work_dir="lingq_work/images",
            )
            note_ids.append(add_collocation_card(anki_client, fields))

        print(f"Wrote note IDs: {note_ids}")

        backlog_note_ids = get_new_backlog_note_ids(
            anki_client, [MODEL_NAME, COLLOCATION_MODEL_NAME], DEFAULT_DECK_NAME
        )
        merged_order = merge_priority_with_backlog(note_ids, backlog_note_ids)
        rewritten = reorder_queue(anki_client, merged_order)
        print(
            f"Reordered {len(rewritten)} still-new card(s) in the queue "
            f"(today's {len(note_ids)} picks first, existing backlog behind them)."
        )


if __name__ == "__main__":
    main()
