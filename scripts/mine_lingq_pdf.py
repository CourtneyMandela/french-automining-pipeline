#!/usr/bin/env python3
"""Run the local Stage 1 filter (§6) on a weekly LingQ PDF drop, then Stage 2
API scoring and (optionally) card generation + writing, if ANTHROPIC_API_KEY
is set.

Requires the French spaCy model, which this sandboxed session couldn't
download (GitHub releases are blocked here) — run this on your own machine
after `python -m spacy download fr_core_news_sm`, with Anki + AnkiConnect
running so real vocabulary state can be read.

    .venv/bin/python scripts/mine_lingq_pdf.py path/to/reading.pdf
    .venv/bin/python scripts/mine_lingq_pdf.py path/to/reading.pdf --write 20
"""
import argparse
import os

from french_mining.anki.connect import AnkiConnectClient
from french_mining.anki.note_type import DEFAULT_DECK_NAME, MODEL_NAME
from french_mining.anki.vocab_state import VocabularyState
from french_mining.frequency import FrequencyList
from french_mining.generation import generate_and_write_cards
from french_mining.lingq.candidates import find_i_plus_1_candidates, select_best_sentences
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
    vocab_state = VocabularyState.build(anki_client, model_names=[MODEL_NAME])

    candidates = find_i_plus_1_candidates(sentences, vocab_state, source=args.pdf_path)
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

    if args.write > 0:
        to_write = ranked[: args.write]
        print(f"\nGenerating and writing {len(to_write)} card(s) to Anki...")
        note_ids = generate_and_write_cards(anki_client, anthropic_client, to_write)
        print(f"Wrote note IDs: {note_ids}")

        backlog_note_ids = get_new_backlog_note_ids(anki_client, MODEL_NAME, DEFAULT_DECK_NAME)
        merged_order = merge_priority_with_backlog(note_ids, backlog_note_ids)
        rewritten = reorder_queue(anki_client, merged_order)
        print(
            f"Reordered {len(rewritten)} still-new card(s) in the queue "
            f"(today's {len(note_ids)} picks first, existing backlog behind them)."
        )


if __name__ == "__main__":
    main()
