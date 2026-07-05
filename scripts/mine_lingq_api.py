#!/usr/bin/env python3
"""Sync the words you looked up on LingQ straight into the pipeline — no PDF
drop. Pulls your LingQs via the LingQ API, mines the ones that are genuine
i+1 given your Anki state, and (with --write) generates + writes + reorders.

This is the "read on LingQ, cards appear" entry point. Run it on the machine
where Anki is open, on a schedule (see README for cron/launchd/Task
Scheduler) for a fully hands-off weekly sync.

Requires (see README): the French spaCy model, ANTHROPIC_API_KEY, and
LINGQ_API_KEY (+ optional LINGQ_LANGUAGE_CODE, default "fr") in .env.

    .venv/bin/python scripts/mine_lingq_api.py
    .venv/bin/python scripts/mine_lingq_api.py --write 20
"""
import argparse
import os

from french_mining.anki.collocation_note_type import MODEL_NAME as COLLOCATION_MODEL_NAME
from french_mining.anki.connect import AnkiConnectClient
from french_mining.anki.note_type import DEFAULT_DECK_NAME, MODEL_NAME
from french_mining.anki.vocab_state import VocabularyState
from french_mining.candidates import select_best_sentences
from french_mining.frequency import FrequencyList
from french_mining.lingq.api import LingQClient, build_lingq_candidates, locate_term
from french_mining.nlp import parse_text
from french_mining.queue_ordering import (
    get_new_backlog_note_ids,
    merge_priority_with_backlog,
    reorder_queue,
)
from french_mining.scoring import build_client, keep_and_rank, score_candidates
from french_mining.text_pipeline import write_ranked_candidates


def _parse_fragment(fragment: str, term: str):
    """Parse a LingQ fragment and return the single sentence containing the
    term (fragments are usually one sentence, but can spill into two)."""
    sentences = parse_text(fragment)
    for sentence in sentences:
        if locate_term(sentence.tokens, term) is not None:
            return sentence
    return sentences[0] if sentences else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        type=int,
        default=0,
        metavar="N",
        help="Generate and write the top N ranked candidates to Anki (default: 0, dry run only).",
    )
    parser.add_argument("--max-pages", type=int, default=10, help="Max LingQ API pages to pull.")
    args = parser.parse_args()

    lingq_client = LingQClient()
    cards = lingq_client.fetch_lingqs(max_pages=args.max_pages)
    print(f"Pulled {len(cards)} LingQ(s) from LingQ ({lingq_client.language_code}).\n")

    items = []
    for card in cards:
        if not card.term or not card.fragment:
            continue
        sentence = _parse_fragment(card.fragment, card.term)
        if sentence is not None:
            items.append((card, sentence))

    anki_client = AnkiConnectClient()
    vocab_state = VocabularyState.build(
        anki_client,
        model_names=[MODEL_NAME],
        collocation_model_names=[COLLOCATION_MODEL_NAME],
    )

    candidates = build_lingq_candidates(items, vocab_state)
    best = select_best_sentences(candidates)

    print(
        f"{len(items)} parseable LingQ(s) -> {len(candidates)} i+1 candidate(s) "
        f"(after Anki dedup) -> {len(best)} after best-sentence selection\n"
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
        print(f"\nGenerating content and resolving images for {len(to_write)} card(s)...")
        note_ids = write_ranked_candidates(
            anki_client, anthropic_client, to_write, image_work_dir="lingq_work/images"
        )
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
