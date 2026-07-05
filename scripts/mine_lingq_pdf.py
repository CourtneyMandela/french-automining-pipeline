#!/usr/bin/env python3
"""Run the local (free) Stage 1 filter (§6) on a weekly LingQ PDF drop.

Requires the French spaCy model, which this sandboxed session couldn't
download (GitHub releases are blocked here) — run this on your own machine
after `python -m spacy download fr_core_news_sm`, with Anki + AnkiConnect
running so real vocabulary state can be read.

    .venv/bin/python scripts/mine_lingq_pdf.py path/to/reading.pdf
"""
import sys

from french_mining.anki.connect import AnkiConnectClient
from french_mining.anki.note_type import MODEL_NAME
from french_mining.anki.vocab_state import VocabularyState
from french_mining.lingq.candidates import find_i_plus_1_candidates, select_best_sentences
from french_mining.lingq.pdf_extract import extract_text
from french_mining.nlp import parse_text


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: mine_lingq_pdf.py path/to/reading.pdf", file=sys.stderr)
        sys.exit(1)

    text = extract_text(sys.argv[1])
    sentences = parse_text(text)

    client = AnkiConnectClient()
    vocab_state = VocabularyState.build(client, model_names=[MODEL_NAME])

    candidates = find_i_plus_1_candidates(sentences, vocab_state, source=sys.argv[1])
    best = select_best_sentences(candidates)

    print(f"{len(sentences)} sentences -> {len(candidates)} i+1 candidates -> {len(best)} after best-sentence selection\n")
    for c in best:
        print(f"[{c.target_lemma}] {c.sentence_text}")


if __name__ == "__main__":
    main()
