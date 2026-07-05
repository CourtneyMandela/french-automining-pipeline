"""Local, free collocation candidate extraction (§7): the chunk-level
mirror of the single-word i+1 filter in `french_mining.candidates`.

Fluency is largely chunk-level, not word-level. Collocations are multi-word
units whose meaning isn't fully predictable from their parts (verb-
preposition pairings, fixed expressions) — a different mining target than
single unknown words, so it needs its own matcher, but produces the same
generic `Candidate` type (`is_collocation=True`) so collocations and single
words share the scoring/generation/queue-ordering pipeline and compete for
the same backlog slots, per §7.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from french_mining.anki.vocab_state import VocabularyState
from french_mining.candidates import (
    DEFAULT_KNOWN_THRESHOLD,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MIN_TOKENS,
    Candidate,
)
from french_mining.nlp import ParsedSentence

DEFAULT_COLLOCATIONS_CSV = resources.files("french_mining.data") / "french_collocations.csv"

# How many extra (non-matching) tokens are allowed between two consecutive
# matched collocation components. French inserts reflexive pronouns,
# negation, adverbs, and clitic objects between a verb and its preposition
# (e.g. "il ne s'en est jamais aperçu de..."), so requiring a contiguous
# match would miss most real occurrences.
MAX_COMPONENT_GAP = 4


@dataclass
class CollocationEntry:
    chunk: str  # canonical display form, e.g. "s'apercevoir de"
    lemmas: list[str]  # ordered component lemmas to match, e.g. ["se", "apercevoir", "de"]
    gloss: str


def load_collocations(path: str | Path | None = None) -> list[CollocationEntry]:
    source = Path(path) if path else DEFAULT_COLLOCATIONS_CSV
    entries = []
    with open(source, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            entries.append(
                CollocationEntry(
                    chunk=row["chunk"].strip().lower(),
                    lemmas=row["lemmas"].strip().lower().split(),
                    gloss=row["gloss"].strip(),
                )
            )
    return entries


def _find_component_match(
    token_lemmas: list[str], target_lemmas: list[str], max_gap: int
) -> tuple[int, int] | None:
    """Find the first in-order, gap-limited occurrence of `target_lemmas`
    within `token_lemmas`. Returns the (start, end) index span (inclusive),
    or None if no match exists within the gap tolerance.
    """
    for start in range(len(token_lemmas)):
        if token_lemmas[start] != target_lemmas[0]:
            continue

        pos = start
        matched_all = True
        for target in target_lemmas[1:]:
            next_pos = None
            for i in range(pos + 1, min(pos + 1 + max_gap + 1, len(token_lemmas))):
                if token_lemmas[i] == target:
                    next_pos = i
                    break
            if next_pos is None:
                matched_all = False
                break
            pos = next_pos

        if matched_all:
            return (start, pos)
    return None


def find_collocation_candidates(
    sentences: list[ParsedSentence],
    vocab_state: VocabularyState,
    collocations: list[CollocationEntry],
    known_threshold: float = DEFAULT_KNOWN_THRESHOLD,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_component_gap: int = MAX_COMPONENT_GAP,
    source: str = "",
) -> list[Candidate]:
    """Keep sentences containing a not-yet-known collocation, where every
    other word in the sentence is already known — the chunk-level version
    of "exactly one unknown unit, no fragile extras."
    """
    candidates = []
    for sentence in sentences:
        if not (min_tokens <= len(sentence.alpha_tokens) <= max_tokens):
            continue

        token_lemmas = [t.lemma for t in sentence.tokens]
        for entry in collocations:
            if vocab_state.chunk_confidence(entry.chunk) >= known_threshold:
                continue  # already known — not i+1 for this chunk

            match = _find_component_match(token_lemmas, entry.lemmas, max_component_gap)
            if match is None:
                continue

            start, end = match
            matched_indices = set(range(start, end + 1))

            other_unknowns = [
                t
                for i, t in enumerate(sentence.tokens)
                if t.is_alpha and i not in matched_indices and vocab_state.confidence(t.lemma) < known_threshold
            ]
            if other_unknowns:
                continue

            surface_form = " ".join(sentence.tokens[i].text for i in range(start, end + 1))
            other_lemmas = [
                sentence.tokens[i].lemma
                for i in range(len(sentence.tokens))
                if i not in matched_indices and sentence.tokens[i].is_alpha
            ]
            candidates.append(
                Candidate(
                    target_lemma=entry.chunk,
                    target_form=surface_form,
                    target_pos="COLLOC",
                    sentence_text=sentence.text,
                    other_lemmas=other_lemmas,
                    target_confidence=vocab_state.chunk_confidence(entry.chunk),
                    source=source,
                    start_time=sentence.start_time,
                    end_time=sentence.end_time,
                    is_collocation=True,
                )
            )
    return candidates
