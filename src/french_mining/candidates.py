"""Stage 1 local filtering (§6): free, mechanical i+1 pre-filter + best-sentence
selection. This is what removes ~90% of candidates before any API tokens are
spent — only genuine i+1 candidates should ever reach the API scoring stage.

Deliberately spaCy-free: operates on `french_mining.nlp.ParsedSentence`, so
this filtering logic is fully unit-testable without the French spaCy model
installed.
"""
from __future__ import annotations

from dataclasses import dataclass

from french_mining.anki.vocab_state import VocabularyState
from french_mining.nlp import ParsedSentence

# A word counts as "known" for i+1 purposes once its gradient confidence
# (§5: FSRS stability/retrievability, frequency floor) clears this bar.
# Below it, treat the word as unknown/fragile — this is what makes
# "one fragile word plus one true unknown" correctly read as i+2, not i+1.
DEFAULT_KNOWN_THRESHOLD = 0.6

# Sentences shorter than this rarely give enough context for transparency;
# longer ones dilute the i+1 signal and cost more to score downstream.
DEFAULT_MIN_TOKENS = 4
DEFAULT_MAX_TOKENS = 30

# Local best-sentence heuristic: prefer sentence lengths near this count.
# Fine-grained quality judgment (context transparency, recency, etc.) is
# Claude's job at Stage 2 (§6) — this is just a cheap free pre-selection so
# we don't send every i+1 hit for the same word to the API.
IDEAL_TOKEN_COUNT = 10


@dataclass
class Candidate:
    target_lemma: str
    target_form: str
    target_pos: str
    sentence_text: str
    other_lemmas: list[str]
    target_confidence: float
    source: str = ""
    days_since_encountered: float | None = None
    # Populated only for timestamped transcript sources (YouTube); used to
    # clip source audio / grab a source frame at the right moment.
    start_time: float | None = None
    end_time: float | None = None
    # Which video this came from, so downstream media steps know which
    # downloaded audio/video file to clip from when a run covers several.
    video_id: str | None = None


def find_i_plus_1_candidates(
    sentences: list[ParsedSentence],
    vocab_state: VocabularyState,
    known_threshold: float = DEFAULT_KNOWN_THRESHOLD,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    source: str = "",
) -> list[Candidate]:
    """Keep only sentences with exactly one unknown-or-fragile word."""
    candidates = []
    for sentence in sentences:
        tokens = sentence.alpha_tokens
        if not (min_tokens <= len(tokens) <= max_tokens):
            continue

        unknowns = [t for t in tokens if vocab_state.confidence(t.lemma) < known_threshold]
        if len(unknowns) != 1:
            continue

        target = unknowns[0]
        candidates.append(
            Candidate(
                target_lemma=target.lemma,
                target_form=target.text,
                target_pos=target.pos,
                sentence_text=sentence.text,
                other_lemmas=[t.lemma for t in tokens if t is not target],
                target_confidence=vocab_state.confidence(target.lemma),
                source=source,
                start_time=sentence.start_time,
                end_time=sentence.end_time,
            )
        )
    return candidates


def _length_score(candidate: Candidate) -> float:
    token_count = len(candidate.other_lemmas) + 1
    return -abs(token_count - IDEAL_TOKEN_COUNT)


def select_best_sentences(candidates: list[Candidate], max_per_lemma: int = 3) -> list[Candidate]:
    """Cap how many sentences per target word survive to reach the API.

    Multiple sentences can be true i+1 for the same word; only the closest-
    to-ideal-length few are worth paying to score. Order among ties/ranking
    is otherwise stable (first-seen order preserved).
    """
    by_lemma: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        by_lemma.setdefault(candidate.target_lemma, []).append(candidate)

    selected: list[Candidate] = []
    for group in by_lemma.values():
        ranked = sorted(group, key=_length_score, reverse=True)
        selected.extend(ranked[:max_per_lemma])
    return selected
