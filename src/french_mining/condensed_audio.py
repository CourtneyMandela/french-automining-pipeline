"""Comprehensibility-filtered condensed audio (a second output alongside
cards): from videos the learner has already watched, keep only the spoken
spans they can follow, so the concatenated result is dense, silence-free,
highly-comprehensible listening review.

Entirely local and free — comprehensibility is a mechanical known-word
coverage computation over the same `VocabularyState` (FSRS-backed) the card
pipeline uses, so no API tokens are spent (§3 cost philosophy).

Deliberately spaCy-free: operates on `french_mining.nlp.ParsedSentence`
(which `nlp.parse_transcript` already produces with per-sentence timing), so
this selection logic is fully unit-testable without the French model.
"""
from __future__ import annotations

from french_mining.anki.vocab_state import VocabularyState
from french_mining.nlp import ParsedSentence

# A sentence is kept only if at least this fraction of its content words are
# already known. High by default: audio has no visual support, and this is
# review of already-watched material, so aim for genuinely comprehensible.
DEFAULT_MIN_COMPREHENSIBILITY = 0.9

# Merge kept sentences whose spoken spans are within this gap (seconds) into
# one span, so the output isn't a stutter of micro-cuts across natural pauses.
DEFAULT_MERGE_GAP_SECONDS = 0.4

# Target length for a shuffle-friendly clip (§ chunking): long enough for a
# coherent mini-scene, short enough that shuffling a playlist and skipping a
# dud clip actually works, unlike one long file per video.
DEFAULT_CLIP_SECONDS = 180.0  # ~3 minutes

# A chunk closes once the next span would push it past target*this multiplier
# -- bounds overshoot while still never splitting an individual span.
DEFAULT_MAX_CLIP_MULTIPLIER = 1.5

# Per-word "known" bar — the same gradient-confidence threshold the rest of
# the pipeline treats as known (§5).
KNOWN_THRESHOLD = 0.6


def sentence_comprehensibility(sentence: ParsedSentence, vocab_state: VocabularyState) -> float:
    """Fraction of a sentence's content (alpha) words that are already known.

    Returns 1.0 for a sentence with no content words (e.g. "..."), so trivial
    utterances are never dropped as "incomprehensible".
    """
    tokens = sentence.alpha_tokens
    if not tokens:
        return 1.0
    known = sum(1 for t in tokens if vocab_state.confidence(t.lemma) >= KNOWN_THRESHOLD)
    return known / len(tokens)


def video_comprehensibility(sentences: list[ParsedSentence], vocab_state: VocabularyState) -> float:
    """Overall known-word coverage across a whole video's content words —
    used to order a multi-video playlist easiest-first. Word-weighted (not a
    mean of per-sentence ratios) so long sentences count proportionally.
    """
    total = 0
    known = 0
    for sentence in sentences:
        for token in sentence.alpha_tokens:
            total += 1
            if vocab_state.confidence(token.lemma) >= KNOWN_THRESHOLD:
                known += 1
    return known / total if total else 1.0


def select_comprehensible_spans(
    sentences: list[ParsedSentence],
    vocab_state: VocabularyState,
    min_comprehensibility: float = DEFAULT_MIN_COMPREHENSIBILITY,
    min_tokens: int = 1,
    merge_gap: float = DEFAULT_MERGE_GAP_SECONDS,
) -> list[tuple[float, float]]:
    """Return the timed `(start, end)` spans to keep: sentences whose
    comprehensibility clears the bar and that carry timing, with adjacent /
    near-adjacent kept spans merged.

    Sentences without timing (`start_time`/`end_time` is None) can't be
    clipped, so they're skipped regardless of comprehensibility.
    """
    kept: list[tuple[float, float]] = []
    for sentence in sentences:
        if sentence.start_time is None or sentence.end_time is None:
            continue
        if len(sentence.alpha_tokens) < min_tokens:
            continue
        if sentence_comprehensibility(sentence, vocab_state) < min_comprehensibility:
            continue
        kept.append((sentence.start_time, sentence.end_time))

    return _merge_spans(kept, merge_gap)


def _merge_spans(spans: list[tuple[float, float]], merge_gap: float) -> list[tuple[float, float]]:
    """Coalesce spans (in time order) that touch or sit within `merge_gap`."""
    if not spans:
        return []
    ordered = sorted(spans)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start - last_end <= merge_gap:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def total_duration(spans: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in spans)


def chunk_spans(
    spans: list[tuple[float, float]],
    target_duration: float = DEFAULT_CLIP_SECONDS,
    max_duration: float | None = None,
) -> list[list[tuple[float, float]]]:
    """Greedily group already-merged spans (in time order) into shuffle-
    friendly, clip-sized chunks — a playlist of a video's own few-minute
    clips rather than one long file, so a player's shuffle is actually
    meaningful and a dud clip is easy to skip past.

    Never splits an individual span (each is already one continuous,
    smoothed piece of speech from `select_comprehensible_spans`'s merge
    step — cutting into one would just chop a sentence group awkwardly). A
    chunk closes as soon as it reaches `target_duration` (typical clips land
    just at/over the target, not drifting toward `max_duration`); separately,
    if the *next* span alone would push a still-under-target chunk past
    `max_duration` (default `target_duration * DEFAULT_MAX_CLIP_MULTIPLIER`)
    it closes early instead, so one big span never gets compounded onto a
    partial chunk. Either way, a single span longer than `max_duration` still
    ends up alone in its own (over-length) chunk rather than being dropped or
    split. The final chunk absorbs whatever's left, which may be shorter than
    the target -- expected for a video's leftover tail.
    """
    if max_duration is None:
        max_duration = target_duration * DEFAULT_MAX_CLIP_MULTIPLIER

    chunks: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    current_duration = 0.0

    for span in sorted(spans):
        span_duration = span[1] - span[0]
        if current and (
            current_duration >= target_duration
            or current_duration + span_duration > max_duration
        ):
            chunks.append(current)
            current = []
            current_duration = 0.0
        current.append(span)
        current_duration += span_duration

    if current:
        chunks.append(current)
    return chunks
