"""LingQ API sync (extends §4): read on LingQ, and the words you look up
(create LingQs for) flow into the pipeline automatically — no PDF drop.

Each LingQ carries the exact sentence it was encountered in (`fragment`),
so it's a strictly better candidate source than the PDF path: the learner
has already told us which word they didn't know, and given us its context.

Per the core architecture, LingQ is only a *candidate source* — Anki's FSRS
state (§5) remains the source of truth for what's known, so a looked-up
word already carded in Anki is skipped, and a fragment is only mined if
that word is its sole unknown (genuine i+1).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import requests

from french_mining.anki.vocab_state import VocabularyState
from french_mining.candidates import (
    DEFAULT_KNOWN_THRESHOLD,
    DEFAULT_MIN_TOKENS,
    Candidate,
)
from french_mining.nlp import ParsedSentence

DEFAULT_BASE_URL = "https://www.lingq.com/api/v3"

# LingQ fragments are natural sentences; allow a little more length than the
# PDF/YouTube default before treating one as too dense to be clean i+1.
DEFAULT_MAX_TOKENS = 40


@dataclass
class LingQCard:
    pk: int
    term: str
    fragment: str
    status: int
    hint: str = ""


def _extract_hint(card: dict) -> str:
    """LingQ `hints` is a list of hint objects; take the first non-empty
    meaning text, tolerating either a `text` or `hint` key.
    """
    for hint in card.get("hints") or []:
        if isinstance(hint, dict):
            text = (hint.get("text") or hint.get("hint") or "").strip()
            if text:
                return text
    return ""


class LingQClient:
    def __init__(
        self,
        api_key: str | None = None,
        language_code: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
    ):
        self.api_key = api_key or os.environ.get("LINGQ_API_KEY")
        self.language_code = language_code or os.environ.get("LINGQ_LANGUAGE_CODE", "fr")
        self.base_url = (base_url or os.environ.get("LINGQ_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise RuntimeError(
                "LINGQ_API_KEY is not set. Get your key from https://www.lingq.com/accounts/apikey/ "
                "and put it in .env (copy .env.example)."
            )
        return {"Authorization": f"Token {self.api_key}"}

    def fetch_lingqs(self, max_pages: int = 10) -> list[LingQCard]:
        """Fetch the learner's LingQs (looked-up words/phrases), following
        pagination up to `max_pages`. Returns them in whatever order the API
        yields (typically most-recent first); the downstream Anki dedup and
        i+1 filter mean fetching more than strictly-new ones is harmless.
        """
        url = f"{self.base_url}/{self.language_code}/cards/"
        cards: list[LingQCard] = []
        pages = 0
        while url and pages < max_pages:
            response = requests.get(url, headers=self._headers(), timeout=self.timeout)
            response.raise_for_status()
            body = response.json()
            for result in body.get("results", []):
                cards.append(
                    LingQCard(
                        pk=result.get("pk"),
                        term=(result.get("term") or "").strip(),
                        fragment=(result.get("fragment") or "").strip(),
                        status=result.get("status", 0),
                        hint=_extract_hint(result),
                    )
                )
            url = body.get("next")
            pages += 1
        return cards


def _normalize(text: str) -> str:
    return text.replace("’", "'").lower()


def locate_term(tokens: list, term: str) -> tuple[int, int] | None:
    """Find the token span (inclusive start/end indices) covering `term`
    within a parsed fragment, tolerant of French tokenization (elisions,
    apostrophes) by matching on the whitespace-stripped concatenation but
    still requiring alignment to token boundaries — so "de" can't match
    inside "monde".

    Falls back to a single-token lemma match when the surface form isn't
    present verbatim (LingQ occasionally stores a base form).
    """
    if not term or not tokens:
        return None

    starts: list[int] = []
    ends: list[int] = []
    joined = ""
    for token in tokens:
        surface = _normalize(token.text)
        starts.append(len(joined))
        joined += surface
        ends.append(len(joined))

    needle = _normalize(term).replace(" ", "")
    if needle:
        search_from = 0
        while True:
            pos = joined.find(needle, search_from)
            if pos == -1:
                break
            end = pos + len(needle)
            if pos in starts and end in ends:
                return (starts.index(pos), ends.index(end))
            search_from = pos + 1

    # Single-word lemma fallback (base form stored instead of surface).
    if " " not in term.strip():
        term_lemma = _normalize(term)
        for i, token in enumerate(tokens):
            if _normalize(token.lemma) == term_lemma:
                return (i, i)
    return None


def build_lingq_candidates(
    items: list[tuple[LingQCard, ParsedSentence]],
    vocab_state: VocabularyState,
    known_threshold: float = DEFAULT_KNOWN_THRESHOLD,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> list[Candidate]:
    """Turn (LingQ card, its parsed fragment) pairs into pipeline candidates.

    Deliberately spaCy-free — the script parses fragments and hands the
    already-parsed sentences in — so this filtering logic is unit-testable
    without the French model.

    A card is kept only when: its term is locatable in the fragment, the
    term isn't already solidly known in Anki, and every *other* alpha token
    in the fragment is known (the looked-up word is the sole unknown, i.e.
    genuine i+1). Multi-word LingQs route to the collocation card type.
    """
    candidates: list[Candidate] = []
    for card, sentence in items:
        tokens = sentence.tokens
        if not (min_tokens <= len(sentence.alpha_tokens) <= max_tokens):
            continue

        span = locate_term(tokens, card.term)
        if span is None:
            continue
        start, end = span
        term_indices = set(range(start, end + 1))

        is_collocation = " " in card.term.strip()
        surface_form = " ".join(tokens[i].text for i in range(start, end + 1))

        if is_collocation:
            target_lemma = _normalize(card.term)
            already_known = vocab_state.chunk_confidence(target_lemma) >= known_threshold
            target_confidence = vocab_state.chunk_confidence(target_lemma)
            target_pos = "COLLOC"
        else:
            target_lemma = tokens[start].lemma
            already_known = vocab_state.confidence(target_lemma) >= known_threshold
            target_confidence = vocab_state.confidence(target_lemma)
            target_pos = tokens[start].pos

        if already_known:
            continue

        other_unknowns = [
            t
            for i, t in enumerate(tokens)
            if t.is_alpha and i not in term_indices and vocab_state.confidence(t.lemma) < known_threshold
        ]
        if other_unknowns:
            continue

        candidates.append(
            Candidate(
                target_lemma=target_lemma,
                target_form=surface_form,
                target_pos=target_pos,
                sentence_text=sentence.text,
                other_lemmas=[t.lemma for i, t in enumerate(tokens) if t.is_alpha and i not in term_indices],
                target_confidence=target_confidence,
                source=f"LingQ: {card.term}",
                is_collocation=is_collocation,
            )
        )
    return candidates
