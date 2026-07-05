from french_mining.anki.vocab_state import VocabularyState, WordKnowledge
from french_mining.collocations import (
    CollocationEntry,
    _find_component_match,
    find_collocation_candidates,
    load_collocations,
)
from french_mining.frequency import FrequencyList
from french_mining.nlp import ParsedSentence, Token


def make_vocab_state(known_words: dict[str, float], known_chunks: dict[str, float] | None = None) -> VocabularyState:
    def to_knowledge(confidences):
        return {
            lemma: WordKnowledge(
                lemma=lemma,
                confidence=c,
                stability_days=None,
                retrievability=None,
                lapses=0,
                in_learning=False,
                card_id=0,
            )
            for lemma, c in confidences.items()
        }

    return VocabularyState(
        known_words=to_knowledge(known_words),
        frequency_list=FrequencyList(ranks={}),
        known_chunks=to_knowledge(known_chunks or {}),
    )


def sentence(text: str, words: list[tuple[str, str]]) -> ParsedSentence:
    """words: list of (surface, lemma) tuples, all alpha tokens."""
    tokens = [Token(text=w, lemma=lemma, pos="X", is_alpha=True) for w, lemma in words]
    return ParsedSentence(text=text, tokens=tokens)


# -- load_collocations --------------------------------------------------------


def test_load_collocations_parses_default_seed_list():
    entries = load_collocations()
    assert len(entries) > 10
    apercevoir = next(e for e in entries if e.chunk == "s'apercevoir de")
    assert apercevoir.lemmas == ["se", "apercevoir", "de"]
    assert "realize" in apercevoir.gloss.lower()


# -- _find_component_match -----------------------------------------------------


def test_find_component_match_contiguous():
    lemmas = ["il", "profiter", "de", "le", "temps"]
    assert _find_component_match(lemmas, ["profiter", "de"], max_gap=4) == (1, 2)


def test_find_component_match_with_gap_within_tolerance():
    # "il ne s' en est jamais aperçu de" -> se ... apercevoir ... de, with gaps
    lemmas = ["il", "ne", "se", "en", "être", "jamais", "apercevoir", "de", "cela"]
    match = _find_component_match(lemmas, ["se", "apercevoir", "de"], max_gap=4)
    assert match == (2, 7)


def test_find_component_match_returns_none_when_gap_exceeds_tolerance():
    lemmas = ["se", "a", "b", "c", "d", "e", "f", "apercevoir", "de"]
    assert _find_component_match(lemmas, ["se", "apercevoir", "de"], max_gap=2) is None


def test_find_component_match_returns_none_when_absent():
    lemmas = ["il", "voir", "le", "chat"]
    assert _find_component_match(lemmas, ["profiter", "de"], max_gap=4) is None


# -- find_collocation_candidates ----------------------------------------------


COLLOCATIONS = [CollocationEntry(chunk="profiter de", lemmas=["profiter", "de"], gloss="to take advantage of")]


def test_finds_collocation_when_all_other_words_are_known():
    vocab = make_vocab_state({"il": 1.0, "le": 1.0, "temps": 1.0}, known_chunks={})
    sentences = [sentence("Il profite de le temps.", [("Il", "il"), ("profite", "profiter"), ("de", "de"), ("le", "le"), ("temps", "temps")])]

    candidates = find_collocation_candidates(sentences, vocab, COLLOCATIONS, min_tokens=1)

    assert len(candidates) == 1
    c = candidates[0]
    assert c.target_lemma == "profiter de"
    assert c.is_collocation is True
    assert c.target_form == "profite de"
    assert c.other_lemmas == ["il", "le", "temps"]


def test_rejects_when_another_word_in_sentence_is_unknown():
    vocab = make_vocab_state({"il": 1.0, "le": 1.0}, known_chunks={})  # "temps" unknown
    sentences = [sentence("Il profite de le temps.", [("Il", "il"), ("profite", "profiter"), ("de", "de"), ("le", "le"), ("temps", "temps")])]

    candidates = find_collocation_candidates(sentences, vocab, COLLOCATIONS, min_tokens=1)

    assert candidates == []


def test_skips_collocation_already_known():
    vocab = make_vocab_state({"il": 1.0, "le": 1.0, "temps": 1.0}, known_chunks={"profiter de": 1.0})
    sentences = [sentence("Il profite de le temps.", [("Il", "il"), ("profite", "profiter"), ("de", "de"), ("le", "le"), ("temps", "temps")])]

    candidates = find_collocation_candidates(sentences, vocab, COLLOCATIONS, min_tokens=1)

    assert candidates == []


def test_no_match_when_collocation_absent_from_sentence():
    vocab = make_vocab_state({"il": 1.0, "voir": 1.0, "chat": 1.0}, known_chunks={})
    sentences = [sentence("Il voit le chat.", [("Il", "il"), ("voit", "voir"), ("le", "le"), ("chat", "chat")])]

    candidates = find_collocation_candidates(sentences, vocab, COLLOCATIONS, min_tokens=1)

    assert candidates == []


def test_fragile_chunk_still_counts_as_the_single_unknown_unit():
    # A low-confidence (fragile) chunk should still be minable -- it's the
    # one unit being tested, same as a fragile single word would be.
    vocab = make_vocab_state({"il": 1.0, "le": 1.0, "temps": 1.0}, known_chunks={"profiter de": 0.1})
    sentences = [sentence("Il profite de le temps.", [("Il", "il"), ("profite", "profiter"), ("de", "de"), ("le", "le"), ("temps", "temps")])]

    candidates = find_collocation_candidates(sentences, vocab, COLLOCATIONS, min_tokens=1)

    assert len(candidates) == 1
