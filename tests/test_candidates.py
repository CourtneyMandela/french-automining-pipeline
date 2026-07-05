from french_mining.anki.vocab_state import VocabularyState
from french_mining.frequency import FrequencyList
from french_mining.candidates import (
    find_i_plus_1_candidates,
    select_best_sentences,
)
from french_mining.nlp import ParsedSentence, Token


def make_vocab_state(known_confidences: dict[str, float]) -> VocabularyState:
    from french_mining.anki.vocab_state import WordKnowledge

    known_words = {
        lemma: WordKnowledge(
            lemma=lemma,
            confidence=confidence,
            stability_days=None,
            retrievability=None,
            lapses=0,
            in_learning=False,
            card_id=0,
        )
        for lemma, confidence in known_confidences.items()
    }
    # Empty frequency list/exclusion set so only explicit known_words count as known,
    # keeping these tests focused purely on the i+1 filtering logic.
    return VocabularyState(known_words=known_words, frequency_list=FrequencyList(ranks={}))


def sentence(text: str, words: list[tuple[str, str, str]]) -> ParsedSentence:
    """words: list of (surface, lemma, pos) tuples, all alpha tokens."""
    tokens = [Token(text=w, lemma=lemma, pos=pos, is_alpha=True) for w, lemma, pos in words]
    return ParsedSentence(text=text, tokens=tokens)


def test_sentence_with_exactly_one_unknown_word_is_a_candidate():
    vocab = make_vocab_state({"chat": 1.0, "noir": 1.0, "dormir": 1.0, "sur": 1.0, "le": 1.0, "canape": 0.0})
    sentences = [
        sentence(
            "Le chat noir dort sur le canape.",
            [
                ("Le", "le", "DET"),
                ("chat", "chat", "NOUN"),
                ("noir", "noir", "ADJ"),
                ("dort", "dormir", "VERB"),
                ("sur", "sur", "ADP"),
                ("le", "le", "DET"),
                ("canape", "canape", "NOUN"),
            ],
        )
    ]
    candidates = find_i_plus_1_candidates(sentences, vocab, min_tokens=1)
    assert len(candidates) == 1
    assert candidates[0].target_lemma == "canape"


def test_sentence_with_two_unknowns_is_rejected_as_i_plus_2():
    vocab = make_vocab_state({"le": 1.0})
    sentences = [
        sentence(
            "Le chat aperçoit le hibou.",
            [
                ("Le", "le", "DET"),
                ("chat", "chat", "NOUN"),  # unknown
                ("apercoit", "apercevoir", "VERB"),  # unknown
                ("le", "le", "DET"),
                ("hibou", "hibou", "NOUN"),  # unknown
            ],
        )
    ]
    candidates = find_i_plus_1_candidates(sentences, vocab, min_tokens=1)
    assert candidates == []


def test_fragile_word_counts_as_unknown_making_sentence_i_plus_2():
    # "chat" is barely known (low FSRS confidence) and "canape" is fully unknown:
    # one true unknown + one fragile word should be rejected, per the gradient model.
    vocab = make_vocab_state({"chat": 0.2, "noir": 1.0, "dormir": 1.0, "sur": 1.0, "le": 1.0, "canape": 0.0})
    sentences = [
        sentence(
            "Le chat noir dort sur le canape.",
            [
                ("Le", "le", "DET"),
                ("chat", "chat", "NOUN"),
                ("noir", "noir", "ADJ"),
                ("dort", "dormir", "VERB"),
                ("sur", "sur", "ADP"),
                ("le", "le", "DET"),
                ("canape", "canape", "NOUN"),
            ],
        )
    ]
    candidates = find_i_plus_1_candidates(sentences, vocab, min_tokens=1)
    assert candidates == []


def test_sentence_all_known_words_is_rejected_as_i_plus_0():
    vocab = make_vocab_state({"le": 1.0, "chat": 1.0, "dormir": 1.0})
    sentences = [
        sentence(
            "Le chat dort.",
            [("Le", "le", "DET"), ("chat", "chat", "NOUN"), ("dort", "dormir", "VERB")],
        )
    ]
    candidates = find_i_plus_1_candidates(sentences, vocab, min_tokens=1)
    assert candidates == []


def test_sentence_length_bounds_are_respected():
    vocab = make_vocab_state({})
    short_sentence = sentence("Chat.", [("Chat", "chat", "NOUN")])
    candidates = find_i_plus_1_candidates([short_sentence], vocab, min_tokens=4, max_tokens=30)
    assert candidates == []


def test_select_best_sentences_caps_per_lemma_and_prefers_ideal_length():
    from french_mining.candidates import Candidate

    def make_candidate(lemma: str, n_words: int) -> Candidate:
        return Candidate(
            target_lemma=lemma,
            target_form=lemma,
            target_pos="NOUN",
            sentence_text=f"sentence with {n_words} words",
            other_lemmas=["x"] * (n_words - 1),
            target_confidence=0.0,
        )

    candidates = [
        make_candidate("canape", 3),   # far from ideal (10)
        make_candidate("canape", 10),  # ideal length
        make_candidate("canape", 20),  # far from ideal
        make_candidate("canape", 9),   # near ideal
        make_candidate("hibou", 10),
    ]
    selected = select_best_sentences(candidates, max_per_lemma=2)

    canape_selected = [c for c in selected if c.target_lemma == "canape"]
    assert len(canape_selected) == 2
    lengths = {len(c.other_lemmas) + 1 for c in canape_selected}
    assert lengths == {10, 9}  # the two closest to IDEAL_TOKEN_COUNT
    assert any(c.target_lemma == "hibou" for c in selected)
