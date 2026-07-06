from french_mining.anki.vocab_state import VocabularyState, WordKnowledge
from french_mining.condensed_audio import (
    select_comprehensible_spans,
    sentence_comprehensibility,
    total_duration,
    video_comprehensibility,
)
from french_mining.frequency import FrequencyList
from french_mining.nlp import ParsedSentence, Token


def make_vocab_state(known: dict[str, float]) -> VocabularyState:
    words = {
        lemma: WordKnowledge(
            lemma=lemma, confidence=c, stability_days=None, retrievability=None, lapses=0, in_learning=False, card_id=0
        )
        for lemma, c in known.items()
    }
    return VocabularyState(known_words=words, frequency_list=FrequencyList(ranks={}))


def sentence(lemmas: list[str], start=None, end=None, include_punct=False) -> ParsedSentence:
    tokens = [Token(text=l, lemma=l, pos="X", is_alpha=True) for l in lemmas]
    if include_punct:
        tokens.append(Token(text=".", lemma=".", pos="PUNCT", is_alpha=False))
    return ParsedSentence(text=" ".join(lemmas), tokens=tokens, start_time=start, end_time=end)


# -- sentence_comprehensibility ----------------------------------------------


def test_comprehensibility_all_known_is_one():
    vocab = make_vocab_state({"le": 1.0, "chat": 1.0, "dormir": 1.0})
    assert sentence_comprehensibility(sentence(["le", "chat", "dormir"]), vocab) == 1.0


def test_comprehensibility_counts_only_content_words():
    # Punctuation is not an alpha token, so it doesn't dilute the ratio.
    vocab = make_vocab_state({"le": 1.0, "chat": 1.0})
    assert sentence_comprehensibility(sentence(["le", "chat"], include_punct=True), vocab) == 1.0


def test_comprehensibility_partial():
    vocab = make_vocab_state({"le": 1.0, "chat": 1.0, "hibou": 0.0})
    # 2 of 3 known
    assert sentence_comprehensibility(sentence(["le", "chat", "hibou"]), vocab) == 2 / 3


def test_comprehensibility_empty_sentence_is_one():
    vocab = make_vocab_state({})
    empty = ParsedSentence(text="...", tokens=[Token(text="...", lemma="...", pos="PUNCT", is_alpha=False)])
    assert sentence_comprehensibility(empty, vocab) == 1.0


# -- select_comprehensible_spans ---------------------------------------------


def test_keeps_high_coverage_sentence_and_drops_low():
    vocab = make_vocab_state({"le": 1.0, "chat": 1.0, "dormir": 1.0, "hibou": 0.0, "voler": 0.0})
    sentences = [
        sentence(["le", "chat", "dormir"], start=0.0, end=2.0),   # 100% -> keep
        sentence(["le", "hibou", "voler"], start=5.0, end=7.0),   # 33% -> drop
    ]
    spans = select_comprehensible_spans(sentences, vocab, min_comprehensibility=0.9)
    assert spans == [(0.0, 2.0)]


def test_merges_adjacent_kept_spans():
    vocab = make_vocab_state({"a": 1.0, "b": 1.0})
    sentences = [
        sentence(["a", "b"], start=0.0, end=2.0),
        sentence(["a", "b"], start=2.2, end=4.0),   # gap 0.2s < merge_gap -> merge
    ]
    spans = select_comprehensible_spans(sentences, vocab, min_comprehensibility=0.9, merge_gap=0.4)
    assert spans == [(0.0, 4.0)]


def test_does_not_merge_far_apart_spans():
    vocab = make_vocab_state({"a": 1.0, "b": 1.0})
    sentences = [
        sentence(["a", "b"], start=0.0, end=2.0),
        sentence(["a", "b"], start=10.0, end=12.0),  # big gap -> stay separate
    ]
    spans = select_comprehensible_spans(sentences, vocab, min_comprehensibility=0.9, merge_gap=0.4)
    assert spans == [(0.0, 2.0), (10.0, 12.0)]


def test_skips_sentences_without_timing():
    vocab = make_vocab_state({"a": 1.0, "b": 1.0})
    sentences = [sentence(["a", "b"], start=None, end=None)]  # comprehensible but untimed
    assert select_comprehensible_spans(sentences, vocab) == []


def test_min_tokens_filters_out_tiny_utterances_when_requested():
    vocab = make_vocab_state({"oui": 1.0, "le": 1.0, "chat": 1.0})
    sentences = [
        sentence(["oui"], start=0.0, end=0.5),
        sentence(["le", "chat"], start=1.0, end=2.0),
    ]
    spans = select_comprehensible_spans(sentences, vocab, min_comprehensibility=0.9, min_tokens=2)
    assert spans == [(1.0, 2.0)]


# -- video_comprehensibility + ordering --------------------------------------


def test_video_comprehensibility_is_word_weighted():
    vocab = make_vocab_state({"a": 1.0, "b": 1.0, "x": 0.0})
    sentences = [sentence(["a", "b"]), sentence(["x"])]  # 2 of 3 content words known
    assert video_comprehensibility(sentences, vocab) == 2 / 3


def test_total_duration_sums_spans():
    assert total_duration([(0.0, 2.0), (3.0, 4.5)]) == 3.5
