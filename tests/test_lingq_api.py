import pytest
import responses

from french_mining.anki.vocab_state import VocabularyState, WordKnowledge
from french_mining.frequency import FrequencyList
from french_mining.lingq.api import (
    LingQCard,
    LingQClient,
    build_lingq_candidates,
    locate_term,
)
from french_mining.nlp import ParsedSentence, Token

CARDS_URL = "https://www.lingq.com/api/v3/fr/cards/"


# -- LingQClient (HTTP, mocked) ----------------------------------------------


@responses.activate
def test_fetch_lingqs_parses_cards_and_hint():
    responses.add(
        responses.GET,
        CARDS_URL,
        json={
            "count": 1,
            "next": None,
            "results": [
                {
                    "pk": 42,
                    "term": "canapé",
                    "fragment": "Le chat dort sur le canapé.",
                    "status": 1,
                    "hints": [{"text": "sofa"}],
                }
            ],
        },
        status=200,
    )
    client = LingQClient(api_key="fake-key", language_code="fr")
    cards = client.fetch_lingqs()

    assert len(cards) == 1
    assert cards[0] == LingQCard(pk=42, term="canapé", fragment="Le chat dort sur le canapé.", status=1, hint="sofa")


@responses.activate
def test_fetch_lingqs_follows_pagination():
    responses.add(
        responses.GET,
        CARDS_URL,
        json={"next": "https://www.lingq.com/api/v3/fr/cards/?page=2", "results": [{"pk": 1, "term": "a", "fragment": "f", "status": 0}]},
        status=200,
    )
    responses.add(
        responses.GET,
        "https://www.lingq.com/api/v3/fr/cards/?page=2",
        json={"next": None, "results": [{"pk": 2, "term": "b", "fragment": "f", "status": 0}]},
        status=200,
    )
    client = LingQClient(api_key="fake-key")
    cards = client.fetch_lingqs()

    assert [c.pk for c in cards] == [1, 2]


@responses.activate
def test_fetch_lingqs_respects_max_pages():
    # Every page points to a next page; max_pages must stop the loop.
    responses.add(
        responses.GET,
        CARDS_URL,
        json={"next": "https://www.lingq.com/api/v3/fr/cards/?page=2", "results": [{"pk": 1, "term": "a", "fragment": "f", "status": 0}]},
        status=200,
    )
    client = LingQClient(api_key="fake-key")
    cards = client.fetch_lingqs(max_pages=1)
    assert len(cards) == 1


def test_fetch_lingqs_requires_api_key(monkeypatch):
    monkeypatch.delenv("LINGQ_API_KEY", raising=False)
    client = LingQClient(language_code="fr")
    with pytest.raises(RuntimeError, match="LINGQ_API_KEY"):
        client.fetch_lingqs()


def test_missing_hints_yields_empty_string():
    from french_mining.lingq.api import _extract_hint

    assert _extract_hint({"term": "x"}) == ""
    assert _extract_hint({"hints": []}) == ""
    assert _extract_hint({"hints": [{"text": ""}, {"text": "meaning"}]}) == "meaning"


# -- locate_term --------------------------------------------------------------


def toks(pairs: list[tuple[str, str]]) -> list[Token]:
    return [Token(text=t, lemma=l, pos="X", is_alpha=t.isalpha()) for t, l in pairs]


def test_locate_term_single_word_surface_match():
    tokens = toks([("Le", "le"), ("chat", "chat"), ("dort", "dormir")])
    assert locate_term(tokens, "chat") == (1, 1)


def test_locate_term_does_not_match_substring_inside_another_word():
    # "de" must not match inside "monde".
    tokens = toks([("le", "le"), ("monde", "monde")])
    assert locate_term(tokens, "de") is None


def test_locate_term_multiword_across_tokens():
    tokens = toks([("Il", "il"), ("profite", "profiter"), ("de", "de"), ("la", "le"), ("vie", "vie")])
    assert locate_term(tokens, "profite de") == (1, 2)


def test_locate_term_lemma_fallback_for_base_form():
    # LingQ stored the infinitive; the fragment has an inflected surface.
    tokens = toks([("Il", "il"), ("profite", "profiter")])
    assert locate_term(tokens, "profiter") == (1, 1)


def test_locate_term_returns_none_when_absent():
    tokens = toks([("Le", "le"), ("chat", "chat")])
    assert locate_term(tokens, "hibou") is None


# -- build_lingq_candidates ---------------------------------------------------


def make_vocab_state(words: dict[str, float], chunks: dict[str, float] | None = None) -> VocabularyState:
    def kn(d):
        return {
            k: WordKnowledge(lemma=k, confidence=v, stability_days=None, retrievability=None, lapses=0, in_learning=False, card_id=0)
            for k, v in d.items()
        }

    return VocabularyState(known_words=kn(words), frequency_list=FrequencyList(ranks={}), known_chunks=kn(chunks or {}))


def sentence(text: str, pairs: list[tuple[str, str]]) -> ParsedSentence:
    return ParsedSentence(text=text, tokens=toks(pairs))


def test_build_keeps_looked_up_word_when_rest_of_fragment_is_known():
    vocab = make_vocab_state({"le": 1.0, "chat": 1.0, "dormir": 1.0, "sur": 1.0, "canapé": 0.0})
    card = LingQCard(pk=1, term="canapé", fragment="Le chat dort sur le canapé.", status=1)
    frag = sentence(card.fragment, [("Le", "le"), ("chat", "chat"), ("dort", "dormir"), ("sur", "sur"), ("le", "le"), ("canapé", "canapé")])

    candidates = build_lingq_candidates([(card, frag)], vocab, min_tokens=1)

    assert len(candidates) == 1
    c = candidates[0]
    assert c.target_lemma == "canapé"
    assert c.target_form == "canapé"
    assert c.source == "LingQ: canapé"
    assert c.is_collocation is False


def test_build_skips_word_already_known_in_anki():
    vocab = make_vocab_state({"le": 1.0, "chat": 1.0, "dormir": 1.0, "sur": 1.0, "canapé": 1.0})
    card = LingQCard(pk=1, term="canapé", fragment="Le chat dort sur le canapé.", status=3)
    frag = sentence(card.fragment, [("Le", "le"), ("chat", "chat"), ("dort", "dormir"), ("sur", "sur"), ("le", "le"), ("canapé", "canapé")])

    assert build_lingq_candidates([(card, frag)], vocab, min_tokens=1) == []


def test_build_skips_fragment_with_another_unknown_word():
    # Looked up "canapé" but "hibou" is also unknown -> not clean i+1.
    vocab = make_vocab_state({"le": 1.0, "voir": 1.0, "canapé": 0.0, "hibou": 0.0})
    card = LingQCard(pk=1, term="canapé", fragment="Le hibou voit le canapé.", status=1)
    frag = sentence(card.fragment, [("Le", "le"), ("hibou", "hibou"), ("voit", "voir"), ("le", "le"), ("canapé", "canapé")])

    assert build_lingq_candidates([(card, frag)], vocab, min_tokens=1) == []


def test_build_routes_multiword_lingq_to_collocation():
    vocab = make_vocab_state({"il": 1.0, "le": 1.0, "temps": 1.0}, chunks={})
    card = LingQCard(pk=1, term="profite de", fragment="Il profite de le temps.", status=1)
    frag = sentence(card.fragment, [("Il", "il"), ("profite", "profiter"), ("de", "de"), ("le", "le"), ("temps", "temps")])

    candidates = build_lingq_candidates([(card, frag)], vocab, min_tokens=1)

    assert len(candidates) == 1
    assert candidates[0].is_collocation is True
    assert candidates[0].target_lemma == "profite de"


def test_build_skips_card_whose_term_cannot_be_located():
    vocab = make_vocab_state({"le": 1.0, "chat": 1.0})
    card = LingQCard(pk=1, term="hibou", fragment="Le chat.", status=1)
    frag = sentence(card.fragment, [("Le", "le"), ("chat", "chat")])

    assert build_lingq_candidates([(card, frag)], vocab, min_tokens=1) == []
