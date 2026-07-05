import time
from unittest.mock import MagicMock

from french_mining.anki.vocab_state import (
    STABILITY_SOLID_DAYS,
    VocabularyState,
    estimate_retrievability,
)
from french_mining.frequency import FrequencyList


def make_client(notes: list[dict], cards: list[dict]) -> MagicMock:
    client = MagicMock()
    client.find_notes.return_value = [n["noteId"] for n in notes]
    client.notes_info.return_value = notes
    client.cards_info.return_value = cards
    return client


def test_estimate_retrievability_decays_over_time():
    fresh = estimate_retrievability(stability_days=30, elapsed_days=0)
    later = estimate_retrievability(stability_days=30, elapsed_days=60)
    assert fresh == 1.0
    assert 0 < later < fresh


def test_solid_mature_card_is_fully_known():
    now = time.time()
    notes = [
        {
            "noteId": 1,
            "fields": {"TargetWord": {"value": "chat", "order": 0}},
            "cards": [101],
        }
    ]
    cards = [
        {
            "cardId": 101,
            "type": 2,
            "lapses": 0,
            "mod": now,
            "interval": 60,
            "memoryState": {"stability": STABILITY_SOLID_DAYS * 2, "difficulty": 3.0},
        }
    ]
    client = make_client(notes, cards)
    state = VocabularyState.build(client, model_names=["FrenchSentence"], now_epoch=now)
    assert state.is_known("chat")
    assert state.confidence("chat") > 0.9


def test_card_still_in_learning_is_treated_almost_as_unknown():
    now = time.time()
    notes = [
        {
            "noteId": 2,
            "fields": {"TargetWord": {"value": "apercevoir", "order": 0}},
            "cards": [201],
        }
    ]
    cards = [
        {"cardId": 201, "type": 1, "lapses": 0, "mod": now, "interval": 1}
    ]
    client = make_client(notes, cards)
    state = VocabularyState.build(client, model_names=["FrenchSentence"], now_epoch=now)
    assert not state.is_known("apercevoir")
    assert state.confidence("apercevoir") < 0.2


def test_repeatedly_lapsing_card_is_practically_unknown():
    now = time.time()
    notes = [
        {
            "noteId": 3,
            "fields": {"TargetWord": {"value": "constater", "order": 0}},
            "cards": [301],
        }
    ]
    cards = [
        {
            "cardId": 301,
            "type": 2,
            "lapses": 5,
            "mod": now,
            "interval": 40,
            "memoryState": {"stability": 40, "difficulty": 5.0},
        }
    ]
    client = make_client(notes, cards)
    state = VocabularyState.build(client, model_names=["FrenchSentence"], now_epoch=now)
    assert state.confidence("constater") < 0.3


def test_tested_field_is_the_only_source_of_known_words():
    # A word appearing only inside SentenceText (context) must NOT count as known.
    now = time.time()
    notes = [
        {
            "noteId": 4,
            "fields": {
                "TargetWord": {"value": "chien", "order": 0},
                "SentenceText": {"value": "Le chien voit le chat.", "order": 1},
            },
            "cards": [401],
        }
    ]
    cards = [
        {
            "cardId": 401,
            "type": 2,
            "lapses": 0,
            "mod": now,
            "interval": 60,
            "memoryState": {"stability": 60, "difficulty": 3.0},
        }
    ]
    client = make_client(notes, cards)
    state = VocabularyState.build(client, model_names=["FrenchSentence"], now_epoch=now)
    assert state.is_known("chien")
    assert not state.is_known("chat")  # only appeared in the sentence, not tested
    assert state.confidence("chat") == 0.0


def test_frequency_floor_treats_pre_anki_words_as_known():
    client = make_client(notes=[], cards=[])
    state = VocabularyState.build(client, model_names=["FrenchSentence"])
    assert state.is_known("être")  # rank 4 in the frequency list, no Anki card needed


def test_exclusion_list_overrides_frequency_floor(tmp_path):
    exclusion_file = tmp_path / "exclusions.txt"
    exclusion_file.write_text("être\n# comment\n")
    client = make_client(notes=[], cards=[])
    state = VocabularyState.build(
        client,
        model_names=["FrenchSentence"],
        exclusion_list_path=str(exclusion_file),
    )
    assert not state.is_known("être")


def test_word_outside_frequency_floor_and_not_in_anki_is_unknown():
    client = make_client(notes=[], cards=[])
    state = VocabularyState.build(client, model_names=["FrenchSentence"])
    assert not state.is_known("chèvrefeuille")


def test_fallback_heuristic_used_when_memory_state_absent():
    now = time.time()
    notes = [
        {
            "noteId": 5,
            "fields": {"TargetWord": {"value": "parler", "order": 0}},
            "cards": [501],
        }
    ]
    # No FSRS memoryState -> falls back to interval-based proxy.
    cards = [{"cardId": 501, "type": 2, "lapses": 0, "mod": now, "interval": int(STABILITY_SOLID_DAYS)}]
    client = make_client(notes, cards)
    state = VocabularyState.build(client, model_names=["FrenchSentence"], now_epoch=now)
    assert state.confidence("parler") == 1.0
