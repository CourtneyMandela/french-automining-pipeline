from unittest.mock import MagicMock

from french_mining.frequency import FrequencyList
from french_mining.hygiene import (
    DEFAULT_INTERVAL_THRESHOLD_DAYS,
    find_hygiene_candidates,
    suspend_candidates,
)


def make_client(notes: list[dict], cards: list[dict]) -> MagicMock:
    client = MagicMock()
    client.find_notes.return_value = [n["noteId"] for n in notes]
    client.notes_info.return_value = notes
    client.cards_info.return_value = cards
    return client


def make_note(note_id: int, lemma: str, card_id: int) -> dict:
    return {"noteId": note_id, "fields": {"TargetWord": {"value": lemma, "order": 0}}, "cards": [card_id]}


def test_flags_mature_high_frequency_card():
    notes = [make_note(1, "avoir", 101)]
    cards = [{"cardId": 101, "type": 2, "interval": 200}]
    client = make_client(notes, cards)
    freq = FrequencyList(ranks={"avoir": 8})

    candidates = find_hygiene_candidates(client, model_names=["FrenchSentence"], frequency_list=freq)

    assert len(candidates) == 1
    c = candidates[0]
    assert c.lemma == "avoir"
    assert c.card_id == 101
    assert c.note_id == 1
    assert c.frequency_rank == 8


def test_does_not_flag_mature_but_rare_word():
    notes = [make_note(1, "chevrefeuille", 101)]
    cards = [{"cardId": 101, "type": 2, "interval": 400}]
    client = make_client(notes, cards)
    freq = FrequencyList(ranks={})  # not in frequency list at all

    candidates = find_hygiene_candidates(client, model_names=["FrenchSentence"], frequency_list=freq)

    assert candidates == []


def test_does_not_flag_frequent_but_immature_card():
    notes = [make_note(1, "avoir", 101)]
    cards = [{"cardId": 101, "type": 2, "interval": 30}]  # well under the threshold
    client = make_client(notes, cards)
    freq = FrequencyList(ranks={"avoir": 8})

    candidates = find_hygiene_candidates(client, model_names=["FrenchSentence"], frequency_list=freq)

    assert candidates == []


def test_does_not_flag_cards_still_in_learning_phase():
    notes = [make_note(1, "avoir", 101)]
    # type=1 (learning) with a suspiciously long interval shouldn't happen,
    # but the type gate should still exclude it defensively.
    cards = [{"cardId": 101, "type": 1, "interval": 300}]
    client = make_client(notes, cards)
    freq = FrequencyList(ranks={"avoir": 8})

    candidates = find_hygiene_candidates(client, model_names=["FrenchSentence"], frequency_list=freq)

    assert candidates == []


def test_respects_frequency_floor_boundary():
    notes = [make_note(1, "rareword", 101)]
    cards = [{"cardId": 101, "type": 2, "interval": 200}]
    client = make_client(notes, cards)
    freq = FrequencyList(ranks={"rareword": 501})  # just past the default floor of 500

    candidates = find_hygiene_candidates(client, model_names=["FrenchSentence"], frequency_list=freq)

    assert candidates == []


def test_respects_custom_interval_threshold():
    notes = [make_note(1, "avoir", 101)]
    cards = [{"cardId": 101, "type": 2, "interval": 100}]
    client = make_client(notes, cards)
    freq = FrequencyList(ranks={"avoir": 8})

    below_default = find_hygiene_candidates(client, model_names=["FrenchSentence"], frequency_list=freq)
    assert below_default == []

    above_custom = find_hygiene_candidates(
        client, model_names=["FrenchSentence"], frequency_list=freq, interval_threshold_days=90
    )
    assert len(above_custom) == 1


def test_default_threshold_is_roughly_six_months():
    assert DEFAULT_INTERVAL_THRESHOLD_DAYS == 180


def test_suspend_candidates_calls_suspend_cards_with_flagged_ids():
    from french_mining.hygiene import HygieneCandidate

    client = MagicMock()
    candidates = [
        HygieneCandidate(card_id=101, note_id=1, lemma="avoir", interval_days=200, frequency_rank=8),
        HygieneCandidate(card_id=102, note_id=2, lemma="etre", interval_days=250, frequency_rank=4),
    ]

    count = suspend_candidates(client, candidates)

    assert count == 2
    client.suspend_cards.assert_called_once_with([101, 102])


def test_suspend_candidates_handles_empty_list_without_calling_client():
    client = MagicMock()
    assert suspend_candidates(client, []) == 0
    client.suspend_cards.assert_not_called()
