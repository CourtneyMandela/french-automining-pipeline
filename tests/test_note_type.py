import pytest
from unittest.mock import MagicMock

from french_mining.anki.note_type import (
    FIELD_NAMES,
    MODEL_NAME,
    add_card,
    build_note,
    ensure_model,
    placeholder_fields,
)


def test_placeholder_fields_are_complete_and_buildable():
    note = build_note(placeholder_fields())
    assert note["modelName"] == MODEL_NAME
    assert set(note["fields"].keys()) == set(FIELD_NAMES)
    assert note["fields"]["TargetWord"] == "apercevoir"


def test_build_note_rejects_missing_required_field():
    fields = placeholder_fields()
    del fields["TargetWordGloss"]
    with pytest.raises(ValueError, match="TargetWordGloss"):
        build_note(fields)


def test_build_note_allows_missing_optional_fields():
    fields = placeholder_fields()
    del fields["Image"]
    del fields["FrequencyRank"]
    note = build_note(fields)
    assert note["fields"]["Image"] == ""
    assert note["fields"]["FrequencyRank"] == ""


def test_ensure_model_creates_when_absent():
    client = MagicMock()
    client.model_names.return_value = []
    created = ensure_model(client)
    assert created is True
    client.create_model.assert_called_once()
    call_kwargs = client.create_model.call_args.kwargs
    assert call_kwargs["model_name"] == MODEL_NAME
    assert call_kwargs["in_order_fields"] == FIELD_NAMES


def test_ensure_model_is_a_noop_when_present():
    client = MagicMock()
    client.model_names.return_value = [MODEL_NAME]
    created = ensure_model(client)
    assert created is False
    client.create_model.assert_not_called()


def test_add_card_creates_deck_and_model_then_adds_note():
    client = MagicMock()
    client.deck_names.return_value = []
    client.model_names.return_value = []
    client.add_note.return_value = 999

    note_id = add_card(client, placeholder_fields())

    assert note_id == 999
    client.invoke.assert_called_once_with("createDeck", deck="French::Mining")
    client.create_model.assert_called_once()
    client.add_note.assert_called_once()


def test_add_card_skips_deck_and_model_creation_when_already_present():
    client = MagicMock()
    client.deck_names.return_value = ["French::Mining"]
    client.model_names.return_value = [MODEL_NAME]
    client.add_note.return_value = 1000

    add_card(client, placeholder_fields())

    client.invoke.assert_not_called()
    client.create_model.assert_not_called()
