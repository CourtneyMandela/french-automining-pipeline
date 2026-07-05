import pytest
from unittest.mock import MagicMock

from french_mining.anki.collocation_note_type import (
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
    assert note["fields"]["TargetChunk"] == "s'apercevoir de"


def test_build_note_rejects_missing_required_field():
    fields = placeholder_fields()
    del fields["TargetChunkGloss"]
    with pytest.raises(ValueError, match="TargetChunkGloss"):
        build_note(fields)


def test_build_note_allows_missing_optional_fields():
    fields = placeholder_fields()
    del fields["Image"]
    note = build_note(fields)
    assert note["fields"]["Image"] == ""


def test_ensure_model_creates_when_absent():
    client = MagicMock()
    client.model_names.return_value = []
    created = ensure_model(client)
    assert created is True
    call_kwargs = client.create_model.call_args.kwargs
    assert call_kwargs["model_name"] == MODEL_NAME
    assert call_kwargs["in_order_fields"] == FIELD_NAMES


def test_ensure_model_is_a_noop_when_present():
    client = MagicMock()
    client.model_names.return_value = [MODEL_NAME]
    assert ensure_model(client) is False
    client.create_model.assert_not_called()


def test_add_card_shares_the_single_word_deck():
    client = MagicMock()
    client.deck_names.return_value = []
    client.model_names.return_value = []
    client.add_note.return_value = 555

    note_id = add_card(client, placeholder_fields())

    assert note_id == 555
    client.invoke.assert_called_once_with("createDeck", deck="French::Mining")


def test_collocation_and_single_word_model_names_are_distinct():
    from french_mining.anki.note_type import MODEL_NAME as SINGLE_WORD_MODEL_NAME

    assert MODEL_NAME != SINGLE_WORD_MODEL_NAME
