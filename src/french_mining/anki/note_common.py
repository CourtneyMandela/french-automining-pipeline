"""Shared mechanics between the single-word (§8) and collocation (§7) note
types: both create a model, build an `addNote` payload, and write one card
the same way — only the field list, templates, and CSS actually differ.
"""
from __future__ import annotations

from dataclasses import dataclass

from french_mining.anki.connect import AnkiConnectClient


@dataclass
class NoteTypeSpec:
    model_name: str
    field_names: list[str]
    optional_fields: set[str]
    front_template: str
    back_template: str
    css: str
    deck_name: str


def ensure_model(client: AnkiConnectClient, spec: NoteTypeSpec) -> bool:
    """Create the note type in Anki if it doesn't already exist.

    Returns True if the model was created, False if it already existed.
    AnkiConnect's `createModel` has no "already exists" idempotent mode, so
    we check `modelNames` first.
    """
    if spec.model_name in client.model_names():
        return False
    client.create_model(
        model_name=spec.model_name,
        in_order_fields=spec.field_names,
        card_templates=[{"Name": "Card 1", "Front": spec.front_template, "Back": spec.back_template}],
        css=spec.css,
    )
    return True


def build_note(fields: dict[str, str], spec: NoteTypeSpec, deck_name: str | None = None) -> dict:
    """Build an AnkiConnect `addNote` payload from field values.

    Raises ValueError if a required (non-optional) field is missing.
    """
    missing_required = [
        name for name in spec.field_names if name not in spec.optional_fields and not fields.get(name)
    ]
    if missing_required:
        raise ValueError(f"Missing required field(s): {', '.join(missing_required)}")

    note_fields = {name: fields.get(name, "") for name in spec.field_names}
    return {
        "deckName": deck_name or spec.deck_name,
        "modelName": spec.model_name,
        "fields": note_fields,
        "options": {"allowDuplicate": False},
        "tags": ["french-mining-pipeline"],
    }


def add_card(
    client: AnkiConnectClient, fields: dict[str, str], spec: NoteTypeSpec, deck_name: str | None = None
) -> int:
    """Create the deck (if needed) and the note type (if needed), then add one card."""
    deck_name = deck_name or spec.deck_name
    if deck_name not in client.deck_names():
        client.invoke("createDeck", deck=deck_name)
    ensure_model(client, spec)
    note = build_note(fields, spec, deck_name=deck_name)
    return client.add_note(note)
