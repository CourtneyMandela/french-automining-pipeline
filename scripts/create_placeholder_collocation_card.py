#!/usr/bin/env python3
"""Run this on the machine where Anki is open, to verify build-order step 9
end-to-end: creates the collocation note type (sharing the single-word
deck) if needed, then adds one placeholder card so you can eyeball the
front/back templates in Anki.

    .venv/bin/python scripts/create_placeholder_collocation_card.py
"""
from french_mining.anki.collocation_note_type import DEFAULT_DECK_NAME, add_card, placeholder_fields
from french_mining.anki.connect import AnkiConnectClient


def main() -> None:
    client = AnkiConnectClient()
    note_id = add_card(client, placeholder_fields())
    print(f"Created note {note_id} in deck '{DEFAULT_DECK_NAME}'.")
    print("Open Anki and check the card's front/back rendering.")


if __name__ == "__main__":
    main()
