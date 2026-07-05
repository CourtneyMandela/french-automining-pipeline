"""Queue ordering via AnkiConnect `due` rewrites (§2, build-order step 6).

Core principle: FSRS owns a card the moment it's first reviewed. Until then
it just sits in the new-card queue, ordered by an integer `due` value — this
module is the only place anything "schedules" a card, and it only ever
touches cards still in that new/unreviewed state. The moment a card enters
review, this pipeline steps back for good.

Note: Anki's new-card *display* order also has to be configured to respect
`due` (Deck Options -> Display Order -> new card gather/sort order set to a
position-based order, not "random") or these rewrites have no visible
effect — that's a one-time setting in Anki itself, not something
AnkiConnect can change on your behalf.
"""
from __future__ import annotations

from french_mining.anki.connect import AnkiConnectClient

NEW_CARD_TYPE = 0  # Anki card `type`: 0 = new, never reviewed.


def get_new_backlog_note_ids(client: AnkiConnectClient, model_names: list[str], deck_name: str) -> list[int]:
    """All still-new notes of the given note type(s) in the deck, in their
    current queue order (ascending `due`).

    Accepts multiple note types so single words and collocations (§7: they
    "share the same backlog") can be reordered together in one pass.
    """
    note_filter = " or ".join(f'note:"{model_name}"' for model_name in model_names)
    card_ids = client.find_cards(f'deck:"{deck_name}" ({note_filter}) is:new')
    cards = client.cards_info(card_ids)
    cards.sort(key=lambda c: c.get("due", 0))
    return [c["note"] for c in cards]


def merge_priority_with_backlog(prioritized_note_ids: list[int], backlog_note_ids: list[int]) -> list[int]:
    """This run's top-ranked notes move to the front of the queue; every
    other still-new note keeps its existing relative order behind them.

    Deliberately doesn't re-score the entire historical backlog every run —
    that would burn API tokens against the cost philosophy (§3) re-judging
    cards that were already reasonably ranked last time.
    """
    prioritized_set = set(prioritized_note_ids)
    rest = [note_id for note_id in backlog_note_ids if note_id not in prioritized_set]
    return list(prioritized_note_ids) + rest


def _new_card_ids_by_note(client: AnkiConnectClient, note_ids: list[int]) -> dict[int, int]:
    if not note_ids:
        return {}
    notes = client.notes_info(note_ids)
    all_card_ids = [card_id for note in notes for card_id in note.get("cards", [])]
    cards_by_id = {c["cardId"]: c for c in client.cards_info(all_card_ids)}

    result: dict[int, int] = {}
    for note in notes:
        for card_id in note.get("cards", []):
            card = cards_by_id.get(card_id)
            if card and card.get("type") == NEW_CARD_TYPE:
                result[note["noteId"]] = card_id
                break  # one card per note by design (§8)
    return result


def reorder_queue(client: AnkiConnectClient, ranked_note_ids: list[int], start_due: int = 0) -> list[int]:
    """Rewrite `due` on each note's still-new card to consecutive integers
    matching `ranked_note_ids` (best-first). Notes whose card has already
    entered review are silently skipped — FSRS owns those now, not us.

    Returns the card IDs actually rewritten, in the order they were set.
    """
    card_id_by_note = _new_card_ids_by_note(client, ranked_note_ids)

    rewritten = []
    due = start_due
    for note_id in ranked_note_ids:
        card_id = card_id_by_note.get(note_id)
        if card_id is None:
            continue
        client.set_specific_value_of_card(card_id, "due", str(due))
        rewritten.append(card_id)
        due += 1
    return rewritten
