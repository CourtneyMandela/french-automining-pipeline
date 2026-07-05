"""Monthly deck hygiene audit (§11, build-order step 10): FSRS has no
window into immersion, so words the learner has fully acquired through
listening/reading still cycle through reviews indefinitely, bloating the
deck. This flags — and can suspend — cards that are both mature (long
scheduled interval) and high-frequency, since a top-N word reaching that
stability is almost certainly acquired by immersion rather than earned
purely through deliberate review.

Suspends, never deletes: suspended cards leave the review queue but stay
in the deck, reactivatable anytime (via Anki's UI or AnkiConnect's
`unsuspend` action).
"""
from __future__ import annotations

from dataclasses import dataclass

from french_mining.anki.connect import AnkiConnectClient
from french_mining.frequency import FrequencyList

# ~6 months, the spec's own example threshold for "projected interval."
DEFAULT_INTERVAL_THRESHOLD_DAYS = 180

# Only words at or above this frequency rank are "almost certainly acquired
# by immersion" — a genuinely rare word that reached the same stability was
# more likely earned through deliberate study, not something to second-guess.
DEFAULT_FREQUENCY_FLOOR = 500

# Anki card `type`: 2 = review (has left the learning phase). Cards still
# in learning/relearning haven't proven anything yet and are never audited.
REVIEW_CARD_TYPE = 2


@dataclass
class HygieneCandidate:
    card_id: int
    note_id: int
    lemma: str
    interval_days: int
    frequency_rank: int


def find_hygiene_candidates(
    client: AnkiConnectClient,
    model_names: list[str],
    tested_field: str = "TargetWord",
    frequency_list: FrequencyList | None = None,
    interval_threshold_days: int = DEFAULT_INTERVAL_THRESHOLD_DAYS,
    frequency_floor: int = DEFAULT_FREQUENCY_FLOOR,
) -> list[HygieneCandidate]:
    """Scan the given note type(s) for mature, high-frequency review cards.

    Frequency cross-referencing needs a frequency list, so this is
    single-word-only in practice — collocations aren't frequency-ranked
    (§7), so there's no equivalent "obviously acquired by immersion" signal
    for them yet.
    """
    frequency_list = frequency_list or FrequencyList.load()

    candidates: list[HygieneCandidate] = []
    for model_name in model_names:
        note_ids = client.find_notes(f'note:"{model_name}"')
        notes = client.notes_info(note_ids)

        card_id_to_note_id: dict[int, int] = {}
        card_id_to_lemma: dict[int, str] = {}
        for note in notes:
            field = note.get("fields", {}).get(tested_field)
            if not field:
                continue
            lemma = field["value"].strip().lower()
            if not lemma:
                continue
            for card_id in note.get("cards", []):
                card_id_to_note_id[card_id] = note["noteId"]
                card_id_to_lemma[card_id] = lemma

        cards = client.cards_info(list(card_id_to_note_id.keys()))
        for card in cards:
            if card.get("type") != REVIEW_CARD_TYPE:
                continue
            if card.get("interval", 0) < interval_threshold_days:
                continue

            lemma = card_id_to_lemma[card["cardId"]]
            rank = frequency_list.rank(lemma)
            if rank is None or rank > frequency_floor:
                continue

            candidates.append(
                HygieneCandidate(
                    card_id=card["cardId"],
                    note_id=card_id_to_note_id[card["cardId"]],
                    lemma=lemma,
                    interval_days=card["interval"],
                    frequency_rank=rank,
                )
            )

    return candidates


def suspend_candidates(client: AnkiConnectClient, candidates: list[HygieneCandidate]) -> int:
    """Suspend the flagged cards. Returns how many were suspended."""
    if not candidates:
        return 0
    client.suspend_cards([c.card_id for c in candidates])
    return len(candidates)
