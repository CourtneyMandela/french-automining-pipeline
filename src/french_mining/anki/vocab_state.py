"""Vocabulary state model (§5) — the most important component in the pipeline.

A word counts as known only if it has its own note that has been reviewed —
never because it showed up in another card's example sentence. So this reads
the *tested* field of each note (the lemma being learned), not sentence text.

"Known" is a gradient, not a binary, weighted by FSRS per-card state:
  - high stability + high retrievability -> solidly known, full weight
  - still in learning / low stability -> fragile, treated almost as unknown
  - repeatedly lapsing -> practically unknown

Words that predate the Anki system entirely (être, avoir, ...) are handled
separately by the frequency floor (french_mining.frequency), not here.
"""
from __future__ import annotations

from dataclasses import dataclass

from french_mining.anki.connect import AnkiConnectClient
from french_mining.frequency import FrequencyList, load_exclusion_list

# Card `type` values used by Anki's scheduler: 0=new, 1=learning, 2=review, 3=relearning.
LEARNING_CARD_TYPES = {0, 1, 3}

# A card whose FSRS stability exceeds this is treated as "solidly known" (full
# confidence, modulo retrievability). Tunable; not a hard Anki/FSRS constant.
STABILITY_SOLID_DAYS = 21.0

# Cards that have lapsed this many times or more are discounted heavily,
# regardless of their current stability — a word that keeps being forgotten
# is not reliably known yet.
LAPSE_THRESHOLD = 3
LAPSE_DISCOUNT = 0.2

# A card still in its learning/relearning phase counts almost as if the word
# were unknown, per the build plan's gradient model.
LEARNING_PHASE_DISCOUNT = 0.15


def estimate_retrievability(stability_days: float, elapsed_days: float) -> float:
    """FSRS-style forgetting-curve approximation: R = (1 + t/(9S))^-1.

    This is the standard simplified FSRS retrievability curve. It's an
    approximation (exact FSRS versions use slightly different decay/factor
    constants) — good enough for confidence *weighting*, not for scheduling.
    """
    if stability_days <= 0:
        return 0.0
    return 1.0 / (1.0 + elapsed_days / (9.0 * stability_days))


@dataclass
class WordKnowledge:
    lemma: str
    confidence: float
    stability_days: float | None
    retrievability: float | None
    lapses: int
    in_learning: bool
    card_id: int


def _confidence_from_card(card: dict, now_epoch: float) -> WordKnowledge:
    card_id = card["cardId"]
    card_type = card.get("type", 0)
    lapses = card.get("lapses", 0)
    in_learning = card_type in LEARNING_CARD_TYPES

    memory_state = card.get("memoryState") or {}
    stability_days = memory_state.get("stability")

    if stability_days is not None:
        elapsed_days = max(0.0, (now_epoch - card.get("mod", now_epoch)) / 86400.0)
        retrievability = estimate_retrievability(stability_days, elapsed_days)
        confidence = retrievability * min(1.0, stability_days / STABILITY_SOLID_DAYS)
    else:
        # Fallback when AnkiConnect/Anki doesn't expose FSRS memory state:
        # use the scheduler's own interval as a stability proxy.
        retrievability = None
        interval_days = max(0, card.get("interval", 0))
        if in_learning:
            confidence = 0.1
        else:
            confidence = min(1.0, interval_days / STABILITY_SOLID_DAYS)

    if in_learning:
        confidence *= LEARNING_PHASE_DISCOUNT
    if lapses >= LAPSE_THRESHOLD:
        confidence *= LAPSE_DISCOUNT

    return WordKnowledge(
        lemma="",  # filled in by caller, who knows which note this card belongs to
        confidence=max(0.0, min(1.0, confidence)),
        stability_days=stability_days,
        retrievability=retrievability,
        lapses=lapses,
        in_learning=in_learning,
        card_id=card_id,
    )


class VocabularyState:
    """Gradient-weighted French vocabulary state, backed by Anki + a frequency floor."""

    def __init__(
        self,
        known_words: dict[str, WordKnowledge],
        frequency_list: FrequencyList | None = None,
        exclusion_list: frozenset[str] = frozenset(),
        frequency_floor: int = 500,
    ):
        self.known_words = known_words
        self.frequency_list = frequency_list
        self.exclusion_list = exclusion_list
        self.frequency_floor = frequency_floor

    def confidence(self, lemma: str) -> float:
        lemma = lemma.strip().lower()
        knowledge = self.known_words.get(lemma)
        if knowledge is not None:
            return knowledge.confidence
        if lemma in self.exclusion_list:
            return 0.0
        if self.frequency_list and self.frequency_list.is_within_floor(lemma, self.frequency_floor):
            return 1.0
        return 0.0

    def is_known(self, lemma: str, threshold: float = 0.6) -> bool:
        return self.confidence(lemma) >= threshold

    @classmethod
    def build(
        cls,
        client: AnkiConnectClient,
        model_names: list[str],
        tested_field: str = "TargetWord",
        frequency_list: FrequencyList | None = None,
        exclusion_list_path: str | None = None,
        frequency_floor: int = 500,
        now_epoch: float | None = None,
    ) -> "VocabularyState":
        """Build vocabulary state by reading note + card state via AnkiConnect.

        `model_names` are the Anki note types whose `tested_field` holds the
        lemma being learned (e.g. the single-word and collocation note types).
        """
        import time

        now_epoch = now_epoch if now_epoch is not None else time.time()

        known_words: dict[str, WordKnowledge] = {}
        for model_name in model_names:
            note_ids = client.find_notes(f'note:"{model_name}"')
            notes = client.notes_info(note_ids)

            card_ids: list[int] = []
            card_id_to_lemma: dict[int, str] = {}
            for note in notes:
                field = note.get("fields", {}).get(tested_field)
                if not field:
                    continue
                lemma = field["value"].strip().lower()
                if not lemma:
                    continue
                for card_id in note.get("cards", []):
                    card_ids.append(card_id)
                    card_id_to_lemma[card_id] = lemma

            cards = client.cards_info(card_ids)
            for card in cards:
                lemma = card_id_to_lemma[card["cardId"]]
                knowledge = _confidence_from_card(card, now_epoch)
                knowledge.lemma = lemma
                existing = known_words.get(lemma)
                if existing is None or knowledge.confidence > existing.confidence:
                    known_words[lemma] = knowledge

        return cls(
            known_words=known_words,
            frequency_list=frequency_list or FrequencyList.load(),
            exclusion_list=load_exclusion_list(exclusion_list_path),
            frequency_floor=frequency_floor,
        )
