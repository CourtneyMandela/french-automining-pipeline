"""Collocation card generation (§7, mirrors §8 build-order step 5 for
chunks): turns a scored collocation candidate into the full field set
`french_mining.anki.collocation_note_type` expects.

The distinguishing field versus the single-word generator is `UsageNote`:
since a collocation's whole reason for being mined is that its meaning
isn't predictable from its parts, the note should say so explicitly (e.g.
contrasting "s'apercevoir de" = "to realize" with transitive "apercevoir"
= "to catch sight of").
"""
from __future__ import annotations

import datetime

import anthropic

from french_mining.anki.collocation_note_type import FIELD_NAMES, add_card
from french_mining.anki.connect import AnkiConnectClient
from french_mining.generation import highlight_target
from french_mining.scoring import DEFAULT_MODEL, ScoredCandidate

GENERATION_TOOL = {
    "name": "submit_collocation_card_content",
    "description": "Submit generated Anki card content for a batch of scored i+1 collocation candidates.",
    "input_schema": {
        "type": "object",
        "properties": {
            "cards": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {
                            "type": "integer",
                            "description": "Matches the candidate's index in the input list.",
                        },
                        "target_chunk_gloss": {
                            "type": "string",
                            "description": "2-4 word English equivalent of the collocation.",
                        },
                        "target_chunk_definition": {
                            "type": "string",
                            "description": "One-sentence contextual definition of the collocation as used in this sentence.",
                        },
                        "usage_note": {
                            "type": "string",
                            "description": "Explains why this is worth learning as a chunk: how its meaning differs from its component words used independently/literally (e.g. a reflexive verb + preposition pairing with a distinct sense).",
                        },
                        "sentence_translation": {
                            "type": "string",
                            "description": "Full English translation of the source sentence.",
                        },
                        "second_example": {
                            "type": "string",
                            "description": "A new French sentence using the collocation in a different context than the source sentence. Wrap its occurrence in <span class=\"target\">...</span>, matching whatever inflected form you use.",
                        },
                    },
                    "required": [
                        "index",
                        "target_chunk_gloss",
                        "target_chunk_definition",
                        "usage_note",
                        "sentence_translation",
                        "second_example",
                    ],
                },
            }
        },
        "required": ["cards"],
    },
}

SYSTEM_PROMPT = """You are generating Anki sentence-mining card content for French collocations.

Each candidate is a sentence containing exactly one collocation the learner
doesn't yet know — a multi-word unit whose meaning isn't fully predictable
from its parts (verb-preposition pairings, fixed expressions). For each one,
produce:
- A short (2-4 word) English gloss and a one-sentence contextual definition
  of the collocation as used in *this* sentence.
- A usage note that explains why this is worth learning as a chunk: contrast
  its meaning with what the component words would mean used independently
  or literally (e.g. "s'apercevoir de" = "to realize," distinct from
  transitive "apercevoir" = "to catch sight of"). This is the single most
  important field on the card — it's the whole reason the chunk is being
  mined instead of the individual words.
- A full, natural English translation of the source sentence.
- A second example sentence using the collocation in a different context,
  so the learner sees it used more than once. Wrap its occurrence in
  <span class="target">...</span> in the second example.

Keep everything concise — this is a spaced-repetition card, not a dictionary
entry. Call submit_collocation_card_content with one entry per candidate index."""


def _format_item(index: int, scored: ScoredCandidate) -> str:
    c = scored.candidate
    return f"[{index}] chunk={c.target_lemma!r} form={c.target_form!r}\n    sentence: {c.sentence_text}"


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def generate_collocation_card_content(
    client: anthropic.Anthropic,
    scored_candidates: list[ScoredCandidate],
    batch_size: int = 10,
    model: str = DEFAULT_MODEL,
    date_mined: str | None = None,
) -> list[dict[str, str]]:
    """Returns one field dict per candidate, ready for
    `collocation_note_type.build_note` (audio/image fields come back blank).
    """
    date_mined = date_mined or datetime.date.today().isoformat()
    results: list[dict[str, str]] = []

    for batch in _chunks(scored_candidates, batch_size):
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=[GENERATION_TOOL],
            tool_choice={"type": "tool", "name": "submit_collocation_card_content"},
            messages=[
                {
                    "role": "user",
                    "content": "\n".join(_format_item(i, s) for i, s in enumerate(batch)),
                }
            ],
        )

        tool_use = next(block for block in response.content if block.type == "tool_use")
        by_index = {entry["index"]: entry for entry in tool_use.input["cards"]}

        for i, scored in enumerate(batch):
            entry = by_index[i]
            c = scored.candidate
            fields = {
                "TargetChunk": c.target_lemma,
                "TargetChunkForm": c.target_form,
                "TargetChunkGloss": entry["target_chunk_gloss"],
                "TargetChunkDefinition": entry["target_chunk_definition"],
                "UsageNote": entry["usage_note"],
                "SentenceText": highlight_target(c.sentence_text, c.target_form),
                "SentenceTranslation": entry["sentence_translation"],
                "SecondExample": entry["second_example"],
                "SecondExampleAudio": "",
                "SentenceAudio": "",
                "ChunkAudio": "",
                "Image": "",
                "Source": c.source,
                "DateMined": date_mined,
            }
            assert set(fields.keys()) == set(FIELD_NAMES)
            results.append(fields)

    return results


def generate_and_write_collocation_cards(
    anki_client: AnkiConnectClient,
    anthropic_client: anthropic.Anthropic,
    scored_candidates: list[ScoredCandidate],
    **kwargs,
) -> list[int]:
    """Generate content for each collocation candidate and write it to Anki.
    Returns the new note IDs, in the same order as `scored_candidates`.
    """
    field_sets = generate_collocation_card_content(anthropic_client, scored_candidates, **kwargs)
    return [add_card(anki_client, fields) for fields in field_sets]
