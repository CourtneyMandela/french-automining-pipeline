"""Card generation (§8, build-order step 5): turns a scored i+1 candidate into
the full field set `french_mining.anki.note_type` expects — gloss,
contextual definition, morphology note, sentence translation, and a second
example in a different context.

Audio (§9) and image (§10) fields are intentionally left blank here; those
are separate, later build-order stages.
"""
from __future__ import annotations

import datetime

import anthropic

from french_mining.anki.note_type import FIELD_NAMES, add_card
from french_mining.anki.connect import AnkiConnectClient
from french_mining.scoring import DEFAULT_MODEL, ScoredCandidate

GENERATION_TOOL = {
    "name": "submit_card_content",
    "description": "Submit generated Anki card content for a batch of scored i+1 sentence-mining candidates.",
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
                        "target_word_gloss": {
                            "type": "string",
                            "description": "2-3 word English equivalent of the target word.",
                        },
                        "target_word_definition": {
                            "type": "string",
                            "description": "One-sentence contextual definition of the target word as used in this sentence.",
                        },
                        "part_of_speech": {"type": "string"},
                        "morphology_note": {
                            "type": "string",
                            "description": "Gender for nouns; conjugation group + key forms + irregularity for verbs; agreement pattern for adjectives.",
                        },
                        "sentence_translation": {
                            "type": "string",
                            "description": "Full English translation of the source sentence.",
                        },
                        "second_example": {
                            "type": "string",
                            "description": "A new French sentence using the target word in a different context than the source sentence. Wrap the target word's occurrence in <span class=\"target\">...</span>, matching whatever inflected form you use.",
                        },
                    },
                    "required": [
                        "index",
                        "target_word_gloss",
                        "target_word_definition",
                        "part_of_speech",
                        "morphology_note",
                        "sentence_translation",
                        "second_example",
                    ],
                },
            }
        },
        "required": ["cards"],
    },
}

SYSTEM_PROMPT = """You are generating Anki sentence-mining card content for a French immersion learner.

For each candidate (a sentence containing exactly one word the learner
doesn't yet know), produce:
- A short (2-3 word) English gloss and a one-sentence contextual definition
  of the target word as it's used in *this* sentence (not every possible
  sense).
- A morphology note: gender for nouns; conjugation group, key conjugated
  forms, and any irregularity for verbs; agreement pattern for adjectives.
- A full, natural English translation of the source sentence.
- A second example sentence using the target word in a different context
  (different sentence structure/topic, ideally a different inflected form
  if it's a verb or agreement variant if an adjective), so the learner sees
  the word used more than one way. Wrap the target word's occurrence in
  <span class="target">...</span> in the second example.

Keep everything concise — this is a spaced-repetition card, not a dictionary
entry. Call submit_card_content with one entry per candidate index."""


def highlight_target(sentence_text: str, target_form: str) -> str:
    """Wrap the first exact occurrence of the target surface form in a
    highlight span. `target_form` comes from tokenizing this exact sentence,
    so it should always be present; if it somehow isn't, fall back to the
    unhighlighted sentence rather than raising.
    """
    if target_form and target_form in sentence_text:
        return sentence_text.replace(target_form, f'<span class="target">{target_form}</span>', 1)
    return sentence_text


def _format_item(index: int, scored: ScoredCandidate) -> str:
    c = scored.candidate
    return (
        f"[{index}] target={c.target_lemma!r} form={c.target_form!r} pos={c.target_pos}\n"
        f"    sentence: {c.sentence_text}"
    )


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def generate_card_content(
    client: anthropic.Anthropic,
    scored_candidates: list[ScoredCandidate],
    batch_size: int = 10,
    model: str = DEFAULT_MODEL,
    date_mined: str | None = None,
) -> list[dict[str, str]]:
    """Returns one field dict per candidate, ready for `note_type.build_note`
    (audio/image fields come back blank — later stages fill those in).
    """
    date_mined = date_mined or datetime.date.today().isoformat()
    results: list[dict[str, str]] = []

    for batch in _chunks(scored_candidates, batch_size):
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=[GENERATION_TOOL],
            tool_choice={"type": "tool", "name": "submit_card_content"},
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
                "TargetWord": c.target_lemma,
                "TargetWordForm": c.target_form,
                "TargetWordGloss": entry["target_word_gloss"],
                "TargetWordDefinition": entry["target_word_definition"],
                "PartOfSpeech": entry["part_of_speech"],
                "MorphologyNote": entry["morphology_note"],
                "SentenceText": highlight_target(c.sentence_text, c.target_form),
                "SentenceTranslation": entry["sentence_translation"],
                "SecondExample": entry["second_example"],
                "SecondExampleAudio": "",
                "SentenceAudio": "",
                "WordAudio": "",
                "Image": "",
                "FrequencyRank": str(scored.frequency_rank) if scored.frequency_rank else "",
                "Source": c.source,
                "DateMined": date_mined,
            }
            assert set(fields.keys()) == set(FIELD_NAMES)
            results.append(fields)

    return results


def generate_and_write_cards(
    anki_client: AnkiConnectClient,
    anthropic_client: anthropic.Anthropic,
    scored_candidates: list[ScoredCandidate],
    **kwargs,
) -> list[int]:
    """Generate content for each candidate and write it to Anki. Returns the
    new note IDs, in the same order as `scored_candidates`.
    """
    field_sets = generate_card_content(anthropic_client, scored_candidates, **kwargs)
    return [add_card(anki_client, fields) for fields in field_sets]
