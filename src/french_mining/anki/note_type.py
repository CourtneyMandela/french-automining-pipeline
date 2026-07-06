"""The single-word sentence-mining note type (§8): one card per note.

Front tests meaning reconstruction in context (the highest-value retrieval
practice). Back delivers confirmation plus audio-visual pairing and
scaffolding, in the fixed order the spec calls for: gloss -> full sentence
translation -> sentence audio -> TargetWordForm -> TargetWord -> morphology
note -> second example with audio -> source line -> frequency rank -> image.

`SentenceText` is expected to already contain the target word wrapped in a
highlight span (e.g. `<span class="target">mot</span>`) by the time it's
written here — highlighting is a card-generation concern, not this module's.
"""
from __future__ import annotations

from french_mining.anki.connect import AnkiConnectClient
from french_mining.anki.note_common import NoteTypeSpec
from french_mining.anki.note_common import add_card as _add_card
from french_mining.anki.note_common import build_note as _build_note
from french_mining.anki.note_common import ensure_model as _ensure_model

MODEL_NAME = "French Sentence Mining"
DEFAULT_DECK_NAME = "French::Mining"

FIELD_NAMES = [
    "TargetWord",
    "TargetWordForm",
    "TargetWordGloss",
    "TargetWordDefinition",
    "PartOfSpeech",
    "MorphologyNote",
    "SentenceText",
    "SentenceTranslation",
    "SecondExample",
    "SecondExampleAudio",
    "SentenceAudio",
    "WordAudio",
    "Image",
    "FrequencyRank",
    "Source",
    "DateMined",
]

# Fields that may legitimately be blank on some cards (e.g. no image tier
# succeeded, or a word predates frequency-rank data). WordAudio/
# SecondExampleAudio come from french_mining.tts (ElevenLabs) and
# SentenceAudio from a clipped source video (youtube.pipeline) -- both stay
# optional since either can go unconfigured/unavailable for a given card.
OPTIONAL_FIELDS = {"Image", "FrequencyRank", "SentenceAudio", "WordAudio", "SecondExampleAudio"}

CSS = """
.card {
  font-family: "Helvetica Neue", Arial, sans-serif;
  font-size: 22px;
  text-align: center;
  color: #1a1a1a;
  background-color: #fafafa;
  max-width: 700px;
  margin: 0 auto;
}
.target { font-weight: bold; color: #1d5fa8; }
.gloss { font-size: 20px; color: #444; margin: 0.5em 0; }
.translation { font-size: 18px; color: #666; margin: 0.5em 0; }
.target-word { font-size: 24px; margin: 0.75em 0; }
.morphology { font-size: 16px; color: #555; margin: 0.5em 0; }
.second-example { font-size: 17px; color: #333; margin: 0.75em 0; }
.source, .frequency { font-size: 13px; color: #999; margin-top: 0.5em; }
.image img { max-width: 320px; margin-top: 0.75em; }
"""

FRONT_TEMPLATE = """
<div class="sentence">{{SentenceText}}</div>
<div class="word-audio">{{WordAudio}}</div>
""".strip()

BACK_TEMPLATE = """
{{FrontSide}}
<hr id="answer">

<div class="gloss">{{TargetWordGloss}}</div>
<div class="translation">{{SentenceTranslation}}</div>
<div class="sentence-audio">{{SentenceAudio}}</div>

<div class="target-word">
  {{TargetWordForm}} &rarr; <span class="target">{{TargetWord}}</span>
  <span class="pos">({{PartOfSpeech}})</span>
</div>
<div class="definition">{{TargetWordDefinition}}</div>
<div class="morphology">{{MorphologyNote}}</div>

<div class="second-example">
  {{SecondExample}}
  {{SecondExampleAudio}}
</div>

<div class="source">{{Source}}</div>
{{#FrequencyRank}}<div class="frequency">Frequency rank: {{FrequencyRank}}</div>{{/FrequencyRank}}
{{#Image}}<div class="image">{{Image}}</div>{{/Image}}
""".strip()


SPEC = NoteTypeSpec(
    model_name=MODEL_NAME,
    field_names=FIELD_NAMES,
    optional_fields=OPTIONAL_FIELDS,
    front_template=FRONT_TEMPLATE,
    back_template=BACK_TEMPLATE,
    css=CSS,
    deck_name=DEFAULT_DECK_NAME,
)


def ensure_model(client: AnkiConnectClient) -> bool:
    """Create the note type in Anki if it doesn't already exist.

    Returns True if the model was created, False if it already existed.
    """
    return _ensure_model(client, SPEC)


def build_note(fields: dict[str, str], deck_name: str = DEFAULT_DECK_NAME) -> dict:
    """Build an AnkiConnect `addNote` payload from field values.

    Raises ValueError if a required (non-optional) field is missing.
    """
    return _build_note(fields, SPEC, deck_name=deck_name)


def add_card(client: AnkiConnectClient, fields: dict[str, str], deck_name: str = DEFAULT_DECK_NAME) -> int:
    """Create the deck (if needed) and the note type (if needed), then add one card."""
    return _add_card(client, fields, SPEC, deck_name=deck_name)


def placeholder_fields() -> dict[str, str]:
    """A complete, valid field set for verifying the end-to-end write path."""
    return {
        "TargetWord": "apercevoir",
        "TargetWordForm": "aperçu",
        "TargetWordGloss": "to catch sight of",
        "TargetWordDefinition": "to see something briefly or from a distance",
        "PartOfSpeech": "verb",
        "MorphologyNote": "3rd group, irregular; past participle: aperçu",
        "SentenceText": 'J\'ai <span class="target">aperçu</span> la tour au loin.',
        "SentenceTranslation": "I caught sight of the tower in the distance.",
        "SecondExample": 'Elle a <span class="target">aperçu</span> son ami dans la foule.',
        "SecondExampleAudio": "",
        "SentenceAudio": "",
        "WordAudio": "",
        "Image": "",
        "FrequencyRank": "",
        "Source": "Placeholder — build-order step 2 verification card",
        "DateMined": "2026-07-05",
    }
