"""The collocation note type (§7): fluency is largely chunk-level, not
word-level. Mirrors the single-word note type's structure and card
template order (see `note_type.py`), but the tested unit is a multi-word
collocation whose meaning isn't fully predictable from its parts, rather
than a single lemma.

Shares the same deck as the single-word note type — §7 is explicit that
single words and collocations "share the same backlog and compete for the
20 daily slots."
"""
from __future__ import annotations

from french_mining.anki.connect import AnkiConnectClient
from french_mining.anki.note_common import NoteTypeSpec
from french_mining.anki.note_common import add_card as _add_card
from french_mining.anki.note_common import build_note as _build_note
from french_mining.anki.note_common import ensure_model as _ensure_model
from french_mining.anki.note_type import DEFAULT_DECK_NAME

MODEL_NAME = "French Collocation Mining"

FIELD_NAMES = [
    "TargetChunk",
    "TargetChunkForm",
    "TargetChunkGloss",
    "TargetChunkDefinition",
    "UsageNote",
    "SentenceText",
    "SentenceTranslation",
    "SecondExample",
    "SecondExampleAudio",
    "SentenceAudio",
    "ChunkAudio",
    "Image",
    "Source",
    "DateMined",
]

# Audio fields are optional for now because audio sourcing (§9) hasn't been
# fully wired into the collocation path yet.
OPTIONAL_FIELDS = {"Image", "SentenceAudio", "ChunkAudio", "SecondExampleAudio"}

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
.target-chunk { font-size: 24px; margin: 0.75em 0; }
.usage-note { font-size: 16px; color: #555; margin: 0.5em 0; }
.second-example { font-size: 17px; color: #333; margin: 0.75em 0; }
.source { font-size: 13px; color: #999; margin-top: 0.5em; }
.image img { max-width: 320px; margin-top: 0.75em; }
"""

FRONT_TEMPLATE = """
<div class="sentence">{{SentenceText}}</div>
<div class="chunk-audio">{{ChunkAudio}}</div>
""".strip()

BACK_TEMPLATE = """
{{FrontSide}}
<hr id="answer">

<div class="gloss">{{TargetChunkGloss}}</div>
<div class="translation">{{SentenceTranslation}}</div>
<div class="sentence-audio">{{SentenceAudio}}</div>

<div class="target-chunk">
  {{TargetChunkForm}} &rarr; <span class="target">{{TargetChunk}}</span>
</div>
<div class="definition">{{TargetChunkDefinition}}</div>
<div class="usage-note">{{UsageNote}}</div>

<div class="second-example">
  {{SecondExample}}
  {{SecondExampleAudio}}
</div>

<div class="source">{{Source}}</div>
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
    return _ensure_model(client, SPEC)


def build_note(fields: dict[str, str], deck_name: str = DEFAULT_DECK_NAME) -> dict:
    return _build_note(fields, SPEC, deck_name=deck_name)


def add_card(client: AnkiConnectClient, fields: dict[str, str], deck_name: str = DEFAULT_DECK_NAME) -> int:
    return _add_card(client, fields, SPEC, deck_name=deck_name)


def placeholder_fields() -> dict[str, str]:
    """A complete, valid field set for verifying the end-to-end write path."""
    return {
        "TargetChunk": "s'apercevoir de",
        "TargetChunkForm": "s'est aperçu de",
        "TargetChunkGloss": "to realize",
        "TargetChunkDefinition": "to become aware of something, often suddenly",
        "UsageNote": "Reflexive + de: distinct from transitive apercevoir (\"to catch sight of\"). "
        "Il a aperçu la tour = he saw the tower; il s'est aperçu de son erreur = he realized his mistake.",
        "SentenceText": 'Il <span class="target">s\'est aperçu de</span> son erreur trop tard.',
        "SentenceTranslation": "He realized his mistake too late.",
        "SecondExample": 'Elle <span class="target">s\'aperçoit de</span> son erreur seulement maintenant.',
        "SecondExampleAudio": "",
        "SentenceAudio": "",
        "ChunkAudio": "",
        "Image": "",
        "Source": "Placeholder — build-order step 9 verification card",
        "DateMined": "2026-07-05",
    }
