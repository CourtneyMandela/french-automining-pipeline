from unittest.mock import MagicMock

from french_mining.anki.collocation_note_type import FIELD_NAMES, build_note
from french_mining.candidates import Candidate
from french_mining.collocation_generation import (
    generate_and_write_collocation_cards,
    generate_collocation_card_content,
)
from french_mining.scoring import ScoredCandidate


def make_scored(lemma: str, form: str, sentence: str) -> ScoredCandidate:
    candidate = Candidate(
        target_lemma=lemma,
        target_form=form,
        target_pos="COLLOC",
        sentence_text=sentence,
        other_lemmas=["il", "son", "erreur"],
        target_confidence=0.0,
        source="Le Petit Prince, p.12",
        is_collocation=True,
    )
    return ScoredCandidate(
        candidate=candidate,
        keep=True,
        i_plus_1_confirmed=True,
        frequency_rank=None,
        unlock_potential=1,
        context_transparency=True,
        interference_risk=None,
        is_concrete_and_visualizable=False,
        reasoning="good candidate",
        priority_score=1.2,
    )


def make_tool_response(cards: list[dict]) -> MagicMock:
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.input = {"cards": cards}
    response = MagicMock()
    response.content = [tool_block]
    return response


def test_generate_collocation_card_content_produces_complete_valid_field_set():
    client = MagicMock()
    client.messages.create.return_value = make_tool_response(
        [
            {
                "index": 0,
                "target_chunk_gloss": "to realize",
                "target_chunk_definition": "to become aware of something",
                "usage_note": "Reflexive + de: distinct from transitive apercevoir (to catch sight of).",
                "sentence_translation": "He realized his mistake too late.",
                "second_example": 'Elle <span class="target">s\'aperçoit de</span> son erreur.',
            }
        ]
    )
    scored = [
        make_scored("s'apercevoir de", "s'est aperçu de", "Il s'est aperçu de son erreur trop tard.")
    ]

    fields_list = generate_collocation_card_content(client, scored, date_mined="2026-07-05")

    assert len(fields_list) == 1
    fields = fields_list[0]
    assert set(fields.keys()) == set(FIELD_NAMES)
    assert fields["TargetChunk"] == "s'apercevoir de"
    assert fields["TargetChunkForm"] == "s'est aperçu de"
    assert "distinct from transitive apercevoir" in fields["UsageNote"]
    assert fields["SentenceText"] == 'Il <span class="target">s\'est aperçu de</span> son erreur trop tard.'
    assert fields["DateMined"] == "2026-07-05"
    assert fields["Source"] == "Le Petit Prince, p.12"
    assert fields["ChunkAudio"] == fields["SentenceAudio"] == fields["Image"] == ""

    note = build_note(fields)
    assert note["fields"]["TargetChunk"] == "s'apercevoir de"


def test_generate_collocation_card_content_batches_requests():
    client = MagicMock()

    def fake_create(**kwargs):
        content = kwargs["messages"][0]["content"]
        n = content.count("chunk=")
        return make_tool_response(
            [
                {
                    "index": i,
                    "target_chunk_gloss": "gloss",
                    "target_chunk_definition": "definition",
                    "usage_note": "note",
                    "sentence_translation": "translation",
                    "second_example": "example",
                }
                for i in range(n)
            ]
        )

    client.messages.create.side_effect = fake_create
    scored = [make_scored(f"chunk{i} de", f"chunk{i} de", f"Une phrase avec chunk{i} de.") for i in range(15)]

    fields_list = generate_collocation_card_content(client, scored, batch_size=10)

    assert client.messages.create.call_count == 2
    assert len(fields_list) == 15


def test_generate_and_write_collocation_cards_calls_add_card_per_candidate():
    anthropic_client = MagicMock()
    anthropic_client.messages.create.return_value = make_tool_response(
        [
            {
                "index": 0,
                "target_chunk_gloss": "gloss",
                "target_chunk_definition": "definition",
                "usage_note": "note",
                "sentence_translation": "translation",
                "second_example": "example",
            }
        ]
    )
    anki_client = MagicMock()
    anki_client.deck_names.return_value = ["French::Mining"]
    anki_client.model_names.return_value = ["French Collocation Mining"]
    anki_client.add_note.return_value = 8080

    scored = [make_scored("profiter de", "profite de", "Il profite de le temps.")]
    note_ids = generate_and_write_collocation_cards(anki_client, anthropic_client, scored)

    assert note_ids == [8080]
    anki_client.add_note.assert_called_once()
