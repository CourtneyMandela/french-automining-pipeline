from unittest.mock import MagicMock

from french_mining.anki.note_type import FIELD_NAMES, build_note
from french_mining.candidates import Candidate
from french_mining.generation import (
    generate_and_write_cards,
    generate_card_content,
    highlight_target,
)
from french_mining.scoring import ScoredCandidate


def make_scored(lemma: str, form: str, sentence: str, frequency_rank=None) -> ScoredCandidate:
    candidate = Candidate(
        target_lemma=lemma,
        target_form=form,
        target_pos="VERB",
        sentence_text=sentence,
        other_lemmas=["il", "la", "tour"],
        target_confidence=0.0,
        source="Le Petit Prince, p.12",
    )
    return ScoredCandidate(
        candidate=candidate,
        keep=True,
        i_plus_1_confirmed=True,
        frequency_rank=frequency_rank,
        unlock_potential=2,
        context_transparency=True,
        interference_risk=None,
        reasoning="good candidate",
        priority_score=1.5,
    )


def make_tool_response(cards: list[dict]) -> MagicMock:
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.input = {"cards": cards}
    response = MagicMock()
    response.content = [tool_block]
    return response


def test_highlight_target_wraps_first_occurrence():
    result = highlight_target("Il a aperçu la tour au loin.", "aperçu")
    assert result == 'Il a <span class="target">aperçu</span> la tour au loin.'


def test_highlight_target_falls_back_when_form_absent():
    result = highlight_target("Some other sentence.", "aperçu")
    assert result == "Some other sentence."


def test_generate_card_content_produces_complete_valid_field_set():
    client = MagicMock()
    client.messages.create.return_value = make_tool_response(
        [
            {
                "index": 0,
                "target_word_gloss": "to catch sight of",
                "target_word_definition": "to see something briefly or from a distance",
                "part_of_speech": "verb",
                "morphology_note": "3rd group, irregular; past participle: aperçu",
                "sentence_translation": "He caught sight of the tower in the distance.",
                "second_example": 'Elle a <span class="target">aperçu</span> son ami dans la foule.',
            }
        ]
    )
    scored = [make_scored("apercevoir", "aperçu", "Il a aperçu la tour au loin.", frequency_rank=650)]

    fields_list = generate_card_content(client, scored, date_mined="2026-07-05")

    assert len(fields_list) == 1
    fields = fields_list[0]
    assert set(fields.keys()) == set(FIELD_NAMES)
    assert fields["TargetWord"] == "apercevoir"
    assert fields["TargetWordForm"] == "aperçu"
    assert fields["SentenceText"] == 'Il a <span class="target">aperçu</span> la tour au loin.'
    assert fields["FrequencyRank"] == "650"
    assert fields["DateMined"] == "2026-07-05"
    assert fields["Source"] == "Le Petit Prince, p.12"
    # Audio/image fields stay blank at this build stage.
    assert fields["WordAudio"] == fields["SentenceAudio"] == fields["Image"] == ""

    # Must be directly usable by the note-writing layer with no further changes.
    note = build_note(fields)
    assert note["fields"]["TargetWord"] == "apercevoir"


def test_generate_card_content_handles_missing_frequency_rank():
    client = MagicMock()
    client.messages.create.return_value = make_tool_response(
        [
            {
                "index": 0,
                "target_word_gloss": "gloss",
                "target_word_definition": "definition",
                "part_of_speech": "noun",
                "morphology_note": "masculine",
                "sentence_translation": "translation",
                "second_example": "example",
            }
        ]
    )
    scored = [make_scored("mot", "mot", "Un mot rare.", frequency_rank=None)]

    fields_list = generate_card_content(client, scored)

    assert fields_list[0]["FrequencyRank"] == ""


def test_generate_card_content_batches_requests():
    client = MagicMock()

    def fake_create(**kwargs):
        content = kwargs["messages"][0]["content"]
        n = content.count("target=")
        return make_tool_response(
            [
                {
                    "index": i,
                    "target_word_gloss": "gloss",
                    "target_word_definition": "definition",
                    "part_of_speech": "noun",
                    "morphology_note": "note",
                    "sentence_translation": "translation",
                    "second_example": "example",
                }
                for i in range(n)
            ]
        )

    client.messages.create.side_effect = fake_create
    scored = [make_scored(f"mot{i}", f"mot{i}", f"Une phrase avec mot{i}.") for i in range(15)]

    fields_list = generate_card_content(client, scored, batch_size=10)

    assert client.messages.create.call_count == 2
    assert len(fields_list) == 15


def test_generate_and_write_cards_calls_add_card_per_candidate():
    anthropic_client = MagicMock()
    anthropic_client.messages.create.return_value = make_tool_response(
        [
            {
                "index": 0,
                "target_word_gloss": "gloss",
                "target_word_definition": "definition",
                "part_of_speech": "verb",
                "morphology_note": "note",
                "sentence_translation": "translation",
                "second_example": "example",
            }
        ]
    )
    anki_client = MagicMock()
    anki_client.deck_names.return_value = ["French::Mining"]
    anki_client.model_names.return_value = ["French Sentence Mining"]
    anki_client.add_note.return_value = 4242

    scored = [make_scored("apercevoir", "aperçu", "Il a aperçu la tour.")]
    note_ids = generate_and_write_cards(anki_client, anthropic_client, scored)

    assert note_ids == [4242]
    anki_client.add_note.assert_called_once()
