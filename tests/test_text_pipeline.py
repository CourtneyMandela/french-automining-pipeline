from unittest.mock import MagicMock

from french_mining.candidates import Candidate
from french_mining.scoring import ScoredCandidate
from french_mining.text_pipeline import write_ranked_candidates


def make_scored(lemma: str, is_collocation: bool) -> ScoredCandidate:
    candidate = Candidate(
        target_lemma=lemma,
        target_form=lemma,
        target_pos="COLLOC" if is_collocation else "NOUN",
        sentence_text=f"Une phrase avec {lemma}.",
        other_lemmas=["une", "phrase", "avec"],
        target_confidence=0.0,
        source="LingQ: " + lemma,
        is_collocation=is_collocation,
    )
    return ScoredCandidate(
        candidate=candidate,
        keep=True,
        i_plus_1_confirmed=True,
        frequency_rank=500,
        unlock_potential=1,
        context_transparency=True,
        interference_risk=None,
        is_concrete_and_visualizable=False,  # keeps resolve_image_field API-free
        reasoning="ok",
        priority_score=1.0,
    )


def tool_response(cards: list[dict]) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.input = {"cards": cards}
    resp = MagicMock()
    resp.content = [block]
    return resp


def test_write_ranked_candidates_routes_words_and_collocations_to_their_note_types():
    anthropic_client = MagicMock()

    def fake_create(**kwargs):
        name = kwargs["tool_choice"]["name"]
        if name == "submit_card_content":
            return tool_response(
                [
                    {
                        "index": 0,
                        "target_word_gloss": "gloss",
                        "target_word_definition": "definition",
                        "part_of_speech": "noun",
                        "morphology_note": "note",
                        "sentence_translation": "translation",
                        "second_example": "example",
                    }
                ]
            )
        return tool_response(
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

    anthropic_client.messages.create.side_effect = fake_create

    anki_client = MagicMock()
    anki_client.deck_names.return_value = ["French::Mining"]
    anki_client.model_names.return_value = ["French Sentence Mining", "French Collocation Mining"]
    anki_client.add_note.side_effect = [111, 222]

    ranked = [make_scored("canapé", is_collocation=False), make_scored("profiter de", is_collocation=True)]
    note_ids = write_ranked_candidates(anki_client, anthropic_client, ranked, image_work_dir="/tmp/imgs")

    assert note_ids == [111, 222]
    assert anki_client.add_note.call_count == 2


def test_write_ranked_candidates_handles_empty_input():
    anthropic_client = MagicMock()
    anki_client = MagicMock()
    note_ids = write_ranked_candidates(anki_client, anthropic_client, [], image_work_dir="/tmp/imgs")
    assert note_ids == []
    anthropic_client.messages.create.assert_not_called()
