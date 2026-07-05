from unittest.mock import MagicMock

import pytest

from french_mining.frequency import FrequencyList
from french_mining.candidates import Candidate
from french_mining.scoring import (
    build_client,
    keep_and_rank,
    score_candidates,
)


def make_candidate(lemma: str, sentence: str = "", days_since_encountered=None) -> Candidate:
    return Candidate(
        target_lemma=lemma,
        target_form=lemma,
        target_pos="NOUN",
        sentence_text=sentence or f"Une phrase avec {lemma}.",
        other_lemmas=["une", "phrase", "avec"],
        target_confidence=0.0,
        days_since_encountered=days_since_encountered,
    )


def make_tool_response(scores: list[dict]) -> MagicMock:
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.input = {"scores": scores}
    response = MagicMock()
    response.content = [tool_block]
    return response


def test_score_candidates_maps_results_back_by_index():
    client = MagicMock()
    client.messages.create.return_value = make_tool_response(
        [
            {
                "index": 0,
                "keep": True,
                "i_plus_1_confirmed": True,
                "unlock_potential": 3,
                "context_transparency": True,
                "interference_risk": None,
                "is_concrete_and_visualizable": False,
                "reasoning": "clear context",
            }
        ]
    )
    freq = FrequencyList(ranks={"canape": 800})
    candidates = [make_candidate("canape")]

    scored = score_candidates(client, candidates, freq)

    assert len(scored) == 1
    assert scored[0].candidate.target_lemma == "canape"
    assert scored[0].keep is True
    assert scored[0].frequency_rank == 800
    assert scored[0].unlock_potential == 3


def test_score_candidates_batches_requests(monkeypatch):
    client = MagicMock()

    def fake_create(**kwargs):
        content = kwargs["messages"][0]["content"]
        indices = [line for line in content.splitlines() if line.startswith("[")]
        return make_tool_response(
            [
                {
                    "index": i,
                    "keep": True,
                    "i_plus_1_confirmed": True,
                    "unlock_potential": 0,
                    "context_transparency": True,
                    "interference_risk": None,
                "is_concrete_and_visualizable": False,
                    "reasoning": "ok",
                }
                for i in range(len(indices))
            ]
        )

    client.messages.create.side_effect = fake_create
    candidates = [make_candidate(f"mot{i}") for i in range(25)]
    freq = FrequencyList(ranks={})

    scored = score_candidates(client, candidates, freq, batch_size=20)

    assert client.messages.create.call_count == 2
    assert len(scored) == 25


def test_interference_risk_heavily_discounts_priority():
    client = MagicMock()

    def fake_create(**kwargs):
        return make_tool_response(
            [
                {
                    "index": 0,
                    "keep": True,
                    "i_plus_1_confirmed": True,
                    "unlock_potential": 2,
                    "context_transparency": True,
                    "interference_risk": "remarquer",
                    "is_concrete_and_visualizable": False,
                    "reasoning": "confusable with remarquer, in queue",
                },
                {
                    "index": 1,
                    "keep": True,
                    "i_plus_1_confirmed": True,
                    "unlock_potential": 2,
                    "context_transparency": True,
                    "interference_risk": None,
                "is_concrete_and_visualizable": False,
                    "reasoning": "no conflict",
                },
            ]
        )

    client.messages.create.side_effect = fake_create
    freq = FrequencyList(ranks={"apercevoir": 400, "constater": 400})
    candidates = [make_candidate("apercevoir"), make_candidate("constater")]

    scored = score_candidates(client, candidates, freq, active_queue_lemmas=["remarquer"])

    flagged, unflagged = scored[0], scored[1]
    assert flagged.interference_risk == "remarquer"
    assert flagged.priority_score < unflagged.priority_score


def test_keep_and_rank_drops_rejected_and_sorts_best_first():
    client = MagicMock()
    client.messages.create.return_value = make_tool_response(
        [
            {
                "index": 0,
                "keep": False,
                "i_plus_1_confirmed": False,
                "unlock_potential": 0,
                "context_transparency": False,
                "interference_risk": None,
                "is_concrete_and_visualizable": False,
                "reasoning": "idiomatic, actually i+0",
            },
            {
                "index": 1,
                "keep": True,
                "i_plus_1_confirmed": True,
                "unlock_potential": 5,
                "context_transparency": True,
                "interference_risk": None,
                "is_concrete_and_visualizable": False,
                "reasoning": "great candidate",
            },
        ]
    )
    freq = FrequencyList(ranks={"rejete": 900, "garde": 50})
    candidates = [make_candidate("rejete"), make_candidate("garde")]

    scored = score_candidates(client, candidates, freq)
    ranked = keep_and_rank(scored)

    assert len(ranked) == 1
    assert ranked[0].candidate.target_lemma == "garde"


def test_build_client_raises_clear_error_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        build_client()
