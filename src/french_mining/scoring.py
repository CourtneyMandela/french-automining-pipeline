"""Stage 2 API scoring (§6): the only place Claude tokens get spent on
candidate judgment. Only sentences that survived the free local i+1 filter
(french_mining.lingq.candidates) ever reach this module.

Scores each candidate on:
  - i+1 fitness (double-checked in context — the local filter is mechanical
    and can't judge idiom/ambiguity the way a model reading the sentence can)
  - frequency rank (higher-frequency unknowns unlock more comprehension)
  - unlock potential (how many other backlog candidates this word's
    introduction would make learnable)
  - context transparency (can meaning be deduced before the answer is shown)
  - recency (recent content carries live episodic memory)
  - interference risk (semantic proximity to words already in the learning
    queue — near-synonyms/confusable forms should be delayed, not discarded)
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import anthropic

from french_mining.frequency import FrequencyList
from french_mining.lingq.candidates import Candidate

DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_BATCH_SIZE = 20

# Interference-flagged candidates aren't discarded (the plan calls this a
# *delay*, not a rejection) but should sink well below unflagged candidates
# of similar quality until the earlier, confusable word matures.
INTERFERENCE_PRIORITY_DISCOUNT = 0.4

SCORING_TOOL = {
    "name": "submit_candidate_scores",
    "description": "Submit sentence-mining quality scores for a batch of i+1 candidate sentences.",
    "input_schema": {
        "type": "object",
        "properties": {
            "scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {
                            "type": "integer",
                            "description": "Matches the candidate's index in the input list.",
                        },
                        "keep": {
                            "type": "boolean",
                            "description": "False if the sentence isn't genuinely good i+1 material on inspection (e.g. idiom makes it i+0 or i+2, awkward phrasing, ambiguous target word).",
                        },
                        "i_plus_1_confirmed": {"type": "boolean"},
                        "unlock_potential": {
                            "type": "integer",
                            "description": "How many other candidates in the provided backlog would become learnable (context-transparent) once this word is known.",
                        },
                        "context_transparency": {
                            "type": "boolean",
                            "description": "Can the target word's meaning be reasonably deduced from the sentence before the answer is shown?",
                        },
                        "interference_risk": {
                            "type": ["string", "null"],
                            "description": "Lemma of a near-synonym/confusable word already in the active learning queue this would interfere with, or null.",
                        },
                        "reasoning": {"type": "string"},
                    },
                    "required": [
                        "index",
                        "keep",
                        "i_plus_1_confirmed",
                        "unlock_potential",
                        "context_transparency",
                        "interference_risk",
                        "reasoning",
                    ],
                },
            }
        },
        "required": ["scores"],
    },
}

SYSTEM_PROMPT = """You are scoring candidate sentences for a French sentence-mining Anki pipeline.

Every candidate already passed a mechanical local filter confirming exactly
one word in the sentence is unknown or fragile to the learner (i+1). Your
job is qualitative judgment the local filter cannot make:

1. i+1 fitness: does the sentence actually read as i+1 in context, not just
   by the mechanical word count (e.g. idiomatic phrases can make a sentence
   effectively i+0 or i+2 even with one "unknown" word)?
2. Frequency: higher-frequency unknowns unlock more downstream comprehension.
3. Unlock potential: given the full backlog word list provided, how many
   other candidates would become learnable (context-transparent) once this
   word is known? Prefer words that clear dependency-tree branches.
4. Context transparency: can the target word's meaning be reasonably
   deduced from the sentence before the answer is revealed? This is the
   biggest quality differentiator between two otherwise-equal i+1 sentences.
5. Interference: check the target word and sentence against the active
   learning queue for near-synonyms or easily-confused forms (e.g.
   apercevoir / remarquer / constater). Flag it, don't just reject it —
   delay, not discard.

Do not shade i+1 toward i+0 by picking only the most obvious, context-giveaway
sentences — slight productive struggle before retrieval is the point.
Call submit_candidate_scores with one entry per candidate index provided."""


@dataclass
class ScoredCandidate:
    candidate: Candidate
    keep: bool
    i_plus_1_confirmed: bool
    frequency_rank: int | None
    unlock_potential: int
    context_transparency: bool
    interference_risk: str | None
    reasoning: str
    priority_score: float


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _format_candidate(index: int, candidate: Candidate) -> str:
    recency = (
        f"{candidate.days_since_encountered:.1f} days ago"
        if candidate.days_since_encountered is not None
        else "unknown"
    )
    return (
        f"[{index}] target={candidate.target_lemma!r} form={candidate.target_form!r} "
        f"pos={candidate.target_pos} recency={recency} source={candidate.source!r}\n"
        f"    sentence: {candidate.sentence_text}"
    )


def _build_user_message(
    batch: list[Candidate], backlog_lemmas: list[str], active_queue_lemmas: list[str]
) -> str:
    candidate_lines = "\n".join(_format_candidate(i, c) for i, c in enumerate(batch))
    return (
        f"Active learning queue (for interference checking): {', '.join(active_queue_lemmas) or '(empty)'}\n\n"
        f"Full backlog target words (for unlock-potential estimation): {', '.join(backlog_lemmas)}\n\n"
        f"Candidates to score:\n{candidate_lines}"
    )


def _priority_score(
    frequency_rank: int | None,
    unlock_potential: int,
    context_transparency: bool,
    days_since_encountered: float | None,
    interference_risk: str | None,
) -> float:
    # Lower rank (more frequent) is better; unranked words get a neutral mid score.
    frequency_score = 1.0 / frequency_rank if frequency_rank else 0.2
    transparency_score = 1.0 if context_transparency else 0.3
    # Recency decays over ~2 weeks; unknown recency is treated as neutral.
    if days_since_encountered is None:
        recency_score = 0.5
    else:
        recency_score = max(0.0, 1.0 - days_since_encountered / 14.0)

    score = (
        frequency_score * 2.0
        + min(unlock_potential, 10) * 0.3
        + transparency_score
        + recency_score
    )
    if interference_risk is not None:
        score *= INTERFERENCE_PRIORITY_DISCOUNT
    return score


def build_client() -> anthropic.Anthropic:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    return anthropic.Anthropic()


def score_candidates(
    client: anthropic.Anthropic,
    candidates: list[Candidate],
    frequency_list: FrequencyList,
    active_queue_lemmas: list[str] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    model: str = DEFAULT_MODEL,
) -> list[ScoredCandidate]:
    """Score every candidate via Claude, in batches. Returns one ScoredCandidate
    per input candidate — filtering on `.keep` and ordering by `.priority_score`
    is left to the caller (queue ordering is a separate build-order stage).
    """
    active_queue_lemmas = active_queue_lemmas or []
    backlog_lemmas = sorted({c.target_lemma for c in candidates})

    results: list[ScoredCandidate] = []
    for batch in _chunks(candidates, batch_size):
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=[SCORING_TOOL],
            tool_choice={"type": "tool", "name": "submit_candidate_scores"},
            messages=[
                {
                    "role": "user",
                    "content": _build_user_message(batch, backlog_lemmas, active_queue_lemmas),
                }
            ],
        )

        tool_use = next(block for block in response.content if block.type == "tool_use")
        scores_by_index = {entry["index"]: entry for entry in tool_use.input["scores"]}

        for i, candidate in enumerate(batch):
            entry = scores_by_index[i]
            frequency_rank = frequency_list.rank(candidate.target_lemma)
            priority_score = _priority_score(
                frequency_rank=frequency_rank,
                unlock_potential=entry["unlock_potential"],
                context_transparency=entry["context_transparency"],
                days_since_encountered=candidate.days_since_encountered,
                interference_risk=entry["interference_risk"],
            )
            results.append(
                ScoredCandidate(
                    candidate=candidate,
                    keep=entry["keep"],
                    i_plus_1_confirmed=entry["i_plus_1_confirmed"],
                    frequency_rank=frequency_rank,
                    unlock_potential=entry["unlock_potential"],
                    context_transparency=entry["context_transparency"],
                    interference_risk=entry["interference_risk"],
                    reasoning=entry["reasoning"],
                    priority_score=priority_score,
                )
            )
    return results


def keep_and_rank(scored: list[ScoredCandidate]) -> list[ScoredCandidate]:
    """Convenience filter: drop rejected candidates, sort best-first."""
    return sorted((s for s in scored if s.keep), key=lambda s: s.priority_score, reverse=True)
