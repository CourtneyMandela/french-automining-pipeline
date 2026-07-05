from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
import responses

from french_mining.candidates import Candidate
from french_mining.images import (
    UNSPLASH_SEARCH_URL,
    UnsplashPhoto,
    check_image_relevance,
    download_image,
    pick_best_unsplash_photo,
    resolve_image_field,
    search_unsplash,
)
from french_mining.scoring import ScoredCandidate


def make_tool_response(tool_input: dict) -> MagicMock:
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.input = tool_input
    response = MagicMock()
    response.content = [tool_block]
    return response


def make_scored(lemma: str, is_concrete: bool) -> ScoredCandidate:
    candidate = Candidate(
        target_lemma=lemma,
        target_form=lemma,
        target_pos="NOUN",
        sentence_text=f"Une phrase avec {lemma}.",
        other_lemmas=["une", "phrase", "avec"],
        target_confidence=0.0,
    )
    return ScoredCandidate(
        candidate=candidate,
        keep=True,
        i_plus_1_confirmed=True,
        frequency_rank=500,
        unlock_potential=1,
        context_transparency=True,
        interference_risk=None,
        is_concrete_and_visualizable=is_concrete,
        reasoning="ok",
        priority_score=1.0,
    )


def write_photo_like_image(path):
    gradient = np.tile(np.linspace(0, 255, 200, dtype=np.uint8), (200, 1))
    image = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(path), image)


def write_text_like_image(path):
    block_grid = np.indices((51, 51)).sum(axis=0) % 2
    pattern = (np.kron(block_grid, np.ones((4, 4))) * 255)[:200, :200].astype(np.uint8)
    image = cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(path), image)


# -- check_image_relevance ---------------------------------------------------


def test_check_image_relevance_sends_image_and_returns_tool_result(tmp_path):
    image_path = tmp_path / "frame.jpg"
    write_photo_like_image(image_path)
    client = MagicMock()
    client.messages.create.return_value = make_tool_response(
        {"relevant": True, "reasoning": "clearly shows a bakery"}
    )

    result = check_image_relevance(client, image_path, "boulangerie", "bakery", "Je vais a la boulangerie.")

    assert result is True
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["tool_choice"]["name"] == "submit_image_relevance"
    content_blocks = kwargs["messages"][0]["content"]
    assert any(b["type"] == "image" for b in content_blocks)


def test_check_image_relevance_returns_false_when_model_rejects(tmp_path):
    image_path = tmp_path / "frame.jpg"
    write_photo_like_image(image_path)
    client = MagicMock()
    client.messages.create.return_value = make_tool_response(
        {"relevant": False, "reasoning": "just a talking head, not the object"}
    )

    assert check_image_relevance(client, image_path, "boulangerie", "bakery", "sentence") is False


# -- Unsplash -----------------------------------------------------------------


@responses.activate
def test_search_unsplash_parses_results():
    responses.add(
        responses.GET,
        UNSPLASH_SEARCH_URL,
        json={
            "results": [
                {"id": "abc", "alt_description": "a french bakery", "urls": {"regular": "https://example.com/a.jpg"}},
                {"id": "def", "alt_description": None, "description": "bread", "urls": {"regular": "https://example.com/b.jpg"}},
            ]
        },
        status=200,
    )

    photos = search_unsplash("boulangerie", api_key="fake-key")

    assert len(photos) == 2
    assert photos[0] == UnsplashPhoto(id="abc", description="a french bakery", url="https://example.com/a.jpg")
    assert photos[1].description == "bread"


def test_search_unsplash_requires_api_key(monkeypatch):
    monkeypatch.delenv("UNSPLASH_ACCESS_KEY", raising=False)
    with pytest.raises(RuntimeError, match="UNSPLASH_ACCESS_KEY"):
        search_unsplash("boulangerie")


@responses.activate
def test_download_image_writes_bytes(tmp_path):
    responses.add(responses.GET, "https://example.com/a.jpg", body=b"fake-jpeg-bytes", status=200)

    output = download_image("https://example.com/a.jpg", tmp_path / "nested" / "a.jpg")

    assert output.read_bytes() == b"fake-jpeg-bytes"


@responses.activate
def test_pick_best_unsplash_photo_returns_chosen_photo(tmp_path):
    responses.add(responses.GET, "https://example.com/a.jpg", body=b"photo-a", status=200)
    responses.add(responses.GET, "https://example.com/b.jpg", body=b"photo-b", status=200)

    client = MagicMock()
    client.messages.create.return_value = make_tool_response(
        {"chosen_index": 1, "reasoning": "more culturally distinctive"}
    )
    photos = [
        UnsplashPhoto(id="a", description="generic bread", url="https://example.com/a.jpg"),
        UnsplashPhoto(id="b", description="traditional french boulangerie storefront", url="https://example.com/b.jpg"),
    ]

    chosen = pick_best_unsplash_photo(client, photos, "boulangerie", "bakery", tmp_path)

    assert chosen == photos[1]
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["tool_choice"]["name"] == "submit_best_photo"


def test_pick_best_unsplash_photo_returns_none_for_empty_list(tmp_path):
    client = MagicMock()
    assert pick_best_unsplash_photo(client, [], "mot", "word", tmp_path) is None
    client.messages.create.assert_not_called()


@responses.activate
def test_pick_best_unsplash_photo_returns_none_when_model_rejects_all(tmp_path):
    responses.add(responses.GET, "https://example.com/a.jpg", body=b"photo-a", status=200)
    client = MagicMock()
    client.messages.create.return_value = make_tool_response(
        {"chosen_index": None, "reasoning": "none are suitable"}
    )
    photos = [UnsplashPhoto(id="a", description="generic", url="https://example.com/a.jpg")]

    assert pick_best_unsplash_photo(client, photos, "mot", "word", tmp_path) is None


# -- resolve_image_field (full 3-tier orchestration) -------------------------


def test_resolve_image_field_returns_blank_for_abstract_word():
    anki_client = MagicMock()
    anthropic_client = MagicMock()
    scored = make_scored("liberte", is_concrete=False)

    result = resolve_image_field(anki_client, anthropic_client, scored, "freedom", None, "/tmp/work")

    assert result == ""
    anki_client.store_media_file.assert_not_called()
    anthropic_client.messages.create.assert_not_called()


def test_resolve_image_field_uses_source_frame_when_it_passes_both_filters(tmp_path):
    frame_path = tmp_path / "frame.jpg"
    write_photo_like_image(frame_path)

    anki_client = MagicMock()
    anki_client.store_media_file.return_value = "boulangerie_source.jpg"
    anthropic_client = MagicMock()
    anthropic_client.messages.create.return_value = make_tool_response(
        {"relevant": True, "reasoning": "shows a bakery storefront"}
    )
    scored = make_scored("boulangerie", is_concrete=True)

    result = resolve_image_field(anki_client, anthropic_client, scored, "bakery", frame_path, tmp_path)

    assert result == '<img src="boulangerie_source.jpg">'
    anki_client.store_media_file.assert_called_once()


def test_resolve_image_field_skips_talking_head_frame_without_any_api_call(tmp_path):
    frame_path = tmp_path / "frame.jpg"
    write_text_like_image(frame_path)  # fails the local pre-filter for real

    anki_client = MagicMock()
    anthropic_client = MagicMock()
    scored = make_scored("boulangerie", is_concrete=True)

    result = resolve_image_field(anki_client, anthropic_client, scored, "bakery", frame_path, tmp_path)

    assert result == ""
    anthropic_client.messages.create.assert_not_called()
    anki_client.store_media_file.assert_not_called()


@responses.activate
def test_resolve_image_field_falls_back_to_unsplash_when_frame_irrelevant(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSPLASH_ACCESS_KEY", "fake-key")
    frame_path = tmp_path / "frame.jpg"
    write_photo_like_image(frame_path)  # passes local pre-filter

    responses.add(
        responses.GET,
        UNSPLASH_SEARCH_URL,
        json={"results": [{"id": "a", "alt_description": "bakery", "urls": {"regular": "https://example.com/a.jpg"}}]},
        status=200,
    )
    responses.add(responses.GET, "https://example.com/a.jpg", body=b"photo-a", status=200)

    anki_client = MagicMock()
    anki_client.store_media_file.return_value = "boulangerie_unsplash.jpg"
    anthropic_client = MagicMock()

    def fake_create(**kwargs):
        if kwargs["tool_choice"]["name"] == "submit_image_relevance":
            return make_tool_response({"relevant": False, "reasoning": "just a talking head"})
        return make_tool_response({"chosen_index": 0, "reasoning": "good enough"})

    anthropic_client.messages.create.side_effect = fake_create
    scored = make_scored("boulangerie", is_concrete=True)

    result = resolve_image_field(anki_client, anthropic_client, scored, "bakery", frame_path, tmp_path)

    assert result == '<img src="boulangerie_unsplash.jpg">'


def test_resolve_image_field_returns_blank_when_no_frame_and_no_unsplash_key(tmp_path, monkeypatch):
    monkeypatch.delenv("UNSPLASH_ACCESS_KEY", raising=False)
    anki_client = MagicMock()
    anthropic_client = MagicMock()
    scored = make_scored("boulangerie", is_concrete=True)

    result = resolve_image_field(anki_client, anthropic_client, scored, "bakery", None, tmp_path)

    assert result == ""
    anthropic_client.messages.create.assert_not_called()
