"""Image tier logic (§10): source frame (with a free local talking-head/
text pre-filter, then a cheap API vision relevance check) -> Unsplash
fallback -> no image.

Concrete, culturally specific nouns only — abstract words never get an
image (irrelevant/generic images actively hurt retention per Mayer's
coherence principle). Concreteness is judged once, during Stage 2 scoring
(`ScoredCandidate.is_concrete_and_visualizable`) rather than with a
dedicated API call here, since Claude already sees the word and sentence
at that stage.
"""
from __future__ import annotations

import base64
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path

import anthropic
import cv2
import requests

from french_mining.scoring import DEFAULT_MODEL, ScoredCandidate

FACE_CASCADE_FILE = "haarcascade_frontalface_default.xml"

# A detected face occupying more than this fraction of the frame area marks
# it a "talking head" shot — not distinctive/relevant per the coherence
# principle. Discard before spending any API tokens on it.
FACE_AREA_RATIO_THRESHOLD = 0.15

# Frames with a high proportion of edge pixels (Canny, which only keeps
# thin non-max-suppressed edges — a smooth photo/gradient measures ~0,
# text/line-art patterns measure noticeably higher) tend to be text-heavy
# (slides, burned-in captions) rather than photographic content. This is a
# coarse heuristic, not real text detection — deliberately conservative to
# avoid discarding legitimately detailed photos.
EDGE_DENSITY_TEXT_THRESHOLD = 0.12

UNSPLASH_SEARCH_URL = "https://api.unsplash.com/search/photos"


def _face_area_ratio_from_boxes(faces, frame_shape: tuple[int, int]) -> float:
    """Pure ratio computation, kept separate from cascade detection so it's
    testable without needing a real photographic face image.
    """
    if len(faces) == 0:
        return 0.0
    frame_area = frame_shape[0] * frame_shape[1]
    largest_face_area = max(w * h for (_, _, w, h) in faces)
    return largest_face_area / frame_area


def _detect_faces(gray_image):
    cascade_path = Path(cv2.data.haarcascades) / FACE_CASCADE_FILE
    cascade = cv2.CascadeClassifier(str(cascade_path))
    return cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)


def _edge_density(gray_image) -> float:
    edges = cv2.Canny(gray_image, 100, 200)
    return float((edges > 0).sum()) / edges.size


def is_talking_head_or_text(frame_path: str | Path) -> bool:
    """Free local pre-filter: True if the frame looks like a talking-head
    shot (large detected face) or a text-heavy frame (slide, burned-in
    captions) — discard before spending any API tokens on it.
    """
    image = cv2.imread(str(frame_path))
    if image is None:
        return True  # unreadable frame — treat conservatively, don't use it

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = _detect_faces(gray)
    face_ratio = _face_area_ratio_from_boxes(faces, image.shape[:2])
    return face_ratio > FACE_AREA_RATIO_THRESHOLD or _edge_density(gray) > EDGE_DENSITY_TEXT_THRESHOLD


IMAGE_RELEVANCE_TOOL = {
    "name": "submit_image_relevance",
    "description": "Judge whether an image is relevant and distinctive enough to pair with a vocabulary word.",
    "input_schema": {
        "type": "object",
        "properties": {
            "relevant": {
                "type": "boolean",
                "description": "True only if the image clearly and distinctively depicts the target word's meaning — not a generic, ambiguous, or unrelated scene.",
            },
            "reasoning": {"type": "string"},
        },
        "required": ["relevant", "reasoning"],
    },
}

PHOTO_PICK_TOOL = {
    "name": "submit_best_photo",
    "description": "Pick the best photo (or none) for a French vocabulary word from a set of candidate images.",
    "input_schema": {
        "type": "object",
        "properties": {
            "chosen_index": {
                "type": ["integer", "null"],
                "description": "0-based index of the best photo, preferring culturally specific/distinctive imagery over generic stock-photo-style images. Null if none are suitable.",
            },
            "reasoning": {"type": "string"},
        },
        "required": ["chosen_index", "reasoning"],
    },
}


def _encode_image(path: str | Path) -> tuple[str, str]:
    media_type, _ = mimetypes.guess_type(str(path))
    media_type = media_type or "image/jpeg"
    data = base64.standard_b64encode(Path(path).read_bytes()).decode("ascii")
    return media_type, data


def check_image_relevance(
    client: anthropic.Anthropic,
    image_path: str | Path,
    target_word: str,
    target_gloss: str,
    sentence_text: str,
    model: str = DEFAULT_MODEL,
) -> bool:
    """Cheap API vision check (§10 tier 1): does this specific frame show
    something concrete and relevant to the target word? Only frames that
    already passed the free local pre-filter should reach this.
    """
    media_type, data = _encode_image(image_path)
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        tools=[IMAGE_RELEVANCE_TOOL],
        tool_choice={"type": "tool", "name": "submit_image_relevance"},
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}},
                    {
                        "type": "text",
                        "text": (
                            f"Target word: {target_word!r} ({target_gloss}). "
                            f"Sentence: {sentence_text}\n\n"
                            "Does this image clearly and distinctively depict the target "
                            "word's meaning? Reject generic, ambiguous, or unrelated frames — "
                            "an irrelevant image actively hurts retention."
                        ),
                    },
                ],
            }
        ],
    )
    tool_use = next(block for block in response.content if block.type == "tool_use")
    return tool_use.input["relevant"]


@dataclass
class UnsplashPhoto:
    id: str
    description: str
    url: str


def search_unsplash(query: str, api_key: str | None = None, per_page: int = 5) -> list[UnsplashPhoto]:
    api_key = api_key or os.environ.get("UNSPLASH_ACCESS_KEY")
    if not api_key:
        raise RuntimeError("UNSPLASH_ACCESS_KEY is not set. Copy .env.example to .env and fill it in.")

    response = requests.get(
        UNSPLASH_SEARCH_URL,
        params={"query": query, "per_page": per_page},
        headers={"Authorization": f"Client-ID {api_key}"},
        timeout=15,
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    return [
        UnsplashPhoto(
            id=r["id"],
            description=r.get("alt_description") or r.get("description") or "",
            url=r["urls"]["regular"],
        )
        for r in results
    ]


def download_image(url: str, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    output_path.write_bytes(response.content)
    return output_path


def pick_best_unsplash_photo(
    client: anthropic.Anthropic,
    photos: list[UnsplashPhoto],
    target_word: str,
    target_gloss: str,
    work_dir: str | Path,
    model: str = DEFAULT_MODEL,
) -> UnsplashPhoto | None:
    """Downloads each candidate photo, then asks Claude to pick the most
    culturally specific/distinctive one — preferring that over generic
    stock-photo-style imagery, per §10 — or none, if none are suitable.
    """
    if not photos:
        return None

    work_dir = Path(work_dir)
    content: list[dict] = [
        {
            "type": "text",
            "text": (
                f"Target word: {target_word!r} ({target_gloss}). Pick the photo that best "
                "and most distinctively depicts it — prefer culturally specific imagery over "
                "generic/stock-photo-style ones. Photos are numbered in the order shown."
            ),
        }
    ]
    for i, photo in enumerate(photos):
        local_path = download_image(photo.url, work_dir / f"unsplash_candidate_{i}.jpg")
        media_type, data = _encode_image(local_path)
        content.append({"type": "text", "text": f"Photo {i}:"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}})

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        tools=[PHOTO_PICK_TOOL],
        tool_choice={"type": "tool", "name": "submit_best_photo"},
        messages=[{"role": "user", "content": content}],
    )
    tool_use = next(block for block in response.content if block.type == "tool_use")
    chosen_index = tool_use.input["chosen_index"]
    return photos[chosen_index] if chosen_index is not None else None


def resolve_image_field(
    anki_client,
    anthropic_client: anthropic.Anthropic,
    scored: ScoredCandidate,
    target_gloss: str,
    frame_path: str | Path | None,
    work_dir: str | Path,
    unsplash_api_key: str | None = None,
) -> str:
    """The full 3-tier image decision (§10): source frame -> Unsplash ->
    no image. Returns the `Image` field value (`<img src="...">`), or ""
    if no image was used.
    """
    candidate = scored.candidate
    if not scored.is_concrete_and_visualizable:
        return ""  # abstract words, generic nouns: never an image

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    name_part = "".join(c if c.isalnum() else "_" for c in candidate.target_lemma) or "word"

    # Tier 1: source frame
    if frame_path is not None and Path(frame_path).exists():
        if not is_talking_head_or_text(frame_path) and check_image_relevance(
            anthropic_client, frame_path, candidate.target_lemma, target_gloss, candidate.sentence_text
        ):
            filename = anki_client.store_media_file(f"{name_part}_source.jpg", str(frame_path))
            return f'<img src="{filename}">'

    # Tier 2: Unsplash fallback
    api_key = unsplash_api_key or os.environ.get("UNSPLASH_ACCESS_KEY")
    if api_key:
        photos = search_unsplash(candidate.target_lemma, api_key=api_key)
        chosen = pick_best_unsplash_photo(
            anthropic_client, photos, candidate.target_lemma, target_gloss, work_dir
        )
        if chosen is not None:
            local_path = download_image(chosen.url, work_dir / f"{name_part}_unsplash.jpg")
            filename = anki_client.store_media_file(local_path.name, str(local_path))
            return f'<img src="{filename}">'

    # Tier 3: no image
    return ""
