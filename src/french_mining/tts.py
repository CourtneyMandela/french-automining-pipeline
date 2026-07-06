"""Text-to-speech for the audio fields clipped source video can't supply
(§9): `WordAudio`/`ChunkAudio` (the target word/chunk spoken alone -- there's
no isolated-word moment in the source video to clip) and
`SecondExampleAudio` (a sentence Claude wrote, so no source clip exists for
it at all). `SentenceAudio` keeps clipping real speech from the source video
as its primary path (`youtube.pipeline.attach_source_media`) wherever timing
is available; this module only fills in what clipping structurally can't.

Opt-in via `ELEVENLABS_API_KEY` (see `.env.example`) -- with no key
configured, these fields stay blank, the same fail-open pattern
`images.resolve_image_field` uses for the Unsplash tier.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import requests

ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

# "Charlotte" -- a premade ElevenLabs voice that reads French well under the
# multilingual model. Override with ELEVENLABS_VOICE_ID for a different
# voice from your own ElevenLabs account.
DEFAULT_VOICE_ID = "XB0fDUnXU5powFXDhCwa"
DEFAULT_MODEL_ID = "eleven_multilingual_v2"

_TAG_RE = re.compile(r"<[^>]+>")


def strip_html(text: str) -> str:
    """Remove HTML tags (e.g. the `<span class="target">` highlight wrapper
    card generation wraps around the target word/chunk) so TTS speaks plain
    text, not markup.
    """
    return _TAG_RE.sub("", text)


def synthesize_speech(
    text: str,
    output_path: str | Path,
    api_key: str | None = None,
    voice_id: str | None = None,
) -> Path:
    """Synthesize `text` to an MP3 via the ElevenLabs API.

    Raises if the key is missing or the request fails -- callers that want
    the fail-open "no TTS configured" behavior should check for the API key
    themselves before calling this (see `resolve_tts_fields`).
    """
    api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is not set. Copy .env.example to .env and fill it in.")
    voice_id = voice_id or os.environ.get("ELEVENLABS_VOICE_ID") or DEFAULT_VOICE_ID

    response = requests.post(
        ELEVENLABS_TTS_URL.format(voice_id=voice_id),
        headers={"xi-api-key": api_key, "Content-Type": "application/json"},
        json={"text": text, "model_id": DEFAULT_MODEL_ID},
        timeout=30,
    )
    response.raise_for_status()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)
    return output_path


def resolve_tts_fields(
    anki_client,
    target_text: str,
    second_example_html: str,
    work_dir: str | Path,
    name_part: str,
    target_field_name: str = "WordAudio",
    api_key: str | None = None,
) -> dict[str, str]:
    """Generate and store the target word/chunk + second-example audio via
    ElevenLabs, returning `{target_field_name: ..., "SecondExampleAudio":
    ...}` as Anki `[sound:...]` field values.

    Both come back as `""` (both note types list them in `OPTIONAL_FIELDS`)
    if no `ELEVENLABS_API_KEY` is configured -- same tier-skip behavior as
    the Unsplash image fallback when `UNSPLASH_ACCESS_KEY` is unset.
    """
    api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        return {target_field_name: "", "SecondExampleAudio": ""}

    work_dir = Path(work_dir)

    word_path = work_dir / f"{name_part}_word.mp3"
    synthesize_speech(target_text, word_path, api_key=api_key)
    word_filename = anki_client.store_media_file(word_path.name, str(word_path))

    example_path = work_dir / f"{name_part}_second_example.mp3"
    synthesize_speech(strip_html(second_example_html), example_path, api_key=api_key)
    example_filename = anki_client.store_media_file(example_path.name, str(example_path))

    return {
        target_field_name: f"[sound:{word_filename}]",
        "SecondExampleAudio": f"[sound:{example_filename}]",
    }
