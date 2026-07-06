import os
from unittest.mock import MagicMock

import pytest
import responses

from french_mining.tts import (
    DEFAULT_MODEL_ID,
    DEFAULT_VOICE_ID,
    ELEVENLABS_TTS_URL,
    resolve_tts_fields,
    strip_html,
    synthesize_speech,
)


# -- strip_html ----------------------------------------------------------


def test_strip_html_removes_highlight_span():
    assert strip_html('Il <span class="target">mange</span> une pomme.') == "Il mange une pomme."


def test_strip_html_leaves_plain_text_untouched():
    assert strip_html("Il mange une pomme.") == "Il mange une pomme."


# -- synthesize_speech -----------------------------------------------------


@responses.activate
def test_synthesize_speech_writes_response_bytes(tmp_path):
    url = ELEVENLABS_TTS_URL.format(voice_id=DEFAULT_VOICE_ID)
    responses.add(responses.POST, url, body=b"fake-mp3-bytes", status=200)

    output_path = tmp_path / "word.mp3"
    result = synthesize_speech("pomme", output_path, api_key="test-key")

    assert result == output_path
    assert output_path.read_bytes() == b"fake-mp3-bytes"

    request = responses.calls[0].request
    assert request.headers["xi-api-key"] == "test-key"
    import json

    assert json.loads(request.body) == {"text": "pomme", "model_id": DEFAULT_MODEL_ID}


@responses.activate
def test_synthesize_speech_uses_custom_voice_id(tmp_path):
    url = ELEVENLABS_TTS_URL.format(voice_id="custom-voice")
    responses.add(responses.POST, url, body=b"bytes", status=200)

    synthesize_speech("pomme", tmp_path / "word.mp3", api_key="test-key", voice_id="custom-voice")

    assert responses.calls[0].request.url.startswith(url)


def test_synthesize_speech_raises_without_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        synthesize_speech("pomme", tmp_path / "word.mp3", api_key=None)


@responses.activate
def test_synthesize_speech_raises_on_http_error(tmp_path):
    url = ELEVENLABS_TTS_URL.format(voice_id=DEFAULT_VOICE_ID)
    responses.add(responses.POST, url, status=401)

    with pytest.raises(Exception):
        synthesize_speech("pomme", tmp_path / "word.mp3", api_key="bad-key")


# -- resolve_tts_fields -----------------------------------------------------


def test_resolve_tts_fields_blank_without_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    anki_client = MagicMock()

    fields = resolve_tts_fields(
        anki_client, "pomme", "Il mange une pomme.", tmp_path, "pomme", api_key=None
    )

    assert fields == {"WordAudio": "", "SecondExampleAudio": ""}
    anki_client.store_media_file.assert_not_called()


@responses.activate
def test_resolve_tts_fields_stores_both_clips_and_returns_sound_tags(tmp_path):
    url = ELEVENLABS_TTS_URL.format(voice_id=DEFAULT_VOICE_ID)
    responses.add(responses.POST, url, body=b"bytes", status=200)

    anki_client = MagicMock()
    anki_client.store_media_file.side_effect = ["pomme_word.mp3", "pomme_second_example.mp3"]

    fields = resolve_tts_fields(
        anki_client,
        "pomme",
        'Il <span class="target">mange</span> une pomme.',
        tmp_path,
        "pomme",
        api_key="test-key",
    )

    assert fields == {
        "WordAudio": "[sound:pomme_word.mp3]",
        "SecondExampleAudio": "[sound:pomme_second_example.mp3]",
    }
    assert anki_client.store_media_file.call_count == 2


@responses.activate
def test_resolve_tts_fields_uses_custom_target_field_name_for_collocations(tmp_path):
    url = ELEVENLABS_TTS_URL.format(voice_id=DEFAULT_VOICE_ID)
    responses.add(responses.POST, url, body=b"bytes", status=200)

    anki_client = MagicMock()
    anki_client.store_media_file.side_effect = ["chunk.mp3", "example.mp3"]

    fields = resolve_tts_fields(
        anki_client,
        "s'apercevoir de",
        "Elle s'aperçoit de son erreur.",
        tmp_path,
        "s_apercevoir_de",
        target_field_name="ChunkAudio",
        api_key="test-key",
    )

    assert fields == {"ChunkAudio": "[sound:chunk.mp3]", "SecondExampleAudio": "[sound:example.mp3]"}
