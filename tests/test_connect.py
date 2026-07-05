import pytest
import responses

from french_mining.anki.connect import AnkiConnectClient, AnkiConnectError

URL = "http://127.0.0.1:8765"


@responses.activate
def test_invoke_returns_result_on_success():
    responses.add(
        responses.POST,
        URL,
        json={"result": ["Deck 1", "French::Mining"], "error": None},
        status=200,
    )
    client = AnkiConnectClient(url=URL)
    assert client.deck_names() == ["Deck 1", "French::Mining"]


@responses.activate
def test_invoke_raises_on_ankiconnect_error():
    responses.add(
        responses.POST,
        URL,
        json={"result": None, "error": "deck was not found"},
        status=200,
    )
    client = AnkiConnectClient(url=URL)
    with pytest.raises(AnkiConnectError, match="deck was not found"):
        client.deck_names()


@responses.activate
def test_invoke_raises_on_malformed_response():
    responses.add(responses.POST, URL, json={"unexpected": "shape"}, status=200)
    client = AnkiConnectClient(url=URL)
    with pytest.raises(AnkiConnectError, match="Unexpected AnkiConnect response"):
        client.deck_names()


def test_invoke_raises_helpful_error_when_anki_unreachable():
    # No responses registered -> connection refused, since Anki isn't running.
    client = AnkiConnectClient(url="http://127.0.0.1:1")
    with pytest.raises(AnkiConnectError, match="Could not reach AnkiConnect"):
        client.deck_names()


@responses.activate
def test_notes_info_empty_list_short_circuits_without_request():
    client = AnkiConnectClient(url=URL)
    assert client.notes_info([]) == []
    assert client.cards_info([]) == []
    assert len(responses.calls) == 0


@responses.activate
def test_add_note_sends_expected_payload():
    def request_callback(request):
        import json

        body = json.loads(request.body)
        assert body["action"] == "addNote"
        assert body["params"]["note"]["deckName"] == "French::Mining"
        return (200, {}, '{"result": 12345, "error": null}')

    responses.add_callback(responses.POST, URL, callback=request_callback)
    client = AnkiConnectClient(url=URL)
    note_id = client.add_note(
        {
            "deckName": "French::Mining",
            "modelName": "FrenchSentence",
            "fields": {"TargetWord": "apercevoir"},
        }
    )
    assert note_id == 12345


@responses.activate
def test_store_media_file_sends_path_and_returns_filename():
    def request_callback(request):
        import json

        body = json.loads(request.body)
        assert body["action"] == "storeMediaFile"
        assert body["params"]["filename"] == "apercevoir_sentence.mp3"
        assert body["params"]["path"] == "/tmp/clip.mp3"
        return (200, {}, '{"result": "apercevoir_sentence.mp3", "error": null}')

    responses.add_callback(responses.POST, URL, callback=request_callback)
    client = AnkiConnectClient(url=URL)
    result = client.store_media_file("apercevoir_sentence.mp3", "/tmp/clip.mp3")
    assert result == "apercevoir_sentence.mp3"
