import pytest
import responses

from french_mining.youtube.metadata import API_URL, fetch_video_metadata, format_source, VideoMetadata


@responses.activate
def test_fetch_video_metadata_parses_snippet():
    responses.add(
        responses.GET,
        API_URL,
        json={
            "items": [
                {
                    "snippet": {
                        "title": "Apprendre le francais - Episode 12",
                        "channelTitle": "Français Facile",
                        "publishedAt": "2026-06-20T10:00:00Z",
                    }
                }
            ]
        },
        status=200,
    )

    metadata = fetch_video_metadata("abc123", api_key="fake-key")

    assert metadata.video_id == "abc123"
    assert metadata.title == "Apprendre le francais - Episode 12"
    assert metadata.channel_title == "Français Facile"
    assert metadata.published_at == "2026-06-20T10:00:00Z"


@responses.activate
def test_fetch_video_metadata_raises_when_video_not_found():
    responses.add(responses.GET, API_URL, json={"items": []}, status=200)

    with pytest.raises(ValueError, match="abc123"):
        fetch_video_metadata("abc123", api_key="fake-key")


def test_fetch_video_metadata_requires_api_key(monkeypatch):
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="YOUTUBE_API_KEY"):
        fetch_video_metadata("abc123")


def test_format_source_includes_timestamp_when_given():
    metadata = VideoMetadata(
        video_id="abc123", title="Episode 12", channel_title="Chaine", published_at="2026-06-20T10:00:00Z"
    )
    assert format_source(metadata, 125) == "Episode 12 (2:05)"


def test_format_source_omits_timestamp_when_none():
    metadata = VideoMetadata(
        video_id="abc123", title="Episode 12", channel_title="Chaine", published_at="2026-06-20T10:00:00Z"
    )
    assert format_source(metadata, None) == "Episode 12"
