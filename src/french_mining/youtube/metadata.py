"""YouTube Data API metadata lookup (§4): video title/channel, feeding the
§8 Source field ("title + timestamp" for video sources).

Watch-history retrieval needs an OAuth scope Google restricts for new API
clients, so this pipeline takes video IDs directly as input (§4: "user
supplies which videos") rather than trying to surface history automatically.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import requests

API_URL = "https://www.googleapis.com/youtube/v3/videos"


@dataclass
class VideoMetadata:
    video_id: str
    title: str
    channel_title: str
    published_at: str


def fetch_video_metadata(video_id: str, api_key: str | None = None) -> VideoMetadata:
    api_key = api_key or os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError("YOUTUBE_API_KEY is not set. Copy .env.example to .env and fill it in.")

    response = requests.get(
        API_URL,
        params={"part": "snippet", "id": video_id, "key": api_key},
        timeout=15,
    )
    response.raise_for_status()
    items = response.json().get("items", [])
    if not items:
        raise ValueError(f"No video found for ID {video_id!r}")

    snippet = items[0]["snippet"]
    return VideoMetadata(
        video_id=video_id,
        title=snippet["title"],
        channel_title=snippet["channelTitle"],
        published_at=snippet["publishedAt"],
    )


def format_source(metadata: VideoMetadata, timestamp_seconds: float | None) -> str:
    """§8 Source field format for video sources: title + timestamp."""
    if timestamp_seconds is None:
        return metadata.title
    minutes, seconds = divmod(int(timestamp_seconds), 60)
    return f"{metadata.title} ({minutes}:{seconds:02d})"
