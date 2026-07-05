"""Timestamped French transcripts for YouTube videos (§4, §9): downloads
auto-generated/manual subtitles via yt-dlp and parses the WebVTT cues into
plain (start, end, text) segments for sentence-timing alignment.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from french_mining.nlp import TranscriptSegment

TIMESTAMP_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})"
)
INLINE_TAG_RE = re.compile(r"<[^>]+>")


def download_subtitles(video_id: str, output_dir: str | Path, lang: str = "fr") -> Path:
    """Download subtitles (manual if available, else auto-generated) for a
    video via yt-dlp. Requires network access to YouTube.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / f"{video_id}.%(ext)s")
    cmd = [
        "yt-dlp",
        "--write-auto-sub",
        "--write-sub",
        "--sub-lang",
        lang,
        "--sub-format",
        "vtt",
        "--skip-download",
        "-o",
        output_template,
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    matches = sorted(output_dir.glob(f"{video_id}*.vtt"))
    if not matches:
        raise RuntimeError(f"yt-dlp did not produce a subtitle file for video {video_id}")
    return matches[0]


def _parse_timestamp(ts: str) -> float:
    parts = ts.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        hours = "0"
        minutes, seconds = parts
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def parse_vtt(vtt_text: str) -> list[TranscriptSegment]:
    """Parse WebVTT cues into timestamped segments.

    YouTube's auto-generated captions are messy: cues often repeat as a
    rolling partial-text effect and carry inline word-level timing tags
    (e.g. `<00:00:01.234><c> word</c>`). This strips those tags and skips
    any cue whose cleaned text is an exact duplicate of the immediately
    preceding one, rather than trying to reconstruct word-level timing.
    """
    segments: list[TranscriptSegment] = []
    lines = vtt_text.splitlines()
    last_text: str | None = None
    i = 0
    while i < len(lines):
        match = TIMESTAMP_RE.search(lines[i])
        if not match:
            i += 1
            continue

        start = _parse_timestamp(match.group(1))
        end = _parse_timestamp(match.group(2))
        i += 1

        text_lines = []
        while i < len(lines) and lines[i].strip():
            text_lines.append(lines[i])
            i += 1

        text = INLINE_TAG_RE.sub("", " ".join(text_lines)).strip()
        text = re.sub(r"\s+", " ", text)

        if text and text != last_text:
            segments.append(TranscriptSegment(start=start, end=end, text=text))
            last_text = text

    return segments


def load_transcript(path: str | Path) -> list[TranscriptSegment]:
    return parse_vtt(Path(path).read_text(encoding="utf-8"))
