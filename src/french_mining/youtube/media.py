"""Audio/video download and clipping for YouTube sources (§9, §10):
downloading is via yt-dlp (requires network access to YouTube), clipping
and frame extraction are via ffmpeg (fully local, no network needed).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

# Clip a little before/after the transcript-timed sentence boundary --
# subtitle timing is rarely frame-accurate, and a hard cut exactly on the
# spoken words tends to clip the first/last syllable.
DEFAULT_PADDING_SECONDS = 0.15


def download_audio(video_id: str, output_dir: str | Path) -> Path:
    """Download the best available audio track via yt-dlp. Requires network
    access to YouTube.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / f"{video_id}.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f",
        "bestaudio",
        "--extract-audio",
        "--audio-format",
        "mp3",
        "-o",
        output_template,
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    matches = sorted(output_dir.glob(f"{video_id}.mp3"))
    if not matches:
        raise RuntimeError(f"yt-dlp did not produce an audio file for video {video_id}")
    return matches[0]


def download_video(video_id: str, output_dir: str | Path, max_height: int = 480) -> Path:
    """Download a modest-resolution video (for frame extraction only, not
    playback) via yt-dlp. Requires network access to YouTube.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / f"{video_id}.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f",
        f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]",
        "--merge-output-format",
        "mp4",
        "-o",
        output_template,
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    matches = sorted(output_dir.glob(f"{video_id}.mp4"))
    if not matches:
        raise RuntimeError(f"yt-dlp did not produce a video file for video {video_id}")
    return matches[0]


def clip_audio(
    source_path: str | Path,
    start: float,
    end: float,
    output_path: str | Path,
    padding: float = DEFAULT_PADDING_SECONDS,
) -> Path:
    """Clip [start-padding, end+padding] from `source_path` to `output_path`
    via ffmpeg, re-encoding (not stream-copying) since the cut points aren't
    keyframe-aligned.
    """
    clip_start = max(0.0, start - padding)
    duration = (end + padding) - clip_start
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{clip_start:.3f}",
        "-i",
        str(source_path),
        "-t",
        f"{duration:.3f}",
        "-vn",
        "-acodec",
        "libmp3lame",
        "-ar",
        "44100",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return output_path


def extract_frame(video_path: str | Path, timestamp: float, output_path: str | Path) -> Path:
    """Grab a single video frame at `timestamp` (seconds) via ffmpeg."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return output_path
