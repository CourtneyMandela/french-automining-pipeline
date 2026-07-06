import subprocess
from unittest.mock import MagicMock, patch

import pytest

from french_mining.youtube.media import (
    clip_audio,
    concat_audio_spans,
    download_audio,
    download_video,
    extract_frame,
)


def ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


requires_ffmpeg = pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not installed")


@pytest.fixture
def synthetic_video(tmp_path):
    """A short synthetic video with a tone, generated locally via ffmpeg's
    lavfi test sources -- no network needed, so this genuinely exercises
    clip_audio/extract_frame against a real ffmpeg binary.
    """
    path = tmp_path / "synthetic.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=6:size=320x240:rate=10",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=6",
            "-shortest",
            str(path),
        ],
        capture_output=True,
        check=True,
    )
    return path


def probe_duration(path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    return float(result.stdout.strip())


@requires_ffmpeg
def test_clip_audio_produces_a_file_of_roughly_the_expected_duration(synthetic_video, tmp_path):
    output = tmp_path / "clip.mp3"
    clip_audio(synthetic_video, start=1.0, end=3.0, output_path=output, padding=0.1)

    assert output.exists()
    duration = probe_duration(output)
    # requested [0.9, 3.1] -> 2.2s, allow encoder slop
    assert 1.8 < duration < 2.6


@requires_ffmpeg
def test_clip_audio_clamps_start_to_zero(synthetic_video, tmp_path):
    output = tmp_path / "clip_start.mp3"
    clip_audio(synthetic_video, start=0.05, end=1.0, output_path=output, padding=0.5)

    assert output.exists()
    duration = probe_duration(output)
    # start clamped to 0 -> duration should be end+padding, not (end+padding)-(start-padding)
    assert duration < 2.0


@requires_ffmpeg
def test_extract_frame_produces_an_image_file(synthetic_video, tmp_path):
    output = tmp_path / "frame.jpg"
    extract_frame(synthetic_video, timestamp=2.0, output_path=output)

    assert output.exists()
    assert output.stat().st_size > 0


@requires_ffmpeg
def test_clip_audio_creates_parent_directories(synthetic_video, tmp_path):
    output = tmp_path / "nested" / "dir" / "clip.mp3"
    clip_audio(synthetic_video, start=0.0, end=1.0, output_path=output)
    assert output.exists()


@requires_ffmpeg
def test_concat_audio_spans_duration_is_sum_of_spans(synthetic_video, tmp_path):
    output = tmp_path / "condensed.mp3"
    # Two spans from a 6s source: [0.5,1.5] and [3.0,4.0] -> ~2s of kept audio
    # (the [1.5, 3.0] gap is dropped). Padding adds a little to each span.
    concat_audio_spans(synthetic_video, [(0.5, 1.5), (3.0, 4.0)], output, padding=0.1)

    assert output.exists()
    duration = probe_duration(output)
    # 2 spans * (1.0 + 2*0.1 padding) = 2.4s nominal; allow encoder slop.
    assert 2.0 < duration < 2.9


@requires_ffmpeg
def test_concat_audio_spans_single_span(synthetic_video, tmp_path):
    output = tmp_path / "one.mp3"
    concat_audio_spans(synthetic_video, [(1.0, 3.0)], output, padding=0.0)

    assert output.exists()
    duration = probe_duration(output)
    assert 1.7 < duration < 2.3


@requires_ffmpeg
def test_concat_audio_spans_creates_parent_directories(synthetic_video, tmp_path):
    output = tmp_path / "nested" / "condensed.mp3"
    concat_audio_spans(synthetic_video, [(0.0, 1.0)], output)
    assert output.exists()


def test_concat_audio_spans_rejects_empty_spans(tmp_path):
    with pytest.raises(ValueError, match="at least one span"):
        concat_audio_spans("source.mp3", [], tmp_path / "out.mp3")


@patch("french_mining.youtube.media.subprocess.run")
def test_download_audio_calls_yt_dlp_and_locates_output(mock_run, tmp_path):
    def fake_run(cmd, **kwargs):
        (tmp_path / "vid123.mp3").write_bytes(b"fake mp3 data")
        return MagicMock(returncode=0)

    mock_run.side_effect = fake_run

    result = download_audio("vid123", tmp_path)

    assert result == tmp_path / "vid123.mp3"
    called_cmd = mock_run.call_args.args[0]
    assert "yt-dlp" in called_cmd
    assert "https://www.youtube.com/watch?v=vid123" in called_cmd


@patch("french_mining.youtube.media.subprocess.run")
def test_download_audio_skips_redownload_when_file_already_exists(mock_run, tmp_path):
    existing = tmp_path / "vid123.mp3"
    existing.write_bytes(b"already downloaded")

    result = download_audio("vid123", tmp_path)

    assert result == existing
    mock_run.assert_not_called()


@patch("french_mining.youtube.media.subprocess.run")
def test_download_audio_raises_when_no_file_produced(mock_run, tmp_path):
    mock_run.return_value = MagicMock(returncode=0)
    with pytest.raises(RuntimeError, match="vid123"):
        download_audio("vid123", tmp_path)


@patch("french_mining.youtube.media.subprocess.run")
def test_download_video_calls_yt_dlp_and_locates_output(mock_run, tmp_path):
    def fake_run(cmd, **kwargs):
        (tmp_path / "vid456.mp4").write_bytes(b"fake mp4 data")
        return MagicMock(returncode=0)

    mock_run.side_effect = fake_run

    result = download_video("vid456", tmp_path, max_height=480)

    assert result == tmp_path / "vid456.mp4"
    called_cmd = mock_run.call_args.args[0]
    assert any("480" in part for part in called_cmd)
