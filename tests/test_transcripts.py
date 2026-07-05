from unittest.mock import MagicMock, patch

from french_mining.youtube.transcripts import download_subtitles, load_transcript, parse_vtt

SAMPLE_VTT = """WEBVTT
Kind: captions
Language: fr

00:00:01.000 --> 00:00:04.000
Bonjour tout le monde.

00:00:04.500 --> 00:00:07.200
Comment ca va aujourd'hui?

00:00:07.200 --> 00:00:07.200

00:00:08.000 --> 00:00:10.000
Je suis tres content.
"""

AUTO_CAPTION_VTT = """WEBVTT
Kind: captions
Language: fr

00:00:01.000 --> 00:00:03.000
Bonjour<00:00:01.500><c> tout</c><00:00:02.000><c> le</c><00:00:02.500><c> monde</c>

00:00:03.000 --> 00:00:05.000
Bonjour tout le monde

00:00:05.000 --> 00:00:07.000
Comment ca va
"""


def test_parse_vtt_extracts_segments_with_timestamps():
    segments = parse_vtt(SAMPLE_VTT)
    assert len(segments) == 3
    assert segments[0].start == 1.0
    assert segments[0].end == 4.0
    assert segments[0].text == "Bonjour tout le monde."
    assert segments[2].text == "Je suis tres content."


def test_parse_vtt_skips_empty_cues():
    segments = parse_vtt(SAMPLE_VTT)
    # The 07.200 --> 07.200 empty cue should not produce a segment.
    assert all(s.text for s in segments)


def test_parse_vtt_strips_inline_word_timing_tags_and_dedupes_rolling_captions():
    segments = parse_vtt(AUTO_CAPTION_VTT)
    texts = [s.text for s in segments]
    # The first cue's inline tags are stripped; the exact-duplicate second
    # cue (YouTube's rolling-caption effect) is skipped.
    assert texts == ["Bonjour tout le monde", "Comment ca va"]


def test_parse_vtt_handles_mm_ss_timestamps_without_hours():
    vtt = "WEBVTT\n\n00:01.000 --> 00:04.000\nUne phrase courte.\n"
    segments = parse_vtt(vtt)
    assert segments[0].start == 1.0
    assert segments[0].end == 4.0


def test_load_transcript_reads_file(tmp_path):
    vtt_path = tmp_path / "sub.fr.vtt"
    vtt_path.write_text(SAMPLE_VTT, encoding="utf-8")
    segments = load_transcript(vtt_path)
    assert len(segments) == 3


@patch("french_mining.youtube.transcripts.subprocess.run")
def test_download_subtitles_calls_yt_dlp_and_finds_output(mock_run, tmp_path):
    def fake_run(cmd, **kwargs):
        (tmp_path / "abc123.fr.vtt").write_text(SAMPLE_VTT, encoding="utf-8")
        return MagicMock(returncode=0)

    mock_run.side_effect = fake_run

    result = download_subtitles("abc123", tmp_path, lang="fr")

    assert result == tmp_path / "abc123.fr.vtt"
    called_cmd = mock_run.call_args.args[0]
    assert "yt-dlp" in called_cmd
    assert "https://www.youtube.com/watch?v=abc123" in called_cmd


@patch("french_mining.youtube.transcripts.subprocess.run")
def test_download_subtitles_raises_when_no_subtitle_file_produced(mock_run, tmp_path):
    mock_run.return_value = MagicMock(returncode=0)
    try:
        download_subtitles("noSubsVideo", tmp_path)
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "noSubsVideo" in str(exc)
