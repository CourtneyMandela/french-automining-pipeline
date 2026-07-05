import subprocess
from unittest.mock import MagicMock

import pytest

from french_mining.candidates import Candidate
from french_mining.scoring import ScoredCandidate
from french_mining.youtube.pipeline import attach_source_media, grab_source_frame


def ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


requires_ffmpeg = pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not installed")


@pytest.fixture
def synthetic_video(tmp_path):
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


def make_scored(lemma: str, start_time=None, end_time=None) -> ScoredCandidate:
    candidate = Candidate(
        target_lemma=lemma,
        target_form=lemma,
        target_pos="NOUN",
        sentence_text=f"Une phrase avec {lemma}.",
        other_lemmas=["une", "phrase", "avec"],
        target_confidence=0.0,
        start_time=start_time,
        end_time=end_time,
        video_id="vid123",
    )
    return ScoredCandidate(
        candidate=candidate,
        keep=True,
        i_plus_1_confirmed=True,
        frequency_rank=500,
        unlock_potential=1,
        context_transparency=True,
        interference_risk=None,
        is_concrete_and_visualizable=False,
        reasoning="ok",
        priority_score=1.0,
    )


def test_attach_source_media_returns_blank_field_when_no_timing():
    anki_client = MagicMock()
    scored = make_scored("canape", start_time=None, end_time=None)

    fields = attach_source_media(anki_client, scored, "audio.mp3", "/tmp/work")

    assert fields == {"SentenceAudio": ""}
    anki_client.store_media_file.assert_not_called()


@requires_ffmpeg
def test_attach_source_media_clips_audio_and_stores_it(synthetic_video, tmp_path):
    anki_client = MagicMock()
    anki_client.store_media_file.side_effect = lambda filename, path: filename
    scored = make_scored("canape", start_time=1.0, end_time=3.0)

    fields = attach_source_media(anki_client, scored, synthetic_video, work_dir=tmp_path)

    assert fields["SentenceAudio"] == "[sound:canape_sentence.mp3]"
    assert (tmp_path / "canape_sentence.mp3").exists()
    anki_client.store_media_file.assert_called_once()


@requires_ffmpeg
def test_attach_source_media_uses_anki_returned_filename_for_dedup(synthetic_video, tmp_path):
    anki_client = MagicMock()
    # AnkiConnect dedupes by content hash and may hand back a different name.
    anki_client.store_media_file.return_value = "canape_sentence-1234.mp3"
    scored = make_scored("canape", start_time=1.0, end_time=3.0)

    fields = attach_source_media(anki_client, scored, synthetic_video, work_dir=tmp_path)

    assert fields["SentenceAudio"] == "[sound:canape_sentence-1234.mp3]"


def test_grab_source_frame_returns_none_when_no_timing():
    scored = make_scored("canape", start_time=None, end_time=None)
    assert grab_source_frame(scored, "video.mp4", "/tmp/work") is None


@requires_ffmpeg
def test_grab_source_frame_extracts_a_real_frame(synthetic_video, tmp_path):
    scored = make_scored("hibou", start_time=1.0, end_time=3.0)

    frame_path = grab_source_frame(scored, synthetic_video, tmp_path)

    assert frame_path == tmp_path / "hibou_frame.jpg"
    assert frame_path.exists()
    assert frame_path.stat().st_size > 0


def test_safe_filename_part_sanitizes_special_characters():
    from french_mining.youtube.pipeline import _safe_filename_part

    assert _safe_filename_part("s'apercevoir") == "s_apercevoir"
    assert _safe_filename_part("") == "word"
