from dataclasses import dataclass

from french_mining.youtube.whisper_transcribe import transcribe_with_whisper


@dataclass
class FakeWhisperSegment:
    start: float
    end: float
    text: str


class FakeWhisperModel:
    def __init__(self, segments):
        self._segments = segments
        self.calls = []

    def transcribe(self, path, language=None):
        self.calls.append((path, language))
        return iter(self._segments), {"language": language}


def test_transcribe_with_whisper_converts_segments():
    fake_model = FakeWhisperModel(
        [
            FakeWhisperSegment(start=0.0, end=2.0, text=" Bonjour tout le monde. "),
            FakeWhisperSegment(start=2.0, end=4.5, text="Comment ca va?"),
        ]
    )

    segments = transcribe_with_whisper("audio.mp3", language="fr", model=fake_model)

    assert len(segments) == 2
    assert segments[0].start == 0.0
    assert segments[0].end == 2.0
    assert segments[0].text == "Bonjour tout le monde."  # stripped
    assert segments[1].text == "Comment ca va?"


def test_transcribe_with_whisper_drops_empty_segments():
    fake_model = FakeWhisperModel(
        [
            FakeWhisperSegment(start=0.0, end=1.0, text="   "),
            FakeWhisperSegment(start=1.0, end=2.0, text="Oui."),
        ]
    )

    segments = transcribe_with_whisper("audio.mp3", model=fake_model)

    assert len(segments) == 1
    assert segments[0].text == "Oui."


def test_transcribe_with_whisper_passes_path_and_language_through():
    fake_model = FakeWhisperModel([])
    transcribe_with_whisper("audio.mp3", language="fr", model=fake_model)
    assert fake_model.calls == [("audio.mp3", "fr")]


def test_transcribe_with_whisper_handles_no_segments():
    fake_model = FakeWhisperModel([])
    assert transcribe_with_whisper("audio.mp3", model=fake_model) == []
