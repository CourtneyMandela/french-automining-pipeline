"""spaCy loading + text-to-sentence parsing.

Kept deliberately thin and separate from the candidate-selection logic in
`french_mining.candidates`: that module operates on the plain
`ParsedSentence`/`Token` dataclasses below, so its i+1 filtering logic is
fully unit-testable without the French spaCy model installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

MODEL_NAME = "fr_core_news_sm"


@dataclass
class Token:
    text: str
    lemma: str
    pos: str
    is_alpha: bool


@dataclass
class ParsedSentence:
    text: str
    tokens: list[Token]
    # Populated only when the source carries timing (YouTube transcripts);
    # left None for plain text sources (LingQ PDFs).
    start_time: float | None = None
    end_time: float | None = None

    @property
    def alpha_tokens(self) -> list[Token]:
        return [t for t in self.tokens if t.is_alpha]


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@lru_cache(maxsize=1)
def load_nlp(model_name: str = MODEL_NAME):
    import spacy

    try:
        return spacy.load(model_name)
    except OSError as exc:
        raise OSError(
            f"spaCy model '{model_name}' isn't installed. Run:\n"
            f"    .venv/bin/python -m spacy download {model_name}\n"
            "(This must be run somewhere with access to GitHub releases — it "
            "may be blocked in some sandboxed environments.)"
        ) from exc


def parse_text(text: str, nlp=None) -> list[ParsedSentence]:
    """Split text into sentences and lemmatize each token via spaCy."""
    nlp = nlp or load_nlp()
    doc = nlp(text)
    sentences = []
    for sent in doc.sents:
        tokens = [
            Token(text=t.text, lemma=t.lemma_.lower(), pos=t.pos_, is_alpha=t.is_alpha)
            for t in sent
        ]
        sentences.append(ParsedSentence(text=sent.text.strip(), tokens=tokens))
    return sentences


def attach_timestamps(
    sentences: list[ParsedSentence], segments: list[TranscriptSegment]
) -> list[ParsedSentence]:
    """Map each sentence to a time range by locating its text within the
    concatenated segment text and finding which segments overlap it.

    Pure and spaCy-free by design: `sentences` just needs to have come from
    running `parse_text` on `" ".join(s.text for s in segments)`, so this is
    fully unit-testable without the French model installed.

    A sentence whose text can't be located (e.g. spaCy normalized
    whitespace in a way that broke an exact match) is left with
    start_time/end_time as None rather than raising — timing is a nice-to-
    have for audio/frame sourcing, not something that should fail the whole
    pipeline over one unmatched sentence.
    """
    concatenated = ""
    segment_spans: list[tuple[int, int, TranscriptSegment]] = []
    offset = 0
    for segment in segments:
        concatenated += segment.text
        segment_spans.append((offset, offset + len(segment.text), segment))
        offset += len(segment.text)
        concatenated += " "
        offset += 1

    cursor = 0
    result = []
    for sentence in sentences:
        char_start = concatenated.find(sentence.text, cursor)
        if char_start == -1:
            char_start = concatenated.find(sentence.text)
        if char_start == -1:
            result.append(sentence)
            continue

        char_end = char_start + len(sentence.text)
        cursor = char_end

        overlapping = [
            seg for span_start, span_end, seg in segment_spans
            if span_start < char_end and span_end > char_start
        ]
        if not overlapping:
            result.append(sentence)
            continue

        result.append(
            ParsedSentence(
                text=sentence.text,
                tokens=sentence.tokens,
                start_time=min(seg.start for seg in overlapping),
                end_time=max(seg.end for seg in overlapping),
            )
        )
    return result


def parse_transcript(segments: list[TranscriptSegment], nlp=None) -> list[ParsedSentence]:
    """Sentence-segment a timestamped transcript and attach a time range to
    each sentence, for audio clipping / frame extraction downstream.
    """
    concatenated_text = " ".join(segment.text for segment in segments)
    sentences = parse_text(concatenated_text, nlp=nlp)
    return attach_timestamps(sentences, segments)
