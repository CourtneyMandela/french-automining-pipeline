from french_mining.nlp import ParsedSentence, Token, TranscriptSegment, attach_timestamps


def make_sentence(text: str) -> ParsedSentence:
    tokens = [Token(text=w, lemma=w.lower(), pos="X", is_alpha=True) for w in text.split()]
    return ParsedSentence(text=text, tokens=tokens)


def test_attach_timestamps_maps_sentence_within_a_single_segment():
    segments = [TranscriptSegment(start=1.0, end=4.0, text="Bonjour tout le monde.")]
    sentences = [make_sentence("Bonjour tout le monde.")]

    result = attach_timestamps(sentences, segments)

    assert result[0].start_time == 1.0
    assert result[0].end_time == 4.0


def test_attach_timestamps_spans_multiple_segments():
    segments = [
        TranscriptSegment(start=0.0, end=2.0, text="Il fait"),
        TranscriptSegment(start=2.0, end=5.0, text="tres beau aujourd'hui."),
    ]
    # Sentence text as spaCy would produce it once segments are joined with spaces.
    sentences = [make_sentence("Il fait tres beau aujourd'hui.")]

    result = attach_timestamps(sentences, segments)

    assert result[0].start_time == 0.0
    assert result[0].end_time == 5.0


def test_attach_timestamps_handles_multiple_sentences_in_order():
    segments = [
        TranscriptSegment(start=0.0, end=3.0, text="Bonjour tout le monde."),
        TranscriptSegment(start=3.0, end=6.0, text="Comment ca va?"),
    ]
    sentences = [make_sentence("Bonjour tout le monde."), make_sentence("Comment ca va?")]

    result = attach_timestamps(sentences, segments)

    assert result[0].start_time == 0.0
    assert result[0].end_time == 3.0
    assert result[1].start_time == 3.0
    assert result[1].end_time == 6.0


def test_attach_timestamps_leaves_unmatched_sentence_with_none():
    segments = [TranscriptSegment(start=0.0, end=2.0, text="Bonjour tout le monde.")]
    sentences = [make_sentence("Une phrase completement differente.")]

    result = attach_timestamps(sentences, segments)

    assert result[0].start_time is None
    assert result[0].end_time is None


def test_attach_timestamps_handles_repeated_sentence_text_using_cursor():
    # The same short phrase appears twice; each occurrence should map to the
    # segment it actually came from, not both mapping to the first.
    segments = [
        TranscriptSegment(start=0.0, end=1.0, text="Oui."),
        TranscriptSegment(start=5.0, end=6.0, text="Oui."),
    ]
    sentences = [make_sentence("Oui."), make_sentence("Oui.")]

    result = attach_timestamps(sentences, segments)

    assert result[0].start_time == 0.0
    assert result[1].start_time == 5.0
