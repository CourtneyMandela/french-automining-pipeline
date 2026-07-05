"""spaCy loading + text-to-sentence parsing.

Kept deliberately thin and separate from the candidate-selection logic in
`french_mining.lingq.candidates`: that module operates on the plain
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

    @property
    def alpha_tokens(self) -> list[Token]:
        return [t for t in self.tokens if t.is_alpha]


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
