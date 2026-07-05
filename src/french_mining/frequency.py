"""Frequency floor for pre-Anki vocabulary (§5).

Words known before the Anki system began (être, avoir, basic adjectives...)
won't have a card and would otherwise look "unknown" to the vocabulary state
model. Anything inside the frequency floor is assumed known unless it's on
the manual exclusion list.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

DEFAULT_FREQUENCY_CSV = resources.files("french_mining.data") / "french_frequency_top500.csv"
DEFAULT_EXCLUSION_LIST: frozenset[str] = frozenset()


@dataclass
class FrequencyList:
    """Maps lemma -> frequency rank (1 = most frequent)."""

    ranks: dict[str, int] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "FrequencyList":
        source = Path(path) if path else DEFAULT_FREQUENCY_CSV
        ranks: dict[str, int] = {}
        with open(source, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                lemma = row["lemma"].strip().lower()
                rank = int(row["rank"])
                # keep the first (best/lowest) rank if a lemma appears twice
                if lemma not in ranks or rank < ranks[lemma]:
                    ranks[lemma] = rank
        return cls(ranks=ranks)

    def rank(self, lemma: str) -> int | None:
        return self.ranks.get(lemma.strip().lower())

    def is_within_floor(self, lemma: str, floor: int = 500) -> bool:
        rank = self.rank(lemma)
        return rank is not None and rank <= floor


def load_exclusion_list(path: str | Path | None = None) -> frozenset[str]:
    """Manual override list: lemmas to NEVER auto-assume known via the frequency
    floor, even though they rank in the top N. One lemma per line, '#' comments
    allowed. Missing file just means an empty exclusion list.
    """
    if path is None:
        return DEFAULT_EXCLUSION_LIST
    p = Path(path)
    if not p.exists():
        return DEFAULT_EXCLUSION_LIST
    lemmas = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        lemmas.add(line.lower())
    return frozenset(lemmas)
