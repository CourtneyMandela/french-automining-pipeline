# French Sentence-Mining Pipeline

Automated pipeline that turns French immersion content (YouTube, LingQ PDFs)
into scaffolded, intelligently ordered Anki cards, with zero manual mining
effort. See the full build spec in the project brief handed to Claude Code
(architecture, card design, cost philosophy, build order).

**Core principle:** FSRS handles review timing; Claude handles curation and
content quality. New cards sit in a queue ordered by an integer `due` value
until their first review — Claude reorders that queue, then steps back the
moment FSRS takes over.

## Status

Build-order step 1 (§13.1 of the spec) is done: the AnkiConnect read layer
and the vocabulary state model.

- [x] AnkiConnect client (`french_mining.anki.connect`)
- [x] Frequency floor for pre-Anki vocabulary (`french_mining.frequency`)
- [x] Gradient vocabulary state model (`french_mining.anki.vocab_state`)
- [ ] Note type + template + write path (§8, build-order step 2)
- [ ] Local LingQ-PDF pipeline (spaCy extraction, i+1 pre-filter)
- [ ] API scoring layer (Claude/Sonnet)
- [ ] Card generation (morphology, second examples, translations)
- [ ] Queue ordering via AnkiConnect `due` rewrites
- [ ] YouTube pipeline (transcripts, audio clipping, frame extraction)
- [ ] Image tier logic
- [ ] Collocation card type
- [ ] Monthly hygiene audit

## Important: this must run on the same machine as Anki

AnkiConnect listens on `http://127.0.0.1:8765` and only accepts connections
from the machine Anki is running on. Anki must be open with the
[AnkiConnect](https://foosoft.net/projects/anki-connect/) add-on installed.
If you're driving this from a cloud/remote Claude Code session, the pipeline
code can be developed and unit-tested there (all AnkiConnect calls are
mocked in `tests/`), but it must actually run against your Anki desktop app
locally to do anything real.

## Setup

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv -e ".[dev]"
cp .env.example .env   # fill in ANTHROPIC_API_KEY etc. as later stages need them
```

Run the tests (no live Anki required — AnkiConnect is mocked):

```bash
.venv/bin/pytest
```

## Vocabulary state model (§5)

`VocabularyState.build(client, model_names=[...], tested_field="TargetWord")`
reads every note of the given Anki note type(s), takes the *tested* field
(the lemma, never sentence-context words), and combines each note's card
review state into a 0–1 confidence score:

- High FSRS stability + high retrievability -> confidence near 1.0.
- Still in the learning/relearning phase -> heavily discounted (fragile).
- Repeatedly lapsed (>= 3 lapses) -> heavily discounted (practically unknown).
- No card at all, but the lemma is inside the top-500 French frequency list
  (`french_mining/data/french_frequency_top500.csv`) and not on the manual
  exclusion list -> assumed known (pre-Anki vocabulary floor).

If your Anki/AnkiConnect version doesn't expose FSRS memory state
(`memoryState.stability`) on `cardsInfo`, the model falls back to an
interval-based proxy automatically — no configuration needed.

The bundled frequency list is a curated approximation of the top ~500 most
frequent French lemmas, not sourced from a canonical corpus-derived list
(e.g. Lexique383). It's good enough to act as a floor; swap in a better list
by passing `frequency_list=FrequencyList.load("path/to/your.csv")` (same
`rank,lemma` CSV format) if you want more precision.

To override words that rank in the top 500 but you *haven't* actually
acquired, create an exclusion file (one lemma per line, `#` comments
allowed) and pass its path as `exclusion_list_path`.

## Project layout

```
src/french_mining/
  anki/
    connect.py      # AnkiConnect JSON-RPC client
    vocab_state.py  # gradient known-word model (§5)
  data/
    french_frequency_top500.csv
  frequency.py       # frequency floor + exclusion list loading
tests/               # all AnkiConnect calls mocked, no live Anki needed
```
