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

Build-order steps 1–2 (§13.1–2 of the spec) are done: the AnkiConnect read
layer, the vocabulary state model, and the note type + write path.

- [x] AnkiConnect client (`french_mining.anki.connect`)
- [x] Frequency floor for pre-Anki vocabulary (`french_mining.frequency`)
- [x] Gradient vocabulary state model (`french_mining.anki.vocab_state`)
- [x] Note type + template + write path (`french_mining.anki.note_type`, §8)
- [x] Local LingQ-PDF pipeline (spaCy extraction, i+1 pre-filter, §6 Stage 1)
- [x] API scoring layer (Claude/Sonnet, §6 Stage 2)
- [x] Card generation (morphology, second examples, translations, §8)
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

## Note type + write path (§8)

`french_mining.anki.note_type` defines the "French Sentence Mining" note
type (one card per note) with the exact field list and front/back template
order from §8: front is just the highlighted sentence + autoplayed word
audio; back reveals gloss -> full translation -> sentence audio ->
`TargetWordForm -> TargetWord` -> morphology note -> second example with
audio -> source -> frequency rank -> image (if present).

`add_card(client, fields)` creates the `French::Mining` deck and the note
type if they don't already exist, then adds one note. To verify this
end-to-end against your real Anki app:

```bash
.venv/bin/python scripts/create_placeholder_card.py
```

This adds one real card with placeholder content so you can check the
front/back rendering directly in Anki. Audio fields (`WordAudio`,
`SentenceAudio`, `SecondExampleAudio`) are optional for now since audio
sourcing (§9) hasn't been built yet — cards will render silently until then.

## Local LingQ-PDF pipeline (§6 Stage 1)

`french_mining.lingq.pdf_extract` pulls raw text out of the weekly PDF drop
(`extract_text`, `extract_text_from_folder`). `french_mining.nlp.parse_text`
lemmatizes it with spaCy's French model into `ParsedSentence`/`Token`
objects. `french_mining.lingq.candidates` then does the free, mechanical
i+1 pre-filter (§6 Stage 1) that's supposed to remove ~90% of candidates
before any API tokens are spent:

- A sentence survives only if **exactly one** word's gradient confidence
  (from `VocabularyState`) falls below `DEFAULT_KNOWN_THRESHOLD` (0.6) — one
  true unknown, no fragile extras. Two weak words means i+2, not i+1, and
  gets dropped.
- Sentences outside a sane token-count band are dropped (too short to give
  context, too long to be worth scoring).
- `select_best_sentences` caps how many sentences per target word move on,
  locally preferring sentence lengths close to `IDEAL_TOKEN_COUNT` — genuine
  quality judgment (context transparency, recency, unlock potential) is
  Stage 2's job (the API layer, not yet built).

The i+1 filtering logic in `candidates.py` is deliberately spaCy-free (it
operates on plain dataclasses) so it's fully unit-tested without the French
model installed. `nlp.py` and `pdf_extract.py` are thin wrappers around
spaCy/pypdf that do need real dependencies to run for real.

**Note on this session:** the French spaCy model (`fr_core_news_sm`) is
distributed via GitHub releases, which this sandboxed session's egress
policy blocks — I couldn't download or exercise it here. On your machine:

```bash
.venv/bin/python -m spacy download fr_core_news_sm
.venv/bin/python scripts/mine_lingq_pdf.py path/to/reading.pdf
```

That script runs the full local Stage 1 chain (PDF -> spaCy -> vocab state
via AnkiConnect -> i+1 filter -> best-sentence selection) and prints the
surviving candidates, so you can sanity-check it against a real reading
before anything gets API-scored.

## API scoring layer (§6 Stage 2)

`french_mining.scoring.score_candidates` sends Stage-1 survivors to Claude
(Sonnet by default) in batches, via a forced tool call (`submit_candidate_scores`)
so the response is structured, not free text. Each candidate comes back with:

- `keep` — a qualitative override for cases the mechanical i+1 filter can't
  catch (idioms that read as i+0/i+2 despite the word count, awkward
  phrasing, ambiguous target words).
- `i_plus_1_confirmed`, `unlock_potential`, `context_transparency` — the
  quality dimensions from §6: the full backlog word list is included in the
  prompt so the model can estimate how many other candidates a given word
  would unlock.
- `interference_risk` — a conflicting lemma from the active learning queue
  (near-synonyms/confusable forms), or null. This *delays* a candidate via a
  steep priority discount (`INTERFERENCE_PRIORITY_DISCOUNT`) rather than
  dropping it — the spec is explicit that interference means "wait," not
  "never."

`_priority_score` combines frequency rank, unlock potential, context
transparency, and recency (`Candidate.days_since_encountered`, currently
unset by the LingQ pipeline since PDFs don't carry per-sentence timestamps —
this becomes meaningful once the YouTube pipeline, with real watch dates,
is built) into the composite score `keep_and_rank` sorts on. Actual queue
reordering via AnkiConnect `due` rewrites is a separate later stage (§2, §6
build-order step 6) — this module only scores and ranks.

Requires `ANTHROPIC_API_KEY` in `.env`. `scripts/mine_lingq_pdf.py` runs
Stage 2 automatically when the key is set, otherwise it just prints Stage 1
survivors. I haven't been able to exercise this against the real API in this
session (no key configured here) — the test suite covers it with the
Anthropic client's `messages.create` mocked; real verification needs your
own API key.

## Card generation (§8, build-order step 5)

`french_mining.generation.generate_card_content` takes `ScoredCandidate`s
(post Stage 2) and asks Claude — one forced tool call (`submit_card_content`)
per batch — for the gloss, contextual definition, morphology note, full
sentence translation, and a second example in a different context. It then
assembles the complete field dict `note_type.py` expects:

- `SentenceText` is highlighted locally (`highlight_target`), not by the
  model — the exact surface form came straight out of spaCy tokenization of
  this sentence, so a plain string replace is more reliable than asking the
  model to reproduce it verbatim.
- `FrequencyRank`, `Source`, `TargetWord`/`TargetWordForm` are carried over
  from earlier stages, not regenerated.
- `DateMined` defaults to today; audio/image fields stay blank (§9/§10 are
  later stages) — `note_type.OPTIONAL_FIELDS` already allows this.

`generate_and_write_cards(anki_client, anthropic_client, scored_candidates)`
chains generation straight into the `add_card` write path from step 2,
producing real, complete cards.

`scripts/mine_lingq_pdf.py --write N` now runs the full chain — PDF extract
-> spaCy -> vocab state -> Stage 1 filter -> Stage 2 scoring -> generation ->
write — capped at the top N ranked candidates so a run can't silently
generate an unbounded number of cards (and API calls) in one go. Omit
`--write` (or leave it at the default 0) to just see the ranked Stage 2
output without writing anything.

Not exercised against the real Anthropic API in this session — covered by
mocking `messages.create` in tests, same as the scoring layer.

## Project layout

```
src/french_mining/
  anki/
    connect.py      # AnkiConnect JSON-RPC client
    vocab_state.py  # gradient known-word model (§5)
    note_type.py    # note type, templates, write path (§8)
  lingq/
    pdf_extract.py  # PDF -> raw text (pypdf)
    candidates.py   # i+1 pre-filter + best-sentence selection (§6 Stage 1)
  nlp.py             # spaCy loading + text -> ParsedSentence/Token
  scoring.py         # Claude/Sonnet API scoring + ranking (§6 Stage 2)
  generation.py      # Claude-generated card content + write path (§8 step 5)
  data/
    french_frequency_top500.csv
  frequency.py       # frequency floor + exclusion list loading
scripts/
  create_placeholder_card.py  # run locally to verify one real card end-to-end
  mine_lingq_pdf.py           # run locally: full pipeline (extract -> filter -> score -> generate -> write) on a real PDF
tests/               # all AnkiConnect/Anthropic calls + spaCy model mocked/avoided
```
