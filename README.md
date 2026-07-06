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
- [x] Queue ordering via AnkiConnect `due` rewrites (§2)
- [x] YouTube pipeline (transcripts, audio clipping, frame extraction)
- [x] Image tier logic (talking-head filter, Unsplash fallback, §10)
- [x] Collocation card type (§7)
- [x] Monthly hygiene audit (§11)

All ten build-order steps from the spec are now implemented.

Beyond the original spec:

- [x] LingQ API sync — read on LingQ, looked-up words auto-flow into the
  pipeline with no PDF drop (`french_mining.lingq.api`, `scripts/mine_lingq_api.py`)
- [x] Condensed audio — comprehensibility-filtered listening material from
  watched videos (`french_mining.condensed_audio`, `scripts/build_condensed_audio.py`)
- [x] Whisper transcription fallback for videos with no captions
  (`french_mining.youtube.whisper_transcribe`, `youtube.pipeline.get_transcript`)

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

The YouTube pipeline also needs `ffmpeg` installed as a system binary (not a
pip package) — `apt install ffmpeg` / `brew install ffmpeg` / equivalent.

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
`SentenceAudio`, `SecondExampleAudio`) stay optional — `SentenceAudio` needs
a clipped source video (YouTube pipeline only), and `WordAudio`/
`SecondExampleAudio` need `ELEVENLABS_API_KEY` configured (see "Word/second-
example audio" below) — cards render silently without them.

## Local LingQ-PDF pipeline (§6 Stage 1)

`french_mining.lingq.pdf_extract` pulls raw text out of the weekly PDF drop
(`extract_text`, `extract_text_from_folder`). `french_mining.nlp.parse_text`
lemmatizes it with spaCy's French model into `ParsedSentence`/`Token`
objects. `french_mining.candidates` then does the free, mechanical i+1
pre-filter (§6 Stage 1) that's supposed to remove ~90% of candidates
before any API tokens are spent — this module is shared with the YouTube
pipeline (below), since none of it is actually LingQ-specific:

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

## LingQ API sync — "read on LingQ, cards appear" (extends §4)

The hands-off alternative to the PDF drop: read on LingQ, look up the words
you don't know (create LingQs), and `scripts/mine_lingq_api.py` syncs them
into Anki automatically. `french_mining.lingq.api` pulls your LingQs from
the LingQ v3 API (`GET /api/v3/{lang}/cards/`, `Authorization: Token <key>`).
Each LingQ already carries the sentence you met it in (`fragment`), so it's
a **stronger** candidate source than the PDF path — you've directly told the
pipeline which word you didn't know, and given it the context.

Crucially, LingQ stays *only a candidate source* — Anki's FSRS state (§5)
remains the source of truth for what's known, exactly as the original §4
argued. So `build_lingq_candidates`:

- **skips a looked-up word you already have a solid Anki card for** (no
  duplicates — the same gradient-confidence check the whole pipeline uses);
- **keeps a LingQ only if its fragment is genuine i+1**: the looked-up word
  must be the sole unknown. If you looked up two words in one sentence, that
  fragment is i+2 and waits in the backlog until each word turns up in a
  cleaner single-unknown sentence — which is the pedagogically correct thing
  to do, not a limitation;
- **routes multi-word LingQs (phrases) to the collocation card type** (§7)
  automatically.

`locate_term` aligns the LingQ `term` to the spaCy-tokenized fragment,
tolerating French elisions/apostrophes (matches on the whitespace-stripped
concatenation but only at token boundaries, so `de` can't match inside
`monde`), with a single-word lemma fallback when LingQ stored a base form.

```bash
# in .env: LINGQ_API_KEY (from https://www.lingq.com/accounts/apikey/),
#          LINGQ_LANGUAGE_CODE=fr, plus ANTHROPIC_API_KEY
.venv/bin/python scripts/mine_lingq_api.py            # dry run: show what would be mined
.venv/bin/python scripts/mine_lingq_api.py --write 20  # generate + write + reorder
```

### Making it actually automatic

"Auto-sync" means a **scheduled job on the machine where Anki runs** — this
pipeline talks to Anki over `localhost:8765`, so it can't run in the cloud.
Schedule the `--write` command however your OS does it, e.g. weekly:

```cron
# crontab -e  (macOS/Linux) — Sundays at 18:00, Anki must be open
0 18 * * 0  cd /path/to/french-automining-pipeline && .venv/bin/python scripts/mine_lingq_api.py --write 20 >> lingq_sync.log 2>&1
```

(macOS launchd or Windows Task Scheduler work equally well.) Because the
Anki dedup + i+1 filter + the daily-slot backlog cap all run every time,
re-pulling the same LingQs is harmless — nothing gets double-carded, and a
large backlog is a feature (§12): it gives Claude more to choose from.

**Not verified against the real LingQ API in this session.** This sandbox
has no LingQ key and restricted egress, so the client is built defensively
against LingQ's documented v3 response shape and mocked in tests — the first
real run on your machine is the true test. If LingQ's field names differ
from what's coded (`term`, `fragment`, `status`, `hints[].text`), adjust
`LingQCard`/`_extract_hint` in `french_mining/lingq/api.py`.

`scripts/mine_lingq_pdf.py` and `scripts/mine_lingq_api.py` share their
downstream write path (`french_mining.text_pipeline.write_ranked_candidates`)
so the two text sources can't drift apart.

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
transparency, and recency (`Candidate.days_since_encountered`) into the
composite score `keep_and_rank` sorts on. This is left unset by both
pipelines currently (LingQ PDFs don't carry timestamps at all; the YouTube
pipeline has real sentence-level timing via `start_time`/`end_time`, but
turning that into "days since the *learner* encountered it" needs a
watched-date, which — per §4 — isn't reliably available from the YouTube
API without OAuth watch-history access). Actual queue reordering via
AnkiConnect `due` rewrites is a separate later stage (§2, §6 build-order
step 6) — this module only scores and ranks.

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

## Queue ordering (§2, build-order step 6)

This is the mechanism behind the core architectural principle: FSRS owns a
card the instant it's first reviewed, but until then it just sits in the
new-card queue ordered by an integer `due` value — and that's the only lever
Claude ever pulls. `french_mining.queue_ordering` never touches a card whose
`type` isn't still `0` (new/unreviewed).

`get_new_backlog_note_ids` reads every still-new note of our note type in
the deck, in current queue order. `merge_priority_with_backlog` puts this
run's freshly-ranked picks at the front and leaves every other still-new
note in its existing relative order behind them — it deliberately does
*not* re-score the whole historical backlog every run (that would burn API
tokens re-judging cards already reasonably ranked last time, against the
cost philosophy in §3). `reorder_queue` then rewrites `due` to consecutive
integers across that merged order via AnkiConnect's
`setSpecificValueOfCard`, skipping any note whose card already left the new
state (FSRS owns it now, not us).

**One-time setup this can't do for you:** Anki's new-card *display* order
also has to be configured to actually respect `due` — in Deck Options ->
Display Order, set the new-card gather/sort order to a position-based
option, not "Random". Without that, rewriting `due` has no visible effect
on what you see in Anki even though the values are correctly updated.

`scripts/mine_lingq_pdf.py --write N` now reorders the queue automatically
right after writing new cards.

## YouTube pipeline (§4, §9, build-order step 7)

Adds source-audio and source-image richness on top of the same shared i+1
filter / Stage 2 scoring / card generation used by the LingQ pipeline —
`french_mining.candidates` was relocated out of `lingq/` to
`french_mining/candidates.py` for this reason (it was never LingQ-specific,
and now two pipelines share it).

- `youtube/transcripts.py` — downloads subtitles via yt-dlp and parses
  WebVTT cues into timestamped segments. YouTube's auto-generated captions
  are messy (rolling partial-text cues, inline per-word timing tags); the
  parser strips those tags and drops exact-duplicate consecutive cues
  rather than trying to reconstruct word-level timing.
- `nlp.parse_transcript` / `nlp.attach_timestamps` — sentence-segments the
  concatenated transcript text via spaCy, then maps each sentence back to a
  time range by locating its text within the transcript and finding which
  segments overlap it. `attach_timestamps` is pure and spaCy-free (fully
  tested without the French model); `ParsedSentence` now carries optional
  `start_time`/`end_time` (`None` for plain-text sources like LingQ PDFs),
  and `Candidate` carries the same plus `video_id`.
- `youtube/media.py` — `download_audio`/`download_video` (yt-dlp, needs
  network) and `clip_audio`/`extract_frame` (ffmpeg, fully local). The
  clip/extract functions are genuinely integration-tested here against
  synthetic media generated locally with ffmpeg's `lavfi` test sources — no
  network needed for that half.
- `youtube/metadata.py` — YouTube Data API video title/channel lookup, for
  the §8 Source field format ("title + timestamp"). Watch-history retrieval
  needs an OAuth scope Google restricts for new API clients, so per §4 this
  takes a video ID directly rather than trying to surface history
  automatically.
- `youtube/pipeline.py` — `attach_source_media` clips the real sentence
  audio and stores it via AnkiConnect's `storeMediaFile`
  (`connect.store_media_file`), returning a `SentenceAudio` field override.
  `grab_source_frame` extracts the raw video frame at the sentence's
  midpoint (with `--with-images`) for the image tier logic below to judge —
  this module doesn't decide whether a frame is any good.

`scripts/mine_youtube_video.py VIDEO_ID --write N [--with-images]` runs the
full chain: metadata -> subtitles -> spaCy+timing -> i+1 filter -> Stage 2
scoring -> generation -> source audio/frame attachment -> write -> queue
reorder.

**Not exercised end-to-end in this session.** Two real network
dependencies are blocked by this sandboxed session's egress policy (same as
the spaCy model download in step 3): `youtube.com` (yt-dlp transcript/audio/
video downloads) and, untested but likely similarly restricted,
`googleapis.com` (YouTube Data API metadata). What *is* verified here:
ffmpeg itself works in this environment (it wasn't preinstalled — `apt-get
install ffmpeg` after fixing a mirror dependency snag got it working), so
`clip_audio`/`extract_frame` are tested against real ffmpeg output, not
just mocks — only the yt-dlp download and Data API calls are mocked. Run
this on your own machine, where none of that is blocked.

## Image tier logic (§10, build-order step 8)

`french_mining.images` implements the full 3-tier decision from §10, gated
by a concreteness judgment that piggybacks on Stage 2 scoring rather than
spending a separate API call:
`ScoredCandidate.is_concrete_and_visualizable` — abstract words, most
verbs/adjectives, and generic nouns never get an image, since an irrelevant
image actively hurts retention (Mayer's coherence principle).

1. **Source frame** (YouTube only) — `is_talking_head_or_text` is a free,
   local OpenCV pre-filter: a Haar-cascade face detector discards frames
   where a face fills too much of the frame (talking-head shots), and a
   Canny edge-density heuristic discards text-heavy frames (slides,
   burned-in captions). Only frames that pass both go to
   `check_image_relevance`, a cheap Claude vision call judging whether the
   frame actually, distinctively shows the target word's referent.
2. **Unsplash fallback** — `search_unsplash` queries by the target lemma;
   `pick_best_unsplash_photo` sends the handful of results to Claude in one
   vision call asking it to prefer culturally specific imagery over generic
   stock-photo-style results (or reject all of them), per §10.
3. **No image** — abstract words (gated before tier 1 even runs), and
   anything that failed both tiers above.

`resolve_image_field` runs all three tiers and returns the `Image` field
value directly. Both `mine_lingq_pdf.py` (Unsplash-only — PDFs have no
video frame) and `mine_youtube_video.py --with-images` (full 3-tier) call
it after card generation, since it needs the generated `TargetWordGloss`.

The OpenCV pre-filter is genuinely tested here against real images (a
smooth gradient vs. a synthetic checkerboard, generated locally with
numpy/cv2 — no network needed) — `EDGE_DENSITY_TEXT_THRESHOLD` was
empirically calibrated against those two cases, not against a real
burned-in-caption frame, so treat it as a reasonable starting point to
retune once you've seen it run on real video frames. One dependency note:
`opencv-python-headless` is pinned below 5.0 in `pyproject.toml` because
that release stopped bundling the Haar cascade XML files this pre-filter
needs (`cv2/data/*.xml` was empty in 5.0.0 when I checked). The vision
check and Unsplash calls themselves are mocked in tests, same as the rest
of the Anthropic API surface.

## Word/second-example audio (§9, beyond the spec so far)

Source video only gives us real speech for `SentenceAudio` (clipped by
`youtube.pipeline.attach_source_media`), and only when a sentence's timing
survived transcript alignment. Two fields have no possible source clip at
all: `WordAudio`/`ChunkAudio` (the target word spoken in isolation — there's
no isolated-word moment in the video to cut) and `SecondExampleAudio` (a
sentence Claude wrote during generation, so it was never actually spoken by
anyone). `french_mining.tts` fills both via the ElevenLabs API
(`eleven_multilingual_v2`, a French-capable premade voice by default,
override with `ELEVENLABS_VOICE_ID`):

- `synthesize_speech(text, output_path, api_key=...)` — one REST call per
  clip, writes the returned MP3 bytes to disk.
- `strip_html` — removes the `<span class="target">` highlight wrapper
  before speaking `SecondExample`, so TTS reads plain text.
- `resolve_tts_fields(anki_client, target_text, second_example_html,
  work_dir, name_part, target_field_name="WordAudio"|"ChunkAudio")` —
  synthesizes both clips, stores them via AnkiConnect's `storeMediaFile`,
  and returns the `[sound:...]` field values.

This is opt-in and fails open, the same tier-skip pattern as the Unsplash
image fallback: with no `ELEVENLABS_API_KEY` set, `resolve_tts_fields`
returns both fields blank instead of raising, so the rest of the write path
is unaffected. It's wired into both `text_pipeline.write_ranked_candidates`
(LingQ PDF/API — `SentenceAudio` stays blank there regardless, no source
video to clip it from) and `mine_youtube_video.py` (alongside source-audio/
image attachment). Tested with the `responses` library mocking the
ElevenLabs endpoint — no real API key or network needed for
`tests/test_tts.py`.

**Not run against the real ElevenLabs API in this session** (no network to
elevenlabs.io in this sandbox) — get a key at https://elevenlabs.io/, set
`ELEVENLABS_API_KEY` in `.env`, and the next mining run will start filling
these fields in.

## Condensed audio from watched videos (beyond the spec)

A second output from the *same* watched YouTube videos, alongside cards:
comprehensibility-filtered **condensed audio** — a playlist of dense,
silence-free, few-minutes-each **clips** made only of the sentences you can
already follow, for passive listening review (commutes, walks).

It's a natural fit because the pipeline already has the two hard
ingredients — subtitle-timed transcripts (`nlp.parse_transcript` gives
sentence-segmented, timed, lemmatized `ParsedSentence`s) and an FSRS-backed
model of what you know (`VocabularyState`). So this is **entirely local and
free — zero Anthropic API cost** (yt-dlp + ffmpeg + spaCy + AnkiConnect
only), fitting the §3 cost philosophy.

`french_mining.condensed_audio`:

- `sentence_comprehensibility` — fraction of a sentence's content words
  that are already known (`confidence >= 0.6`, the pipeline's standard
  known bar).
- `select_comprehensible_spans` — keeps sentences at/above
  `--min-comprehensibility` (default **0.9** — audio has no visual support,
  and this is review of already-watched material, so aim high), drops the
  rest, and **merges adjacent kept spans** (within ~0.4s) so the result
  isn't a stutter of micro-cuts. Sentences without timing are skipped.
- `video_comprehensibility` — overall coverage, used to order the playlist
  easiest-video-first.
- `chunk_spans` — groups a video's merged spans into shuffle-friendly clips
  around `--clip-minutes` (default **3**), *without ever splitting a span*.
  One big file (per video, or the whole corpus) has the same "always listen
  to the start, scrub to find your place" problem as any long audio file —
  clips are what make a player's shuffle actually useful, and a dud clip is
  easy to skip past. A clip always stays within one video (never splices two
  videos together); a video's leftover tail shorter than the target just
  becomes its own shorter final clip. A single span longer than the target
  on its own (rare) still isn't split — it becomes its own over-length clip.

`youtube.media.concat_audio_spans` does the actual condensing (once per
clip): a single ffmpeg pass (`atrim`+`concat` via a `-filter_complex_script`
file, so many spans can't blow the command-line limit), mirroring
`clip_audio`'s codec and padding choices.

```bash
.venv/bin/python scripts/build_condensed_audio.py VIDEO_ID [VIDEO_ID ...]
.venv/bin/python scripts/build_condensed_audio.py --ids-file watched.txt --clip-minutes 3 --single-file
```

Produces `<video_id>_clip01.mp3`, `_clip02.mp3`, ... per video, and an
easiest-video-first `condensed_playlist.m3u` listing every clip across every
video — shuffle *that* in your player. `--single-file` is an **opt-in
extra** (stitches every clip into one `condensed_all.mp3`) for anyone who
still wants that; it's not the default, since it's the exact one-big-file
pattern the clips are meant to replace. Reads Anki vocab state but writes no
cards.

**Tradeoff (your choice):** dropping individual hard sentences maximizes
comprehensible density but sacrifices some narrative continuity — the
adjacent-merge + padding soften the cuts, and a low-comprehension video
will yield few or no clips at the 0.9 bar (expected). Videos without
captions have no timing to cut on out of the box — see the Whisper
fallback below, which covers exactly that case.

**Verified here:** the comprehensibility/selection/chunking logic is fully
unit-tested (no spaCy model needed), and `concat_audio_spans` is tested
against **real ffmpeg** using synthetic audio (output duration ≈ sum of
kept spans). Not runnable end-to-end in this sandbox (yt-dlp network + the
spaCy model are blocked) — first real run is on your machine.

## Whisper transcription fallback (beyond the spec)

Not every watched video has captions. Without a fallback, those videos
would silently contribute nothing to either cards or condensed audio — no
transcript, no timing, no candidates. `french_mining.youtube.pipeline.get_transcript`
is the single entry point both `mine_youtube_video.py` and
`build_condensed_audio.py` now use: try YouTube's own captions first
(`transcripts.download_subtitles`, free and instant), and only if that
fails, fall back to local transcription with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper)
(`youtube.whisper_transcribe`) — CTranslate2-based, no PyTorch dependency,
int8 CPU inference. Still **zero API cost**, just slower than downloading
captions (real speech-to-text inference vs. a file download).

`get_transcript` returns `(segments, source)` so both scripts can report
which path was taken. Flags on both scripts: `--whisper-model` (default
`"small"` — balances accuracy/speed/download size for a background job;
`"base"` is faster/smaller, `"medium"`/`"large-v3"` more accurate) and
`--no-whisper-fallback` (skip captionless videos instead of transcribing
them, if you'd rather not pay the CPU time).

`media.download_audio` now skips re-downloading if the target file already
exists in the working directory — needed because the Whisper fallback path
and (for `mine_youtube_video.py --write`) the later card-audio-clipping
step can both want the same downloaded audio within one run.

**Verified here:** `transcribe_with_whisper` accepts an injectable `model`
(duck-typed: anything with a `.transcribe()` method), so it's tested
without downloading real model weights; `get_transcript`'s captions-first/
Whisper-fallback branching is tested with both paths mocked. The real
faster-whisper model download (first use, ~500MB for "small", from Hugging
Face) and actual transcription are not exercised in this sandbox (blocked
network) — first real run is on your machine.

## Collocation card type (§7, build-order step 9)

Fluency is largely chunk-level, not word-level. Single words and
collocations are mined by different criteria (§6 vs. §7) but explicitly
**share the same backlog and compete for the same daily slots** — so
rather than forking the pipeline, a collocation candidate is just a
`Candidate` with `is_collocation=True` and `target_lemma` set to the
chunk's canonical form. It flows through the exact same Stage 2 scoring,
`select_best_sentences`, and queue-ordering code as single words; only
local extraction, card generation, and the note type are collocation-
specific.

- `french_mining.collocations` — `find_collocation_candidates` matches a
  curated seed list (`data/french_collocations.csv`: ~40 common verb-
  preposition pairings and fixed expressions, e.g. `s'apercevoir de` /
  "to realize" vs. transitive `apercevoir` / "to catch sight of") against
  sentence lemma sequences. Matching allows a small gap between components
  (`MAX_COMPONENT_GAP`, default 4 tokens) since French inserts reflexive
  pronouns, negation, and clitics between a verb and its preposition — an
  exact contiguous match would miss most real occurrences. The same i+1
  gating idea applies: a sentence qualifies only if the matched chunk is
  the sole unknown/fragile unit and every other word is already known.
- `VocabularyState.known_chunks` / `chunk_confidence` / `is_chunk_known` —
  collocations already mastered are tracked in a separate namespace from
  single words (added via `VocabularyState.build(..., collocation_model_names=[...])`),
  so a chunk's text never collides with an unrelated single word, and (unlike
  single words) chunks get no frequency-floor fallback — an untested chunk
  is just unknown.
- `french_mining.anki.collocation_note_type` — a second note type
  ("French Collocation Mining") sharing the single-word note type's field
  order and template structure (`note_common.py` now holds the mechanics
  both note types share), but keyed on `TargetChunk` instead of
  `TargetWord`, with a `UsageNote` field replacing `MorphologyNote` — this
  is the card's most important field, since it's expected to explain why
  the chunk is worth learning as a unit rather than as its component words.
  It shares the single-word note type's deck, per §7.
- `french_mining.collocation_generation` — mirrors `generation.py` for the
  collocation fields; its system prompt explicitly asks for that
  component-word contrast in `UsageNote`.

Both mining scripts now extract word and collocation candidates side by
side, merge them before Stage 2 scoring/ranking (one shared, ranked list),
then split back out by `is_collocation` only at generation/write time
(different note types need different generated fields) before reordering
the queue across both note types together
(`get_new_backlog_note_ids`/`reorder_queue` now accept multiple note types).

`scripts/create_placeholder_collocation_card.py` verifies the collocation
template end-to-end, same as `create_placeholder_card.py` does for single
words.

The seed collocation list is a starting point (~40 entries), not
exhaustive — same caveat as the frequency list: swap in a bigger one via
`load_collocations("path/to/your.csv")` (same `chunk,lemmas,gloss` format)
as you find gaps.

## Monthly hygiene audit (§11, build-order step 10)

FSRS has no window into immersion: a word the learner has fully acquired
by listening/reading keeps cycling through reviews indefinitely, since
nothing tells the scheduler it's done. `french_mining.hygiene` flags —
and, with `--suspend`, suspends (never deletes) — cards that are both:

- **mature**: Anki's own scheduled `interval` on the card is already past
  `DEFAULT_INTERVAL_THRESHOLD_DAYS` (180 days, ~6 months, the spec's own
  example threshold) — using Anki's already-computed interval directly
  rather than re-deriving a "projected interval" from FSRS stability.
- **high-frequency**: the word's frequency rank is within
  `DEFAULT_FREQUENCY_FLOOR` (top 500) — a word that common would almost
  certainly be reinforced by immersion long before deliberate review got it
  this mature, whereas a genuinely rare word reaching the same stability
  was more likely earned through deliberate study and shouldn't be
  second-guessed.

This is single-word-only in practice: collocations aren't frequency-ranked
(§7), so there's no equivalent "obviously acquired by immersion" signal for
them yet.

```bash
.venv/bin/python scripts/monthly_hygiene_audit.py            # dry run, just lists candidates
.venv/bin/python scripts/monthly_hygiene_audit.py --suspend   # actually suspends them
```

Suspended cards leave the review queue but stay in the deck — reactivatable
anytime via Anki's own UI or AnkiConnect's `unsuspend` action, nothing is
destroyed.

## Project layout

```
src/french_mining/
  anki/
    connect.py               # AnkiConnect JSON-RPC client + storeMediaFile
    vocab_state.py           # gradient known-word/known-chunk model (§5)
    note_common.py           # shared note-type mechanics (createModel/addNote)
    note_type.py              # single-word note type, templates, write path (§8)
    collocation_note_type.py  # collocation note type, shares the single-word deck (§7)
  lingq/
    pdf_extract.py  # PDF -> raw text (pypdf)
    api.py          # LingQ v3 API client + looked-up-word candidate builder
  youtube/
    transcripts.py         # yt-dlp subtitle download + WebVTT parsing
    whisper_transcribe.py  # local Whisper fallback transcription (faster-whisper)
    media.py               # yt-dlp audio/video download + ffmpeg clip/concat/frame-extract
    metadata.py            # YouTube Data API video metadata
    pipeline.py             # get_transcript (captions->Whisper fallback); attach_source_media/grab_source_frame per card
  candidates.py             # i+1 pre-filter + best-sentence selection (§6 Stage 1, shared)
  collocations.py            # collocation matcher -> generic Candidate (§7)
  condensed_audio.py         # comprehensibility scoring + span selection for condensed audio
  nlp.py                     # spaCy loading + text/transcript -> ParsedSentence/Token
  scoring.py                  # Claude/Sonnet API scoring + ranking (§6 Stage 2, shared)
  generation.py                # single-word card content + write path (§8 step 5)
  collocation_generation.py    # collocation card content + write path (§7)
  text_pipeline.py             # shared score->generate->write path for PDF + LingQ-API sources
  images.py          # 3-tier image logic: source frame / Unsplash / none (§10)
  queue_ordering.py  # AnkiConnect `due` rewrites for the new-card queue (§2)
  hygiene.py         # monthly mature+high-frequency card suspension audit (§11)
  data/
    french_frequency_top500.csv
    french_collocations.csv
  frequency.py       # frequency floor + exclusion list loading
scripts/
  create_placeholder_card.py              # verify the single-word card end-to-end
  create_placeholder_collocation_card.py  # verify the collocation card end-to-end
  mine_lingq_pdf.py           # run locally: full pipeline (words + collocations) on a real PDF
  mine_lingq_api.py           # run locally (or scheduled): sync looked-up LingQs -> Anki
  mine_youtube_video.py       # run locally: full pipeline (words + collocations) on a real YouTube video
  build_condensed_audio.py    # run locally: comprehensibility-filtered condensed audio from watched videos
  monthly_hygiene_audit.py    # run locally (or on a schedule): flag/suspend mature cards
tests/               # all AnkiConnect/Anthropic/yt-dlp calls mocked; ffmpeg tested for real
```
