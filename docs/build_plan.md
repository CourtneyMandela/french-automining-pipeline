# Automated French Sentence-Mining Pipeline — Build Plan

## Purpose of this document

This is a build specification for an automated pipeline that turns French immersion content into well-scaffolded, intelligently ordered Anki cards with zero manual mining effort. Hand this to Claude Code as the project brief. It describes *what* to build and *why* each piece exists, so design decisions aren't relitigated mid-build.

---

## 1. The problem this solves

Manual sentence mining costs immersion time. Every minute spent looking up a word, creating a card, and formatting a note is a minute not spent consuming French. Worse, manually created cards are minimal under time pressure, created at the moment of encounter rather than when the learner is optimally ready, and ordered by chance rather than by what the learner's current vocabulary can absorb.

This pipeline removes the learner from the mining loop entirely. They consume content; cards appear. Immersion and review become complementary rather than competing activities.

## 2. The core architectural principle

**FSRS handles timing. Claude handles curation and content quality.**

The system never replaces or fights Anki's scheduler. New cards haven't entered FSRS yet — they sit in a queue ordered by an integer `due` value. Claude reorders that queue. The moment a card gets its first review, FSRS takes over and Claude steps back. Clean handoff, no contention.

A second principle governs scope: **the card system is a scaffold for immersion, not a parallel track to it.** Every decision optimizes for making immersion more comprehensible faster — not for optimizing the card experience as an end in itself. The user also does significant immersion and output separately; the pipeline does not need to compensate for transfer gaps those activities already address.

## 3. Cost philosophy

Keep API usage minimal by doing all mechanical and first-pass filtering work locally for free, so only genuine candidates ever reach the API. Target monthly cost: under $2. The structure that achieves this is not an optimization to add later — it is the architecture.

Local (free): Anki DB reads, transcript extraction, audio download/clipping, frame extraction, spaCy lemmatization, frequency filtering, talking-head detection.

API (paid, cheap): sentence quality scoring, interference detection, collocation judgment, morphology notes, second examples, frame relevance assessment, final queue ordering. Use Sonnet for production generation.

---

## 4. Inputs

### YouTube
- Watch history pulled via the YouTube Data API (user supplies which videos, or the API surfaces recent history).
- Transcripts + timestamped subtitles via `yt-dlp --write-auto-sub` (and manual subs when available).
- Audio downloaded via `yt-dlp` for sentence-audio clipping.

### LingQ reading
- User drops a **PDF** of what they read into a weekly folder. That's the only required input.
- No CSV needed. The Anki database is a better source of truth for vocabulary state than LingQ's 1–4 status scale, because it carries FSRS stability behind every word.

---

## 5. The vocabulary state model (most important component)

Everything downstream depends on this being correct. Build it carefully.

### Build the "known" set from target words only
A word counts as known **only if it has its own card that has been successfully reviewed** — never because it appeared incidentally in another card's example sentence. Words in context positions were background noise during encoding, not the thing being learned. Extract the known set from the **tested field** of each note, not from sentence fields.

### Make "known" a gradient, not a binary
Use FSRS per-card state (stability, retrievability, learning phase) to weight confidence:
- High stability + high retrievability → solidly known, counts fully.
- Low stability / still in learning phase → fragile; treat its presence in a candidate sentence almost as if the word were unknown.
- Repeatedly lapsing → practically unknown.

A sentence with one true unknown plus one fragile word is effectively i+2 and should be treated as such.

### Frequency floor for pre-Anki vocabulary
Words known before the Anki system began (e.g. *être*, *avoir*, basic adjectives) won't be in Anki and would look unknown. Skip any word in the top ~300–500 most frequent French words that isn't already in Anki. Maintain a short manual exclusion list for anything that slips through.

### Access
Use **AnkiConnect** (the only add-on needed) to read card state while Anki runs. It bridges cleanly across versions and handles sync state. No raw SQLite manipulation needed.

---

## 6. Candidate selection and scoring

### Stage 1 — local filtering (free)
1. Extract candidate words from transcripts/PDF via spaCy lemmatization (lemma + surface form).
2. Cross-reference against the confidence-weighted vocabulary map.
3. Apply the frequency floor and exclusion list.
4. Drop any sentence that isn't true i+1 given the gradient known-set (one genuine unknown, no fragile extras). This typically removes ~90% of candidates before any tokens are spent.

### Stage 2 — API scoring of survivors
For sentences that survive, score on:
- **i+1 fitness** — confirmed single unknown given the gradient model (non-negotiable first filter).
- **Frequency rank** — higher-frequency unknowns unlock more downstream comprehension per card slot.
- **Unlock potential** — how many *other* backlog candidates become learnable if this word is introduced. Clear dependency-tree branches, don't pick randomly.
- **Context transparency** — can the meaning be deduced from the sentence before the answer is shown? Prefer sentences where it can. Two sentences can both be i+1 yet differ sharply in quality; this is the differentiator.
- **Recency** — recent content carries live episodic memory ("I heard this in that video"). Favor it, not because newer is better in principle, but because the episodic hook fades.

### Desirable difficulty guardrail
i+1 ordering must not shade into i+0. The slight struggle before successful retrieval is the consolidation mechanism. If review feels effortless, that's a warning sign. Don't artificially inflate difficulty, but don't optimize the struggle out either.

### Interference avoidance
Track what's currently in the learning queue. Delay introducing near-synonyms and easily confused forms (e.g. *apercevoir* / *remarquer* / *constater*) until earlier ones reach maturity. Scheduling logic must account for semantic proximity, not just sentence-level i+1.

---

## 7. Collocations as a distinct card type

Fluency is largely chunk-level, not word-level. The selection criterion for chunks differs from single words: **mine multi-word units whose meaning is not fully predictable from the component words** — verb–preposition pairings, fixed expressions, high-frequency collocates (e.g. *s'apercevoir de* = "realize," distinct from transitive *apercevoir* = "catch sight of").

Single words and collocations share the same backlog and compete for the 20 daily slots. Claude balances the mix based on the learner's current level. Build a second note template for chunks.

---

## 8. Card design

**One card per note.** The front tests meaning reconstruction in context — the highest-value retrieval practice. The back delivers confirmation plus the audio-visual pairing and scaffolding.

### Fields
- `TargetWord` — lemma form
- `TargetWordForm` — form as it appeared in source
- `TargetWordGloss` — 2–3 word English equivalent
- `TargetWordDefinition` — one-sentence contextual definition
- `PartOfSpeech`
- `MorphologyNote` — gender for nouns; group + key conjugations + irregularity for verbs; agreement pattern for adjectives
- `SentenceText` — full source sentence, target word wrapped in highlight span
- `SentenceTranslation` — full English translation of the whole sentence
- `SecondExample` + `SecondExampleAudio` — Claude-generated, word in a different context
- `SentenceAudio` — clipped from source where possible, TTS fallback
- `WordAudio` — TTS
- `Image` — conditional (see §10)
- `FrequencyRank`
- `Source` — title + timestamp (video) or title + excerpt (text)
- `DateMined`

### Template
**Front:** highlighted sentence + autoplayed isolated word audio. Nothing else.
**Back, in order:** gloss → full sentence translation → sentence audio (pairs fresh comprehension with audio) → `TargetWordForm → TargetWord` → morphology note → second example with audio → source line → frequency rank → image if present.

### Deliberately excluded
Production cards (different skill, doubles review burden — automate promotion later once a word matures). Cloze deletion (tests the wrong thing for immersion learning). Images on abstract words (see §10).

---

## 9. Audio sourcing

- **Sentence audio** — clip the exact segment from source via `yt-dlp` + `ffmpeg` using subtitle timestamps. Native speaker saying the actual sentence. Best case, free. (Same approach as the user's Latin/Whisper pipeline.) TTS fallback for text-only LingQ content.
- **Word audio** — TTS. `gTTS` is free and adequate for French; ElevenLabs if quality matters.
- **Second example audio** — TTS.

---

## 10. Image sourcing

Images help retention via the picture superiority effect and dual coding — but only when relevant and distinctive (Mayer's coherence principle: irrelevant images actively hurt). Concrete, culturally specific nouns only; never abstract words.

### Tiered decision
1. **Source frame** (best) — grab the video frame at the sentence timestamp. It adds episodic reinstatement (encoding specificity) on top of dual coding. But assess first: OpenCV talking-head pre-filter locally (free) to discard face/text frames, then a cheap API vision check for whether the frame shows anything concrete and relevant to the target word.
2. **Unsplash** (fallback) — free API, for concrete nouns where the source frame failed. Prefer culturally specific images over generic ones; distinctiveness is its own encoding mechanism.
3. **No image** — abstract words, and anything that failed both above.

---

## 11. Deck hygiene (monthly audit)

FSRS has no window into immersion, so words the learner has fully acquired through listening still cycle through reviews indefinitely, bloating the deck. A separate monthly script:
- Scans cards above a stability threshold (e.g. projected interval > 6 months).
- Cross-references against frequency — a top-500 word at that stability is almost certainly acquired by immersion.
- Flags candidates for **suspension** (not deletion — suspended cards leave the queue but stay in the deck, reactivatable anytime).

Keeps daily sessions focused on genuine consolidation rather than maintenance.

---

## 12. User interaction model

- **During the week:** nothing. Watch French content; read in LingQ; drop PDFs into the weekly folder.
- **Weekly (e.g. Sunday evening):** trigger the pipeline in a Claude Code session, or run it on a schedule.
- **Monday:** open Anki. The new-card queue is populated, ordered, and scaffolded. Do the usual 20 reviews.
- **Tuning:** if review burden feels off, adjust new-cards-per-day in Anki's own interface. The backlog absorbs the change silently. A large backlog is a feature — it gives Claude more to select from.

---

## 13. Suggested build order

1. **AnkiConnect read layer + vocabulary state model** (§5). Nothing works without this; build and verify it first.
2. **Note type + template** (§8) created in Anki, plus AnkiConnect write path that produces one correct card end-to-end with placeholder content.
3. **Local LingQ-PDF pipeline** — spaCy extraction, frequency floor, i+1 pre-filter, best-sentence selection. Easiest content source, no audio/video complexity.
4. **API scoring layer** (§6) on the filtered candidates.
5. **Card generation** — morphology notes, second examples, translations, gloss.
6. **Queue ordering via AnkiConnect** `due` rewrites (§2).
7. **YouTube pipeline** — transcripts, audio clipping, frame extraction. Adds the source-audio and source-image richness.
8. **Image tier logic** (§10).
9. **Collocation card type** (§7).
10. **Monthly hygiene audit** (§11).

Build incrementally; each stage should produce verifiable output before the next is added.

---

## 14. Stack summary

- **Language:** Python
- **Anki bridge:** AnkiConnect (only add-on)
- **NLP:** spaCy (French model) for lemmatization
- **Content:** YouTube Data API, `yt-dlp`
- **Media:** `ffmpeg` (audio clip + frame grab), OpenCV (talking-head pre-filter), `gTTS`/ElevenLabs (TTS), Unsplash API (images)
- **Intelligence:** Anthropic API, Sonnet for production generation
- **Frequency:** a French frequency list (top-N for the floor + per-word rank)
