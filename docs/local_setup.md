# Local setup runbook (for a Claude Code session)

This file is written as instructions **for an AI agent** (a `claude` session
running directly on the user's machine, not a cloud sandbox) to execute —
not as end-user prose. If you're a human reading this, it still works as a
manual checklist; just run each command yourself instead of "ask the user."

Read the whole file before starting. You have real shell, filesystem, and
(once Anki is open) localhost network access. Work through the steps in
order, running each numbered step's own "Verify" check before moving to the
next one. Wherever a step says "ask the user," stop and ask — don't guess
API key values or make destructive choices on their behalf.

## 0. Detect the environment

Run `uname -s 2>/dev/null || echo Windows` to determine OS, and confirm
whether you're already inside a clone of this repo
(`git rev-parse --is-inside-work-tree`). Several steps below branch on OS —
Windows via Git Bash/MINGW64 needs different commands than macOS/Linux in a
few places (venv paths, package manager).

## 1. Clone + branch

If not already inside the repo:

```bash
git clone https://github.com/courtneymandela/french-automining-pipeline.git
cd french-automining-pipeline
```

```bash
git checkout claude/project-setup-8s4509
git pull
```

Verify: `git branch --show-current` prints `claude/project-setup-8s4509`.

## 2. System prerequisites

Check each is already on PATH before attempting an install
(`command -v ffmpeg`, `git --version`, `python3 --version` or
`python --version`) — don't reinstall something that's already there.

- **git** — should already be present if step 1 worked.
- **Python 3.11+**
  - Windows: if `python`/`python3` prints something like "Python was not
    found... Microsoft Store", the Store alias is shadowing a real
    install. Ask the user to install Python from
    https://python.org/downloads (check "Add python.exe to PATH" during
    install), then go to Settings → Apps → Advanced app settings → App
    execution aliases and turn OFF "python.exe"/"python3.exe". Wait for
    them to confirm, then re-check `python --version`.
  - macOS: `brew install python@3.11` (install Homebrew first from
    https://brew.sh if `brew` isn't found).
  - Linux: `sudo apt install python3.11 python3.11-venv` (Debian/Ubuntu) or
    the equivalent for the user's distro.
- **ffmpeg**
  - Windows: `winget install ffmpeg` (winget ships with Windows 10
    2004+/11). If `winget` isn't found, try `choco install ffmpeg`
    (Chocolatey), or ask the user to grab a static build and add its
    `bin/` folder to PATH.
  - macOS: `brew install ffmpeg`.
  - Linux: `sudo apt install ffmpeg`.
  - Verify: `ffmpeg -version` prints a version banner.

## 3. Python environment

```bash
python -m venv .venv     # or python3, whichever step 2 confirmed works
```

Binary location differs by OS — use the right one in every command below:

- Windows (Git Bash): `.venv/Scripts/pip.exe`, `.venv/Scripts/python.exe`
  (or `source .venv/Scripts/activate` once, then plain `pip`/`python`).
- macOS/Linux: `.venv/bin/pip`, `.venv/bin/python`.

```bash
<venv-pip> install -e ".[dev]"
```

Verify: `<venv-pip> show french-mining-pipeline` succeeds without error.

## 4. spaCy French model

```bash
<venv-python> -m spacy download fr_core_news_sm
```

This pulls from GitHub releases — needs real outbound network access.
Verify:

```bash
<venv-python> -c "import spacy; spacy.load('fr_core_news_sm'); print('ok')"
```

## 5. Run the test suite (no live Anki or API keys needed yet)

```bash
<venv-python> -m pytest
```

Expect around 193 passed. If anything fails here, stop and diagnose before
continuing — every later step assumes this baseline is clean.

## 6. Anki + AnkiConnect

Ask the user to confirm both of these before continuing:

1. Anki desktop is installed (https://apps.ankiweb.net/) and currently
   open.
2. The AnkiConnect add-on is installed (inside Anki: Tools → Add-ons → Get
   Add-ons → paste code `2055492159`) and Anki has been restarted since
   installing it.

Do not proceed past this point until they confirm Anki is open — every
step after this needs `http://127.0.0.1:8765` to answer.

## 7. Configure `.env`

```bash
cp .env.example .env
```

Ask the user which of these they want to fill in right now. Only
`ANTHROPIC_API_KEY` is required to do anything at all — the rest each
unlock one specific feature and can be added later, whenever the user gets
around to it:

| Variable | Unlocks | Get it at |
|---|---|---|
| `ANTHROPIC_API_KEY` (required) | scoring, generation, everything API-touching | https://console.anthropic.com/settings/keys |
| `LINGQ_API_KEY` | LingQ auto-sync | https://www.lingq.com/accounts/apikey/ |
| `YOUTUBE_API_KEY` | video title/metadata lookups | Google Cloud Console, YouTube Data API v3 enabled |
| `UNSPLASH_ACCESS_KEY` | fallback card images | https://unsplash.com/developers |
| `ELEVENLABS_API_KEY` | word/second-example audio (TTS) | https://elevenlabs.io/ |

Edit `.env` in place with whatever values the user gives you. Never print
key values back to the terminal or log them, and never commit `.env` —
confirm it's actually gitignored with `git check-ignore .env` before moving
on.

## 8. Verify the AnkiConnect write path for real

With Anki open:

```bash
<venv-python> scripts/create_placeholder_card.py
<venv-python> scripts/create_placeholder_collocation_card.py
```

Ask the user to check Anki's card browser for a `French::Mining` deck
containing two new placeholder cards. Don't proceed until they confirm this
worked — a failure here almost always means AnkiConnect isn't actually
reachable (Anki not open, add-on missing, or a local firewall blocking
`localhost:8765`).

## 9. First real content run

Ask the user which of these they have on hand right now:

- **A YouTube video they've watched** — get the video ID (the part after
  `v=` in the URL), then:
  ```bash
  <venv-python> scripts/mine_youtube_video.py VIDEO_ID --write 5
  ```
- **A LingQ PDF export** — check `scripts/mine_lingq_pdf.py --help` for
  where to drop it, then run with a small `--write` count first.
- **LingQ API sync** (only if `LINGQ_API_KEY` is set) — see
  `scripts/mine_lingq_api.py --help`.

Use a small `--write` count (5 or fewer) on this first run so the user can
review the actual generated cards in Anki before pointing it at a full
backlog.

## 10. Wrap up

Summarize for the user: which steps completed, which API keys are
configured vs. still blank (by name only — never repeat key values back),
and what to try next (e.g. `scripts/build_condensed_audio.py` once they've
mined a few videos, `scripts/monthly_hygiene_audit.py` for deck upkeep).
Do not set up a cron job/scheduler unless the user explicitly asks for one
in this session.
