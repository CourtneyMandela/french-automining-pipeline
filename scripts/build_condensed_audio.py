#!/usr/bin/env python3
"""Build comprehensibility-filtered condensed audio from YouTube videos
you've watched — a dense, silence-free listening file/playlist made of only
the sentences you can already follow, for passive review on the go.

Entirely local and free (yt-dlp + ffmpeg + spaCy + AnkiConnect) — no
Anthropic API calls. Run on the machine where Anki is open.

Requires (see README): the French spaCy model, network for yt-dlp, ffmpeg,
and Anki running with AnkiConnect (read-only — no cards are written).

    .venv/bin/python scripts/build_condensed_audio.py VIDEO_ID [VIDEO_ID ...]
    .venv/bin/python scripts/build_condensed_audio.py --ids-file watched.txt --single-file
"""
import argparse
from pathlib import Path

from french_mining.anki.collocation_note_type import MODEL_NAME as COLLOCATION_MODEL_NAME
from french_mining.anki.connect import AnkiConnectClient
from french_mining.anki.note_type import MODEL_NAME
from french_mining.anki.vocab_state import VocabularyState
from french_mining.condensed_audio import (
    DEFAULT_MIN_COMPREHENSIBILITY,
    select_comprehensible_spans,
    total_duration,
    video_comprehensibility,
)
from french_mining.nlp import parse_transcript
from french_mining.youtube.media import concat_audio_spans, download_audio
from french_mining.youtube.transcripts import download_subtitles, load_transcript


def _read_video_ids(args) -> list[str]:
    ids = list(args.video_ids)
    if args.ids_file:
        for line in Path(args.ids_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                ids.append(line)
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_ids", nargs="*", help="YouTube video IDs you've watched.")
    parser.add_argument("--ids-file", help="File with one video ID per line (# comments allowed).")
    parser.add_argument("--lang", default="fr")
    parser.add_argument("--min-comprehensibility", type=float, default=DEFAULT_MIN_COMPREHENSIBILITY)
    parser.add_argument("--order", choices=["easiest", "source"], default="easiest",
                        help="Playlist order: easiest-first (default) or as given.")
    parser.add_argument("--single-file", action="store_true",
                        help="Also concatenate everything into one condensed MP3.")
    parser.add_argument("--work-dir", default="condensed_audio")
    args = parser.parse_args()

    video_ids = _read_video_ids(args)
    if not video_ids:
        parser.error("provide at least one VIDEO_ID or --ids-file")

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    anki_client = AnkiConnectClient()
    vocab_state = VocabularyState.build(
        anki_client,
        model_names=[MODEL_NAME],
        collocation_model_names=[COLLOCATION_MODEL_NAME],
    )

    outputs: list[tuple[str, Path, float]] = []  # (video_id, condensed_path, comprehensibility)
    for video_id in video_ids:
        vid_dir = work_dir / video_id
        audio_path = download_audio(video_id, vid_dir)
        vtt_path = download_subtitles(video_id, vid_dir, lang=args.lang)
        sentences = parse_transcript(load_transcript(vtt_path))

        spans = select_comprehensible_spans(
            sentences, vocab_state, min_comprehensibility=args.min_comprehensibility
        )
        coverage = video_comprehensibility(sentences, vocab_state)
        if not spans:
            print(f"[{video_id}] {coverage:.0%} comprehensible — no sentences cleared the bar, skipping.")
            continue

        condensed_path = work_dir / f"{video_id}_condensed.mp3"
        concat_audio_spans(audio_path, spans, condensed_path)
        kept = total_duration(spans)
        print(f"[{video_id}] {coverage:.0%} comprehensible -> {kept:.0f}s kept across {len(spans)} span(s) -> {condensed_path.name}")
        outputs.append((video_id, condensed_path, coverage))

    if not outputs:
        print("\nNothing cleared the comprehensibility bar — try a lower --min-comprehensibility.")
        return

    if args.order == "easiest":
        outputs.sort(key=lambda o: o[2], reverse=True)

    playlist_path = work_dir / "condensed_playlist.m3u"
    playlist_path.write_text(
        "#EXTM3U\n" + "".join(f"#EXTINF:-1,{vid} ({cov:.0%})\n{path.name}\n" for vid, path, cov in outputs),
        encoding="utf-8",
    )
    print(f"\nWrote playlist ({len(outputs)} file(s), easiest-first): {playlist_path}")

    if args.single_file:
        # Stitch the per-video condensed files into one, in playlist order.
        single_path = work_dir / "condensed_all.mp3"
        _concat_files([path for _, path, _ in outputs], single_path)
        print(f"Wrote single file: {single_path}")


def _concat_files(paths: list[Path], output: Path) -> None:
    """Concatenate already-encoded per-video condensed MP3s into one file via
    ffmpeg's concat demuxer (uniform codec from concat_audio_spans, so -c copy)."""
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        for p in paths:
            fh.write(f"file '{p.resolve()}'\n")
        list_path = fh.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", str(output)],
            check=True, capture_output=True, text=True,
        )
    finally:
        Path(list_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
