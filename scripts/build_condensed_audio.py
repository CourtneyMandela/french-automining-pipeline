#!/usr/bin/env python3
"""Build comprehensibility-filtered condensed audio from YouTube videos
you've watched — a playlist of dense, silence-free, few-minutes-each clips
made of only the sentences you can already follow, for passive review on
the go. Chunked (not one long file per video) so shuffling actually works
and a dud clip is easy to skip past, instead of always starting at the
beginning and scrubbing to find your place.

Entirely local and free (yt-dlp + ffmpeg + spaCy + AnkiConnect) — no
Anthropic API calls. Run on the machine where Anki is open.

Requires (see README): the French spaCy model, network for yt-dlp, ffmpeg,
and Anki running with AnkiConnect (read-only — no cards are written).

    .venv/bin/python scripts/build_condensed_audio.py VIDEO_ID [VIDEO_ID ...]
    .venv/bin/python scripts/build_condensed_audio.py --ids-file watched.txt --clip-minutes 3 --single-file
"""
import argparse
from pathlib import Path

from french_mining.anki.collocation_note_type import MODEL_NAME as COLLOCATION_MODEL_NAME
from french_mining.anki.connect import AnkiConnectClient
from french_mining.anki.note_type import MODEL_NAME
from french_mining.anki.vocab_state import VocabularyState
from french_mining.condensed_audio import (
    DEFAULT_CLIP_SECONDS,
    DEFAULT_MIN_COMPREHENSIBILITY,
    chunk_spans,
    select_comprehensible_spans,
    total_duration,
    video_comprehensibility,
)
from french_mining.nlp import parse_transcript
from french_mining.youtube.media import concat_audio_spans, download_audio
from french_mining.youtube.pipeline import get_transcript
from french_mining.youtube.whisper_transcribe import DEFAULT_MODEL_SIZE as DEFAULT_WHISPER_MODEL_SIZE


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
    parser.add_argument(
        "--clip-minutes", type=float, default=DEFAULT_CLIP_SECONDS / 60,
        help="Target clip length in minutes (a video's own kept spans are chunked into clips this long).",
    )
    parser.add_argument("--order", choices=["easiest", "source"], default="easiest",
                        help="Playlist order: easiest-video-first (default) or as given; a video's own clips always stay together and in order.")
    parser.add_argument("--single-file", action="store_true",
                        help="Also concatenate every clip into one big MP3 (opt-in -- defeats the point of shuffling).")
    parser.add_argument(
        "--no-whisper-fallback",
        action="store_true",
        help="Skip videos with no captions instead of transcribing them locally with Whisper.",
    )
    parser.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL_SIZE)
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

    # (video_id, clip_index, clip_path, video_coverage, clip_duration) per clip.
    clips: list[tuple[str, int, Path, float, float]] = []
    for video_id in video_ids:
        vid_dir = work_dir / video_id
        audio_path = download_audio(video_id, vid_dir)
        # Falls back to local Whisper transcription (no API cost, just
        # slower) when the video has no captions at all -- otherwise those
        # videos would silently contribute nothing.
        segments, transcript_source = get_transcript(
            video_id,
            vid_dir,
            lang=args.lang,
            use_whisper_fallback=not args.no_whisper_fallback,
            whisper_model_size=args.whisper_model,
        )
        sentences = parse_transcript(segments)
        print(f"[{video_id}] transcript source: {transcript_source}")

        spans = select_comprehensible_spans(
            sentences, vocab_state, min_comprehensibility=args.min_comprehensibility
        )
        coverage = video_comprehensibility(sentences, vocab_state)
        if not spans:
            print(f"[{video_id}] {coverage:.0%} comprehensible — no sentences cleared the bar, skipping.")
            continue

        video_chunks = chunk_spans(spans, target_duration=args.clip_minutes * 60)
        for i, chunk in enumerate(video_chunks, start=1):
            clip_path = work_dir / f"{video_id}_clip{i:02d}.mp3"
            concat_audio_spans(audio_path, chunk, clip_path)
            clips.append((video_id, i, clip_path, coverage, total_duration(chunk)))

        print(
            f"[{video_id}] {coverage:.0%} comprehensible -> {total_duration(spans):.0f}s kept "
            f"across {len(video_chunks)} clip(s)"
        )

    if not clips:
        print("\nNothing cleared the comprehensibility bar — try a lower --min-comprehensibility.")
        return

    if args.order == "easiest":
        # Stable sort by video coverage descending -- a video's own clips
        # (already in chronological/clip-index order) stay grouped together.
        clips.sort(key=lambda c: c[3], reverse=True)

    playlist_path = work_dir / "condensed_playlist.m3u"
    playlist_path.write_text(
        "#EXTM3U\n"
        + "".join(
            f"#EXTINF:{int(duration)},{video_id} clip {index:02d} ({coverage:.0%})\n{path.name}\n"
            for video_id, index, path, coverage, duration in clips
        ),
        encoding="utf-8",
    )
    print(f"\nWrote playlist ({len(clips)} clip(s) across {len(video_ids)} video(s)): {playlist_path}")
    print("Shuffle this playlist in your player of choice.")

    if args.single_file:
        # Stitch every clip into one file, in playlist order -- opt-in only;
        # this is the one-big-file pattern the chunking above is meant to move away from.
        single_path = work_dir / "condensed_all.mp3"
        _concat_files([path for _, _, path, _, _ in clips], single_path)
        print(f"Wrote single file: {single_path}")


def _concat_files(paths: list[Path], output: Path) -> None:
    """Concatenate already-encoded clip MP3s into one file via ffmpeg's
    concat demuxer (uniform codec from concat_audio_spans, so -c copy)."""
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
