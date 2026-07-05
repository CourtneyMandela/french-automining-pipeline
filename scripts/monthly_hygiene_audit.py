#!/usr/bin/env python3
"""Monthly deck hygiene audit (§11, build-order step 10): flags -- and,
with --suspend, suspends -- mature, high-frequency cards that immersion
has almost certainly already taught, so daily reviews stay focused on
genuine consolidation rather than maintenance. Suspends, never deletes.

    .venv/bin/python scripts/monthly_hygiene_audit.py
    .venv/bin/python scripts/monthly_hygiene_audit.py --suspend
"""
import argparse

from french_mining.anki.connect import AnkiConnectClient
from french_mining.anki.note_type import MODEL_NAME
from french_mining.hygiene import (
    DEFAULT_FREQUENCY_FLOOR,
    DEFAULT_INTERVAL_THRESHOLD_DAYS,
    find_hygiene_candidates,
    suspend_candidates,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suspend",
        action="store_true",
        help="Actually suspend flagged cards (default: dry run, just lists them).",
    )
    parser.add_argument("--interval-threshold-days", type=int, default=DEFAULT_INTERVAL_THRESHOLD_DAYS)
    parser.add_argument("--frequency-floor", type=int, default=DEFAULT_FREQUENCY_FLOOR)
    args = parser.parse_args()

    client = AnkiConnectClient()
    candidates = find_hygiene_candidates(
        client,
        model_names=[MODEL_NAME],
        interval_threshold_days=args.interval_threshold_days,
        frequency_floor=args.frequency_floor,
    )

    if not candidates:
        print("No hygiene candidates found.")
        return

    print(f"{len(candidates)} card(s) flagged as likely acquired via immersion:\n")
    for c in candidates:
        print(f"  [{c.lemma}] interval={c.interval_days}d frequency_rank={c.frequency_rank}")

    if args.suspend:
        suspended = suspend_candidates(client, candidates)
        print(f"\nSuspended {suspended} card(s) (reactivatable anytime in Anki).")
    else:
        print("\nDry run — pass --suspend to actually suspend these cards.")


if __name__ == "__main__":
    main()
