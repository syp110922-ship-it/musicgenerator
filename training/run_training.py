#!/usr/bin/env python3
"""Stage 1 CLI entry point: offline data collection + model training.

Usage examples
--------------
  # Full live run, all 4 genres, writing straight into the web app's model dir:
  python training/run_training.py --source live

  # Smoke test with bundled fixture ABC tunes, no network access required
  # (what this sandbox uses to validate the pipeline logic):
  python training/run_training.py --source fixtures --genres JRPG_FIELD

  # Faster iteration: fewer tunes/epochs, custom output dir.
  python training/run_training.py --tunes-per-genre 20 --epochs 3 --output-dir /tmp/models
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.src.data_sources import FixtureSource
from training.src.pipeline import load_config, run_pipeline
from training.src.tokens import GENRES
from training.tests.fixtures.sample_tunes import (
    OCTAVE_AND_REST,
    SAMPLE_RESPONSE_ABC_REEL,
    TIE_AND_KEY_SIGNATURE,
    TWINKLE_TWINKLE,
)

_FIXTURE_TUNES_FOR_SMOKE_TEST = [
    TWINKLE_TWINKLE,
    TIE_AND_KEY_SIGNATURE,
    OCTAVE_AND_REST,
    SAMPLE_RESPONSE_ABC_REEL,
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source",
        choices=["live", "fixtures"],
        default="live",
        help="'live' streams the real public-domain APIs in config/sources.yaml; "
        "'fixtures' replays a handful of bundled ABC snippets with no network access, "
        "for smoke-testing the pipeline wiring (NOT real training data).",
    )
    parser.add_argument("--genres", nargs="+", choices=GENRES, default=None, help="Subset of genres to train (default: all).")
    parser.add_argument("--config", default=None, help="Path to sources.yaml (default: training/config/sources.yaml).")
    parser.add_argument("--output-dir", default=None, help="Override the model output directory.")
    parser.add_argument("--tunes-per-genre", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    sources_override = None
    if args.source == "fixtures":
        selected = args.genres or GENRES
        sources_override = {genre: FixtureSource(_FIXTURE_TUNES_FOR_SMOKE_TEST) for genre in selected}
        logging.getLogger(__name__).warning(
            "--source fixtures uses a handful of bundled ABC snippets, not real genre data. "
            "This only validates that the pipeline runs end-to-end; do not ship the resulting model."
        )

    try:
        result = run_pipeline(
            config_path=args.config,
            genres=args.genres,
            output_dir=args.output_dir,
            tunes_per_genre=args.tunes_per_genre,
            epochs=args.epochs,
            sources_override=sources_override,
        )
    except Exception:
        logging.getLogger(__name__).exception("Stage 1 training pipeline failed")
        return 1

    print("\n=== Stage 1 complete ===")
    print(f"Markov bank:  {result.markov_bank_path}")
    print(f"Base model:   {result.base_model_path}")
    for gr in result.genre_results:
        print(
            f"  {gr.genre:<20} fetched={gr.fetched:<4} parsed={gr.parsed:<4} "
            f"parse_errors={gr.skipped_parse_errors:<4} filtered={gr.skipped_filtered}"
        )
    print("\nEphemeral SQLite database has been closed and deleted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
