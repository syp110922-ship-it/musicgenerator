"""Stage 1 orchestration: stream -> tokenize -> ephemeral DB -> train -> export -> cleanup.

This is the single entry point the rest of the project (and `run_training.py`)
calls. It intentionally keeps every stage small and independently testable
(see `tests/test_pipeline_offline.py` for a full offline run using fixture
data and mocked HTTP).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import yaml

from .abc_tokenizer import AbcParseError, parse_abc
from .data_sources import FixtureSource, MelodySource, build_source
from .ephemeral_db import EphemeralDatabase
from .markov_model import build_markov_table, write_markov_bank
from .neural_model import TinyGRUConfig
from .tokens import NOTE_VOCAB_SIZE, duration_bucket_to_beats, note_token
from .train_neural import export_safetensors, train_tiny_gru

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "sources.yaml")


@dataclass
class GenreResult:
    genre: str
    fetched: int = 0
    parsed: int = 0
    skipped_parse_errors: int = 0
    skipped_filtered: int = 0


@dataclass
class PipelineResult:
    output_dir: str
    markov_bank_path: str
    base_model_path: str
    genre_results: list[GenreResult] = field(default_factory=list)


def load_config(config_path: str | None = None) -> dict:
    path = config_path or DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _average_duration_beats(steps) -> float:
    sounded = [s for s in steps if s.dur_bucket is not None]
    if not sounded:
        return 0.0
    return sum(duration_bucket_to_beats(s.dur_bucket) for s in sounded) / len(sounded)


def _passes_density_filter(steps, density: str | None) -> bool:
    if not density:
        return True
    avg = _average_duration_beats(steps)
    if density == "dense":
        return avg < 0.6
    if density == "sparse":
        return avg >= 1.0
    return True


def _fetch_and_tokenize_genre(
    genre: str,
    genre_config: dict,
    db: EphemeralDatabase,
    max_tunes: int,
    min_tune_notes: int,
    source_override: MelodySource | None = None,
) -> GenreResult:
    result = GenreResult(genre=genre)
    source = source_override or build_source(genre_config)
    density = (genre_config.get("filters") or {}).get("density")

    for abc_text in source.stream_abc(genre_config.get("params", {}), max_tunes):
        result.fetched += 1
        try:
            parsed = parse_abc(abc_text)
        except AbcParseError as exc:
            result.skipped_parse_errors += 1
            logger.debug("Skipping unparsable %s tune: %s", genre, exc)
            continue

        if len(parsed.steps) < min_tune_notes:
            result.skipped_filtered += 1
            continue
        if not _passes_density_filter(parsed.steps, density):
            result.skipped_filtered += 1
            continue

        db.insert_tune(genre=genre, steps=parsed.steps)
        result.parsed += 1

    logger.info(
        "[%s] fetched=%d parsed=%d skipped_parse_errors=%d skipped_filtered=%d",
        genre,
        result.fetched,
        result.parsed,
        result.skipped_parse_errors,
        result.skipped_filtered,
    )
    return result


def run_pipeline(
    config_path: str | None = None,
    genres: list[str] | None = None,
    output_dir: str | None = None,
    tunes_per_genre: int | None = None,
    epochs: int | None = None,
    sources_override: dict[str, MelodySource] | None = None,
    neural_config: TinyGRUConfig | None = None,
) -> PipelineResult:
    """Run the full Stage 1 pipeline. Returns a `PipelineResult` describing
    what was written and basic per-genre fetch/parse telemetry.

    The ephemeral SQLite database is guaranteed to be deleted before this
    function returns or raises (see `EphemeralDatabase`), regardless of
    whether training succeeds.
    """
    config = load_config(config_path)
    all_genre_configs: dict = config["genres"]
    selected_genres = genres or list(all_genre_configs.keys())
    training_cfg = config.get("training", {})

    resolved_output_dir = output_dir or training_cfg.get("output_dir", "apps/web-studio/public/models")
    resolved_tunes_per_genre = tunes_per_genre or training_cfg.get("tunes_per_genre", 80)
    resolved_epochs = epochs or training_cfg.get("epochs", 8)
    min_tune_notes = training_cfg.get("min_tune_notes", 8)
    smoothing = training_cfg.get("markov_smoothing", 0.1)

    os.makedirs(resolved_output_dir, exist_ok=True)
    markov_bank_path = os.path.join(resolved_output_dir, "markov_bank.bin")
    base_model_path = os.path.join(resolved_output_dir, "base_model.safetensors")

    genre_results: list[GenreResult] = []

    with EphemeralDatabase() as db:
        for genre in selected_genres:
            genre_config = all_genre_configs[genre]
            source_override = (sources_override or {}).get(genre)
            genre_results.append(
                _fetch_and_tokenize_genre(
                    genre,
                    genre_config,
                    db,
                    resolved_tunes_per_genre,
                    min_tune_notes,
                    source_override=source_override,
                )
            )

        genres_with_data = db.distinct_genres()
        if not genres_with_data:
            raise RuntimeError(
                "No tunes were successfully tokenized for any genre — nothing to train on. "
                "Check network access / API response shapes (see README 'Verifying live data sources')."
            )

        # -- Hybrid statistical training: per-genre 2nd-order Markov tables --
        markov_tables = {}
        sequences_by_genre: dict[str, list[list[tuple[int, int, int]]]] = {}
        for genre in genres_with_data:
            tunes = db.fetch_sequences(genre)
            note_token_sequences = [
                [note_token(row.pitch_bucket, row.dur_bucket) for row in tune] for tune in tunes
            ]
            markov_tables[genre] = build_markov_table(note_token_sequences, smoothing=smoothing)
            sequences_by_genre[genre] = [
                [(row.pitch_bucket, row.dur_bucket, row.pos_bucket) for row in tune] for tune in tunes
            ]

        write_markov_bank(markov_bank_path, markov_tables, vocab_size=NOTE_VOCAB_SIZE)

        # -- Neural training: one shared TinyGRU across all genres --------
        model = train_tiny_gru(
            sequences_by_genre,
            config=neural_config,
            epochs=resolved_epochs,
            min_tune_length=min_tune_notes,
        )
        export_safetensors(model, base_model_path)

        # -- Verify outputs before the ephemeral DB is torn down ----------
        _verify_output(markov_bank_path, base_model_path)

    return PipelineResult(
        output_dir=resolved_output_dir,
        markov_bank_path=markov_bank_path,
        base_model_path=base_model_path,
        genre_results=genre_results,
    )


def _verify_output(markov_bank_path: str, base_model_path: str) -> None:
    for path in (markov_bank_path, base_model_path):
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            raise RuntimeError(f"Expected training output missing or empty: {path}")
    # Cheap structural sanity check beyond "file exists".
    from .markov_model import read_markov_bank

    vocab_size, genres = read_markov_bank(markov_bank_path)
    if vocab_size != NOTE_VOCAB_SIZE or not genres:
        raise RuntimeError("markov_bank.bin failed post-write verification")
