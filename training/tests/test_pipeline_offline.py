"""End-to-end offline test of the full Stage 1 pipeline: fixture ABC tunes ->
in-memory tokenization -> ephemeral SQLite -> Markov + neural training ->
safetensors/.bin export -> guaranteed cleanup. No network access required."""
import os
import tempfile

import pytest

torch = pytest.importorskip("torch")

from training.src.data_sources import FixtureSource
from training.src.markov_model import read_markov_bank
from training.src.neural_model import TinyGRUConfig
from training.src.pipeline import run_pipeline
from training.src.tokens import NOTE_VOCAB_SIZE

# A handful of short, hand-written, unambiguously public-domain-style ABC
# fragments reused across all four genres purely to exercise the pipeline's
# control flow end-to-end (NOT meant to produce musically meaningful output
# at these volumes).
_SAMPLE_TUNES = [
    """X:1
T:Fixture Tune A
M:4/4
L:1/8
K:D
DEFG ABcd | efed cBAG | FEDC DEFG | A2 F2 D4 |
""",
    """X:2
T:Fixture Tune B
M:4/4
L:1/8
K:G
GABc defg | abag fedc | BAGF GABc | d2 B2 G4 |
""",
    """X:3
T:Fixture Tune C
M:4/4
L:1/8
K:Ador
A2BA GABd | eaag edBd | A2BA GABd | eaag edBA |
""",
    """X:4
T:Fixture Tune D
M:4/4
L:1/8
K:C
CDEF GABc | dcBA GFED | CDEF GABc | d2 c2 c4 |
""",
]


# LOFI_CHILL's genre config filters for note-sparse (long-duration) tunes
# (avg duration >= 1.0 beat), so it needs its own fixture set distinct from
# the eighth-note-heavy `_SAMPLE_TUNES` above (which average ~0.59 beats/note
# and are deliberately used to satisfy CHIPTUNE_8BIT's "dense" filter).
_SPARSE_SAMPLE_TUNES = [
    """X:5
T:Fixture Tune Lofi A
M:4/4
L:1/4
K:C
C D E F | G A B2 | c B A G | F2 E2 |
""",
    """X:6
T:Fixture Tune Lofi B
M:4/4
L:1/4
K:G
G A B c | d2 B2 | A G F E | D2 C2 |
""",
]


def _fixture_sources_for_all_genres():
    return {
        "JRPG_FIELD": FixtureSource(_SAMPLE_TUNES),
        "CHIPTUNE_8BIT": FixtureSource(_SAMPLE_TUNES),
        "LOFI_CHILL": FixtureSource(_SPARSE_SAMPLE_TUNES),
        "FANTASY_ORCHESTRA": FixtureSource(_SAMPLE_TUNES),
    }


def test_full_pipeline_runs_offline_and_cleans_up():
    tiny_config = TinyGRUConfig(hidden_size=16, genre_dim=4, pitch_dim=8, dur_dim=4, pos_dim=4)

    with tempfile.TemporaryDirectory() as tmp_out:
        result = run_pipeline(
            genres=["JRPG_FIELD", "CHIPTUNE_8BIT", "LOFI_CHILL", "FANTASY_ORCHESTRA"],
            output_dir=tmp_out,
            tunes_per_genre=4,
            epochs=2,
            sources_override=_fixture_sources_for_all_genres(),
            neural_config=tiny_config,
        )

        assert os.path.exists(result.markov_bank_path)
        assert os.path.exists(result.base_model_path)
        assert os.path.getsize(result.markov_bank_path) > 0
        assert os.path.getsize(result.base_model_path) > 0

        vocab_size, genre_tables = read_markov_bank(result.markov_bank_path)
        assert vocab_size == NOTE_VOCAB_SIZE
        assert set(genre_tables.keys()) == {
            "JRPG_FIELD",
            "CHIPTUNE_8BIT",
            "LOFI_CHILL",
            "FANTASY_ORCHESTRA",
        }
        for table in genre_tables.values():
            assert len(table) > 0

        for genre_result in result.genre_results:
            assert genre_result.parsed > 0

    # No stray ephemeral database files left anywhere under the system temp dir.
    temp_root = tempfile.gettempdir()
    leftover = [f for f in os.listdir(temp_root) if f.startswith("rustmidi_forge_train_")]
    assert leftover == []


def test_pipeline_cleans_up_even_when_training_fails(monkeypatch):
    """If something downstream blows up, the ephemeral DB must still be deleted."""
    from training.src import pipeline as pipeline_module

    def boom(*args, **kwargs):
        raise RuntimeError("simulated neural training crash")

    monkeypatch.setattr(pipeline_module, "train_tiny_gru", boom)

    with tempfile.TemporaryDirectory() as tmp_out:
        with pytest.raises(RuntimeError, match="simulated neural training crash"):
            run_pipeline(
                genres=["JRPG_FIELD"],
                output_dir=tmp_out,
                tunes_per_genre=4,
                epochs=1,
                sources_override={"JRPG_FIELD": FixtureSource(_SAMPLE_TUNES)},
            )

    temp_root = tempfile.gettempdir()
    leftover = [f for f in os.listdir(temp_root) if f.startswith("rustmidi_forge_train_")]
    assert leftover == []


def test_pipeline_raises_when_no_genre_has_usable_data():
    with tempfile.TemporaryDirectory() as tmp_out:
        with pytest.raises(RuntimeError, match="No tunes were successfully tokenized"):
            run_pipeline(
                genres=["JRPG_FIELD"],
                output_dir=tmp_out,
                tunes_per_genre=4,
                epochs=1,
                sources_override={"JRPG_FIELD": FixtureSource(["not valid abc at all"])},
            )
