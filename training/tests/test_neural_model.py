import os
import tempfile

import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors")

from safetensors.torch import load_file

from training.src.neural_model import TinyGRU, TinyGRUConfig
from training.src.train_neural import export_safetensors, train_tiny_gru
from training.src.tokens import NUM_DUR_BUCKETS, NUM_PITCH_BUCKETS


def test_tiny_gru_is_actually_tiny():
    model = TinyGRU()
    assert model.num_parameters() < 1_500_000


def test_forward_pass_output_shapes():
    model = TinyGRU()
    batch, seq_len = 2, 5
    genre_ids = torch.zeros(batch, dtype=torch.long)
    pitch_ids = torch.randint(0, NUM_PITCH_BUCKETS, (batch, seq_len))
    dur_ids = torch.randint(0, NUM_DUR_BUCKETS, (batch, seq_len))
    pos_ids = torch.randint(0, 8, (batch, seq_len))

    pitch_logits, dur_logits, hidden = model(genre_ids, pitch_ids, dur_ids, pos_ids)
    assert pitch_logits.shape == (batch, seq_len, NUM_PITCH_BUCKETS)
    assert dur_logits.shape == (batch, seq_len, NUM_DUR_BUCKETS)
    assert hidden.shape == (model.config.num_layers, batch, model.config.hidden_size)


def _toy_sequences_by_genre():
    # Small deterministic "melodies" per genre, long enough to train on.
    jrpg = [[(1, 3, 0), (14, 3, 1), (16, 3, 2), (12, 5, 3), (14, 3, 4)] for _ in range(6)]
    lofi = [[(1, 2, 0), (11, 4, 2), (13, 4, 4), (11, 6, 6)] for _ in range(6)]
    return {"JRPG_FIELD": jrpg, "LOFI_CHILL": lofi}


def test_training_loop_reduces_loss_and_exports_safetensors():
    sequences = _toy_sequences_by_genre()
    tiny_config = TinyGRUConfig(hidden_size=16, genre_dim=4, pitch_dim=8, dur_dim=4, pos_dim=4)

    model = train_tiny_gru(sequences, config=tiny_config, epochs=3, min_tune_length=3)
    assert model.num_parameters() < 1_500_000

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "base_model.safetensors")
        export_safetensors(model, path)
        assert os.path.getsize(path) > 0

        reloaded_state = load_file(path)
        original_state = model.state_dict()
        assert set(reloaded_state.keys()) == set(original_state.keys())
        for key in original_state:
            assert torch.allclose(reloaded_state[key], original_state[key])


def test_training_raises_on_empty_input():
    with pytest.raises(ValueError):
        train_tiny_gru({"JRPG_FIELD": []}, epochs=1)
