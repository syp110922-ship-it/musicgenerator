import math
import os
import tempfile

from training.src.markov_model import build_markov_table, read_markov_bank, write_markov_bank


def test_build_markov_table_basic_probabilities():
    # Sequence: 1,2,3,2,3,1  -> contexts (1,2)->3 (twice), (2,3)->2 (once), (3,2)->3(once), (2,3)? etc
    sequences = [[1, 2, 3, 2, 3, 1]]
    table = build_markov_table(sequences)
    # context (1,2) is followed by 3 both times it occurs? Let's check actual pairs:
    # i=2: ctx(1,2)->3 ; i=3: ctx(2,3)->2 ; i=4: ctx(3,2)->3 ; i=5: ctx(2,3)->1
    assert table[(1, 2)] == {3: 1.0}
    assert table[(3, 2)] == {3: 1.0}
    assert set(table[(2, 3)].keys()) == {2, 1}
    assert math.isclose(sum(table[(2, 3)].values()), 1.0)
    assert math.isclose(table[(2, 3)][2], 0.5)
    assert math.isclose(table[(2, 3)][1], 0.5)


def test_markov_bank_binary_round_trip():
    seq_lofi = [[1, 2, 3, 2, 3, 1, 4, 5]]
    seq_chip = [[9, 8, 7, 8, 7, 9]]
    tables = {
        "LOFI_CHILL": build_markov_table(seq_lofi),
        "CHIPTUNE_8BIT": build_markov_table(seq_chip),
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "markov_bank.bin")
        write_markov_bank(out_path, tables, vocab_size=243)
        assert os.path.getsize(out_path) > 0

        vocab_size, loaded = read_markov_bank(out_path)
        assert vocab_size == 243
        assert set(loaded.keys()) == {"LOFI_CHILL", "CHIPTUNE_8BIT"}

        for genre, original_table in tables.items():
            loaded_table = loaded[genre]
            assert set(loaded_table.keys()) == set(original_table.keys())
            for context, next_probs in original_table.items():
                for next_tok, prob in next_probs.items():
                    assert math.isclose(loaded_table[context][next_tok], prob, rel_tol=1e-5)


def test_smoothing_keeps_probabilities_normalized():
    sequences = [[1, 2, 3, 2, 3, 1]]
    table = build_markov_table(sequences, smoothing=0.5)
    for next_probs in table.values():
        assert math.isclose(sum(next_probs.values()), 1.0, rel_tol=1e-6)
