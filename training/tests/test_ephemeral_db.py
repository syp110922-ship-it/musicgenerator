import os

import pytest

from training.src.ephemeral_db import EphemeralDatabase
from training.src.tokens import NoteStep


def _steps(n):
    return [NoteStep(pitch_bucket=i % 5, dur_bucket=1, pos_bucket=0) for i in range(n)]


def test_insert_and_fetch_round_trip():
    with EphemeralDatabase() as db:
        db.insert_tune(genre="LOFI_CHILL", steps=_steps(4))
        db.insert_tune(genre="LOFI_CHILL", steps=_steps(3))
        db.insert_tune(genre="CHIPTUNE_8BIT", steps=_steps(5))

        assert set(db.distinct_genres()) == {"LOFI_CHILL", "CHIPTUNE_8BIT"}
        assert db.tune_count("LOFI_CHILL") == 2
        assert db.tune_count() == 3

        lofi = db.fetch_sequences("LOFI_CHILL")
        assert len(lofi) == 2
        assert [len(t) for t in sorted(lofi, key=len, reverse=True)] == [4, 3]


def test_file_is_deleted_on_normal_exit():
    with EphemeralDatabase() as db:
        path = db.path
        assert os.path.exists(path)
    assert not os.path.exists(path)


def test_file_is_deleted_even_on_exception():
    path_holder = {}
    with pytest.raises(RuntimeError):
        with EphemeralDatabase() as db:
            path_holder["path"] = db.path
            assert os.path.exists(db.path)
            raise RuntimeError("simulated training failure")
    assert not os.path.exists(path_holder["path"])


def test_cleanup_is_idempotent():
    db = EphemeralDatabase()
    path = db.path
    db.cleanup()
    assert not os.path.exists(path)
    db.cleanup()  # must not raise even though already closed/deleted
