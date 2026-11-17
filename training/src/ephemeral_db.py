"""Temporary SQLite staging database for abstracted token sequences.

Design goals (per the ephemeral-pipeline spec):
  * The database file lives in the OS temp directory (never inside the
    project/repo), and only ever holds abstracted integer token sequences —
    never raw ABC/MIDI text or audio.
  * The connection is always explicitly closed before the file is deleted
    (required on Windows to release the file lock), inside a guaranteed
    teardown path (``try/finally``) so a crash mid-training can never leave
    a stray database file behind.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import tempfile
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS token_sequences (
    tune_id     INTEGER NOT NULL,
    genre       TEXT    NOT NULL,
    seq_index   INTEGER NOT NULL,
    pitch_bucket INTEGER NOT NULL,
    dur_bucket   INTEGER NOT NULL,
    pos_bucket   INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_genre ON token_sequences(genre, tune_id, seq_index);
"""


@dataclass(frozen=True)
class TokenRow:
    tune_id: int
    genre: str
    seq_index: int
    pitch_bucket: int
    dur_bucket: int
    pos_bucket: int


class EphemeralDatabase:
    """A SQLite database file that exists only for the lifetime of one
    training run. Use as a context manager::

        with EphemeralDatabase() as db:
            db.insert_tune(genre="LOFI_CHILL", tune_id=1, steps=[...])
            sequences = db.fetch_sequences("LOFI_CHILL")
        # file is guaranteed gone here, even if an exception was raised.
    """

    def __init__(self) -> None:
        fd, path = tempfile.mkstemp(prefix="rustmidi_forge_train_", suffix=".sqlite")
        os.close(fd)
        self.path = path
        self._conn: sqlite3.Connection | None = sqlite3.connect(self.path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._next_tune_id = 1
        logger.debug("Ephemeral training database created at %s", self.path)

    # -- context manager -------------------------------------------------
    def __enter__(self) -> "EphemeralDatabase":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.cleanup()

    # -- data access -------------------------------------------------------
    def insert_tune(self, genre: str, steps, tune_id: int | None = None) -> int:
        assert self._conn is not None, "database already closed"
        if tune_id is None:
            tune_id = self._next_tune_id
            self._next_tune_id += 1
        rows = [
            (tune_id, genre, idx, step.pitch_bucket, step.dur_bucket, step.pos_bucket)
            for idx, step in enumerate(steps)
        ]
        self._conn.executemany(
            "INSERT INTO token_sequences "
            "(tune_id, genre, seq_index, pitch_bucket, dur_bucket, pos_bucket) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
        return tune_id

    def fetch_sequences(self, genre: str) -> list[list[TokenRow]]:
        """Return one ordered list of `TokenRow` per tune for the given genre."""
        assert self._conn is not None, "database already closed"
        cur = self._conn.execute(
            "SELECT tune_id, genre, seq_index, pitch_bucket, dur_bucket, pos_bucket "
            "FROM token_sequences WHERE genre = ? ORDER BY tune_id, seq_index",
            (genre,),
        )
        tunes: dict[int, list[TokenRow]] = {}
        for tune_id, g, seq_index, pitch_bucket, dur_bucket, pos_bucket in cur.fetchall():
            tunes.setdefault(tune_id, []).append(
                TokenRow(tune_id, g, seq_index, pitch_bucket, dur_bucket, pos_bucket)
            )
        return [tunes[k] for k in sorted(tunes)]

    def distinct_genres(self) -> list[str]:
        assert self._conn is not None, "database already closed"
        cur = self._conn.execute("SELECT DISTINCT genre FROM token_sequences")
        return [row[0] for row in cur.fetchall()]

    def tune_count(self, genre: str | None = None) -> int:
        assert self._conn is not None, "database already closed"
        if genre is None:
            cur = self._conn.execute("SELECT COUNT(DISTINCT tune_id) FROM token_sequences")
        else:
            cur = self._conn.execute(
                "SELECT COUNT(DISTINCT tune_id) FROM token_sequences WHERE genre = ?", (genre,)
            )
        return cur.fetchone()[0]

    # -- guaranteed teardown ----------------------------------------------
    def cleanup(self) -> None:
        """Explicitly close the connection (releasing Windows file locks)
        then permanently delete the temp file. Safe to call multiple times."""
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None
        if self.path and os.path.exists(self.path):
            try:
                os.remove(self.path)
                logger.debug("Ephemeral training database deleted: %s", self.path)
            except OSError as exc:
                logger.warning("Could not delete ephemeral database %s: %s", self.path, exc)

    def __del__(self) -> None:  # pragma: no cover - best-effort safety net
        try:
            self.cleanup()
        except Exception:
            pass
