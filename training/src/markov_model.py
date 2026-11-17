"""2nd-order Markov transition table training and compact binary export.

See `training/FORMAT.md` for the exact `markov_bank.bin` byte layout this
module reads and writes.
"""
from __future__ import annotations

import struct
from collections import defaultdict

from .tokens import NOTE_VOCAB_SIZE

MAGIC = b"RMFM"
FORMAT_VERSION = 1

# context -> {next_token: probability}
MarkovTable = dict[tuple[int, int], dict[int, float]]


def build_markov_table(sequences: list[list[int]], smoothing: float = 0.0) -> MarkovTable:
    """Compute 2nd-order transition probabilities `P(next | prev2, prev1)`
    from note-token sequences (each inner list is one tune's tokens, in
    order). `smoothing` is an optional additive (Laplace) smoothing constant
    applied only across the next-tokens actually observed for a context
    (keeping the table sparse rather than allocating the full vocab)."""
    counts: dict[tuple[int, int], dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for seq in sequences:
        for i in range(2, len(seq)):
            context = (seq[i - 2], seq[i - 1])
            counts[context][seq[i]] += 1.0

    table: MarkovTable = {}
    for context, next_counts in counts.items():
        total = sum(next_counts.values()) + smoothing * len(next_counts)
        table[context] = {
            next_tok: (count + smoothing) / total for next_tok, count in next_counts.items()
        }
    return table


def write_markov_bank(path: str, genre_tables: dict[str, MarkovTable], vocab_size: int = NOTE_VOCAB_SIZE) -> None:
    """Write multiple genres' tables into one compact sparse binary file."""
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<H", FORMAT_VERSION))
        f.write(struct.pack("<H", vocab_size))
        f.write(struct.pack("<H", len(genre_tables)))
        for genre_id, (name, table) in enumerate(genre_tables.items()):
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<H", genre_id))
            f.write(struct.pack("<H", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<I", len(table)))
            for (prev2, prev1), next_probs in table.items():
                f.write(struct.pack("<HHH", prev2, prev1, len(next_probs)))
                for next_token, prob in next_probs.items():
                    f.write(struct.pack("<Hf", next_token, prob))


def read_markov_bank(path: str) -> tuple[int, dict[str, MarkovTable]]:
    """Reference reader, used for round-trip testing (and as the spec for
    the future Rust loader). Returns `(vocab_size, {genre_name: table})`."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(f"bad magic bytes: {magic!r}")
        (version,) = struct.unpack("<H", f.read(2))
        if version != FORMAT_VERSION:
            raise ValueError(f"unsupported markov_bank.bin version: {version}")
        (vocab_size,) = struct.unpack("<H", f.read(2))
        (num_genres,) = struct.unpack("<H", f.read(2))

        genres: dict[str, MarkovTable] = {}
        for _ in range(num_genres):
            f.read(2)  # genre_id (position in file order already gives this)
            (name_len,) = struct.unpack("<H", f.read(2))
            name = f.read(name_len).decode("utf-8")
            (num_contexts,) = struct.unpack("<I", f.read(4))
            table: MarkovTable = {}
            for _ in range(num_contexts):
                prev2, prev1, num_next = struct.unpack("<HHH", f.read(6))
                next_probs: dict[int, float] = {}
                for _ in range(num_next):
                    next_token, prob = struct.unpack("<Hf", f.read(6))
                    next_probs[next_token] = prob
                table[(prev2, prev1)] = next_probs
            genres[name] = table
        return vocab_size, genres
