"""Shared token-vocabulary definitions for the ephemeral training pipeline.

Every melody is reduced, in memory, to a sequence of *key-invariant* note
tokens: a relative pitch interval (semitones from the previous note) bucketed
into a small vocabulary, combined with a duration bucket. A separate
beat-position feature records where in the bar each note falls. None of this
retains the original absolute pitches/key or the source notation.
"""
from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Genres. Keeping the original spec's genre labels; Stage 1's `config/
# sources.yaml` maps each one onto a *real* public-domain data source /
# filter, since the literal "JRPG field" style tagging doesn't exist in the
# public domain ABC datasets we stream from.
# ---------------------------------------------------------------------------
GENRES: list[str] = [
    "JRPG_FIELD",
    "CHIPTUNE_8BIT",
    "LOFI_CHILL",
    "FANTASY_ORCHESTRA",
]
GENRE_TO_ID: dict[str, int] = {g: i for i, g in enumerate(GENRES)}
NUM_GENRES = len(GENRES)

# ---------------------------------------------------------------------------
# Pitch interval bucket: clamp relative semitone motion to [-12, +12].
# Two reserved buckets:
#   REST_PITCH_BUCKET  - no pitched note sounds on this step (a rest).
#   START_PITCH_BUCKET - the first sounded note of a sequence, which has no
#                         preceding pitch to compute an interval from.
# Keeping these distinct (rather than collapsing "no previous pitch" into the
# rest bucket) avoids teaching the model that "start of melody" and "silence"
# are the same musical event.
# ---------------------------------------------------------------------------
PITCH_INTERVAL_CLAMP = 12
REST_PITCH_BUCKET = 0
START_PITCH_BUCKET = 1
_INTERVAL_OFFSET = PITCH_INTERVAL_CLAMP + 2  # first real interval bucket starts at index 2
NUM_PITCH_BUCKETS = PITCH_INTERVAL_CLAMP * 2 + 3  # rest + start + [-12..+12] = 27


def pitch_interval_to_bucket(interval: int) -> int:
    """Map a real (non-None) semitone interval to its bucket, clamped to
    +/-12 semitones. Use `REST_PITCH_BUCKET` / `START_PITCH_BUCKET` directly
    for rests / sequence-initial notes instead of calling this."""
    clamped = max(-PITCH_INTERVAL_CLAMP, min(PITCH_INTERVAL_CLAMP, interval))
    return clamped + _INTERVAL_OFFSET


def pitch_bucket_to_interval(bucket: int) -> int | None:
    if bucket in (REST_PITCH_BUCKET, START_PITCH_BUCKET):
        return None
    return bucket - _INTERVAL_OFFSET


# ---------------------------------------------------------------------------
# Duration bucket, in quarter-note beats. Values chosen to cover the vast
# majority of durations found in monophonic dance-tune / hymn / folk ABC
# transcriptions (sixteenth notes through dotted whole notes), with bucket 8
# acting as a catch-all for anything longer.
# ---------------------------------------------------------------------------
DURATION_CLASSES_BEATS: list[float] = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
NUM_DUR_BUCKETS = len(DURATION_CLASSES_BEATS) + 1  # +1 catch-all "long" bucket = 9


def duration_beats_to_bucket(duration_beats: float) -> int:
    if duration_beats <= 0:
        return 0
    if duration_beats > DURATION_CLASSES_BEATS[-1] * 1.25:
        return len(DURATION_CLASSES_BEATS)  # catch-all bucket
    best_idx = min(
        range(len(DURATION_CLASSES_BEATS)),
        key=lambda i: abs(DURATION_CLASSES_BEATS[i] - duration_beats),
    )
    return best_idx


def duration_bucket_to_beats(bucket: int) -> float:
    if bucket >= len(DURATION_CLASSES_BEATS):
        return DURATION_CLASSES_BEATS[-1]
    return DURATION_CLASSES_BEATS[bucket]


# ---------------------------------------------------------------------------
# Beat position bucket: where within the bar a note starts, as a fraction of
# the bar length, quantized into 8 slices.
# ---------------------------------------------------------------------------
NUM_POS_BUCKETS = 8


def beat_position_to_bucket(position_in_bar: float, bar_length_beats: float) -> int:
    if bar_length_beats <= 0:
        return 0
    fraction = (position_in_bar % bar_length_beats) / bar_length_beats
    bucket = int(fraction * NUM_POS_BUCKETS)
    return max(0, min(NUM_POS_BUCKETS - 1, bucket))


# ---------------------------------------------------------------------------
# Composite "note token" used by the Markov model: pitch bucket and duration
# bucket packed into a single integer so a 2nd-order Markov chain only needs
# to track a single previous-token id for each of the two prior steps.
# ---------------------------------------------------------------------------
NOTE_VOCAB_SIZE = NUM_PITCH_BUCKETS * NUM_DUR_BUCKETS


def note_token(pitch_bucket: int, dur_bucket: int) -> int:
    return pitch_bucket * NUM_DUR_BUCKETS + dur_bucket


def note_token_to_buckets(token: int) -> tuple[int, int]:
    return token // NUM_DUR_BUCKETS, token % NUM_DUR_BUCKETS


@dataclass(frozen=True)
class NoteStep:
    """One tokenized note/rest event, fully key- and tempo-invariant."""

    pitch_bucket: int
    dur_bucket: int
    pos_bucket: int

    @property
    def note_token(self) -> int:
        return note_token(self.pitch_bucket, self.dur_bucket)
