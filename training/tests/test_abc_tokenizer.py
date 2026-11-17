import pytest

from training.src.abc_tokenizer import AbcParseError, parse_abc
from training.src.tokens import (
    NUM_DUR_BUCKETS,
    NUM_POS_BUCKETS,
    PITCH_INTERVAL_CLAMP,
    REST_PITCH_BUCKET,
    START_PITCH_BUCKET,
    pitch_bucket_to_interval,
    pitch_interval_to_bucket,
)
from .fixtures.sample_tunes import (
    HEADER_ONLY_NO_BODY,
    OCTAVE_AND_REST,
    SAMPLE_RESPONSE_ABC_REEL,
    TIE_AND_KEY_SIGNATURE,
    TWINKLE_TWINKLE,
)


def test_twinkle_twinkle_intervals_and_durations():
    tune = parse_abc(TWINKLE_TWINKLE)
    steps = tune.steps
    assert len(steps) == 14

    intervals = [pitch_bucket_to_interval(s.pitch_bucket) for s in steps]
    # First note has no previous pitch -> dedicated START bucket, not an
    # interval of 0 and not confused with a rest.
    assert steps[0].pitch_bucket == START_PITCH_BUCKET
    assert intervals[0] is None

    expected_intervals_from_second = [0, 7, 0, 2, 0, -2, -2, 0, -1, 0, -2, 0, -2]
    assert intervals[1:] == expected_intervals_from_second

    # Quarter notes (L:1/4, no suffix) -> 1.0 beat -> bucket index 3 in
    # DURATION_CLASSES_BEATS; the "G2"/"C2" half notes -> 2.0 beats -> idx 5.
    durations = [s.dur_bucket for s in steps]
    assert durations[0] == 3  # C (1 beat)
    assert durations[3] == 3  # G (1 beat)
    assert durations[6] == 5  # G2 (2 beats)
    assert durations[-1] == 5  # C2 (2 beats)


def test_key_signature_sharp_applied_and_tie_merges_duration():
    tune = parse_abc(TIE_AND_KEY_SIGNATURE)
    steps = tune.steps
    # F2 F2 F4 | G2-G2 A2 z2  ->  F F F (tie doesn't apply, same pitch no tie
    # between them, so 3 separate tokens) then tied G (merged into one token
    # covering both G2's), then A, then a rest.
    assert len(steps) == 6  # F, F, F, G(merged), A, rest

    assert steps[0].pitch_bucket == START_PITCH_BUCKET
    # F -> F -> F: zero relative motion.
    assert pitch_bucket_to_interval(steps[1].pitch_bucket) == 0
    assert pitch_bucket_to_interval(steps[2].pitch_bucket) == 0

    # D major key signature sharpens F by default (no explicit accidental in
    # the source), so the sounding pitch is F#. F# -> G (natural, no sharp in
    # D major) is only a semitone (minor second).
    assert pitch_bucket_to_interval(steps[3].pitch_bucket) == 1

    # The tied G2-G2 must merge into a single token of combined duration
    # (1.0 + 1.0 = 2.0 beats -> bucket index 5), not two separate tokens.
    assert steps[3].dur_bucket == 5

    # G -> A is +2 semitones (whole tone).
    assert pitch_bucket_to_interval(steps[4].pitch_bucket) == 2

    # Final token is the trailing rest.
    assert steps[5].pitch_bucket == REST_PITCH_BUCKET


def test_octave_marks_and_rest_values():
    tune = parse_abc(OCTAVE_AND_REST)
    steps = tune.steps
    assert len(steps) == 6
    intervals = [pitch_bucket_to_interval(s.pitch_bucket) for s in steps]
    assert steps[0].pitch_bucket == START_PITCH_BUCKET
    # C, (MIDI 48) -> E (MIDI 64): +16 semitones, clamped to +12.
    assert intervals[1] == PITCH_INTERVAL_CLAMP
    # E (64) -> G (67): +3
    assert intervals[2] == 3
    # G (67) -> c (72): +5
    assert intervals[3] == 5
    # c (72) -> e (76): +4
    assert intervals[4] == 4
    assert steps[5].pitch_bucket == REST_PITCH_BUCKET


def test_header_only_tune_raises_parse_error():
    with pytest.raises(AbcParseError):
        parse_abc(HEADER_ONLY_NO_BODY)


def test_empty_string_raises_parse_error():
    with pytest.raises(AbcParseError):
        parse_abc("")


def test_sample_reel_parses_and_all_buckets_in_range():
    tune = parse_abc(SAMPLE_RESPONSE_ABC_REEL)
    assert len(tune.steps) > 8
    for step in tune.steps:
        assert 0 <= step.dur_bucket < NUM_DUR_BUCKETS
        assert 0 <= step.pos_bucket < NUM_POS_BUCKETS


def test_pitch_interval_bucket_roundtrip_within_clamp():
    for interval in range(-PITCH_INTERVAL_CLAMP, PITCH_INTERVAL_CLAMP + 1):
        bucket = pitch_interval_to_bucket(interval)
        assert pitch_bucket_to_interval(bucket) == interval


def test_pitch_interval_bucket_clamps_extremes():
    assert pitch_bucket_to_interval(pitch_interval_to_bucket(999)) == PITCH_INTERVAL_CLAMP
    assert pitch_bucket_to_interval(pitch_interval_to_bucket(-999)) == -PITCH_INTERVAL_CLAMP
