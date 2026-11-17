"""In-memory ABC notation parsing and tokenization.

This module never touches disk: it takes a raw ABC tune string (as returned
by a streaming HTTP API) and reduces it, purely in memory, to a list of
`NoteStep` tokens — key-invariant relative pitch intervals paired with
quantized note durations and beat positions. Absolute pitch, key, tempo and
the original notation text are all discarded once tokenization completes.

Known MVP limitations (documented rather than silently wrong):
  * Chords (`[CEG]`) take only their first note.
  * Broken rhythm markers (`>`, `<`) are not yet rebalanced between notes.
  * In-bar accidental propagation (an accidental "sticking" for the rest of
    the measure in classical ABC semantics) is not modeled; only the
    explicit accidental on each note or the key signature default is used.
  * Multi-voice (`V:`) tunes are flattened into a single stream since the
    pipeline only models monophonic melody.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .key_signature import NATURAL_SEMITONES, parse_key_header
from .tokens import (
    NoteStep,
    REST_PITCH_BUCKET,
    START_PITCH_BUCKET,
    beat_position_to_bucket,
    duration_beats_to_bucket,
    duration_bucket_to_beats,
    pitch_interval_to_bucket,
)


class AbcParseError(ValueError):
    """Raised when an ABC tune cannot be reduced to any usable note tokens."""


_HEADER_RE = re.compile(r"^([A-Za-z]):\s?(.*)$")
_LENGTH_SUFFIX = r"\d*/*\d*"
_TOKEN_RE = re.compile(
    r"(?P<tie>-)"
    r"|(?P<rest>[zxZX])(?P<rlen>" + _LENGTH_SUFFIX + r")"
    r"|(?P<acc>\^\^|__|\^|_|=)?(?P<note>[A-Ga-g])(?P<oct>[,']*)(?P<nlen>" + _LENGTH_SUFFIX + r")"
)
_ACCIDENTAL_VALUE = {"^": 1, "^^": 2, "_": -1, "__": -2, "=": 0}

_CHORD_SYMBOL_RE = re.compile(r'"[^"]*"')
_GRACE_NOTE_RE = re.compile(r"\{[^}]*\}")
_DECORATION_BANG_RE = re.compile(r"!([^!]*)!")
_LINE_CONTINUATION_RE = re.compile(r"\\\s*$")


def _strip_comment(line: str) -> str:
    idx = line.find("%")
    if idx != -1:
        line = line[:idx]
    return line


def _split_header_body(abc_text: str) -> tuple[dict[str, str], list[str]]:
    """Any line of the form `<letter>:<value>` is metadata; everything else
    is tune body. This is simpler (and robust enough for monophonic dance
    tune / hymn transcriptions) than tracking an explicit in-header state
    machine, because body lines never legitimately start with `letter:`."""
    headers: dict[str, str] = {}
    body_lines: list[str] = []
    for raw_line in abc_text.splitlines():
        line = _strip_comment(raw_line)
        line = _LINE_CONTINUATION_RE.sub("", line).strip()
        if not line:
            continue
        if line[:2].lower() == "w:":
            continue  # lyrics line, irrelevant to melody tokens
        m = _HEADER_RE.match(line)
        if m:
            headers.setdefault(m.group(1).upper(), m.group(2).strip())
            continue
        body_lines.append(line)
    return headers, body_lines


def _parse_fraction(text: str, default: tuple[int, int] = (1, 4)) -> tuple[int, int]:
    text = (text or "").strip()
    if "/" in text:
        num_s, _, den_s = text.partition("/")
        try:
            return int(num_s or 1), int(den_s or 1)
        except ValueError:
            return default
    try:
        return int(text), 1
    except ValueError:
        return default


def _parse_meter_beats_per_bar(meter_text: str) -> float:
    text = (meter_text or "4/4").strip()
    if text.upper() == "C":
        return 4.0
    if text in ("C|", "c|"):
        return 4.0
    num, den = _parse_fraction(text, default=(4, 4))
    if den <= 0:
        den = 4
    return num * (4.0 / den)


def _default_unit_beats(meter_text: str, length_text: str | None) -> float:
    if length_text:
        l_num, l_den = _parse_fraction(length_text, default=(1, 8))
        if l_den <= 0:
            l_den = 8
        return (l_num / l_den) * 4.0
    # Per ABC convention: if the meter's numeric ratio is >= 0.75 the implied
    # default unit note length is 1/8, otherwise 1/16.
    text = (meter_text or "4/4").strip()
    if text.upper() in ("C", "C|"):
        ratio = 1.0
    else:
        num, den = _parse_fraction(text, default=(4, 4))
        ratio = num / den if den else 1.0
    return 0.5 if ratio >= 0.75 else 0.25


def _parse_length_multiplier(suffix: str) -> float:
    suffix = suffix or ""
    if not suffix:
        return 1.0
    if "/" not in suffix:
        try:
            return float(int(suffix))
        except ValueError:
            return 1.0
    num_part, _, rest = suffix.partition("/")
    numerator = int(num_part) if num_part else 1
    extra_slashes = rest.count("/")
    digits = rest.replace("/", "")
    denominator = int(digits) if digits else 2
    denominator *= 2 ** extra_slashes
    if denominator <= 0:
        denominator = 1
    return numerator / denominator


def _clean_body(body_text: str) -> str:
    body_text = _CHORD_SYMBOL_RE.sub(" ", body_text)
    body_text = _GRACE_NOTE_RE.sub(" ", body_text)
    body_text = _DECORATION_BANG_RE.sub(" ", body_text)
    # Chords [CEG...]: keep only the first pitch letter+accidental+octave.
    body_text = re.sub(
        r"\[(\^\^|__|\^|_|=)?([A-Ga-g])([,']*)[^\]]*\]",
        lambda m: f"{m.group(1) or ''}{m.group(2)}{m.group(3) or ''}",
        body_text,
    )
    return body_text


@dataclass
class ParsedTune:
    steps: list[NoteStep]
    bar_length_beats: float
    unit_beats: float


def parse_abc(abc_text: str) -> ParsedTune:
    """Parse one ABC tune (headers + body, as a single string) into a list
    of key-invariant `NoteStep` tokens. Raises `AbcParseError` if no usable
    notes are found."""
    if not abc_text or not abc_text.strip():
        raise AbcParseError("empty ABC text")

    headers, body_lines = _split_header_body(abc_text)
    meter_text = headers.get("M", "4/4")
    bar_length_beats = _parse_meter_beats_per_bar(meter_text)
    unit_beats = _default_unit_beats(meter_text, headers.get("L"))
    key_sig = parse_key_header(headers.get("K", "C"))

    body_text = _clean_body(" ".join(body_lines))

    steps: list[NoteStep] = []
    prev_pitch_midi: int | None = None
    cumulative_beats = 0.0
    pending_tie = False

    for m in _TOKEN_RE.finditer(body_text):
        if m.group("tie"):
            pending_tie = True
            continue

        if m.group("rest"):
            dur = unit_beats * _parse_length_multiplier(m.group("rlen"))
            pos_bucket = beat_position_to_bucket(cumulative_beats, bar_length_beats)
            steps.append(
                NoteStep(
                    pitch_bucket=REST_PITCH_BUCKET,
                    dur_bucket=duration_beats_to_bucket(dur),
                    pos_bucket=pos_bucket,
                )
            )
            cumulative_beats += dur
            pending_tie = False
            continue

        letter = m.group("note")
        acc_str = m.group("acc") or ""
        oct_marks = m.group("oct") or ""
        length_suffix = m.group("nlen") or ""

        semitone = NATURAL_SEMITONES[letter.upper()]
        acc_val = _ACCIDENTAL_VALUE[acc_str] if acc_str else key_sig.default_accidental(letter.upper())
        base_octave = 4 if letter.isupper() else 5
        octave = base_octave + oct_marks.count("'") - oct_marks.count(",")
        midi_pitch = (octave + 1) * 12 + semitone + acc_val
        dur = unit_beats * _parse_length_multiplier(length_suffix)

        if pending_tie and steps and prev_pitch_midi == midi_pitch:
            prev_step = steps[-1]
            merged_beats = duration_bucket_to_beats(prev_step.dur_bucket) + dur
            steps[-1] = NoteStep(
                pitch_bucket=prev_step.pitch_bucket,
                dur_bucket=duration_beats_to_bucket(merged_beats),
                pos_bucket=prev_step.pos_bucket,
            )
            cumulative_beats += dur
            pending_tie = False
            continue

        if prev_pitch_midi is None:
            pitch_bucket = START_PITCH_BUCKET
        else:
            pitch_bucket = pitch_interval_to_bucket(midi_pitch - prev_pitch_midi)

        pos_bucket = beat_position_to_bucket(cumulative_beats, bar_length_beats)
        steps.append(
            NoteStep(
                pitch_bucket=pitch_bucket,
                dur_bucket=duration_beats_to_bucket(dur),
                pos_bucket=pos_bucket,
            )
        )
        cumulative_beats += dur
        prev_pitch_midi = midi_pitch
        pending_tie = False

    if not steps:
        raise AbcParseError("no note/rest tokens found in ABC body")

    return ParsedTune(steps=steps, bar_length_beats=bar_length_beats, unit_beats=unit_beats)
