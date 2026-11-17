"""ABC key-signature resolution.

ABC notation only writes accidentals that deviate from the key signature
implied by the ``K:`` header (e.g. ``K:D`` implicitly sharpens every F and C
note). To turn ABC pitches into correct absolute MIDI numbers we must know,
for a given tonic + mode, which of the seven natural note letters are
sharpened or flattened by default.

This module implements the standard "circle of fifths + modal offset"
algorithm used by reference tools such as ``abc2midi``.
"""
from __future__ import annotations

from dataclasses import dataclass

# Number of sharps (negative = flats) for each major-key tonic, keyed by the
# tonic's pitch class name as it would be written in ABC (natural, sharp or
# flat spellings all map onto the classic circle of fifths position).
_MAJOR_KEY_SHARPS: dict[str, int] = {
    "C": 0,
    "G": 1,
    "D": 2,
    "A": 3,
    "E": 4,
    "B": 5,
    "F#": 6,
    "C#": 7,
    "F": -1,
    "Bb": -2,
    "Eb": -3,
    "Ab": -4,
    "Db": -5,
    "Gb": -6,
    "Cb": -7,
}

# How many sharps a mode adds/removes relative to the Ionian (major) scale
# built on the same tonic letter. Derived from each mode's position as a
# rotation of the major scale (see module docstring / spec discussion).
_MODE_OFFSET: dict[str, int] = {
    "ion": 0,
    "maj": 0,
    "dor": -2,
    "phr": -4,
    "lyd": 1,
    "mix": -1,
    "aeo": -3,
    "min": -3,
    "m": -3,
    "loc": -5,
}

# Order in which sharps/flats are added to a key signature.
_SHARP_ORDER = ["F", "C", "G", "D", "A", "E", "B"]
_FLAT_ORDER = ["B", "E", "A", "D", "G", "C", "F"]

# Base semitone value (0-11, C=0) for the natural note letters.
NATURAL_SEMITONES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


@dataclass(frozen=True)
class KeySignature:
    tonic_letter: str          # "A".."G"
    tonic_accidental: int      # -1, 0, +1 (relative to the natural letter)
    mode: str                  # normalized mode key, e.g. "maj", "dor"
    sharps: int                # signed count, -7..7 (negative = flats)
    accidental_map: dict[str, int]  # e.g. {"F": 1, "C": 1} for D major

    def default_accidental(self, letter: str) -> int:
        return self.accidental_map.get(letter, 0)


def _normalize_tonic_name(letter: str, accidental: int) -> str:
    """Map a (letter, accidental) pair onto the keys of `_MAJOR_KEY_SHARPS`,
    using enharmonic fallbacks for spellings we don't have a direct table
    entry for (e.g. G# major -> treat like Ab major, D# -> Eb, A# -> Bb)."""
    suffix = {1: "#", -1: "b", 0: ""}[accidental]
    name = f"{letter}{suffix}"
    if name in _MAJOR_KEY_SHARPS:
        return name
    enharmonic = {
        "G#": "Ab",
        "D#": "Eb",
        "A#": "Bb",
        "E#": "F",
        "B#": "C",
        "Fb": "E",
        "Cb": "Cb",  # already valid (7 flats), kept for clarity
    }
    if name in enharmonic:
        return enharmonic[name]
    # Last resort: ignore the accidental rather than crash on odd input.
    return letter


def resolve_key_signature(tonic_letter: str, tonic_accidental: int, mode: str) -> KeySignature:
    """Compute the full key signature for a tonic + mode.

    Args:
        tonic_letter: one of "A".."G" (case-insensitive).
        tonic_accidental: -1 (flat), 0 (natural), +1 (sharp) applied to the tonic.
        mode: ABC mode string, e.g. "maj", "min", "dor", "mix", "lyd", "phr", "loc".
    """
    letter = tonic_letter.upper()
    if letter not in NATURAL_SEMITONES:
        raise ValueError(f"Invalid tonic letter: {tonic_letter!r}")
    mode_key = mode.lower()[:3] if mode else "maj"
    mode_key = {"ion": "ion", "maj": "maj"}.get(mode_key, mode_key)
    if mode_key not in _MODE_OFFSET:
        mode_key = "maj"

    major_name = _normalize_tonic_name(letter, tonic_accidental)
    base_sharps = _MAJOR_KEY_SHARPS.get(major_name, 0)
    sharps = base_sharps + _MODE_OFFSET[mode_key]
    sharps = max(-7, min(7, sharps))

    accidental_map: dict[str, int] = {}
    if sharps > 0:
        for note_letter in _SHARP_ORDER[:sharps]:
            accidental_map[note_letter] = 1
    elif sharps < 0:
        for note_letter in _FLAT_ORDER[: -sharps]:
            accidental_map[note_letter] = -1

    return KeySignature(
        tonic_letter=letter,
        tonic_accidental=tonic_accidental,
        mode=mode_key,
        sharps=sharps,
        accidental_map=accidental_map,
    )


def parse_key_header(key_field: str) -> KeySignature:
    """Parse the value of an ABC ``K:`` header, e.g. ``"D"``, ``"Ador"``,
    ``"Gmix"``, ``"F#m"``, ``"Bb"``, ``"C major"``, ``"HP"`` (bagpipe -> treat as D major fallback)."""
    text = (key_field or "C").strip()
    if not text or text.upper() in {"HP", "NONE"}:
        return resolve_key_signature("C", 0, "maj")

    i = 0
    letter = text[i].upper()
    if letter not in NATURAL_SEMITONES:
        letter = "C"
        i = 0
    else:
        i += 1
    accidental = 0
    if i < len(text) and text[i] == "#":
        accidental = 1
        i += 1
    elif i < len(text) and text[i].lower() == "b":
        accidental = -1
        i += 1

    rest = text[i:].strip()
    # Normalize things like "major", "Major", "minor", "dorian" down to the
    # 3-letter mode codes used internally.
    rest_lower = rest.lower().replace(" ", "")
    if rest_lower.startswith("maj") or rest_lower == "":
        mode = "maj"
    elif rest_lower.startswith("min") or rest_lower == "m":
        mode = "min"
    else:
        mode = rest_lower[:3] if rest_lower else "maj"

    return resolve_key_signature(letter, accidental, mode)
