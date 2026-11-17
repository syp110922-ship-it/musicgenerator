"""RustMidi-Forge offline training pipeline (Stage 1).

Everything in this package runs entirely in memory / against a single
ephemeral SQLite file in the OS temp directory. No raw audio, MIDI, or ABC
source text is ever written permanently to disk by this package: only the
final, abstracted model artifacts (``*.safetensors`` and ``*.bin``) are
persisted, directly into the web app's static model directory.
"""
