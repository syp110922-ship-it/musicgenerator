# Stage 1 — Offline Data Collection & Model Training

This is the first of the project's three-stage local execution workflow:

1. **Stage 1 (this directory):** stream public-domain melody data, train the
   models, export weight files into `apps/web-studio/public/models/`.
2. **Stage 2 (not built yet):** compile the Rust core engine to WASM via `wasm-pack`.
3. **Stage 3 (not built yet):** run the Vite dev server for the web studio.

Everything in Stage 1 is designed around one rule: **nothing but the final
abstracted model weights is ever permanently written to disk.** No raw ABC
text, no MIDI, no audio, and only a single ephemeral SQLite file (in the OS
temp dir, never inside this repo) that is guaranteed to be deleted — on
success *or* failure — before the process exits.

## The five pipeline components

| # | Component | File |
| - | --- | --- |
| 1 | Compact neural network (2-layer Tiny-GRU) | `src/neural_model.py` |
| 2 | In-memory ABC parsing & key-invariant tokenization | `src/abc_tokenizer.py`, `src/key_signature.py`, `src/tokens.py` |
| 3 | API streaming into a temporary SQLite DB | `src/data_sources.py`, `src/ephemeral_db.py` |
| 4 | Hybrid statistical (Markov) + neural training | `src/markov_model.py`, `src/train_neural.py` |
| 5 | Guaranteed ephemeral cleanup | `src/ephemeral_db.py` (`EphemeralDatabase`, used as a context manager in `src/pipeline.py`) |

`src/pipeline.py` wires all five together; `run_training.py` is the CLI.

## Setup

```bash
cd training
python3 -m venv ../.venv       # or any venv location you like
source ../.venv/bin/activate
pip install -r requirements.txt
```

No `miditok` dependency: ABC parsing/tokenization is implemented directly in
`src/abc_tokenizer.py` so the exact `(genre, pitch-interval, duration,
beat-position)` token format described in the spec is fully under our
control and unit-tested (`tests/test_abc_tokenizer.py`).

## Running it

```bash
# Full live run (needs normal internet access — see the warning below):
python training/run_training.py --source live

# Fast offline smoke test using a few bundled ABC snippets (no network,
# NOT real training data — just proves the pipeline wiring works):
python training/run_training.py --source fixtures --tunes-per-genre 4 --epochs 2 --output-dir /tmp/smoke_models
```

Useful flags: `--genres JRPG_FIELD CHIPTUNE_8BIT`, `--tunes-per-genre N`,
`--epochs N`, `--output-dir path`, `-v` for debug logging. Defaults come from
`config/sources.yaml`.

Output (written straight into `apps/web-studio/public/models/` by default):

- `base_model.safetensors` — shared Tiny-GRU weights, genre-conditioned (~0.8 MB at default size).
- `markov_bank.bin` — per-genre 2nd-order Markov transition tables (sparse binary, see `FORMAT.md`).

## ⚠️ Verifying live data sources

This repository was developed in a network-restricted sandbox (outbound
access limited to github/npm/pypi hosts), so **the exact JSON response
shapes of `thesession.org` and `datasets-server.huggingface.co` were
implemented from their documented/expected API contracts, not confirmed
against a live call.** `src/data_sources.py` is unit-tested against mocked
responses that mirror those docs (`tests/test_data_sources.py`), and is
written defensively (it tries several candidate field names and skips
anything that doesn't look like ABC text), but **before trusting a real
training run**, please:

1. Run `python training/run_training.py --source live --genres JRPG_FIELD -v`
   and check the `fetched=.. parsed=..` log line. If `parsed` is 0 while
   `fetched` is > 0, the ABC field name differs from what's configured —
   check the raw JSON once (e.g. `curl` the URLs in `config/sources.yaml`'s
   comments) and update `abc_fields:` in `config/sources.yaml` accordingly
   (no code changes needed).
2. Spot check a few genres' `skipped_filtered` counts — if a genre's
   "dense"/"sparse" duration filter in `config/sources.yaml` is rejecting
   almost everything, loosen the thresholds in
   `src/pipeline.py::_passes_density_filter`.

## Genre → data source mapping (v1 best-effort)

The original spec's genre names (JRPG field, chiptune, lo-fi, fantasy
orchestra) describe a *feel*; real public-domain ABC/MIDI datasets are
mostly traditional folk dance tunes and public-domain classical scores. See
the comment block at the top of `config/sources.yaml` for the current
mapping and rationale — it's intentionally just a config file so you can
retune it without touching code once you've listened to real output.

## Known tokenizer limitations (MVP, documented rather than silently wrong)

- Chords (`[CEG]`) use only their first note.
- Broken rhythm (`A>B`, `A<B`) is not yet rebalanced between the two notes.
- In-bar accidental propagation (an accidental "sticking" for the rest of a
  measure, per classical ABC semantics) is not modeled — only an explicit
  per-note accidental or the key signature default is used.
- Multi-voice (`V:`) tunes are flattened into a single note stream.

## Tests

```bash
cd /path/to/musicgenerator
.venv/bin/python -m pytest training/ -q
```

All 27 tests run fully offline (HTTP calls are mocked; the neural model
tests use `pytest.importorskip("torch")` so the suite still passes green in
an environment without `torch` installed, just skipping those cases).
