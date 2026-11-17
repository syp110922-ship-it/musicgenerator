# Stage 1 output artifact formats

These are the two files the offline training pipeline writes into the web
app's static model directory (default: `apps/web-studio/public/models/`).
They are the **only** artifacts that persist; the ephemeral SQLite staging
database is always deleted before the pipeline returns (see
`src/ephemeral_db.py`).

This document is the contract the future Rust/WASM loader (Stage 2) must
implement a reader for.

## Token vocabulary (shared by both files)

Defined in `src/tokens.py`:

| Feature | Size | Meaning |
| --- | --- | --- |
| Pitch bucket | 27 | `0`=rest, `1`=start-of-sequence (no previous pitch), `2..26` = clamped relative semitone interval `-12..+12` |
| Duration bucket | 9 | nearest of `[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]` quarter-note beats, `8` = catch-all "longer" bucket |
| Beat-position bucket | 8 | which 1/8 slice of the bar the note starts in |
| Note token (Markov vocab) | `27 * 9 = 243` | `pitch_bucket * 9 + dur_bucket` |
| Genre id | 4 | index into `["JRPG_FIELD", "CHIPTUNE_8BIT", "LOFI_CHILL", "FANTASY_ORCHESTRA"]` |

## `markov_bank.bin` — genre-indexed 2nd-order Markov transition tables

Sparse binary format (only observed `(prev2, prev1) -> next` contexts are
stored; the vast majority of the dense `243^3` table is never visited by
real melodies, so a dense table would be wasteful). All integers
little-endian.

```
MAGIC               4 bytes   ASCII "RMFM"
VERSION             u16       format version, currently 1
NOTE_VOCAB_SIZE      u16       e.g. 243 (sanity check for the reader)
NUM_GENRES          u16       number of genre tables that follow
-- repeated NUM_GENRES times --
  GENRE_ID          u16       index into the genre list
  NAME_LEN          u16       byte length of the UTF-8 genre name
  NAME              bytes     UTF-8 genre name (e.g. "LOFI_CHILL")
  NUM_CONTEXTS      u32       number of distinct (prev2, prev1) contexts
  -- repeated NUM_CONTEXTS times --
    PREV2           u16
    PREV1           u16
    NUM_NEXT        u16       number of possible next tokens from this context
    -- repeated NUM_NEXT times --
      NEXT_TOKEN    u16
      PROBABILITY   f32       in (0, 1], sums to ~1.0 across one context's entries
```

Reference writer/reader: `src/markov_model.py` (`write_markov_bank` /
`read_markov_bank`).

## `base_model.safetensors` — shared Tiny-GRU neural weights

A single [safetensors](https://github.com/huggingface/safetensors) file
holding the `state_dict()` of the `TinyGRU` module defined in
`src/neural_model.py`. Tensor names (subject to the `TinyGRU` definition,
listed here for convenience and kept in sync by `tests/test_neural_model.py`):

- `genre_embedding.weight`      — `[NUM_GENRES, genre_dim]`
- `pitch_embedding.weight`      — `[NUM_PITCH_BUCKETS, pitch_dim]`
- `dur_embedding.weight`        — `[NUM_DUR_BUCKETS, dur_dim]`
- `pos_embedding.weight`        — `[NUM_POS_BUCKETS, pos_dim]`
- `input_proj.weight` / `.bias` — projects concatenated embeddings to `hidden_size`
- `gru.weight_ih_l0`, `gru.weight_hh_l0`, `gru.bias_ih_l0`, `gru.bias_hh_l0` (layer 0)
- `gru.weight_ih_l1`, `gru.weight_hh_l1`, `gru.bias_ih_l1`, `gru.bias_hh_l1` (layer 1)
- `pitch_head.weight` / `.bias` — `hidden_size -> NUM_PITCH_BUCKETS`
- `dur_head.weight` / `.bias`   — `hidden_size -> NUM_DUR_BUCKETS`

A small `safetensors` metadata header also stores `hidden_size`, embedding
dims, and the vocab sizes so the Rust/WASM loader can reconstruct the exact
network shape without hard-coding it twice.
