# RustMidi-Forge

A 100%-local AI music generation system: a pure Rust core engine (Markov +
tiny neural inference), compiled to WebAssembly, powering a Gemini-style web
studio — with zero backend servers and zero retained copyrighted training
data.

## Three-stage local execution workflow

1. **Stage 1 — Offline data collection & model training** *(current status: implemented)*
   See [`training/README.md`](training/README.md). Streams public-domain
   melody data by genre, tokenizes it in memory, trains a 2nd-order Markov
   model + a Tiny-GRU neural model, and exports both directly into
   `apps/web-studio/public/models/`. No raw source data or audio is ever
   retained — a single ephemeral SQLite file is guaranteed to be deleted
   before the process exits.
2. **Stage 2 — WebAssembly engine compilation** *(not built yet)*
   Will compile the pure-Rust `rustmidi-core` engine to a browser-ready
   WASM binary via `wasm-pack build --release`.
3. **Stage 3 — Web studio server launch** *(not built yet)*
   Will run the Vite dev server for the Gemini-style web UI. Stages 2 and 3
   will be chained into one combined script (e.g. `npm run dev`) for local
   development convenience.

Each stage must complete, in order, before the next: the web app's model
directory and WASM package are build inputs, not things it generates itself.

## Repository layout (grows one stage at a time)

```
training/              Stage 1 — Python offline training pipeline (done)
apps/web-studio/        Stage 3 — web app; currently just public/models/
                        as the Stage 1 output target. UI/Rust/WASM come later.
```

The full target architecture (Cargo workspace with `rustmidi-core` /
`rustmidi-wasm` / `rustmidi-ffi`, the `@rustmidi/forge-sdk` NPM package, the
Netlify deployment config, etc.) is being built incrementally, stage by
stage, rather than scaffolded all at once.
