# Project: Hum a Beat, Find a Loop

Four-person ML class project: retrieve drum loops from a library using a beatboxed or tapped audio query. A neural network maps queries and drum loops into a shared rhythm embedding space, so recordings with the same rhythm land close together.

Training data is synthetic. Drum patterns from GMD / E-GMD (MIDI) are rendered through drum kits to build the library, and rendered with real amateur beatbox sounds from AVP (plus added sloppiness) to build fake queries. Both sides come from the same MIDI pattern, so ground-truth matches are known exactly.

## Read these first

- `docs/README.md`: the full team plan (data, models, baselines, evaluation, experiments, roles, timeline, file contracts). Read the sections relevant to the current task before starting.
- `docs/music_spec.md`: every musical decision, written as rules. Owned by the music lead (Member A).

## Roles (who owns what)

- **A, music lead:** musical decisions, `docs/music_spec.md`, `config/music_spec.yaml`, the "same rhythm" function, listening tools, recording sessions.
- **B, data pipeline:** library and query manifests, query generator, splits, data loaders.
- **C, model:** audio features, encoder, training, embeddings.
- **D, evaluation:** answer key (`relevance.json`), `evaluate.py`, baselines, results tables.

Don't change another role's interfaces without flagging it to the user. The file formats in README section 12 are a contract between roles.

## Hard rules

- **All musical constants live in `config/music_spec.yaml`.** Load them through `src/music/spec.py`. Never hardcode thresholds, drum mappings, tempos, grid sizes, or sloppiness ranges anywhere else.
- **Never make a musical judgment call.** If a decision depends on how something sounds or what counts musically as "the same," stop and ask the user (or flag it for the music lead).
- **You can't hear audio.** When a result depends on sound, generate audio files and a simple way to audition them, and ask the user to listen.
- **Never commit audio, datasets, model checkpoints, or rendered files.** They live under `data/` and `results/` subfolders that are gitignored.
- **Never release audio rendered with commercial drum kits.** Only MIDI, manifests, and splits may be shared.
- **Keep splits disjoint.** Test patterns, test drum kits, and test AVP participants must never appear in training data.
- **Verify dataset facts against official docs** (GMD note mapping and metadata, AVP file layout, licenses, download URLs) rather than relying on memory.
- **Ask before** downloading large datasets or adding heavy dependencies.

## Repo layout

```
config/       music_spec.yaml and experiment configs
data/         raw and rendered audio (gitignored)
docs/         README.md, music_spec.md
manifests/    library_manifest.csv, query_manifest.csv, relevance.json
src/music/    spec loader, "same rhythm" grid function (A)
src/library/  loop cutting and rendering (B)
src/queries/  beatbox and tap query generation (B)
src/models/   features, encoder, training (C)
src/baselines/ CLAP, onset matching, transcribe-then-match (D)
src/eval/     evaluate.py and metrics (D)
tools/        listening and audition tools (A)
results/      embeddings, tables, figures (gitignored except final tables)
tests/
```

## Commands

<!-- Fill in once set up -->
- Environment setup (Python 3.11+; macOS/Linux shown, on Windows use `.venv\Scripts\`):
  `python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"`
  Exact versions that are known to work are in `requirements-lock.txt`.
- Run tests: `.venv/bin/pytest`
- Lint/format: `.venv/bin/ruff check . && .venv/bin/ruff format .`
- Imports: `src` and `tools` are packages, e.g. `from src.music.spec import load_spec`.

## Working style

- Propose a plan and wait for approval before large changes.
- Work in small, runnable steps. After each, say exactly what command to run and what to check.
- Write unit tests for anything other roles depend on, especially the "same rhythm" function and file-format writers.
- For purely technical choices, pick something sensible and state what you chose.
