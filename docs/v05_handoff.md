# v0.5 data checkpoint (2026-11-09)

Member B: Junior. This is a small, reproducible review batch for Joshua and a
loader compatibility check for C, before full seven-kit generation.

## What exists now

- 10 drum loops: seven train patterns and three validation patterns, on the
  installed training kit, AVL Black Pearl.
- 20 sloppy beatbox queries: Personal and Fixed versions per pattern.
- 20 clean references using the same mouth-sound selections for comparison.
- A saved seeded 20/3/5 participant assignment and split-resolved GMD catalog.
- Real AVP improvisations alongside the generated sounds on the listening page.
- A successful run through C's actual manifest loader, audio loader, R1 feature
  extractor, contrastive batch sampler, and relevance lookup.

The page is `results/v05_checkpoint/preview.html`. Generated audio and reports
are gitignored. Open it in a browser; nothing else needs running to listen on
the machine where this checkpoint was generated. The real AVP improvisation
is a reference for sound/feel, **not the same rhythm** as the target.

This batch is not the full v0 freeze. It has no test renders and only one kit.
Joshua still needs to review the sloppy queries. Internal vocal sample tails
still overlap: the approved cut affects the clip ending, not transitions between
mouth sounds. This is documented behavior, pending further listening guidance.

## Reproduce from a checkout

Use the existing project virtual environment and downloaded GMD/AVP data.
The normal core/dev install is sufficient for generation. For the independent
model-loader check, install the optional SciPy extra:

```bash
.venv/bin/python -m pip install -e ".[dev,handoff]"
```

If missing, fetch the small preview kit using the existing downloader:

```bash
.venv/bin/python -m tools.download_datasets avl_black_pearl
```

**One seeded generation command:**

```bash
.venv/bin/python -m src.build_checkpoint --seed 0
```

Or open `src/build_checkpoint.py` in the IDE and choose **Run Python File**.
It rebuilds the catalog, refreshes extracted AVP sounds if the spec version has
changed, checks the saved participant partition, and generates manifests,
audio, and the listening page. Same inputs/settings/seed regenerate identical
WAV bytes, verified by hashes. It does not download kits or dependencies.

`config/dataset_v0.yaml` holds B's build seed, split sizes, review batch sizes,
and normalization settings. All musical rules remain in `music_spec.yaml`.
The default batch uses train and validation only. `--splits train validation test`
can include test sources once an approved test kit is installed; this is still
a small review selection, not an all-pattern/all-kit generation command.
`--out` can put a separate review in another folder.

The primary assignment is `manifests/avp_participant_splits.csv`. It should be
committed alongside code/config. The generator checks an existing assignment
and rejects conflicting seeds/partitions rather than silently moving people
between splits. Keep seed 0 for this dataset version; reseeding participants
requires an explicit separate dataset/assignment decision.

## Audio and imperfection behavior

Every loop and query follows `spec.loops.audio`: 0.25 seconds before the first
downbeat, a cut at the exact end of the second bar, and a 20 ms fade inside that
ending. File length is `round((lead_in + beats * 60 / bpm) * sample_rate)`.
Source tempo is unchanged. Both old preview entry points now read this framing
instead of relying on the listening tools' pre-roll/tail defaults.

The production level policy chosen by B is a single gain toward -20 dBFS RMS,
limited by a -1 dBFS peak ceiling. These values live in `config/dataset_v0.yaml`.
The ceiling takes precedence when a clip has strong peaks; not every file
therefore achieves -20 dBFS RMS. Final end fading can lower RMS slightly. No
per-hit normalization is used, preserving within-recording relative levels.
This is RMS normalization, not perceptual/LUFS loudness matching. Joshua will
check hat audibility during listening.

Each sloppy query draws reproducible parameters from the approved ranges:

- One uniform steady offset for the clip.
- One uniform jitter bound, then an independent uniform displacement within
  +/- that bound for each hit.
- One drop probability per voice, then independent Bernoulli hit decisions.
- A target signal-to-noise ratio (SNR), with Gaussian white noise scaled using
  measured whole-clip signal/noise RMS before level normalization and end fade.

These distribution choices are documented implementation choices for Joshua's
v0 review. No tempo drift, swing drift, or mic EQ is added. Sample selections
are held fixed between a clean reference and its sloppy version; the event
record preserves both the clean and displaced frame positions and drop flags.
If every sound is dropped/outside the clip, generation fails explicitly rather
than producing an unlabeled noise-only query.

In `query_manifest.csv`, `drop_prob` is the average expected drop probability
over that query's original voiced events. `drop_prob_by_voice` is a JSON object
containing the actual voice-specific probabilities. `dropped_hits` is the
realized count. `jitter_ms` is the displacement bound; each event's actual jitter
is in `events.jsonl`. `snr_db` is requested SNR, and
`measured_snr_db_before_fade` confirms the implemented scaling. Do not interpret
the average probability as the observed dropped-hit fraction.

## Split rules and catalog changes

`src/splits.py` owns reproducible participant assignment and exact-pattern
precedence: test outranks validation, which outranks train. It retains all
copies inside the winning split, not only one recording. Patterns keep GMD's
original split; lower-ranked copies are removed rather than relabeled.

The regenerated GMD catalog has 8,643 clips: 6,904 train, 846 validation,
893 test. Three training copies were removed, leaving zero exact-pattern
clashes. `original_split_clashes.csv` and `split_removals.csv` retain the audit;
`split_conflicts.csv` now has headers only.

Participants are assigned 20 train / 3 validation / 5 test with seed 0. Query
participants always come from the target pattern's assigned split, with one
participant and one mode per recording. Not every assigned participant appears
in this small batch, but the full 28-person assignment is saved.

Kits are read through the spec. Train/validation sources use train kits; test
sources use test kits; null-split kits are skipped. The checkpoint chooses the
first installed compatible kit per split. Full generation across all five
training kits and both held-out kits remains to implement/run. Only Black Pearl
is currently installed locally; no large kit download was done for this batch.

## Files to inspect or hand off

Under `results/v05_checkpoint/`:

| File | Meaning |
|---|---|
| `preview.html` | Drum target, clean query, sloppy query, and real AVP reference players |
| `manifests/library_manifest.csv` | Ten rendered library rows in the team's contract format |
| `manifests/query_manifest.csv` | Twenty sloppy query rows, including actual effect parameters |
| `manifests/avp_participant_splits.csv` | Copy of the primary saved assignment |
| `audio/` | Library, sloppy queries, and clean reference WAVs |
| `references/` | Original AVP improvisations for listening comparison |
| `events.jsonl` | Sample provenance, clean/actual frames, and dropped-hit decisions |
| `listening_notes.csv` | Joshua's review fields; existing notes survive regeneration |
| `summary.json`, `build_settings.json` | Counts, status, settings, and source spec fingerprint |
| `audio_sha256.json` | Current generated audio hashes for reproducibility checks |
| `loader_check.json` | C's tested Git revision and per-file feature-check results |
| `SOURCES.md` | Copied dataset/kit source and attribution record |

Manifests use repository-relative audio paths for normal builds, and absolute
paths for an external output directory. C's loader resolves relative paths from
the repository root. Original MIDI paths remain relative to the GMD root. The
HTML uses relative audio links, so the complete checkpoint folder can be shared
for listening; manifest paths may need regeneration if the dataset is relocated.
Current manifests define the active set; older WAVs may remain after changes.

## C's loader check

The model is currently on the fetched `origin/c/model-core` branch. Do not merge
or copy it into `main` just to test the data. Run:

```bash
.venv/bin/python -m tools.check_model_loader --model-ref origin/c/model-core
```

The checker exports C's real config/data/features files from the exact local Git
revision to a temporary directory, loads them in the checker process, and removes
the temporary files afterward. Main's model package remains untouched. No mocks
or substitute loader are used. If the branch is unavailable locally, fetch the
team's branch before running. This is the locally fetched revision, not a live
verification of PR #1's latest head.

Verified revision: `2730c81101ab276080a60c066a497f7d5c16d629`.
The loader accepted all 30 manifest clips, all 20 queries had a matching target,
all 30 audio files produced finite R1 features, pattern-disjoint checks passed,
and the training sampler produced three contrastive batches. This is data/feature
compatibility, not a model training run or a retrieval performance result.

## Next checkpoint

Send Joshua this listening folder for Tuesday's review. Then obtain the remaining
approved kit samples, expand generation across every retained pattern and kit,
run loader/integrity checks on the expanded manifests, and give C the Wednesday
dataset handoff. The generator and split helpers are reusable foundations; the
complete seven-kit dataset and final listening approval are not finished here.
