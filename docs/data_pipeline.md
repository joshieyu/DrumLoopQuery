# Data pipeline: MIDI clips and AVP sound index

Owner: Member B (Junior). This documents the first pipeline steps added after the
week 1 handoff. It does not replace the team plan or Joshua's music spec.

## What this step does

Read one downloaded Groove MIDI Dataset (GMD) performance, export its drum
events, and extract the first content-qualified raw loop window.
MIDI stores performance instructions, not
recorded sound: pitch identifies a drum, velocity describes hit strength, and
timing describes when to play it.

Joshua's inventory explores the datasets and compares possible musical rules.
This step starts B's pipeline by following the approved configuration for source
selection. It does not produce training data yet.

## Run from the IDE

1. Select the repository's `.venv/bin/python` as your Python interpreter.
2. Open `src/library/inspect_midi.py` and choose **Run Python File**.
3. Open `results/midi_inspection/report.html` in a browser. In VS Code, you can
   right-click the file, reveal it in Finder, then double-click it.

The script uses repository-relative defaults, so its data paths do not depend on
the IDE's working directory. It needs the downloaded GMD data and the editable
project installation from environment setup.

Terminal equivalent, for reference:

```bash
.venv/bin/python -m src.library.inspect_midi
```

Choose another performance using its exact `id` from `info.csv`:

```bash
.venv/bin/python -m src.library.inspect_midi --performance-id drummer1/eval_session/1
```

## Source selection

The script reads `data/raw/gmd/groove/info.csv`, applies the time signature and
performance tempo range from `load_spec()`, and excludes fill recordings when
configured. It chooses the first eligible performance sorted by ID. An explicit
ID must also satisfy those filters. This is an inspection sample, not a training
split selection: the source may belong to the official test split.

No kit mapping, musical threshold, or tempo range is copied into this script.
It does not change the original MIDI or another role's interfaces.

## Outputs and how to read them

The gitignored `results/midi_inspection/` folder contains:

| File | Contents |
|---|---|
| `report.html` | Metadata and a preview of the first 30 drum hits |
| `hits.csv` | All original drum hits, in time order |
| `metadata.json` | Source path, original split, tempo, timing resolution, spec version, and unknown pitches |

Each hit has these fields:

| Field | Meaning |
|---|---|
| `pitch` | MIDI drum identifier; the spec assigns it to a voice |
| `velocity` | Original hit strength |
| `tick` | Absolute position in MIDI timing units |
| `beat` | Absolute position in beats: `tick / ticks_per_beat` |
| `start_sec`, `end_sec` | Original timing in seconds |
| `spec_group` | Configured voice, `drop`, or `unknown` |

Beat positions start at zero and are relative to the whole performance. Only
drum tracks are exported. Quiet hits and hits mapped to `drop` are deliberately
preserved here: the later label-building stage applies the rhythm rules, while
drum audio can still contain dropped voices. Unknown pitches are flagged rather
than silently discarded. Re-running replaces the inspection outputs.

This step does not play audio. To audition MIDI, load it into a drum-instrument
track in GarageBand or Logic; Joshua should review instrument mappings before
we generate training audio.

## What to check

- The source path exists and the report contains drum hits.
- Hits are ordered by time and beat positions are nondecreasing.
- The original train/validation/test split is visible and preserved.
- Unknown pitches are reviewed before downstream processing.

## First raw clip (implemented)

`src/library/clips.py` supplies `first_clip()`, returning a `Clip` containing
`Note(pitch, velocity, beat)` objects with positions relative to the window start.
The inspection script runs this automatically and adds a clip section to the
same HTML report. The current configured two-bar 4/4 window spans eight beats.

The implementation:

1. Reads the length, required starting voices, ghost thresholds, drum mapping,
   and ending exclusion from the shared spec.
2. Counts only complete bars within the available MIDI note span, then reserves
   the configured number of final complete bars. A trailing partial bar is not
   counted. This conservative source-span policy is explicit and may need
   refinement when the production dataset cutter is agreed.
3. Finds the first remaining bar containing all required voices, counting hits
   at or above their voice's quiet-hit threshold.
4. Requires enough space for the full window at that start. It reports failure
   rather than shortening the clip.
5. Includes original onsets in `[start_beat, end_beat)`: the start is included,
   the end is excluded. Subtracts `start_beat` to produce relative positions.
   Quiet hits and dropped voices remain in this raw clip.

Window boundaries currently use raw timing, without quantization. A hit just
before the start stays outside, even if it was intended as an early downbeat.
This is a provisional inspection boundary policy, not the final rhythmic
membership rule. Confirm it with Joshua before scaling up the dataset. No flam
merging, grid labels, duplicate removal, or musical equivalence is implemented.

Additional outputs under `results/midi_inspection/`:

| File | Contents |
|---|---|
| `clip_hits.csv` | Pitch, velocity, relative beat, original source beat, configured group |
| `clip_metadata.json` | Source details, zero-based start bar, window boundaries, spec version, boundary policy |

For example, source beat 9.5 in a clip starting at beat 8 becomes relative beat
1.5. `start_bar=0` means the first bar; `start_bar=1` means the second bar.
The source split is copied unchanged. This does not make a test performance
eligible for training.

If a performance is too short, has no qualifying bar, or contains unknown
pitches, the report explains why no clip was produced. Old clip outputs are
removed so they cannot be mistaken for the current performance's clip.

Validation: `tests/test_clips.py` checks count-in skipping, relative timing,
exact end exclusion, preserved quiet/dropped hits, insufficient source length,
and unknown pitches. `tests/test_midi_inspection.py` checks the raw export.

## Next step and dependencies

Before final loop selection and labels, coordinate with Joshua on loop stride
(`loops.hop_bars`), the retained flam timing (`same_rhythm.flam_keep`), and the
remaining rhythm-matching decisions. `clip_grid()` and `dualize()` are still
unimplemented shared interfaces. Do not substitute private labeling logic.

Drum rendering, synthetic beatbox generation, final manifests, duplicate
removal, and split-leakage checks are later checkpoints. This inspection does
not claim that a dataset is ready for model training.

## AVP sound index (implemented)

AVP is the Amateur Vocal Percussion dataset: recordings of people imitating drum
sounds with their mouths. The [official version 4 description](https://zenodo.org/records/5036529)
documents onset and class annotations, Personal/Fixed modes, and both isolated
sound repetitions and improvisations. The local file layout is handled by the
existing `tools.inventory.avp.load_avp()` loader, reused without changing its
interface. This index is a catalog, not a collection of newly cut audio samples.

### Run and inspect

Open `src/queries/index_avp.py` in the IDE and select **Run Python File**, using
the project's `.venv` interpreter. The script has already been run on the local
download. Terminal equivalent:

```bash
.venv/bin/python -m src.queries.index_avp
```

Generated files are under the gitignored `results/avp_index/` directory:

| File | Purpose |
|---|---|
| `utterance_index.csv` | One row per usable annotated onset |
| `annotation_issues.csv` | Exact source rows with validation or normalization issues |
| `counts.csv` | Sound counts by participant, mode, recording kind, and label |
| `summary.json` | Dataset root, source link, totals, and unresolved split/sample status |

Open the index CSV to see the catalog; open the summary JSON for overall counts.
Re-running replaces these four outputs. No audio is copied, edited, or combined.

### Index fields

| Field | Meaning |
|---|---|
| `utterance_id` | Relative annotation path plus one-based parsed CSV row; stable for unchanged source files |
| `participant_id` | Source participant, such as `P3`; retained for future participant-disjoint splits |
| `mode`, `kind` | Original Personal/Fixed mode and recording type, including Improvisation |
| `label`, `sound_class` | Source class code (`kd`, `sd`, `hhc`, `hho`) and readable name |
| `audio_path`, `annotation_path` | Paths relative to the AVP root recorded in `summary.json` |
| `annotation_row` | One-based parsed annotation row; not a sample index into the audio |
| `onset_s` | Annotated start time in seconds within the source recording |
| `samplerate`, `channels`, `duration_s` | Source recording properties; duration is the whole file, not the individual sound |
| `onset_phoneme`, `coda_phoneme` | Original optional mouth-sound annotations, retained when available |

The index includes both modes and all recording kinds for exploration. It does
not choose which sounds should be used for synthesis; that choice belongs to
the music spec. There is deliberately no end-time or split column: neither
sample boundaries nor participant split assignments have been decided here.

### Validation and local results

Whitespace around valid labels is removed and logged; blank/unknown labels,
nonfinite or out-of-range onsets, and all rows sharing a duplicate onset within
one recording are excluded and logged. Missing companion annotation files are
reported. Missing roots, empty inputs, malformed CSVs, or unreadable audio fail
with an error rather than producing an apparently complete catalog.

The local run produced 9,776 indexed annotations from 9,778 loaded annotation
rows, across 280 recordings and 28 participants. Three labels required whitespace
trimming. One blank label and one onset outside its audio were excluded; no
annotation files were missing. These are measured local counts, rather than
the publication's headline total or a reproduction of the older handoff counts.
Consult `annotation_issues.csv` for the exact affected rows.

`tests/test_avp_index.py` uses tiny generated recordings to check source identity,
stable identifiers, whitespace normalization, invalid labels/onsets, missing
annotations, duplicate onsets, and unchanged original audio bytes.

### What this enables next

A future generator can select, for example, a kick annotation from one participant
and locate it in the source WAV. Before extracting samples, define the end-time
and decay policy with Joshua. Before training, assign participant-disjoint splits
and select the approved mode/recording kinds. Cataloging everyone together is
not permission to use everyone's sounds for training.
