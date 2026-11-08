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

## Labeled loop extraction (updated for spec v0.4)

The original single-window checkpoint has been replaced with spec-based cutting
in `src/library/clips.py`. `cut_clips()` returns all qualifying clips from one
performance. `first_clip()` returns the first for the inspection report. Both
require the performance BPM because flam merging uses milliseconds.

The cutter now:

1. Considers complete source bars, reserves `loops.drop_last_bars`, and ignores
   a trailing partial bar. The source span is still measured from the MIDI's last
   note end, matching the listening prototype's duration convention.
2. Finds the first bar containing the required start voices **on the shared
   quantized grid**, so an early downbeat can qualify the intended bar.
3. Starts successive windows every `loops.hop_bars` bars (currently two).
4. Passes original notes from **one beat before to one beat after** each window
   to `clip_grid()`, with positions relative to the window start. This follows
   the newer week 3 handoff, superseding the smaller v0.3 boundary margin.
5. Uses `has_min_hits()` to reject insufficient content and `pattern_id()` to
   remove duplicate patterns within the same performance, as configured.
6. Stores the three-voice grid, its pattern ID, and the dualized tap-pattern ID.
   No alternative hashing, rounding, or flam-merging logic is used.

A `Clip` contains `start_bar`, `start_beat`, `end_beat`, `notes`, `grid`,
`pattern_id`, and `dualized_pattern_id`. Bars are zero-based. The window duration
excludes context: a two-bar 4/4 clip is eight beats even though `notes` may span
negative positions through positions greater than eight. Context is needed for
labeling boundary hits and flams; **do not blindly render all context as part of
an eight-beat audio clip**. Audio boundary handling is a separate upcoming step.

The raw context retains original velocity, dropped drum categories, and quiet
hits. `clip_grid()` owns filtering, merging, timing quantization, and label
membership. The grid can therefore have fewer events than the raw context.

The single-performance report now includes pattern IDs and a grid-hit count.
Its `clip_hits.csv` intentionally includes context, and `clip_metadata.json`
records that boundary policy. Re-running an older report can change counts and
pattern labels because its original raw-boundary extraction was provisional.

## Catalog the local GMD collection

Open `src/library/catalog_gmd.py` and select **Run Python File**. The script has
already been run locally. Terminal equivalent:

```bash
.venv/bin/python -m src.library.catalog_gmd
```

It selects eligible performances from `info.csv`, loads their MIDI drum tracks,
and applies the cutter. It logs failed sources explicitly; variable-tempo files
and time signatures inconsistent with metadata require review. The command
returns a nonzero exit if any sources failed. It does not download data.

Outputs are gitignored under `results/gmd_catalog/`:

| File | Contents |
|---|---|
| `loop_catalog.csv` | One row per retained MIDI window: source, original split, zero-based starting bar, boundaries, BPM, pattern IDs, context/grid counts |
| `loops.jsonl` | One JSON object per line with the same metadata, original context `Note` inputs, and boolean three-voice grid |
| `split_conflicts.csv` | Catalog rows whose exact three-voice pattern occurs in more than one official split |
| `failures.csv` | Sources that failed processing and their errors; headers only when none failed |
| `summary.json` | Spec version, dataset root, selection counts, split counts, and unresolved status |

MIDI paths in the catalog are relative to `gmd_root` in the summary. A diagnostic
clip ID combines the source performance ID and starting bar; future audio IDs
will also need to distinguish kits and render tempos. This catalog is **not**
`library_manifest.csv`: there is no audio path, kit assignment, or rendered audio
at this checkpoint. Source splits are preserved rather than shuffled.

### Verified local run (2026-11-05)

- Spec v0.4; 460 eligible performances; zero processing failures.
- 72 eligible performances produced no retained complete, content-qualified clip.
- 8,646 retained clips and 8,626 distinct three-voice patterns.
- 6,907 train clips, 846 validation clips, and 893 test clips.
- Three exact rhythm patterns occur across official splits.

These are measurements of this cutter and local data, not the older handoff's
provisional counts. The split-conflict report is an audit: **the catalog is not
approved for training until conflicts are resolved**. The final cross-split
pattern rule is still a team decision. Tap relevance uses dualized IDs and will
need its own review when constructing splits and the answer key.

Tests cover early downbeats, boundary context, relative positions, the configured
stride, within-performance duplicates, minimum content, final-bar exclusion,
source preservation, and catalog label round trips. A synthetic catalog test
ensures shared patterns are flagged without silently changing source splits.

## Next pipeline checkpoints

Joshua's formerly missing rules are now in spec v0.4 and `clip_grid()`,
`has_min_hits()`, and `dualize()` are implemented. Use `docs/week3_handoff.md`
as the current coordination reference.

AVP sample extraction is now implemented below. Next, use those samples to
build a small synthetic beatbox batch. Drum rendering can use the new
`tools/render/` sampler once kit samples are available locally; the approximately
1.9 GB kit download is a separate step requiring confirmation. Kit note maps
and train/test assignments still await the music lead's sign-off. Production
loudness normalization, paired audio manifests, participant splits, and the
cross-split pattern policy remain to implement. E-GMD audio is on hold.

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
and locate it in the source WAV. The approved `beatbox.sound_end` policy now specifies 30 dB below peak,
the next annotated onset, or 500 ms, whichever comes first, plus a 10 ms fade.
Sample extraction using that policy is implemented below. Before training, assign participant-disjoint splits
and select the approved mode/recording kinds. Cataloging everyone together is
not permission to use everyone's sounds for training.

## AVP individual sound extraction (implemented)

`src/queries/extract_avp.py` turns the onset catalog into individual playable WAV
files. These are the mouth-sound ingredients for the future beatbox generator;
they are not yet complete rhythmic queries or paired training examples.

### Listen or regenerate

The local extraction has already run. Open
`results/avp_samples/preview.html` in a browser and press the audio play buttons.
The page provides 16 examples: one per source class, mode, and recording category.
Examples are selected deterministically from the first successfully extracted
sound in each group, rather than chosen by ear. Waveform heights are scaled
individually for visibility; playback preserves the original sound level.

To regenerate from the IDE, open `src/queries/extract_avp.py` and select
**Run Python File** with the project's `.venv` interpreter. Terminal equivalent:

```bash
.venv/bin/python -m src.queries.extract_avp
```

The extractor refreshes its own onset index under `results/avp_samples/index/`
using the existing index builder, then uses approved `avp_modes`,
`avp_recordings`, and `avp_labels` from the spec. Both Personal and Fixed modes
and isolated/improvisation recordings are currently approved. Each sample keeps
its source participant and mode; there are no participant split assignments yet.

### How sound endings are measured

The duration and fade settings come exclusively from `beatbox.sound_end`:
30 dB below peak, next annotated onset, or 500 ms, whichever comes first. The
recording end is also a hard boundary.

To measure sound level reliably rather than stopping at a waveform zero crossing,
the implementation follows the existing inventory's **10 ms RMS envelope**
approach. RMS is a short-window average signal-level measure. The window size is
an explicit technical measurement choice, not an additional musical threshold.
For multichannel inputs, energy is pooled across channels for this measurement;
the actual output retains all channels.

Within the allowed interval, it finds the highest-level envelope frame, then
stops at the start of the first subsequent frame at or below the configured
relative threshold. Initial silence or the quieter start of an attack therefore
does not immediately end a sample. Measurement precision is one envelope frame;
any final partial frame is included in the analysis.

The 10 ms linear fade is applied **inside the retained tail**, ending at the
chosen cutoff, rather than extending past the next onset or duration cap. A
shorter sound uses a shorter fade. Joshua should audition this concrete envelope
and fade interpretation before the dataset freeze. Samples that are empty,
silent, nonfinite, or unusable after fading are logged and excluded.

For boundary safety, next-onset limits are read from **all finite source
annotations**, even if the next sound had an invalid/blank label and was excluded
from the searchable index. We never silently stretch a sound over such an onset.
The first sample at/after the annotated start is selected, and output does not
cross the first sample at/after the next onset.

### Outputs

Everything stays under the gitignored `results/avp_samples/` directory:

| File/folder | Purpose |
|---|---|
| `samples/` | Individual float-format WAV sounds, retaining source sample rate and channels |
| `sample_catalog.csv` | Sample paths, source identities, participant/mode/class, annotations, actual boundaries, duration, and stopping reason |
| `extraction_issues.csv` | Any selected sounds that could not be extracted; headers only when none failed |
| `summary.json` | Spec version, applied settings, source root, counts, and pending split/listening status |
| `preview.html` | Playable examples and waveform previews for human review |
| `index/` | Refreshed annotation index and validation logs used by this extraction |

Sample paths are relative to the output directory; original audio paths are
relative to `source_root` in the summary. `start_frame` and exclusive `end_frame`
are positions in the original recording. `frames` counts retained samples per
channel, `duration_s` is the extracted sound's duration, and `fade_frames` records
how much of its tail was faded. Participant and mode remain separate fields so
future query generation can keep one participant and one mode per recording.

Float WAV avoids introducing PCM quantization or clipping when writing samples.
No resampling, loudness normalization, or beginning fade is applied, and source
recordings are never modified. Re-running replaces current outputs at stable
paths. Older WAVs may remain if source selection changes; the **current sample
catalog**, not a recursive folder scan, defines the usable collection.

### Verified local run (2026-11-05)

- 9,776 selected annotations; 9,776 sounds extracted; zero skipped extractions.
- 28 participants, with both modes and recording categories represented.
- Stop reasons: 7,209 decay, 2,288 next onset, 206 duration cap, 73 recording end.
- Sixteen playable audition examples. No participant splits assigned.

`tests/test_avp_samples.py` verifies delayed attacks, decay measurement, duration
and next-onset caps, recording ends, multichannel preservation, short fades,
silent/invalid inputs, and boundaries at annotations with blank labels. It also
checks catalog provenance, unchanged source bytes, and preview generation.

Listening is still required: check that the attack is present, the tail sounds
natural, and another utterance has not leaked in. Technical tests cannot approve
how these sounds are perceived. After that review, the next implementation is
arranging these samples according to labeled MIDI patterns, followed by paired
drum renders, participant/kit splits, and final manifests.

## First drum-library audition batch (implemented)

This is the first checkpoint toward library audio: **nine playable drum loops**
rendered from our labeled GMD catalog. There are three source performances per
official split, through one configured kit (currently AVL Black Pearl). Source
selection is deterministic and uses different performances within each batch.
Known cross-split pattern conflicts are omitted from this preview only; the
final conflict-removal policy remains a team decision.

### Listen or regenerate

Open `results/library_preview/preview.html` in a browser. The nine audio players
let Joshua review the drum mappings and cut boundaries before we scale up.
There is a short lead-in before the pattern and natural decay afterward.
Technical checks do not establish whether the kit sounds musically right.

The 14.2 MB Black Pearl archive was downloaded using the existing downloader;
the source/license/credit record is in `data/raw/SOURCES.md`. The full kit
collection was not downloaded. Mappings are still drafts and no kit train/test
assignment has been made. This checkpoint uses only the already-configured kit,
without changing musical decisions or other teammates' interfaces.

To regenerate, open `src/library/render_preview.py` and select **Run Python
File** using `.venv`, or use:

```bash
.venv/bin/python -m src.library.render_preview
```

The script defaults to the first kit in the shared spec and three clips per
source split. `--kit <kit_id>`, `--per-split N`, and `--seed N` are optional.
Kit samples must already exist locally. Unsupported sampler programs, missing
samples, mapping problems, failed catalogs, or stale spec versions cause an
explicit failure. Labels are rechecked against the current spec before rendering.

### What is rendered

The renderer reuses Joshua's `original_hits()` listening helper to select raw
hits owned by the window, then `kit_hits()` applies the configured note map and
`Sampler` generates stereo audio. It never plays the entire one-beat labeling
context. Original timing, velocity, quiet hits, flams, and dropped-label drum
categories are preserved in drum audio, as required by the spec.

The helper's half-grid-step onset boundary convention is reused unchanged. Audio
gets the existing listening tool's 0.25-second pre-roll so early hits are audible.
Output extends at least through the whole pattern, and longer if drum tails
require it. Thus audio duration can exceed the eight-beat pattern duration.
This is an audition representation, not the final fixed-window model input.

Loudness uses the **existing listening normalization**, targeting -20 dB RMS
with a -1 dB peak ceiling. This makes previews convenient to compare; production
library loudness normalization is still a separate decision/implementation.
The script does not change `config/music_spec.yaml` or normalize the source kits.

### Outputs and manifest

All generated files are gitignored under `results/library_preview/`:

| File/folder | Purpose |
|---|---|
| `audio/` | Stereo WAV loop previews |
| `library_manifest.csv` | Team-contract base columns plus preview timing, source bar, kit split, level, and status fields |
| `listening_notes.csv` | Empty verdict/notes fields for Joshua's review; existing matching-ID notes are preserved on reruns |
| `summary.json` | Kit/spec provenance, selection and boundary policies, path bases, and pending status |
| `preview.html` | Nine playable examples with source and pattern identities |

This manifest is deliberately kept in the preview directory, **not** the final
`manifests/` directory. Every row is marked `audition_only`. `split` is the source
pattern split, while `kit_split` remains empty until Joshua assigns it. This kit
appearing against all three source splits does not establish a valid kit-disjoint
training set. Do not treat this preview as ready for training/evaluation.

`audio_path` is relative to the repository when inside it, or absolute for an
external output folder; `midi_path` remains relative to the GMD root. Both bases
are stated in `summary.json`. Render IDs distinguish source window, kit,
recorded tempo mode, and seed. `pattern_duration_s` excludes pre-roll/tails;
`duration_s` measures the actual audio file. Reruns replace audio at stable paths;
the current manifest defines the active set if older files remain on disk.

### Verification and next checkpoint

`tests/test_library_preview.py` renders a real tiny SFZ fixture and checks early
hits, exclusion of neighbouring context, preservation of dropped-label sounds,
stored-label consistency, audio timing/channels, peak ceiling, manifest paths,
source splits, stale catalogs, conflict omission, and retained review notes.

The local checkpoint produced nine non-silent stereo WAVs at 44.1 kHz with no
kit mapping-check warnings. Listen before expanding. Next, build a small
beatbox/drum paired preview; after Joshua's review, add the remaining kits,
final audio formatting/normalization, split enforcement, and full manifests.
