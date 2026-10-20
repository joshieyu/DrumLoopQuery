# Week 1 handoff (2026-10-20)

What exists in the repo, what is decided, what we found in the data, and what to settle on today's call.

## Get set up

```bash
git clone https://github.com/joshieyu/DrumLoopQuery.git
cd DrumLoopQuery
python3.12 -m venv .venv                      # Python 3.11+ works
.venv/bin/pip install -e ".[dev]"             # Windows: .venv\Scripts\pip
.venv/bin/python -m tools.download_datasets   # GMD MIDI, E-GMD MIDI, AVP v4: ~340 MB into data/
.venv/bin/pytest
```

Optional: `.venv/bin/python -m tools.inventory.report` writes a dataset report to `results/inventory/report.html`.

The big audio downloads (GMD audio 4.76 GB, E-GMD audio 90 GB) aren't scripted yet. Ask the team before pulling them. Never commit anything under `data/` or `results/` (git already ignores them).

## What's in the repo

| Path | What it is | Who uses it |
|---|---|---|
| `config/music_spec.yaml` | Every musical constant (drum map, grid, thresholds, loop rules, sloppiness ranges). Mirrors `docs/music_spec.md`. | Everyone |
| `src/music/spec.py` | `load_spec()` reads and validates the YAML. Never hardcode a musical number; read it from the spec. `spec.undecided()` lists values still at their default. | Everyone |
| `src/music/rhythm.py` | The "same rhythm" interface. `Grid` and `pattern_id()` are final. `clip_grid()` (MIDI clip to grid) and `dualize()` (tap reduction) have their final signatures but raise `NotImplementedError` until spec §2 and §7 are decided. | B, D |
| `tools/download_datasets.py` | Downloads to `data/raw/{gmd,egmd_midi,avp}/` and checks the published checksums. Please reuse this layout. | B, D |
| `tools/inventory/` | The music lead's dataset report (exploration only). | A |

Licenses: GMD, E-GMD, and AVP are all CC BY 4.0 (checked on the official pages).

## Spec status

Decided by the music lead (spec §1–4, details in `config/music_spec.yaml`):

- **Voices:** kick = 36; snare = 38, 40 (rimshot), 37 (cross-stick); hats = closed, open, and pedal hi-hat plus ride and ride bell. Toms and crashes are dropped from the label (they stay in the audio).
- **Same rhythm:** 12 steps per beat; hits below velocity 15 are ignored (every voice); same-voice hits closer than 30 ms merge into one; swung and straight are different rhythms; half-time and double-time are different rhythms; a loop needs at least one kick and one snare.
- **Loops:** 2 bars, 4/4 only, starting on a downbeat; GMD fill recordings are excluded; start at the first bar that has a kick and a snare (skips count-ins) and drop each performance's last bar (skips endings); leave out performances recorded outside 70–160 BPM; remove duplicate patterns within a performance, no other limit.
- **Rendering:** 70–160 BPM, loudness-normalized, performer velocities kept.

Still open: which hit's timing survives a flam merge, whether loops start every 1 or 2 bars, how far a loop may be rendered from its original tempo, the kit list, and spec §5–7 (still defaults). `load_spec().undecided()` lists them.

**Open question for everyone: what counts as "the same rhythm"?** Matching grids exactly makes almost every loop unique. A provisional count with the decisions above (460 performances qualify):

| Grid | New loop every | 2-bar loops | Unique patterns | Test loops whose pattern also appears in train |
|---|---|---|---|---|
| 12 steps/beat | 2 bars | 8,806 | 99.8% | 0.2% |
| 12 steps/beat | 1 bar | 17,430 | 99.8% | 0.3% |
| 16ths (4 steps/beat), for comparison | 2 bars | 8,831 | 98.9% | 1.9% |

Many hits land between straight-16th and triplet positions (human timing, light swing), and GMD drummers vary a lot bar to bar. The options:
1. **Accept it.** A match means the same clip on a different kit, tempo, or voice; D's "how many hits off" score covers near misses.
2. **Loosen the grid,** so musically identical bars quantize the same way.
3. **Count near matches.** A loop is relevant if its grid is within N hits of the query's.

The music lead makes the musical call. It affects how D builds `relevance.json`, how C defines classes for the contrastive loss, and whether B's duplicate removal matters.

## Data facts found so far

**GMD (MIDI)**
- 1,150 files: 503 beats, 647 fills; 99% in 4/4.
- 480 ticks per beat, one tempo per file (matching the BPM in `info.csv`), all on MIDI channel 10.
- Fill files are separate short recordings (median 1 bar). Fills inside beat recordings are not labeled.
- The zip includes macOS `Icon` files. Skip anything that isn't `.mid`.

**E-GMD (MIDI)**
- 43 kits, all recorded from one Roland TD-17 module. Every sequence is rendered on every kit, in every split.
- Covers 1,059 of the 1,150 GMD performances. **71 GMD beats (195 minutes) are missing.**
- **Some kits remap notes** (checked on a sample of 100 performances × 43 kits):
  - Compact Lite and Dark Hybrid: hi-hats become tambourine (54).
  - 808 Simple, 909 Simple, Custom 1–3: cross-stick (37) becomes rimshot (40).
  - Big Room: rimshot (40) becomes clap (39).
  - West Coast, Jazz Funk, Second Line: tom rim notes (47, 58) become cowbell (56).

  In the same sample, 11 performances had 1–17 fewer notes on every kit. So labels for E-GMD audio should come from that kit's own E-GMD MIDI, or these kits need special handling.

**AVP v4**
- 28 participants × {Personal, Fixed} × {Kick, Snare, HHclosed, HHopened, Improvisation}.
- 44.1 kHz, 16-bit, mono WAV. Each WAV has a CSV with no header: `onset_sec, label[, onset_phoneme, coda_phoneme]`.
- Labels are `kd`, `sd`, `hhc`, `hho`. Phoneme columns exist only in Personal.
- Four labels have stray spaces, and one is blank (P6 and P23, Fixed improvisations). Strip whitespace and skip the blank one.
- Many files contain exact digital silence (median 2% of samples; 43 files have more than 5%), so the gaps between sounds aren't always real room tone.

## To settle on today's call (milestone M1)

1. **File formats (README §12).** Agree on them. Proposed additions:
   - a `start_bar` column in `library_manifest.csv`, so every loop can be traced back to its MIDI for listening checks
   - `pattern_id` always comes from `src.music.rhythm.pattern_id`; nobody hashes grids their own way
2. **Patterns in more than one GMD split.** With the current decisions, 0.2% of test loops share a pattern with training (more with a coarser grid). Pick a rule, for example drop them from test.
3. **`config/` or `configs/`.** The repo uses `config/`, but README §12 says `configs/`. Pick one.
4. **This week:**
   - **B:** load GMD, E-GMD, and AVP; write loop cutting that produces `Note` lists for `clip_grid()`.
   - **C:** model prototype on toy data. Clips are 2 bars (3.0–6.9 s at 70–160 BPM); AVP audio is 44.1 kHz mono.
   - **D:** `evaluate.py` and metric definitions, tested against a fake `relevance.json`.
5. **Musical questions go to A,** who adds the answer to `docs/music_spec.md` and the YAML within a day.
