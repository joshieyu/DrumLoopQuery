# Week 3 handoff (2026-11-05)

What's new since week 2, what the data freeze still needs, and what to settle on today's call.

> **Update 2026-11-06:** the class presentation (around Nov 19) changes this week's plan. See [presentation_plan.md](presentation_plan.md). It settles call items 1 (the M2 date) and 4 (recording sessions):
>
> - The v0 data is handed to C on Nov 11, and M2 moves to after the talk.
> - A short recording session, about 10 beats each, runs Nov 13–16. The full sessions come after the talk.
> - Since these notes were written, B's `cut_clips` in `src/library/clips.py` labels loops with the shared rhythm functions, and D's `src/eval/evaluate.py` is on GitHub.
>
> The rest of these notes still apply.

## Where we are

Week 3 of 8. **M2, the data and query-generator freeze after the music lead's sign-off, is due at the end of this week** (README §11).

| Role | On GitHub | Last push |
|---|---|---|
| A, music lead | Spec v0.4, the final "same rhythm" functions, kit downloads, a drum sampler with 8 draft kits, listening tools | Nov 4 |
| B, data pipeline | GMD inspection, first loop extraction, AVP sound index | Oct 27 |
| C, model | Nothing yet | — |
| D, evaluation | Nothing yet | — |

## Get up to date

```bash
git pull
.venv/bin/python -m tools.download_datasets kits   # free drum kits, about 1.9 GB, into data/raw/kits/
.venv/bin/pytest
```

`python -m tools.download_datasets --list` shows each download's size and license. Only B needs the kits this week.

## New since week 2

### The "same rhythm" functions are final (B, C, D)

In `src/music/rhythm.py`:

- `clip_grid(notes, bpm, spec)`: one loop's MIDI hits to its 3-voice grid (kick, snare, hats; 12 steps per beat).
- `has_min_hits(grid, spec)`: false for loops without at least one kick and one snare. Leave those out.
- `dualize(grid, spec)`: the two-hand tap version (low = kick; high = snare and hats).
- `pattern_id(grid)`: the ID for the manifests and the answer key.

Two clips are the same rhythm only when their `pattern_id`s are equal (`same_rhythm.match: exact`). Taps match on the dualized ID. Near misses are D's separate score, not matches.

**For B's loop cutter:** pass `clip_grid` every hit from a beat before the loop's first downbeat to a beat after its end, with `Note.beat` measured from that downbeat. It keeps the hits whose nearest grid step falls inside the loop (`loops.boundary: nearest_step`), so a downbeat played a little early stays the downbeat. `src/library/clips.py` cuts by raw time in [start, end), which drops those early downbeats, so it should change before the freeze. A new loop starts every 2 bars (`loops.hop_bars: 2`), so loops never overlap.

Spec v0.4 added two tie-break rules that haven't been announced yet. `clip_grid` already applies both:

- A run of flam hits merges into one hit at the loudest hit's time.
- A hit exactly halfway between two steps goes to the simpler position: beat, then 8th, then 16th or triplet.

### Kit rendering (B; new interface)

`rendering.kits` in `config/music_spec.yaml` lists the kits. Each kit has:

- a `kit_id`, the same ID that goes in `library_manifest.csv`
- a `type` and its SFZ `program`
- a `split`
- a `note_map` from every GMD drum note to what the kit plays

Load kits through the spec rather than hardcoding IDs or notes.

```python
from src.music.spec import load_spec
from tools.render.kits import kit_by_id, kit_hits, load_program
from tools.render.sampler import Sampler

spec = load_spec()
kit = kit_by_id(spec, "drskit")
sampler = Sampler(load_program(kit), seed=0)
hits = kit_hits(kit, [(0.0, 36, 100), (0.5, 38, 90)])  # (time_s, GMD note, velocity)
audio = sampler.render(hits)  # (frames, 2) float32 at sampler.sample_rate
```

- **7 of the 8 kits render.** `gogodze_phu_vol_ii` can't yet: it needs pitch shifting, filters and mic delays, which the sampler doesn't do.
- **The note maps are drafts.** The music lead is checking them by ear this week, and the final maps will change some keys. Code that goes through `kit.play(note)` (as `kit_hits` does) picks up the changes automatically.
- **`split` is empty for every kit** until the music lead assigns train and test kits.
- **`tools.render.kits.normalize` is only a listening level** for the audition tools. The library's loudness normalization (spec §4) is B's.
- **None of the kits is commercial.** Before releasing any audio, follow each license's terms. The credits are in `data/raw/SOURCES.md`.

| kit_id | Type | License | Renders |
|---|---|---|---|
| avl_black_pearl, avl_red_zeppelin, avl_blonde_bop, avl_blonde_bop_hr | acoustic | CC BY-SA 3.0 | yes |
| drskit, muldjordkit | acoustic | CC BY 4.0 | yes |
| salamander | acoustic | CC BY-SA 3.0 | yes |
| gogodze_phu_vol_ii | lo-fi | CC0 1.0 | not yet |

To hear one GMD groove through every kit: `.venv/bin/python -m tools.render.sampler --all-kits --groove`. The renders go to `results/listening/render/`, and Gogodze is skipped with a message.

## Freeze checklist (spec §9)

The music lead signs off after five listening checks, and none of them can start until its input exists:

| Check | Needs | From | On GitHub |
|---|---|---|---|
| 20 loops from each kit | rendered library loops | B, using A's final note maps | Not yet |
| 20 loop cuts | the loop cutter, labelled with `clip_grid` | B | First extraction only |
| 20 fake beatbox clips next to real AVP | the beatbox generator (spec §5–6) | B | AVP sound index |
| 10 fake tap clips | the tap generator and tap sounds (spec §7) | B | Not yet |
| 10 answer-key pairs | `relevance.json` | D | Not yet |

## To settle on today's call

1. **The M2 date.** There are two options:
   - Keep the date. B and D push a first batch by an agreed day, and the music lead listens within a day of it landing.
   - Move M2 later. C's first full training run moves with it. The recording sessions don't have to move: patterns come from GMD's official test split (README §5.2), so the music lead can choose the beats now.
2. **What everyone has locally.** C and D have nothing on GitHub yet. Push work in progress to a branch, even if it's rough, so the others can build on it.
3. **Kit variety.** All 7 kits that render are acoustic, but spec §4 wants a mix, with some kits of each type held out for test. The music lead chooses the kits; the team should weigh in on effort and the release rule. The options:
   - Electronic kits from MuseScore_General.sf2 (MIT, already in `tools/download_datasets`). This needs an SF2 reader.
   - Logic Pro kits, rendered on the music lead's Mac. Their audio stays internal, because audio from commercial kits is never released.
   - Getting the lo-fi Gogodze kit working in the sampler.
4. **Recording sessions next week (week 4).** Everyone records about 50 beatbox queries (README §13), so book times now. The music lead fills in spec §8 (mic, click, takes, enrollment sounds) before then.
5. **Carried over from week 1 but not written down anywhere yet:**
   - a `start_bar` column in `library_manifest.csv`
   - the rule for test loops whose pattern also appears in training
   - `config/` or `configs/` (README §12 still says `configs/`)
   - from README §17: where the GPU comes from (C needs one for the first full run), the role names, and the decisions log

## Spec status

The spec is at version 0.4. `load_spec().undecided()` lists what's still open:

- `rendering.kits`: note maps and the train/test assignment, after this week's listening.
- `sloppiness.swing_drift_per_bar` and `sloppiness.mic_eq`: off until the music lead hears fake queries next to real AVP.

The values in spec §5–7 are provisional until the §9 listening checks. The tap-sound row in §7 and every row in §8 have no decision yet.

## This week (proposed)

- **A:** finish the kit note maps by ear, assign train and test kits, choose recording beats from the GMD test split, and plan the sessions. Listen to each batch within a day of it landing.
- **B:** move the loop cutter onto `clip_grid` and render a first batch through the 7 kits. Build the beatbox and tap generators and the splits (GMD's split for patterns; 20/3/5 AVP participants).
- **C:** train on B's first batch when it lands. Until then, the sampler can render practice data from any GMD loop.
- **D:** `relevance.json` from `pattern_id`, the CLAP and onset baselines, and TapTamDrum.

Musical questions still go to the music lead, who answers within a day.
