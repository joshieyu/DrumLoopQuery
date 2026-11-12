# Presentation plan (talk around 2026-11-19)

The talk is 12 minutes: about 8 minutes of slides, then questions. Every member presents a part. It isn't graded on how complete the project is. It's graded on four things:

- a clear problem definition and motivation
- our methods, with technical details and innovations
- the experiments and results we have so far
- remaining work and next steps

So for the next 13 days, the goal is a small end-to-end result in two rounds:

- **Round 1, Thu Nov 12:** a fake beatbox query is embedded, the library is ranked, and the ranking is scored. That result gets reported for the trained encoder and for the onset baseline.
- **Round 2, Mon Nov 16:** one round of fixes, plus about 10 real beatbox queries from each of us, scored the same way.

Everything else becomes the "remaining work" slide. After the talk, we go back to the full plan (README §9–11).

## The v0 data snapshot

v0 is a small, early version of the data, handed to C on Wed Nov 11. It is not M2: the music lead gives it only a quick listen, and the full spec §9 sign-off waits until after the talk.

| | v0 |
|---|---|
| Loops | All GMD loops that pass the spec. Patterns use GMD's own split. |
| Library audio | Every loop at its recorded tempo, through the 7 kits that render. Other tempos come later. Test kits: Salamander and MuldjordKit; the other five train, and validation uses the training kits. Every clip starts 0.25 s before its downbeat and is cut at the end of bar 2. |
| Queries | Fake beatbox only, from AVP with the spec §5 rules. Mouth sounds never overlap, and a speed limit keeps them about as fast as real AVP beatboxers. The only sloppiness is steady rush or drag, per-hit wobble, dropped hits and background noise; all values come from `config/music_spec.yaml`. Queries are cut like the loops. AVP participants split 20 / 3 / 5. |
| Answer key | Exact `pattern_id` match (`same_rhythm.match: exact`) |
| Split clashes | A pattern found in more than one GMD split keeps only its most held-out copy (test, then validation). The 3 clashes today all lose their training copies, so the official test split stays whole. |

Sizes, measured on 2026-11-06 with the music lead's prototype cutter (`tools/listen/grid_ab.cut_loops`, spec v0.4). B's pipeline may come out slightly different.

| GMD split | 2-bar loops | Unique patterns | Drummers | Audio |
|---|---|---|---|---|
| train | 6,914 | 6,899 | 10 | 521 min |
| validation | 847 | 846 | 9 | 64 min |
| test | 894 | 893 | 6 | 68 min |

- **Only 1 test pattern also appears in training.**
- **Each query has exactly one correct loop per held-out kit.** That makes the task finding one clip among about 900, where random guessing gets the right loop into the top 10 about 1% of the time.
- **Rendering is fast.** Each loop takes 0.005–0.08 s, depending on the kit, so all 8,655 loops through all 7 kits take under half an hour. The beatbox generator is the slow piece, not rendering.

## Schedule

| Dates | A, music lead | B, data | C, model | D, evaluation |
|---|---|---|---|---|
| Fri 6 – Mon 9 | Every v0 call made on Mon 9 (spec v0.5) | Loop rendering and clean beatbox previews | Model and training pipeline on practice data (PR #1) | `evaluate.py` (Hit@1/5/10, MRR, mAP, chance level) |
| Tue 10 | Listen to about 20 sloppy fake queries next to real AVP improvisations, and a few loops per kit | Full library render, sloppiness, AVP participant split, manifests; run C's loader on a small manifest first | Merge PR #1 | `relevance.json` and the onset baseline |
| Wed 11 | **Data handed to C** | **Data handed to C:** manifests plus one seeded command that regenerates the audio | Train on beatbox–loop pairs | Run the onset baseline on the v0 test set |
| Thu 12 | Choose the beats for the recording session and settle the spec §8 basics. Listen to the top 5 results for a few queries. | Dataset numbers and an audio example for the slides | Train. **Round 1 numbers Thu.** | Score the encoder and the onset baseline; results table. **Round 1 numbers Thu.** |
| Fri 13 – Mon 16 | Run a short recording session: about 10 beats each, with the click (README §13). Take a first look at which patterns fail. | Add the real queries to `query_manifest.csv` | One round of fixes, from the round 1 results | Score the real queries with every system. **Final numbers Mon.** |
| Tue 17 – Wed 18 | Everyone: slides, then a rehearsal by Wed | | | |

## Results, in priority order

1. **Must have (Thu 12):** dataset numbers, audio examples, and the onset baseline against random chance. This is enough for a "results so far" slide.
2. **Target (Thu 12, improved by Mon 16):** the trained encoder against the onset baseline, with fake beatbox queries, on held-out kits and held-out AVP participants.
3. **Target (Mon 16):** the team's real beatbox queries, scored with every system. This is the first answer to "does it work on real people?", even if the numbers are low.
4. **Fallback if the beatbox generator is late:** cross-kit retrieval. The query is a loop on one held-out kit, and the library is the test loops on another. It needs only the library, and it still shows the model matching rhythm rather than sound.
5. **Stretch:**
   - The CLAP baseline. It needs a model download, so ask the team first.
   - A small version of E5: the encoder and the onset baseline as dropped hits and noise get worse. It needs only regenerated queries.
   - A small version of E8: which kinds of pattern fail (swung or straight, busy or sparse).

## After the talk

- **Full plan:** the full spec §9 sign-off and M2 freeze, then the full recording sessions (about 50 beats each, with free-tempo takes and enrollment clips).
- **Moved to later:**
  - tap queries and TapTamDrum
  - the "identify each hit, then match" baseline
  - MERT/LoRA and the rest of the experiment matrix
  - more kit types (electronic, lo-fi)
  - rendering at other tempos
  - tempo drift, swing drift and mic EQ in the query generator

## Calls for the music lead

Made on Mon Nov 9 (spec v0.5; see the change log in `docs/music_spec.md`):

- the drafted kit note maps are kept for v0
- Salamander and MuldjordKit are held out for test, the other five kits train, and validation uses the training kits. Every kit in use is acoustic, and the slides should say so.
- recorded tempo only
- every clip starts 0.25 s before its downbeat and is cut at the end of bar 2 with a 20 ms fade
- AVP sound endings and per-beat hat thinning approved as B built them
- tempo drift off for v0

Made on Tue Nov 10 (spec v0.6):

- mouth sounds never overlap: each is cut off at the next sound played
- the kit listen found no wrong articulations

Made on Wed Nov 11 (spec v0.7):

- the sloppiness ranges stay as they are
- a speed limit on the fake beatboxes: kick and snare at least 120 ms apart, hats 250 ms. Before it they ran about twice as fast as real AVP beatboxers.

Still to come:

- **By Thu 12:** choose the beats for the recording session from the GMD test split, and settle the spec §8 basics (mic, click, takes) for it.
- **Fri 13 – Mon 16:** decide which real takes are usable.

## Talk outline (about 2 minutes each)

| Speaker | Covers |
|---|---|
| A | The problem and why it matters. Audio of a beatbox and its matching loop. What "same rhythm" means (a 3-voice grid at 12 steps per beat), and why almost every 2-bar pattern is unique. |
| B | How the library and the fake queries are built (GMD → kits; AVP sounds, simplification rules, sloppiness), with numbers and an audio example |
| C | The encoder and contrastive training, with the training curve |
| D | How we score (held-out kits and participants; Hit@k, MRR, chance level), the results table on fake and real queries, and next steps |

## Open for the team

- **Recording session times,** Fri 13 – Mon 16. Book them now.
- **The talk date.** This plan assumes around Nov 19. Shift the schedule if the slot is different.
