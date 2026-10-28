# Music Spec

**Owner:** Member A, music lead (Joshua Yu)
**Status:** v0.4 (2026-10-28), matching `config/music_spec.yaml` version 0.4. §1–7 decided except the kit list, swing drift, mic coloration, and tap sounds. §5–7 values are provisional until the §9 listening checks.

## How to use this document

This file turns every musical decision in the project into a **concrete rule with numbers** that someone without music experience can code directly.

- **Member B** implements the library and fake-search rules exactly as written.
- **Member D** builds the answer key from the "same rhythm" rules.
- **Everyone** asks A when something musical is unclear, and A adds the answer here.

The values marked *default* are starting points, not decisions. A should change them after listening and fill in every `DECISION:` line. When a rule changes after the week 3 freeze, note it in the change log at the bottom and tell the team.

---

## 1. Drum grouping (which parts become kick, snare, or hats)

Every drum hit in GMD gets assigned to one of three voices (**kick**, **snare**, **hats**) or dropped.

MIDI note numbers are from the [GMD documentation](https://magenta.tensorflow.org/datasets/groove) (Roland TD-11 mapping). Notes 22, 26, and 58 are not General MIDI drum notes, so a GM drum instrument may play them as something else.

| Articulation | MIDI note(s) | Default group | DECISION |
|---|---|---|---|
| Kick | 36 | kick | kick |
| Snare, head | 38 | snare | snare |
| Snare, rim shot | 40 | snare | snare |
| Snare, cross-stick | 37 | snare | snare |
| Toms (all) | 48, 50 (tom 1 head, rim); 45, 47 (tom 2 head, rim); 43, 58 (tom 3 head, rim) | drop | drop |
| Hi-hat closed | 42 (bow), 22 (edge) | hats | hats |
| Hi-hat open | 46 (bow), 26 (edge) | hats | hats |
| Hi-hat pedal | 44 | drop | hats |
| Ride (bow, edge) | 51 (bow), 59 (edge) | hats | hats |
| Ride bell | 53 | hats | hats |
| Crash | 49, 55 (crash 1 bow, edge); 57, 52 (crash 2 bow, edge) | drop | drop |

**Notes for B:** apply this table before anything else. Dropped hits are removed from both the rhythm label and the fake beatbox, but they stay in the rendered drum audio so loops still sound natural.

DECISION (open vs. closed hi-hat: same voice, or should open hats count as a different rhythm?): They should count as the same voice, because it's difficult to articulate open vs. closed hi-hat while beatboxing. 

---

## 2. What counts as "the same rhythm"

These rules decide which loops count as a match. D builds the answer key from them.

| Rule | Default | DECISION |
|---|---|---|
| Timing grid | 12 steps per beat (fits straight 16ths and triplets) | Yes |
| Ghost notes | Ignore hits with velocity below **40** (check by ear across a few kits) | Ignore below 15 |
| Flams | Merge hits of the same voice closer than **30 ms** into one hit | Yes |
| Flam timing | — | The merged hit takes the **loudest** hit's timing (the earlier one on a tie) |
| Flam runs (ruffs, drags) | — | A run of hits, each closer than 30 ms to the next, merges into one hit, even when the whole run spans more than 30 ms |
| Hits exactly between two steps | — | Go to the simpler position: beat, then 8th, then 16th or triplet, then finer steps. When both rank the same, the later step. (2.4% of hits sit exactly on a midpoint.) |
| Swing | Swung and straight versions of a beat are **different** rhythms | Yes |
| Minimum content | A loop needs at least one kick and one snare to be kept | Yes |

**Half-time vs. double-time:**
- Default: treat them as **different** rhythms, and only use tempos where the beat's intended feel is clear.
- DECISION: They should be treated as different rhythms. 

**How close counts as a match:**
- With the rules above, almost every loop has its own grid (about 99.8% unique), so a match is mostly the same loop on another kit, at another tempo, or performed by another voice.
- DECISION: Exact. Two loops are the same rhythm only when their grids are identical (same pattern ID), as README §5.4 says. D scores near misses (how many hits off) separately; they don't count as matches.

---

## 3. Cutting loops

| Rule | Default | DECISION |
|---|---|---|
| Length | 2 bars | Yes |
| Time signature | 4/4 only | Yes |
| Start point | Always on a downbeat | Yes |
| Hits near a bar line | — | A hit belongs to the bar its nearest grid step falls in, so a downbeat played a little early stays the downbeat |
| Fills | Excluded (use GMD's beat/fill labels, then spot-check) | Yes |
| Count-ins and endings | Excluded | Excluded, start at first bar that has kick and snare; drop the last bar |
| Spacing | — | A new loop every 2 bars (bars 1–2, 3–4, …), never overlapping |
| Loops per performance | No limit, but duplicates removed | Yes, within one performance |

---

## 4. Drum kits and rendering

**Kit list:**

| Kit name | Type (acoustic / electronic / lo-fi / other) | Source and license | Train or test | Notes |
|---|---|---|---|---|
| | | | | |

- **Aim:** a varied set, so the model can't identify a match just by kit sound. Hold out a few kits of each type for testing only.
- **Kit sources:** DECISION: Freely licensed multisampled kits, rendered by a small Python sampler, are the backbone. Each kit's license is checked before use, and their audio may be released. A drum VST adds extra kits (good candidates for held-out test kits), rendered on A's Mac; its audio stays internal. E-GMD audio is optional later. A auditions the kits and assigns train/test here.
- **Tempos:** default range 80–140 BPM. DECISION: Expand range to 70-160 bpm, remove performances outside
- **Tempo change:** how far from its recorded tempo a loop may be rendered. DECISION: within ±20%, inside 70–160 BPM. Provisional: check one groove at ±10/20/30% by ear once a kit renders.
- **Level balancing:** default is to loudness-normalize each rendered loop and check that hats are clearly audible in every kit. DECISION: Yes. 
- **Velocity handling:** keep the performer's velocities, unless a kit has poor velocity layers. DECISION: Yes.
- **Licensing:** if a kit comes from a commercial library, we release only MIDI and splits, never its rendered audio.

---

## 5. How people beatbox a beat (simplification rules)

Real amateurs don't voice every hit. B's generator applies these rules before placing the mouth sounds from AVP.

| Rule | Default | DECISION |
|---|---|---|
| Voices used | kick, snare, closed hi-hat, open hi-hat (the AVP sound types) | Yes |
| Hits at the same moment | Voice only one sound, with priority **kick > snare > hats** | Yes |
| Ghost notes | Never voiced | Skip snare hits below velocity **10**; kick and hats are always voiced. Snares down to 10 still sound intentional. The label already ignores hits below 15 (§2), so for now every labelled hit is voiced. |
| Busy hats | If hats play 16ths, thin to 8ths with probability **0.5** | Yes |
| Cymbals and toms | Never voiced | Yes |
| Open hi-hat | Voice with the AVP open-hat sound | Yes |
| Other hats hits | — | Closed, pedal, ride, and ride bell all use the AVP closed-hat sound |

**Mouth sounds:** use AVP's "personal" recordings first, where people chose their own sounds. DECISION: Use Personal and Fixed, both the single-sound takes and the improvisations (all 9,776 usable sounds). Each fake recording uses one participant's sounds from one mode.

**Where a mouth sound ends:** AVP only marks where each sound starts. DECISION: A sound ends where its level falls 30 dB below its peak, at the next annotated sound, or after 500 ms, whichever comes first, then fades out over 10 ms. (Measured: the median sound decays 30 dB in 160–275 ms, and 29% of improvisation sounds are still ringing when the next one starts.)

---

## 6. Realistic sloppiness

Each fake beatbox recording gets a random amount of each effect within these ranges. B builds each as a separate adjustable setting, so D can also test them at extreme values.

| Effect | Default range | What it models | DECISION |
|---|---|---|---|
| Steady rush or drag | −20 to +20 ms, same for the whole clip | Playing consistently ahead of or behind the beat | Yes (provisional) |
| Per-hit timing wobble | 0 to 40 ms | Imprecise timing on individual hits | Yes (provisional) |
| Tempo drift | 0 to 8% over the clip | Slowly speeding up or slowing down | Yes (provisional) |
| Swing drift | small random change in swing amount per bar | Inconsistent swing feel | Off until A has listened |
| Dropped hits | kick 0–0.05, snare 0–0.05, hats 0–0.4 probability | Forgetting or skipping hits | Yes (provisional) |
| Background noise | 10 to 40 dB SNR | Room and laptop noise | Yes (provisional) |
| Mic coloration | mild random EQ | Different laptop mics | Off until A has listened |

These ranges are guesses. A should listen to a handful of fake recordings next to real AVP improvisations and adjust until they sound similar.

---

## 7. How people tap a beat

Taps have only two "hands": low and high.

| Rule | Default | DECISION |
|---|---|---|
| Low hand | kick | Yes (provisional) |
| High hand | snare and hats merged | Yes (provisional) |
| Hits at the same moment | One tap on the low hand | Yes (provisional) |
| Tap sounds | Recorded desk knocks, two different sounds | |

After the first version works, compare these rules against how the drummers in TapTamDrum actually tapped, and note the main differences here.

---

## 8. Recording sessions

| Item | Default | DECISION |
|---|---|---|
| Mic | Built-in laptop mic (matches AVP) | |
| Click | Headphones only, at the loop's tempo | |
| Takes per beat | One with click, one without | |
| Beats per person | About 50, chosen by A from the test split | |
| Enrollment sounds | 5 each of kick, snare, and hi-hat per person | |
| Usable take | Keep imperfect takes; discard only if the beat is clearly lost or the audio is broken | |

**Beat selection:** balance straight and swung, sparse and busy. Skip beats that are unreasonable to beatbox.

---

## 9. Listening sign-off checklist (before the week 3 freeze)

- [ ] 20 random loops from each kit: hits sound natural, hats audible, no clipping or wrong articulations
- [ ] 20 random loop cuts: every loop starts on a downbeat, no fills or count-ins
- [ ] 20 fake beatbox recordings next to real AVP improvisations: similar feel and sloppiness
- [ ] 10 fake tap recordings: believable
- [ ] 10 random pairs from the answer key: the matched loops really are "the same rhythm"

**Signed off by A on:** YYYY-MM-DD

---

## 10. Change log

| Date | Change | Why | Who was told |
|---|---|---|---|
| 2026-10-20 | §1–4 decided (spec v0.2) | First version for the team call | Team call |
| 2026-10-28 | Decided: exact matching; flam timing from the loudest hit; hits near a bar line join their nearest step's bar; a new loop every 2 bars; renders within ±20% of the recorded tempo; kit sources; §5 mouth-sound choice and cut-off, snares below velocity 10 not voiced (none reach the beatbox while the label floor is 15), other hats use the closed-hat sound; §5–7 defaults approved as provisional (spec v0.3) | B's questions; unblocks `clip_grid()`, the loop cutter, the beatbox generator, and D's answer key | B |
| 2026-10-28 | Decided: flam runs merge as one hit; exact midpoints go to the simpler position (spec v0.4) | Tie-breaks needed to build `clip_grid()` | Not yet |
