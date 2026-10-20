# Music Spec

**Owner:** Member A, music lead (`[Member A Name]`)
**Status:** v0 template. Share v1 by the end of week 1.

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
| Swing | Swung and straight versions of a beat are **different** rhythms | Yes |
| Minimum content | A loop needs at least one kick and one snare to be kept | Yes |

**Half-time vs. double-time:**
- Default: treat them as **different** rhythms, and only use tempos where the beat's intended feel is clear.
- DECISION: They should be treated as different rhythms. 

---

## 3. Cutting loops

| Rule | Default | DECISION |
|---|---|---|
| Length | 2 bars | Yes |
| Time signature | 4/4 only | Yes |
| Start point | Always on a downbeat | Yes |
| Fills | Excluded (use GMD's beat/fill labels, then spot-check) | Yes |
| Count-ins and endings | Excluded | Excluded, start at first bar that has kick and snare; drop the last bar |
| Loops per performance | No limit, but duplicates removed | Yes, within one performance |

---

## 4. Drum kits and rendering

**Kit list:**

| Kit name | Type (acoustic / electronic / lo-fi / other) | Source and license | Train or test | Notes |
|---|---|---|---|---|
| | | | | |

- **Aim:** a varied set, so the model can't identify a match just by kit sound. Hold out a few kits of each type for testing only.
- **Tempos:** default range 80–140 BPM. DECISION: Expand range to 70-160 bpm, remove performances outside
- **Level balancing:** default is to loudness-normalize each rendered loop and check that hats are clearly audible in every kit. DECISION: Yes. 
- **Velocity handling:** keep the performer's velocities, unless a kit has poor velocity layers. DECISION: Yes.
- **Licensing:** if a kit comes from a commercial library, we release only MIDI and splits, never its rendered audio.

---

## 5. How people beatbox a beat (simplification rules)

Real amateurs don't voice every hit. B's generator applies these rules before placing the mouth sounds from AVP.

| Rule | Default | DECISION |
|---|---|---|
| Voices used | kick, snare, closed hi-hat, open hi-hat (the AVP sound types) | |
| Hits at the same moment | Voice only one sound, with priority **kick > snare > hats** | |
| Ghost notes | Never voiced | |
| Busy hats | If hats play 16ths, thin to 8ths with probability **0.5** | |
| Cymbals and toms | Never voiced | |
| Open hi-hat | Voice with the AVP open-hat sound | |

**Mouth sounds:** use AVP's "personal" recordings first, where people chose their own sounds. DECISION:

---

## 6. Realistic sloppiness

Each fake beatbox recording gets a random amount of each effect within these ranges. B builds each as a separate adjustable setting, so D can also test them at extreme values.

| Effect | Default range | What it models | DECISION |
|---|---|---|---|
| Steady rush or drag | −20 to +20 ms, same for the whole clip | Playing consistently ahead of or behind the beat | |
| Per-hit timing wobble | 0 to 40 ms | Imprecise timing on individual hits | |
| Tempo drift | 0 to 8% over the clip | Slowly speeding up or slowing down | |
| Swing drift | small random change in swing amount per bar | Inconsistent swing feel | |
| Dropped hits | kick 0–0.05, snare 0–0.05, hats 0–0.4 probability | Forgetting or skipping hits | |
| Background noise | 10 to 40 dB SNR | Room and laptop noise | |
| Mic coloration | mild random EQ | Different laptop mics | |

These ranges are guesses. A should listen to a handful of fake recordings next to real AVP improvisations and adjust until they sound similar.

---

## 7. How people tap a beat

Taps have only two "hands": low and high.

| Rule | Default | DECISION |
|---|---|---|
| Low hand | kick | |
| High hand | snare and hats merged | |
| Hits at the same moment | One tap on the low hand | |
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
| | | | |
