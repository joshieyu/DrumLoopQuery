# Hum a Beat, Find a Loop — Team README

Few-shot retrieval of drum loops from beatboxed and tapped queries.

This README is the team's single source of truth for **what we are building, why, who owns what, and how our pieces fit together**. The proposal itself is `main.tex`; references are in `references.bib`. All musical decisions live in `music_spec.md`, owned by the music lead (Member A).

> **Status:** Proposal draft. Update the "Decisions log" at the bottom whenever we change scope.

---

## Contents

1. [The project in one paragraph](#1-the-project-in-one-paragraph)
2. [Research questions and hypotheses](#2-research-questions-and-hypotheses)
3. [Background: what already exists](#3-background-what-already-exists)
4. [System overview](#4-system-overview)
5. [Data](#5-data)
6. [Models](#6-models)
7. [Baselines](#7-baselines)
8. [Evaluation protocol](#8-evaluation-protocol)
9. [Experiment matrix](#9-experiment-matrix)
10. [Team roles](#10-team-roles)
11. [Timeline and milestones](#11-timeline-and-milestones)
12. [Interfaces between roles (file contracts)](#12-interfaces-between-roles-file-contracts)
13. [Real-query recording protocol](#13-real-query-recording-protocol)
14. [Risk register](#14-risk-register)
15. [Working in this Overleaf project](#15-working-in-this-overleaf-project)
16. [Glossary](#16-glossary)
17. [Open decisions and decisions log](#17-open-decisions-and-decisions-log)

---

## 1. The project in one paragraph

Producers searching a sample library for "that kind of groove" usually type text tags, which describe genre and mood but not *where the kicks, snares, and hats actually land*. People naturally communicate grooves by performing them: beatboxing or tapping on a desk. We build a system where a user beatboxes or taps a 1–2 bar rhythm into a laptop mic and gets back a ranked list of drum loops with the same rhythm. The core technical idea is to learn an embedding space where a query and a drum loop land close together **if and only if they share a rhythm pattern**, regardless of timbre (voice vs. drum kit). The core scientific question is whether this learned approach beats the classic 2004-era pipeline of *transcribing* the query into drum hits and matching symbolically, and whether a few seconds of **user enrollment** (few-shot adaptation to one person's voice) helps.

**Input:** imperfect beatbox or tap audio (1–2 bars).
**Output:** top-k drum-loop audio clips, ranked by rhythmic similarity.

Why it is hard:

- **Timbre gap.** A mouth "pf" and a kick drum share timing, not spectrum.
- **Imperfect performers.** Amateurs drift in tempo, drop hi-hats, and merge voices. A tap has at most two "voices".
- **Voice variability.** Every person's beatbox kick sounds different, which is why we test on people the model has never heard.

---

## 2. Research questions and hypotheses

| ID | Question | Hypothesis | Main metric |
|---|---|---|---|
| RQ1 | Do learned embeddings beat transcribe-then-match (B3) and CLAP (B1) on unseen users? | Learned embeddings beat CLAP by a wide margin; the comparison with B3 is genuinely open, and B3 may win on clean beatbox but lose on messy or tapped queries. | MRR |
| RQ2 | Which front-end (R1–R4) transfers best from synthetic to real queries? | Rhythm-focused front-ends (R1, R2) transfer better than log-mel (R3). The pretrained encoder (R4) is the wildcard. | MRR on real queries |
| RQ3 | How much does few-shot enrollment help unseen voices? | Gains are largest from k=0 to k=1 and flatten by k=5. | ΔMRR vs. k=0 |

A negative result on any RQ is still a valid result. What matters is that the comparison is fair and the error analysis explains *why*.

---

## 3. Background: what already exists

Read at least the first three before week 2.

| Paper | Why it matters to us |
|---|---|
| Kapur, Benning, Tzanetakis — *Query-by-Beat-Boxing: Music Retrieval for the DJ* (ISMIR 2004) | The original idea. Classifies beatbox sounds and detects tempo to browse drum loops. Our B3 baseline is a modern version of this design. |
| Nakano, Ogata, Goto, Hiraga — *A Drum Pattern Retrieval Method by Voice Percussion* (ISMIR 2004) | Matches the timing between voiced drum sounds against symbolic patterns. Also transcribe-then-match. |
| O'Reilly et al. — *The Rhythm In Anything (TRIA)* (ISMIR 2025) | Generates drum audio from beatbox and tap prompts. Its adaptive **two-band "dualized"** rhythm features are our front-end R2. Evaluates on AVP and TapTamDrum, the same datasets we use. |
| Ramires, Penha, Davies — *User Specific Adaptation in Automatic Transcription of Vocalised Percussion* (arXiv 2018) | Shows that per-user adaptation helps *transcription*. Our novelty is few-shot adaptation for *retrieval embeddings*. |
| Wu et al. — LAION-CLAP (ICASSP 2023) | Our B1 baseline. Trained on text–audio semantics, so we expect it to be weak on fine rhythm. We test this; we don't assume it. |
| Delgado et al. — *A New Dataset for Amateur Vocal Percussion Analysis* (Audio Mostly 2019) | The AVP dataset, our source of real amateur vocal drum sounds. |
| Haki et al. — *TapTamDrum* (ISMIR 2023) | Real two-hand tapped imitations of drum patterns by experienced drummers. Our real tap test set. |
| Gillick et al. — *Learning to Groove* (ICML 2019); Callender et al. — *E-GMD* (arXiv 2020) | GMD (MIDI and audio drum performances) and E-GMD (the same performances rendered on many kits). Our library source. |
| Khosla et al. — *Supervised Contrastive Learning* (NeurIPS 2020) | Our training loss. |
| Li et al. — MERT (ICLR 2024); Hu et al. — LoRA (ICLR 2022) | Pretrained music encoder (R4) and the adapter method for few-shot enrollment. |

**Our gap in one sentence:** nobody has recently compared learned and pretrained-encoder rhythm embeddings against transcribe-then-match for query-by-vocal-percussion retrieval, with controlled query conditions, participant-disjoint evaluation, and few-shot enrollment.

---

## 4. System overview

```
GMD / E-GMD MIDI patterns
   ├── render with drum kits (several tempos) ──────────────► LOOP LIBRARY ─┐
   └── render with AVP voices + perturbations / tap reduction ► QUERIES ────┤
                                     REAL QUERIES (team, TapTamDrum) ───────┤
                                                                            ▼
                                   FRONT-END (R1 onset bands | R2 dualized | R3 log-mel | R4 MERT)
                                                                            ▼
                                   SHARED ENCODER f_θ  (+ per-user LoRA on the query branch)
                                                                            ▼
                                   128-d RHYTHM EMBEDDING  ──►  cosine top-k retrieval
                         training signal: SupCon loss, class = rhythm pattern ID
```

**The key trick:** library loops and synthetic queries are both rendered from the *same MIDI pattern*. That gives us exact, free ground-truth relevance labels, which is what makes this feasible in eight weeks.

---

## 5. Data

### 5.1 Sources

| Dataset | What it is | How we use it | Owner |
|---|---|---|---|
| **GMD** | Expressive drum performances with aligned MIDI and audio | Source of rhythm patterns (MIDI) | A decides, B builds |
| **E-GMD** | GMD performances re-rendered with many drum kits | Timbre-diverse library audio; kit-disjoint test split | A decides, B builds |
| **AVP** | 28 amateur participants, about 9.8k annotated vocal drum utterances (kick, snare, closed and open hi-hat), laptop mic, "personal" and "fixed" imitation modes | Voice sounds used to synthesize beatbox queries; enrollment examples for the few-shot experiment | A decides, B builds |
| **TapTamDrum** | Two-hand tapped imitations of drum patterns by 4 experienced drummers | Real tap test set | D |
| **Team recordings** | About 200 real beatbox queries (about 50 per member) | Real test set and leave-one-member-out fine-tuning | A leads sessions, B organizes files, everyone records |

**Check licenses before week 1 ends.** B notes the terms of each dataset here. Our use is non-commercial coursework.

### 5.2 Library construction (A decides, B builds)

The musical choices below are defaults. The final versions live in `music_spec.md`.

1. Keep 4/4 material only. Cut performances into **2-bar clips**.
2. Map drum notes to **3 voices**: kick; snare (plus rimshot); hats and cymbals. The full mapping, including toms and cymbals, is in `music_spec.md`.
3. Quantize each clip to a grid of **12 ticks per beat**. This resolution represents both straight 16ths and triplet swing, so "straight" and "swung" patterns stay distinct.
4. **Pattern ID** = hash of the 3-voice quantized grid. Clips with identical grids share an ID (this deduplicates repeated bars within a performance).
5. Render audio. Use E-GMD audio where available, plus our own renders of the MIDI with open drum-sample kits at several tempos (for example 80–140 BPM).
6. **Splits:** use GMD's official train/val/test split for patterns. Hold out a set of kits for test only.

### 5.3 Query synthesis (A decides, B builds)

A writes the rules for how people beatbox and tap, and the ranges for realistic sloppiness, in `music_spec.md`. B turns them into code with adjustable settings.

- **Beatbox:** for each MIDI hit, place one AVP utterance of the matching drum class from a single participant. Use the "personal" subset first; "fixed" is an option.
- **Perturbations**, each with a controllable strength so D can sweep them:
  - timing jitter (ms)
  - tempo drift (% over the clip)
  - dropped-hit probability, highest for hi-hats
  - background noise (SNR)
  - mic coloration (EQ)
- **Taps:** reduce each pattern to 2 voices (low = kick; high = snare ∪ hats), then render with tap or knock sounds. This deterministic reduction is our simplification. TapTamDrum's human reductions tell us what it costs.
- **Participant split (AVP):** 20 train / 3 val / 5 test participants. **No participant appears in more than one split.**

### 5.4 Relevance definition (rules from A, answer key built by D)

- **Beatbox query:** relevant items are all library clips with the same pattern ID.
- **Tap query:** relevant items are all clips whose **2-voice reduced grid** matches the query's pattern. Many 3-voice patterns share one 2-voice reduction, and we don't penalize the model for that ambiguity.
- **Graded signal:** also report the grid Hamming distance between the query pattern and the top-1 result ("how wrong is the wrong answer?").

---

## 6. Models

**Front-ends (C):**

| ID | Front-end | Why |
|---|---|---|
| R1 | 3-band onset-strength envelopes (low/mid/high ≈ kick/snare/hat) | Hand-designed, timbre-light, cheap |
| R2 | TRIA-style adaptive 2-band dualized features | Shown to bridge beatbox/taps and drums for generation |
| R3 | 64-bin log-mel spectrogram | Generic learned-feature control |
| R4 | Frozen MERT-95M frame features (and a LoRA-adapted variant) | Pretrained music encoder; ties to course topics |

**Encoder:**
- 4-layer Transformer (d = 128), a learned query-type token (library / beatbox / tap), attention pooling, and an L2-normalized 128-d output.
- Weights are shared between the query and library branches.

**Loss:** supervised contrastive (SupCon). Class = pattern ID. Each batch contains several "views" of each pattern: different kits, different AVP voices, different perturbations. Triplet loss is an ablation only.

**Few-shot enrollment (B runs the experiment, C provides the adapter code):**
1. A new user provides k ∈ {1, 3, 5} utterances per drum class.
2. Synthesize queries in that voice from training patterns.
3. Fine-tune a LoRA adapter on the **query branch only** for a few minutes.
4. Evaluate on that user's held-out queries.

**Compute budget:** under 5M trainable parameters excluding MERT. MERT features are precomputed once and cached. Everything should train on one GPU in hours.

---

## 7. Baselines (D)

| ID | Baseline | Notes |
|---|---|---|
| B1 | LAION-CLAP audio embeddings, cosine similarity | Tests the "semantic embeddings miss rhythm" claim |
| B2 | Cross-correlation of onset envelopes (max over circular lags) | Pure DSP, no learning |
| B3 | **Transcribe-then-match:** onset detection, then a drum-sound classifier trained on AVP train participants, then a 3-voice grid, then Hamming distance to library grids | The 2004-style paradigm. The library side uses ground-truth MIDI grids, so B3 is *optimistic*. Say so in the report. |
| (B4) | Tempogram autoencoder | Optional; only if time allows |

B3 is the most important comparison in the project. Build it carefully, not as a strawman.

---

## 8. Evaluation protocol (D)

- **Metrics:** MRR (primary), Recall@1/5/10, mAP, and top-1 grid distance.
- **Statistics:** bootstrap 95% confidence intervals over queries. Paired permutation tests between systems on the same query set.
- **Test sets:** reported separately, never pooled:
  1. synthetic beatbox (test AVP participants, test kits)
  2. synthetic taps
  3. real team beatbox
  4. TapTamDrum
- **Leave-one-member-out:** when fine-tuning on real team queries, train on 3 members and test on the 4th. Rotate and average.
- **One harness rule:** every system is scored by D's single evaluation script from an embedding file (or score matrix, for B3). Nobody computes metrics in their own notebook for the report.

---

## 9. Experiment matrix

| Exp | Varies | Fixed | Answers | Owner |
|---|---|---|---|---|
| E1 Main comparison | System: B1, B2, B3, best learned model | All 4 test sets | RQ1 | D |
| E2 Front-end | R1 / R2 / R3 / R4 (frozen) / R4 (LoRA) | Loss, data | RQ2 | C |
| E3 Synthetic→real | Synthetic only vs. plus real fine-tuning (leave-one-member-out) | Best front-end | RQ2 | C |
| E4 Few-shot | k = 0, 1, 3, 5 | Best model | RQ3 | B |
| E5 Robustness | Tempo drift, dropped-hit rate, SNR sweeps | Best model and B3 | Where each approach breaks | D |
| E6 Tempo handling | Tempo-normalized vs. tempo-agnostic input | Best model | Design choice | C |
| E7 Loss | SupCon vs. triplet | Best front-end | Minor ablation | C |
| E8 Error analysis | Pattern properties: swing, density, tempo, style tags | Best model and B3 | Why failures happen | A |

---

## 10. Team roles

**In one line:** A decides what everything should sound like, B turns those decisions into data, C builds the search engine, and D checks whether it works.

**The music rule:** A is the team's only musician, so every decision that needs musical judgment goes to A and gets written down in `music_spec.md` as a concrete rule with numbers. Everyone else works from that written spec. Nobody else should ever need to decide what "sounds right." If you run into a musical question, ask A and add the answer to the spec.

Each role has an early job and a late job, so nobody is stuck waiting at the start or idle at the end.

### Member A: music lead (`[Member A Name]`)

A makes every musical decision in the project and checks the results by ear. That includes how drum parts are grouped, what counts as "the same rhythm," which drum kits we use, how real people simplify beats when they beatbox or tap, and what realistic sloppiness sounds like. A writes these down as clear rules so B can code them without needing any music background.

- **Week 1:** write the first version of `music_spec.md`. Pick the drum kits, and try test renders in a DAW to hear how they sound together.
- **Weeks 2–3:** set up scripted rendering of the chosen kits with B (DawDreamer or pedalboard can play VST instruments from Python). Listen to samples of B's loops and fake beatbox recordings and approve them or send back fixes. **Musical sign-off is required before the week 3 data freeze.**
- **Weeks 4–5:** lead the recording sessions: choose which beats to record, set up the click, headphones, and levels, and judge whether each take is usable. Tag beats with musical properties (swung or straight, busy or sparse, style) for later analysis. Listen to the top search results from the first models and flag anything that sounds wrong even when the numbers look fine.
- **Weeks 6–7:** study the system's mistakes and explain them in musical terms (experiment E8), with figures.
- **Hands off to the team:** `music_spec.md`, the list of drum kits, approved listening checks, and the recording session plan.
- **Proposal:** Section 1 (Problem and Motivation) and the Datasets part of Section 4.
- **Report:** the Data section's musical design, and the mistake analysis in Results.

### Member B: data pipeline (`[Member B Name]`)

B writes the code that turns A's rules into the actual data. B builds the loop library by cutting the drum recordings into loops, rendering them through the chosen kits, and splitting them into training and test sets. B also builds the generator that makes fake beatbox and tap recordings, with the sloppiness settings as adjustable knobs. No music experience is needed: every musical choice comes from the spec, and A checks the output by ear.

- **Weeks 1–3:** load GMD, E-GMD, and AVP; write the code that cuts loops, groups drum parts, labels rhythms, and removes duplicates; build the fake beatbox and tap generator from the spec; make the training and test splits; check dataset licenses. **Freeze the data at the end of week 3** (after A's sign-off).
- **Weeks 4–5:** handle the files from the recording sessions (naming, trimming, entering them into the manifest) and prepare each person's enrollment sounds.
- **Weeks 6–7:** test whether letting a new person record a few sample sounds first makes the system work better for them (experiment E4), using C's adapter code.
- **Hands off to the team:** the loop list (`library_manifest.csv`), the list of all searches (`query_manifest.csv`), the generator code, and code that loads the data.
- **Proposal:** Section 6 (Feasibility and Fallback).
- **Report:** how the data and searches were built, the new-user results, and a **final editing pass so the report reads in one voice**.

### Member C: builds and trains the AI model (`[Member C Name]`)

C writes the neural network that listens to a recording and summarizes its rhythm, so that a beatbox and a drum loop playing the same beat get similar summaries. C trains it on B's data and tries a few different ways of feeding the audio in to see which works best. *This role needs the most hands-on machine learning experience, so give it to whoever is most comfortable training models in PyTorch.* No music experience needed.

- **Weeks 1–3:** build the model and training code using small practice data while A and B finish the real data. Prepare the pretrained music model (MERT) so its outputs are computed once and saved.
- **Weeks 4–5:** train the full models, one for each way of feeding in the audio.
- **Weeks 6–7:** run the model comparisons (experiments E2, E3, E6, E7) and give B the code for the new-user adjustment.
- **Hands off to the team:** training code, saved models, and the rhythm summaries for every search and loop (`embeddings/*.npz`), in the format D's scoring code expects.
- **Proposal:** Sections 3 (Proposed Method, including Figure 1) and 5 (Novelty).
- **Report:** the Method section and the model-comparison results.

### Member D: judges how well everything works (`[Member D Name]`)

D writes the scoring code that measures how often the right loop shows up at or near the top of the results. D builds the answer key (which loops count as a match for each search) directly from the rules in the spec, and prepares the TapTamDrum tapping recordings as an extra test set. D also builds the simpler systems we compare against, including an older-style method that identifies each individual drum hit first and then looks up a match. D runs every system through the same scoring so the comparison is fair, tests how each one holds up when the recordings get messier, and makes the final results tables. No music experience needed.

- **Weeks 1–3:** write the scoring code; build the answer key (`relevance.json`) from the spec's rules; build the two simplest comparison systems (CLAP and onset matching); prepare TapTamDrum.
- **Weeks 4–5:** build the older-style "identify each hit, then match" system; score the comparison systems on the fake searches.
- **Weeks 6–7:** run the main comparison of all systems (E1), test them on messier recordings (E5), check which differences are real and not luck, and make the results tables.
- **Hands off to the team:** one scoring script (`evaluate.py`) that everyone uses, the answer key, the comparison systems, and the final tables and figures.
- **Proposal:** Section 2 (Related Work), the Baselines and Metrics part of Section 4, and Section 7 (Timeline).
- **Report:** Related work, Experimental setup, and the main results.

### Shared by everyone

- Record about 50 real beatbox searches each (week 4). A will coach you; no beatboxing skill is needed, and amateur recordings are exactly what we want.
- Meet for 30 minutes each week. Each person **shows something that runs**, not just a status update. A plays a few audio examples each week so everyone hears what the data sounds like.
- Review one other person's part of the report (A reviews B, C reviews D, and the reverse).

---

## 11. Timeline and milestones

| Week | A (music lead) | B (data pipeline) | C (model) | D (evaluation) |
|---|---|---|---|---|
| 1 | Write `music_spec.md` v1; pick kits; test renders in a DAW | Load GMD, E-GMD, AVP; loop-cutting code; licenses | Prototype model on toy data | Scoring code, metric definitions |
| 2 | Set up scripted kit rendering with B; listen to first loops and fake beatboxes | Drum grouping, rhythm labels, duplicates; beatbox and tap generator | Training loop, MERT caching | Answer key from spec rules; CLAP and onset baselines |
| 3 | **Musical sign-off**; plan recording sessions | Sloppiness knobs, splits; **freeze data** | First full training run | "Identify each hit, then match" baseline; TapTamDrum |
| 4 | Lead recording sessions | Organize and trim recordings | Model variants | Baselines on fake searches |
| 5 | Tag beats musically; listen to early search results | Enrollment sounds; final real-query manifest | MERT/LoRA arm | Evaluate all models so far |
| 6 | Mistake analysis (E8) | New-user experiment (E4) | E2, E6 | Messier-recording tests (E5) |
| 7 | Mistake analysis, figures | New-user results | E3, E7, final runs | Final tables, significance tests |
| 8 | Report | Report and edit pass | Report | Report |

**Milestones:**

- **M1, end of week 1:** file contracts (Section 12) agreed and written down; first version of `music_spec.md` shared.
- **M2, end of week 3: dataset and query generator frozen, after A's musical sign-off.** This is the critical path. C and D work on preliminary data until then rather than waiting.
- **M3, end of week 5:** every system has numbers on every test set, even if bad.
- **M4, end of week 7:** all tables and figures final; no new experiments after this.

---

## 12. Interfaces between roles (file contracts)

Agree on these in week 1. Changing a contract after M2 requires telling the whole team.

**`library_manifest.csv`** (owner: B, following `music_spec.md`)
```
clip_id, pattern_id, dualized_pattern_id, split, kit_id, tempo_bpm, audio_path, midi_path, source_performance
```

**`query_manifest.csv`** (owner: B, following `music_spec.md`)
```
query_id, pattern_id, dualized_pattern_id, query_type{beatbox,tap}, source{synthetic,team,taptamdrum},
participant_id, split, tempo_bpm, jitter_ms, drift_pct, drop_prob, snr_db, audio_path
```

**`relevance.json`** (owner: D, following the rules in `music_spec.md`)
```
{ "query_id": ["clip_id", ...], ... }   # generated from pattern_id / dualized_pattern_id rules
```

**`embeddings/<system>_<testset>.npz`** (owner: D defines, C produces)
```
query_ids:  (Nq,)       library_ids: (Nl,)
query_emb:  (Nq, d)     library_emb: (Nl, d)     # L2-normalized
# Non-embedding systems (B2, B3) instead save  scores: (Nq, Nl)
```

**Code lives in a shared Git repo, not in Overleaf.** Overleaf is for writing only. Suggested repo layout:
```
data/        (gitignored raw + rendered audio)
manifests/   library_manifest.csv, query_manifest.csv, relevance.json
src/library/ src/queries/ src/models/ src/baselines/ src/eval/
configs/     one YAML per experiment
results/     npz files, tables, figures
```

---

## 13. Real-query recording protocol

A leads the sessions and makes the musical calls. B handles the files. Everyone records.

- **Consent:** only team members are recorded. If we ever add outside participants, check with the instructor first about human-subjects rules.
- **Setup:**
  - built-in laptop mic, to match AVP conditions
  - quiet room
  - headphones for the click, so it doesn't bleed into the recording
- **Procedure:**
  1. The participant hears the target loop twice.
  2. They beatbox it along to a click at the loop's tempo, for 2 bars.
  3. Repeat once without the click. This gives the "free tempo" subset.
- **Per member:** about 50 patterns, chosen by A from the **test split** and balanced across straight and swung and across pattern density. A skips beats that are unreasonable to beatbox.
- **Take quality:** A decides whether a take is usable. Imperfect is fine and expected; only discard takes where the person clearly lost the beat or the audio is broken.
- **Enrollment clips:** each member also records 5 isolated kicks, snares, and hi-hats for E4.
- **Filenames:** `team_<member>_<pattern_id>_<click|free>_<take>.wav`. Log everything in `query_manifest.csv`.

---

## 14. Risk register

| Risk | Likelihood | Impact | Mitigation / fallback | Owner |
|---|---|---|---|---|
| Synthetic queries don't resemble real ones | Medium | High | A's listening checks before the freeze; measured directly by the real test set; plus-real fine-tuning (E3); report the gap honestly | A, B, C |
| Dataset freeze slips past week 3 | Medium | High | C and D use preliminary data; cut tempo variants before cutting kits | B |
| Tempo estimation fails on beatboxing | High | Medium | Click-track recordings; tempo-agnostic model; drift ablation (E5/E6) | C, D |
| TapTamDrum patterns can't be aligned to our library | Medium | Low | Report taps on synthetic queries only; describe TapTamDrum qualitatively | D |
| Learned models lose to B3 | Possible | Low | Still a publishable-style finding; B3 becomes the main system and analysis explains why | D |
| MERT too slow or large | Low | Medium | Precompute features once; drop the LoRA variant, keep frozen | C |
| Music lead becomes a bottleneck | Medium | Medium | Spec v1 by end of week 1 with default values so B can start; A answers spec questions within a day; listening checks use small random samples | A |
| Scope creep | High | Medium | Anything not in the experiment matrix goes to "Future work" | Everyone |

**Emergency scope cut, in this order:** drop B4, drop E7, drop R3, shrink to 1-bar patterns, drop taps.

---

## 15. Working in this Overleaf project

- **Files:** `main.tex` (proposal), `references.bib`, this `README.md`, and `music_spec.md` (musical decisions, owned by A).
- **Compiler:** pdfLaTeX. BibTeX runs automatically. The figure is drawn in TikZ inside `main.tex`, so there are no image files.
- **Page limit:** 1–2 pages **excluding references**. The current draft's body ends on page 2 with some room left. Check the PDF after every edit. If you add text, cut the same amount elsewhere.
- **Editing your section:** leave comments as `% TODO[A]: ...` with your letter. Use Overleaf's review/track-changes mode when editing someone else's section.
- **Citations:** add BibTeX entries to `references.bib`, cite with `\cite{key}`, and **verify every entry** (authors, venue, year) against the actual paper. Never cite something none of us has opened.
- **Before submitting:**
  - replace all `[Member X Name]`, `[Course ...]`, `[Instructor Name]`, and `[Submission Date]` placeholders
  - recompile
  - confirm the page count
  - export the PDF

---

## 16. Glossary

**Music terms** (for non-musicians; A can demo any of these):

- **Bar:** one repeating unit of a beat. In 4/4 time, a bar has four beats. Our loops are two bars long.
- **Downbeat:** the first beat of a bar. Loops should start there.
- **Velocity:** how hard a drum is hit (0–127 in MIDI).
- **Ghost note:** a very quiet hit that adds feel but isn't really part of the pattern.
- **Flam:** two hits of the same drum played almost at once, heard as one thick hit.
- **Fill:** a short break from the main beat, usually at the end of a phrase. We keep fills out of the library.
- **Swing:** playing off-beat notes slightly late, so the rhythm "bounces" instead of sounding even ("straight").
- **Half-time / double-time:** the same beat felt at half or twice the speed. Two MIDI files can look different yet feel like the same beat, or the reverse.
- **Articulation:** a specific way of hitting a drum, for example open vs. closed hi-hat, or cross-stick vs. full snare hit.
- **Click:** a metronome sound played in headphones so a performer keeps a steady tempo.

**Project terms:**

- **Query-by-vocal-percussion (QVP):** retrieving music using a beatboxed query.
- **Pattern ID:** hash of a clip's quantized 3-voice drum grid. Clips with the same ID are "the same rhythm".
- **Dualized / 2-voice reduction:** collapsing a drum pattern to low and high voices, the way a tap represents it.
- **Transcribe-then-match:** convert the query into symbolic drum hits first, then compare symbols. This is the classic approach (B3).
- **SupCon:** supervised contrastive loss; pulls together embeddings that share a class label.
- **LoRA:** low-rank adapter; a small set of trainable weights added to a frozen model, used here for per-user enrollment.
- **MRR:** mean reciprocal rank; average of 1/(rank of the first correct result).
- **Participant-disjoint:** no AVP participant or team member appears in both training and testing for the same result.

---

## 17. Open decisions and decisions log

**Open (decide by M1):**

- [ ] Who takes each role (A is our musician; C should be the strongest PyTorch person)
- [ ] Musical decisions (kits, drum grouping, toms, ghost notes, beatbox rules): tracked in `music_spec.md`, decided by A
- [ ] Clip length: 2 bars, or 1 bar if dedup leaves too few patterns?
- [ ] GPU source (lab machine, university cluster, Colab)

**Log** (date — decision — who):

- YYYY-MM-DD — Chose drum-loop retrieval over the synth-patch project — team
- YYYY-MM-DD — …
