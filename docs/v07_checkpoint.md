# v0.7 vocal rules and review checkpoint

Joshua's spec v0.7 supplies two new rules. This checkpoint implements both for
clean and sloppy queries, without changing source rhythm labels or participant
assignments. It is still the small train/validation review batch, not the full
seven-kit dataset.

## What changed

### Speed limit, before sloppiness

`limit_event_rate()` runs after the existing simplifications. It considers all
kicks first, then snares, then hats, in `simultaneous_priority` order. Within
each voice it considers earlier grid times first. A candidate is accepted only
if its clean time is at least `beatbox.min_gap_ms[voice]` from **every** sound
already accepted, including sounds later in musical time. Equality is allowed.
The final retained list is sorted chronologically for rendering.

Current gaps: 120 ms for kicks and snares; 250 ms for hats. These come from the
shared spec, not from new thresholds in the generator. This is intentionally
not a simple chronological filter and not a symmetric maximum of two voices'
thresholds. It uses the candidate voice's threshold, as specified.

The source pattern ID remains the retrieval target. `speed_limited_hits` records
the number of voiced candidates removed; this does not alter the drum library.

### No overlapping mouth sounds, after sloppiness

`src/queries/mixing.py` supplies one shared mixer for clean and sloppy queries.
For a sloppy query, jitter/offset and dropped-hit decisions happen first. Only
retained placements can interrupt another sound. The mixer sorts actual frame
positions and shortens an overlapping sample to the next played onset, fading
inside its tail using `beatbox.sound_end.fade_ms`. A sound that naturally ends
sooner is never stretched to fill a gap. A dropped or out-of-window hit cannot
choke an audible sample. Coincident starts are handled deterministically, with
the later input winning rather than summing two samples.

This happens **before noise and normalization**. Background noise is expected
between utterances, but the vocal contributions do not overlap. The normal
whole-clip ending cut/fade from `loops.audio` is applied separately.

Provenance records now include `played`, `audio_start_frame`, exclusive
`audio_end_frame`, `next_played_frame`, `choked_at_next_sound`, and
`choke_fade_frames`. Dropped events retain their decision and timing record but
are marked unplayed. Clean and sloppy queries each compute their own choke
boundaries; the sloppy mixer reloads the original extracted sample rather than
reusing a sample shortened for the clean query.

## Regenerate and listen

From the new laptop's repository root:

```bash
.venv/bin/python -m src.build_checkpoint --seed 0
.venv/bin/python -m tools.check_model_loader --model-ref origin/c/model-core
```

The generation command now defaults to a directory named for the current spec:
`results/v07_checkpoint/`. Versioned render/query IDs prevent the regenerated
batch from masquerading as the previous v0.5 audio. The loader checker follows
the same current-spec default. Use `--out` for a separate listening folder and
`--dataset` for its loader check.

Open `results/v07_checkpoint/preview.html` in a browser. Expected size remains
10 drum loops, 20 clean references, and 20 sloppy queries, plus real AVP
improvisations for listening comparison. Older preview folders are retained for
comparison; their audio does not automatically update when source code changes.

The command refreshes the catalog/extracted samples when required by the new
spec version and keeps the existing saved 20/3/5 participant partition. No new
dataset download is required for this small Black Pearl train/validation batch.
The existing source data, sample kit, SciPy loader dependency, and the locally
fetched model branch must be present. See `v05_handoff.md` for those prerequisites.

`summary.json` records the current speed/choke settings and removal counts.
`events.jsonl` records actual retained vocal intervals, which can be checked for
zero overlap independently of the added background noise. `audio_sha256.json`
supports a byte-for-byte repeat-generation check. The current manifests remain
the active-file list; generated audio/reports are gitignored.

## Verification and remaining work

`tests/test_v07_voice_rules.py` checks priority-first spacing, exact threshold
equality, future-in-time conflicts, actual-time sorting, endpoint fades, short
samples, coincident starts, the disabled-choke setting, and a dropped middle
hit after jitter. Existing simplification, SNR, source provenance, framing, and
model-loader tests/checks also apply.

Listen for a less frantic single-person performance and natural transitions.
Technical non-overlap does not prove musical realism; Joshua's go/no-go listen
is still required.

Next, implement/run full-pattern generation across the five training kits and
the two held-out kits, generate matching queries, and run full-manifest checks.
That is a separate checkpoint: this command still builds a small review batch.
No commit, push, or message to another teammate is performed by regeneration.
