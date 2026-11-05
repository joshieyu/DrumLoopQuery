"""Offline drum sampler: note-on events through an SFZ program, out comes a stereo mix.

    program = read_program(path)
    sampler = Sampler(program, seed=0)
    audio = sampler.render([Hit(0.25, key=38, velocity=100)])  # (frames, 2) float32
    sampler.sample_rate

For each note-on, in time order:

- Regions: key, velocity, CC, and keyswitch conditions, then round robin (a counter per
  region, advanced by every note-on the region matches) and one random number per
  note-on for lorand/hirand.
- Level: velocity curve, amplitude / volume / pan with their CC modulations, phase.
  Pan law for mono samples: -3 dB constant power; for stereo samples, a balance control.
- Onset alignment: many kits leave a few ms of silence before each hit (Salamander up to
  22 ms). Each note-on is moved earlier by the lead-in of its earliest voice, so the hit
  lands on time while the natural delays between microphones of one hit stay.
- Chokes: a new voice whose `group` is G stops earlier voices with `off_by` G
  (off_mode fast: a 6 ms fade, ARIA's value; normal: ampeg_release; time: off_time).
- Polyphony: `note_polyphony` limits earlier voices of the same group and key, sparing
  louder ones (the reference's default self-masking); `polyphony` limits a group.
- Envelope: ampeg attack (linear), hold, decay (exponential, ARIA's curve), sustain.

Hits are one-shots: no note-offs are sent, so a voice plays to its sample's end unless
something stops it. Nothing is resampled, so every sample must share one rate.

Run `python -m tools.render.sampler --help` for the listening command line.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.render.sfz import Program, Region, SfzError

FAST_OFF_S = 0.006  # off_mode=fast fade (ARIA)
ONSET_DB = -40.0  # a sample's onset: its first frame within 40 dB of the sample's peak
FLOOR = 1e-5  # exponential release tails stop below -100 dB


@dataclass(frozen=True)
class Hit:
    time_s: float
    key: int
    velocity: int
    ccs: dict[int, int] = field(default_factory=dict)  # CC values (0..127) for this hit only


@dataclass
class _Voice:
    region: Region
    event: int
    key: int
    velocity: int
    start: int  # frame where the sample's first played frame lands
    gains: tuple[float, float]
    length: int  # frames of sample played
    stop: int | None = None  # frame where a fade-out starts
    stop_mode: str = "fast"
    stop_s: float = FAST_OFF_S

    def sounding_at(self, frame: int) -> bool:
        return self.start <= frame < self.start + self.length and (
            self.stop is None or frame < self.stop
        )


class Sampler:
    def __init__(self, program: Program, seed: int = 0, align_onsets: bool = True):
        if program.unsupported:
            raise SfzError(
                f"{program.path.name} uses opcodes the sampler doesn't implement: "
                f"{sorted(program.unsupported)}"
            )
        self.program = program
        self.seed = seed
        self.align_onsets = align_onsets
        self._audio: dict[str, np.ndarray] = {}
        self._onsets: dict[tuple[str, int, int | None], int] = {}
        first = next((r.sample for r in program.regions if not r.silent), None)
        if first is None:
            raise SfzError(f"{program.path.name} has no audible regions")
        self.sample_rate = sf.info(first).samplerate

    # ------------------------------------------------------------------ samples

    def audio(self, path: str) -> np.ndarray:
        """A sample as (frames, channels) float32, decoded once and cached."""
        if path not in self._audio:
            data, rate = sf.read(path, dtype="float32", always_2d=True)
            if rate != self.sample_rate:
                raise SfzError(
                    f"{Path(path).name} is {rate} Hz but the kit is {self.sample_rate} Hz; "
                    "resampling isn't implemented"
                )
            self._audio[path] = data
        return self._audio[path]

    def played(self, r: Region) -> np.ndarray:
        """The frames of the region's sample that play: offset to end, inclusive."""
        data = self.audio(r.sample)
        stop = len(data) if r.end is None else min(len(data), r.end + 1)
        return data[min(r.offset, stop) : stop]

    def onset(self, r: Region) -> int:
        """Frames of near-silence before the hit in this region's played audio."""
        key = (r.sample, r.offset, r.end)
        if key not in self._onsets:
            level = np.abs(self.played(r)).max(axis=1)
            peak = float(level.max()) if level.size else 0.0
            if peak == 0:
                self._onsets[key] = 0
            else:
                self._onsets[key] = int(np.argmax(level >= peak * 10 ** (ONSET_DB / 20)))
        return self._onsets[key]

    # ------------------------------------------------------------------ voices

    def voices(self, hits: list[Hit]) -> list[_Voice]:
        """Every voice the hits start, with chokes and polyphony applied."""
        prog = self.program
        rng = np.random.default_rng(self.seed)
        counters: dict[int, int] = defaultdict(int)
        voices: list[_Voice] = []
        order = sorted(range(len(hits)), key=lambda i: (hits[i].time_s, i))
        for event in order:
            hit = hits[event]
            values = prog.cc_values({cc: v / 127 for cc, v in hit.ccs.items()})
            rand = rng.random()
            chosen = []
            for i in prog.candidates(hit.key, hit.velocity, values):
                r = prog.regions[i]
                counters[i] += 1
                position = (counters[i] - 1) % r.seq_length + 1
                if position == r.seq_position and r.lorand <= rand < r.hirand:
                    chosen.append(r)
            start = round(hit.time_s * self.sample_rate)
            new = []
            for r in chosen:
                if r.silent:
                    gains, length = (0.0, 0.0), 0
                else:
                    gain, pan = prog.level(r, hit.velocity, values)
                    channels = self.audio(r.sample).shape[1]
                    gains = _pan_gains(gain, pan, channels)
                    length = len(self.played(r))
                new.append(_Voice(r, event, hit.key, hit.velocity, start, gains, length))
            audible = [v for v in new if v.length]
            if self.align_onsets and audible:
                shift = min(self.onset(v.region) for v in audible)
                for v in new:
                    v.start -= shift
            for v in new:
                self._stop_others(v, voices)
            voices.extend(new)
        return voices

    def _stop_others(self, v: _Voice, earlier: list[_Voice]) -> None:
        r = v.region
        at = v.start
        live = [u for u in earlier if u.event != v.event and u.sounding_at(at)]
        if r.group != 0:
            for u in live:
                if u.region.off_by is not None and u.region.off_by == r.group:
                    _stop(u, at, u.region.off_mode, u.region)
        same_group = [u for u in live if u.region.group == r.group and u.stop is None]
        if r.note_polyphony is not None:
            same_note = [u for u in same_group if u.key == v.key]
            excess = len(same_note) - r.note_polyphony + 1
            for u in [u for u in same_note if u.velocity <= v.velocity][: max(excess, 0)]:
                _stop(u, at, "time", u.region)
        if r.polyphony is not None:
            same_group = [u for u in same_group if u.stop is None]
            for u in same_group[: max(len(same_group) - r.polyphony + 1, 0)]:
                _stop(u, at, "time", u.region)

    # ------------------------------------------------------------------ mixing

    def render(self, hits: list[Hit], length_s: float | None = None) -> np.ndarray:
        """Stereo mix as (frames, 2) float32, from time 0 to length_s (default: the end)."""
        voices = [v for v in self.voices(hits) if v.length]
        sr = self.sample_rate
        ends = [v.start + _voice_frames(v, sr) for v in voices]
        frames = round(length_s * sr) if length_s is not None else max(ends, default=0)
        out = np.zeros((frames, 2), dtype=np.float64)
        for v in voices:
            sig = self._voice_signal(v)
            a, b = v.start, v.start + len(sig)
            lo, hi = max(a, 0), min(b, frames)
            if lo < hi:
                out[lo:hi] += sig[lo - a : hi - a]
        return out.astype(np.float32)

    def _voice_signal(self, v: _Voice) -> np.ndarray:
        r, sr = v.region, self.sample_rate
        data = self.played(r)
        n = _voice_frames(v, sr)
        env = _envelope(r, n, sr)
        if v.stop is not None:
            env *= _stop_curve(v, n, sr)
        mono = data[:n, 0] if data.shape[1] == 1 else None
        if mono is not None:
            sig = np.stack([mono * v.gains[0], mono * v.gains[1]], axis=1)
        else:
            sig = data[:n, :2] * np.asarray(v.gains, dtype=np.float32)
        return sig * env[:, None]


def _pan_gains(gain: float, pan: float, channels: int) -> tuple[float, float]:
    if channels == 1:
        theta = (pan + 100) / 200 * math.pi / 2
        return gain * math.cos(theta), gain * math.sin(theta)
    return gain * min(1.0, 1 - pan / 100), gain * min(1.0, 1 + pan / 100)


def _stop(u: _Voice, at: int, mode: str, r: Region) -> None:
    if u.stop is not None and u.stop <= at:
        return
    u.stop = at
    if mode == "normal":
        u.stop_mode, u.stop_s = "release", r.ampeg_release
    elif mode == "time":
        u.stop_mode, u.stop_s = "fade", r.off_time if r.off_time is not None else FAST_OFF_S
    else:
        u.stop_mode, u.stop_s = "fade", FAST_OFF_S


def _release_frames(seconds: float, sr: int) -> int:
    """Frames for ARIA's exponential release to fall below FLOOR."""
    if seconds <= 0:
        return 0
    return math.ceil(seconds * sr * math.log(1 / FLOOR) / 8)


def _voice_frames(v: _Voice, sr: int) -> int:
    """Frames the voice actually sounds, including its fade or release tail."""
    if v.stop is None:
        return v.length
    rel = v.stop - v.start
    tail = (
        _release_frames(v.stop_s, sr) if v.stop_mode == "release" else max(1, round(v.stop_s * sr))
    )
    return max(0, min(v.length, rel + tail))


def _stop_curve(v: _Voice, n: int, sr: int) -> np.ndarray:
    rel = v.stop - v.start
    curve = np.ones(n, dtype=np.float32)
    if rel < 0:
        return np.zeros(n, dtype=np.float32)
    k = np.arange(max(n - rel, 0), dtype=np.float64)
    if v.stop_mode == "release":
        mu = math.exp(-8.0 / (v.stop_s * sr)) if v.stop_s > 0 else 0.0
        curve[rel:] = mu**k
    else:
        fade = max(1, round(v.stop_s * sr))
        curve[rel:] = np.clip(1 - k / fade, 0, 1)
    return curve


def _envelope(r: Region, n: int, sr: int) -> np.ndarray:
    env = np.ones(n, dtype=np.float32)
    attack = round(r.ampeg_attack * sr)
    hold = round(r.ampeg_hold * sr)
    sustain = r.ampeg_sustain / 100
    if attack:
        a = min(attack, n)
        env[:a] = np.arange(a) / attack
    d0 = min(attack + hold, n)
    if sustain < 1:
        k = np.arange(n - d0, dtype=np.float64)
        if r.ampeg_decay > 0:
            mu = math.exp(-8.0 / (r.ampeg_decay * sr))
            env[d0:] = sustain + (1 - sustain) * mu**k
        else:
            env[d0:] = sustain
    return env


# --------------------------------------------------------------------------- command line


def main(argv: list[str] | None = None) -> int:
    from src.music.spec import load_spec
    from tools.render import kits  # imported here: the kit helpers import this module

    ap = argparse.ArgumentParser(
        description="Render test hits or a GMD groove through a kit from the music spec."
    )
    ap.add_argument("--kit", action="append", help="kit_id from rendering.kits (repeatable)")
    ap.add_argument("--all-kits", action="store_true", help="every kit in rendering.kits")
    what = ap.add_mutually_exclusive_group()
    what.add_argument("--note", type=int, help="GMD note, played through the kit's note_map")
    what.add_argument("--key", type=int, help="the kit's own key, bypassing note_map")
    what.add_argument("--groove", action="store_true", help="one GMD loop (picked by --seed)")
    ap.add_argument("--vel", type=int, nargs="+", default=[100], help="velocities to play")
    ap.add_argument("--repeat", type=int, default=4, help="hits per velocity (round robins)")
    ap.add_argument("--every", type=float, default=0.5, help="seconds between hits")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=kits.OUT_DIR)
    args = ap.parse_args(argv)

    spec = load_spec()
    chosen = [k.kit_id for k in spec.rendering.kits] if args.all_kits else (args.kit or [])
    if not chosen:
        ap.error("name a --kit or pass --all-kits")
    args.out.mkdir(parents=True, exist_ok=True)
    groove = kits.pick_groove(spec, args.seed) if args.groove else None
    if groove is not None:
        print(f"Groove: {groove.performance} bar {groove.start_bar + 1} at {groove.bpm:.0f} BPM")
    for kit_id in chosen:
        kit = kits.kit_by_id(spec, kit_id)
        try:
            program = kits.load_program(kit)
            sampler = Sampler(program, seed=args.seed)
        except SfzError as e:
            print(f"[{kit_id}] skipped: {e}")
            continue
        for problem in kits.check_kit(kit, program, spec):
            print(f"[{kit_id}] note map: {problem}")
        if groove is not None:
            audio = kits.render_groove(kit, sampler, groove, spec)
            name = f"{kit_id}_groove_seed{args.seed}.wav"
        else:
            times = [i * args.every for i in range(len(args.vel) * args.repeat)]
            vels = [v for v in args.vel for _ in range(args.repeat)]
            if args.key is not None:
                hits = [Hit(t + 0.1, args.key, v) for t, v in zip(times, vels, strict=True)]
                name = f"{kit_id}_key{args.key}.wav"
            else:
                note = 38 if args.note is None else args.note
                hits = kits.kit_hits(
                    kit, [(t + 0.1, note, v) for t, v in zip(times, vels, strict=True)]
                )
                name = f"{kit_id}_note{note}.wav"
            audio = kits.normalize(sampler.render(hits))
        kits.write_wav(args.out / name, audio, sampler.sample_rate)
        print(f"[{kit_id}] wrote {args.out / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
