"""Kits from the music spec, played through the sampler: the glue for listening tools.

    spec = load_spec()
    kit = kit_by_id(spec, "avl_black_pearl")
    program = load_program(kit)                 # its SFZ program under data/raw/kits
    print(check_kit(kit, program, spec))        # note_map entries the program can't play
    sampler = Sampler(program, seed=0)
    audio = normalize(sampler.render(kit_hits(kit, [(0.1, 38, 100)])))  # GMD notes in
    write_wav(OUT_DIR / "snare.wav", audio, sampler.sample_rate)

Loudness matching here is for listening: RMS to TARGET_RMS_DB with a peak ceiling. The
library's own normalization is B's to define.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import soundfile as sf

from src.music.spec import Kit, MusicSpec
from tools.listen import grid_ab
from tools.render.sampler import Hit, Sampler
from tools.render.sfz import Program, read_program

REPO = Path(__file__).resolve().parents[2]
KITS_DIR = REPO / "data" / "raw" / "kits"
OUT_DIR = REPO / "results" / "listening" / "render"
TARGET_RMS_DB = -20.0
PEAK_CEILING_DB = -1.0
PRE_ROLL_S = 0.25  # silence before a groove's first downbeat, so early hits aren't cut


def kit_by_id(spec: MusicSpec, kit_id: str) -> Kit:
    for kit in spec.rendering.kits:
        if kit.kit_id == kit_id:
            return kit
    known = ", ".join(k.kit_id for k in spec.rendering.kits)
    raise KeyError(f"no kit {kit_id!r} in rendering.kits (known: {known})")


def load_program(kit: Kit, kits_dir: Path = KITS_DIR) -> Program:
    path = kits_dir / kit.program
    if not path.is_file():
        raise FileNotFoundError(
            f"{kit.kit_id}: {path} is missing; fetch kits with "
            "`python -m tools.download_datasets kits`"
        )
    return read_program(path, kits_dir / kit.samples if kit.samples else None)


def check_kit(kit: Kit, program: Program, spec: MusicSpec) -> list[str]:
    """Mapped keys that play nothing in the program, or that are keyswitches."""
    groups = spec.note_to_group
    problems = []
    for note, played in sorted(kit.note_map.items()):
        if played.key is None:
            continue
        ccs = {cc: v / 127 for cc, v in {**kit.controls, **played.cc}.items()}
        where = f"GMD {note} ({groups[note]}) -> key {played.key}"
        sw = program.sw_keys
        if sw and sw[0] <= played.key <= sw[1]:
            problems.append(f"{where} is a keyswitch, not a sound")
        elif played.key not in program.keys(ccs):
            problems.append(f"{where}: the program plays nothing there")
    return problems


def sound_name(sample: str) -> str:
    """A readable name for a sample: its instrument folder (DrumGizmo kits keep each
    instrument in Samples/<name>/), or its file name without the round robin number,
    Salamander's mic and dynamic, and AVL's leading key number."""
    p = Path(sample)
    if f"-{p.parent.name}-" in p.stem:
        return p.parent.name
    stem = re.sub(r"[-_]\d+$", "", p.stem)
    stem = re.sub(r"_OH(_[A-Za-z]+)?$", "", stem)
    return re.sub(r"^\d+-", "", stem)


def describe(program: Program, key: int, values: dict[int, float]) -> tuple[str, str]:
    """(sound name, "N layers, M variations") of what a key plays at these CC values."""
    regions = [
        r
        for r in program.regions
        if r.lokey <= key <= r.hikey
        and not r.silent
        and program.switch_allows(r)
        and r.cc_allows(values)
    ]
    if not regions:
        return "", "no sound"
    name = Counter(sound_name(r.sample) for r in regions).most_common(1)[0][0]
    layers = sorted({(r.lovel, r.hivel) for r in regions})
    variations = max(
        len(
            {
                (r.seq_position if r.seq_length > 1 else 0, r.lorand, r.hirand)
                for r in regions
                if (r.lovel, r.hivel) == layer
            }
        )
        for layer in layers
    )
    return name, f"{_count(len(layers), 'layer')}, {_count(variations, 'variation')}"


def _count(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def kit_hits(kit: Kit, notes: Iterable[tuple[float, int, int]]) -> list[Hit]:
    """(time_s, GMD note, velocity) to the kit's keys; notes it doesn't play are skipped."""
    hits = []
    for time_s, note, velocity in notes:
        played = kit.play(note)
        if played.key is not None:
            hits.append(Hit(time_s, played.key, velocity, {**kit.controls, **played.cc}))
    return hits


def pick_groove(spec: MusicSpec, seed: int, gmd_dir: Path = grid_ab.GMD_DIR) -> grid_ab.Loop:
    """One GMD loop, chosen the same way as the grid A/B tool's first clip."""
    perfs = grid_ab.eligible_performances(spec, gmd_dir)
    rng = np.random.default_rng(seed)
    for idx in rng.permutation(len(perfs)):
        p = perfs.iloc[idx]
        notes, length = grid_ab.performance_notes(gmd_dir / p["midi_filename"])
        loops = grid_ab.cut_loops(notes, length, float(p["bpm"]), p["id"], spec)
        if loops:
            return loops[rng.integers(len(loops))]
    raise RuntimeError("no GMD performance has a usable loop")


def groove_notes(loop: grid_ab.Loop, spec: MusicSpec) -> list[tuple[float, int, int]]:
    """Every hit the drummer played in the loop as (time_s, GMD note, velocity)."""
    sec = 60.0 / loop.bpm
    return [
        (PRE_ROLL_S + beat * sec, pitch, velocity)
        for pitch, velocity, beat in grid_ab.original_hits(loop, spec)
    ]


def render_groove(kit: Kit, sampler: Sampler, loop: grid_ab.Loop, spec: MusicSpec) -> np.ndarray:
    return normalize(sampler.render(kit_hits(kit, groove_notes(loop, spec))))


def normalize(audio: np.ndarray) -> np.ndarray:
    """Scale to TARGET_RMS_DB, or less if that would push a peak past the ceiling."""
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if audio.size else 0.0
    if rms == 0:
        return audio
    gain = 10 ** (TARGET_RMS_DB / 20) / rms
    peak = float(np.abs(audio).max()) * gain
    ceiling = 10 ** (PEAK_CEILING_DB / 20)
    if peak > ceiling:
        gain *= ceiling / peak
    return (audio * gain).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int, subtype: str = "PCM_24") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate, subtype=subtype)
