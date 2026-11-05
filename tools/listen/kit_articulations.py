"""Audition the kits' note maps: every GMD note through every kit, with alternatives.

    .venv/bin/python -m tools.listen.kit_articulations
    open results/listening/kit_articulations/index.html

The page has one row per drum_map note and one column per kit in rendering.kits. Each
cell plays the kit's draft sound for that note first, then the alternatives listed for
it in config/kit_auditions.yaml, each as one hit at every --vel velocity. Pick the sound
that fits, or type any other key from the page's "All sounds" list. Picks are kept in
the browser and export as a CSV for updating the note maps.

Each clip is one hit with PRE_ROLL_S of silence before it, cut at --max-seconds with a
short fade. Within a kit nothing is rescaled, so velocities and articulations keep the
kit's own balance. Across kits, one gain per kit puts its draft snare at the top
velocity at REF_RMS_DB (over REF_WINDOW_S from the hit), or lower if that would push any
of the kit's clips past the peak ceiling. Clips are 16-bit WAV at the kit's own rate.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import yaml

from src.music.spec import DROP, Kit, KitNote, MusicSpec, load_spec
from tools.inventory.gmd import GMD_NOTES
from tools.render import kits
from tools.render.sampler import Hit, Sampler
from tools.render.sfz import Program, SfzError

REPO = Path(__file__).resolve().parents[2]
AUDITIONS_PATH = REPO / "config" / "kit_auditions.yaml"
OUT_DIR = REPO / "results" / "listening" / "kit_articulations"
VELOCITIES = (30, 80, 120)
MAX_SECONDS = 4.0
PRE_ROLL_S = 0.05
FADE_S = 0.05
REF_RMS_DB = -18.0
REF_WINDOW_S = 0.2
SILENT = "silent"
STORE_KEY = "kit_articulations.v1"  # localStorage key the page keeps picks under
PAGE_TEMPLATE = Path(__file__).with_suffix(".html")  # the page; build() supplies its data

Alternatives = dict[str, dict[int, tuple[KitNote, ...]]]


# --------------------------------------------------------------------------- values


def note_value(played: KitNote) -> str:
    """A note_map value as page and CSV text: "38", "42 cc4=0", or "silent"."""
    if played.key is None:
        return SILENT
    return " ".join([str(played.key), *(f"cc{cc}={v}" for cc, v in sorted(played.cc.items()))])


def parse_value(text: str) -> KitNote:
    """Inverse of note_value."""
    words = text.split()
    if words == [SILENT]:
        return KitNote(None, {})
    if not words or not words[0].isdigit() or not 0 <= int(words[0]) <= 127:
        raise ValueError(f"expected a key 0-127, optionally with ccN=V, or {SILENT!r}: {text!r}")
    cc = {}
    for word in words[1:]:
        m = re.fullmatch(r"cc(\d+)=(\d+)", word)
        if not m or int(m[1]) > 127 or int(m[2]) > 127:
            raise ValueError(f"expected ccN=V with N and V in 0-127, got {word!r} in {text!r}")
        cc[int(m[1])] = int(m[2])
    return KitNote(int(words[0]), cc)


def _kit_note(value: object, where: str) -> KitNote:
    if isinstance(value, int) and not isinstance(value, bool):
        return parse_value(str(value))
    if isinstance(value, dict) and set(value) == {"key", "cc"} and isinstance(value["cc"], dict):
        cc = " ".join(f"cc{c}={v}" for c, v in value["cc"].items())
        try:
            return parse_value(f"{value['key']} {cc}")
        except ValueError as e:
            raise ValueError(f"{where}: {e}") from None
    raise ValueError(f"{where}: expected a key or {{key: K, cc: {{CC: value}}}}, got {value!r}")


def load_auditions(path: Path, spec: MusicSpec) -> Alternatives:
    """config/kit_auditions.yaml, checked against the spec's kits and drum_map notes."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping of kit_id to alternatives")
    known = {k.kit_id for k in spec.rendering.kits}
    out: Alternatives = {}
    for kit_id, notes in raw.items():
        if kit_id not in known:
            raise ValueError(f"{path}: unknown kit {kit_id!r} (rendering.kits has {sorted(known)})")
        if not isinstance(notes, dict):
            raise ValueError(f"{path}: {kit_id}: expected a mapping of GMD note to a list")
        out[kit_id] = {}
        for note, values in notes.items():
            where = f"{path.name}: {kit_id}.{note}"
            if note not in spec.note_to_group:
                raise ValueError(f"{where}: {note!r} is not a drum_map note")
            if not isinstance(values, list):
                raise ValueError(f"{where}: expected a list of kit sounds")
            out[kit_id][note] = tuple(_kit_note(v, where) for v in values)
    return out


def cell_options(kit: Kit, note: int, alternatives: Alternatives) -> list[KitNote]:
    """The sounds a cell plays: the draft first, then its alternatives, without repeats."""
    out: list[KitNote] = []
    for played in (kit.play(note), *alternatives.get(kit.kit_id, {}).get(note, ())):
        if played.key is not None and played not in out:
            out.append(played)
    return out


# --------------------------------------------------------------------------- audio


def render_clip(
    sampler: Sampler, kit: Kit, played: KitNote, velocity: int, max_seconds: float
) -> np.ndarray:
    """One hit at PRE_ROLL_S, cut at max_seconds with a FADE_S fade; empty if it's silent."""
    sr = sampler.sample_rate
    audio = sampler.render([Hit(PRE_ROLL_S, played.key, velocity, {**kit.controls, **played.cc})])
    if not np.any(audio):
        return np.zeros((0, 2), dtype=np.float32)
    n = round(max_seconds * sr)
    if len(audio) > n:
        audio = audio[:n].copy()
        fade = min(n, round(FADE_S * sr))
        audio[n - fade :] *= np.linspace(1, 0, fade, dtype=np.float32)[:, None]
    return audio


def kit_gain(clips: list[np.ndarray], reference: np.ndarray | None, sample_rate: int) -> float:
    """One gain for a kit's clips: the reference hit at REF_RMS_DB, under the peak ceiling."""
    peak = max((float(np.abs(c).max()) for c in clips if c.size), default=0.0)
    if peak == 0:
        return 1.0
    gain = 10 ** (kits.PEAK_CEILING_DB / 20) / peak
    if reference is not None and reference.size:
        start = round(PRE_ROLL_S * sample_rate)
        window = reference[start : start + round(REF_WINDOW_S * sample_rate)]
        rms = float(np.sqrt(np.mean(np.square(window, dtype=np.float64)))) if window.size else 0
        if rms > 0:
            gain = min(gain, 10 ** (REF_RMS_DB / 20) / rms)
    return gain


def _slug(played: KitNote) -> str:
    return "k" + note_value(played).replace(" ", "_").replace("=", "-")


# --------------------------------------------------------------------------- page data


@dataclass
class _KitAudio:
    program: Program | None
    sampler: Sampler | None
    reason: str  # why the kit can't be played, if it can't


def _open_kit(kit: Kit, kits_dir: Path, seed: int) -> _KitAudio:
    try:
        program = kits.load_program(kit, kits_dir)
    except FileNotFoundError:
        return _KitAudio(None, None, "not downloaded")
    except SfzError as e:
        return _KitAudio(None, None, str(e))
    try:
        return _KitAudio(program, Sampler(program, seed=seed), "")
    except SfzError as e:
        unsupported = ", ".join(sorted({o.split("=")[0] for o in program.unsupported})[:6])
        return _KitAudio(program, None, f"uses {unsupported}" if unsupported else str(e))


def _rows(spec: MusicSpec) -> tuple[list[dict], list[dict]]:
    groups, rows = [], []
    for name, group in spec.drum_map.items():
        groups.append({"name": name, "voice": group.group, "notes": list(group.notes)})
        for note in group.notes:
            rows.append(
                {
                    "note": note,
                    "name": GMD_NOTES[note][0] if note in GMD_NOTES else str(note),
                    "group": name,
                    "voice": group.group,
                    "dropped": group.group == DROP,
                }
            )
    return groups, rows


def build(
    spec: MusicSpec,
    kit_list: list[Kit],
    alternatives: Alternatives,
    out: Path,
    velocities: tuple[int, ...] = VELOCITIES,
    max_seconds: float = MAX_SECONDS,
    kits_dir: Path = kits.KITS_DIR,
    seed: int = 0,
) -> dict:
    """Render every clip the page plays into out/audio and return the page's data."""
    groups, rows = _rows(spec)
    for stale in out.glob("audio/*/k*_v*.wav"):
        stale.unlink()
    snare = spec.drum_map["snare_head"].notes[0] if "snare_head" in spec.drum_map else None
    data: dict = {
        "generated": date.today().isoformat(),
        "store_key": STORE_KEY,
        "velocities": list(velocities),
        "max_seconds": max_seconds,
        "groups": groups,
        "rows": rows,
        "kits": [],
        "cells": {},
        "browser": {},
        "warnings": [],
    }
    for kit in kit_list:
        ka = _open_kit(kit, kits_dir, seed)
        drafts = {r["note"]: note_value(kit.play(r["note"])) for r in rows}
        sounds: dict[str, KitNote] = {}
        for r in rows:
            for played in cell_options(kit, r["note"], alternatives):
                sounds.setdefault(note_value(played), played)
        controls = {cc: v / 127 for cc, v in kit.controls.items()}
        if ka.program is not None:
            for key in sorted(ka.program.keys(controls)):
                sounds.setdefault(str(key), KitNote(key, {}))

        info: dict[str, tuple[str, str]] = {}
        clips: dict[str, list[str | None]] = {v: [None] * len(velocities) for v in sounds}
        entry = {"id": kit.kit_id, "type": kit.type, "renderable": ka.sampler is not None}
        entry["reason"] = ka.reason
        if ka.program is not None:
            for value, played in sounds.items():
                values = ka.program.cc_values(
                    {**controls, **{c: v / 127 for c, v in played.cc.items()}}
                )
                info[value] = kits.describe(ka.program, played.key, values)
                if info[value][1] == "no sound":
                    data["warnings"].append(f"{kit.kit_id}: {value} plays nothing")
        if ka.sampler is not None:
            audio = {
                (value, i): render_clip(ka.sampler, kit, played, vel, max_seconds)
                for value, played in sounds.items()
                for i, vel in enumerate(velocities)
            }
            reference = audio.get((drafts.get(snare, ""), len(velocities) - 1))
            sr = ka.sampler.sample_rate
            gain = kit_gain(list(audio.values()), reference, sr)
            for (value, i), clip in audio.items():
                if clip.size:
                    rel = f"audio/{kit.kit_id}/{_slug(sounds[value])}_v{velocities[i]}.wav"
                    kits.write_wav(out / rel, clip * gain, sr, subtype="PCM_16")
                    clips[value][i] = rel
            entry["gain_db"] = round(20 * math.log10(gain), 1)
            entry["sample_rate"] = sr
        data["kits"].append(entry)

        def option(value: str, info=info, clips=clips, drafts=drafts) -> dict:
            label, details = info.get(value, ("", ""))
            return {
                "value": value,
                "label": label,
                "info": details,
                "clips": clips[value],
                "used_by": [note for note, draft in drafts.items() if draft == value],
            }

        for r in rows:
            note = r["note"]
            same = [
                n["note"] for n in rows if n["note"] != note and drafts[n["note"]] == drafts[note]
            ]
            data["cells"][f"{kit.kit_id}|{note}"] = {
                "draft": drafts[note],
                "same_as": same if drafts[note] != SILENT else [],
                "options": [option(note_value(p)) for p in cell_options(kit, note, alternatives)],
            }
        data["browser"][kit.kit_id] = [
            option(v) for v in sorted(sounds, key=lambda v: (sounds[v].key, v)) if not sounds[v].cc
        ]
    return data


def write_page(data: dict, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    page = out / "index.html"
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    page.write_text(PAGE_TEMPLATE.read_text().replace("/*DATA*/{}", payload))
    return page


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--kit", action="append", help="kit_id from rendering.kits (default: all)")
    ap.add_argument("--vel", type=int, nargs="+", default=list(VELOCITIES))
    ap.add_argument("--max-seconds", type=float, default=MAX_SECONDS, help="clip length cap")
    ap.add_argument("--auditions", type=Path, default=AUDITIONS_PATH)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)

    spec = load_spec()
    chosen = [kits.kit_by_id(spec, k) for k in args.kit] if args.kit else list(spec.rendering.kits)
    alternatives = load_auditions(args.auditions, spec)
    data = build(
        spec, chosen, alternatives, args.out, tuple(args.vel), args.max_seconds, seed=args.seed
    )
    page = write_page(data, args.out)
    for k in data["kits"]:
        if k["renderable"]:
            print(f"[{k['id']}] {k['sample_rate']} Hz, level {k['gain_db']:+.1f} dB")
        else:
            print(f"[{k['id']}] not playable: {k['reason']}")
    for w in data["warnings"]:
        print(f"warning: {w}")
    clips = sum(1 for _ in args.out.glob("audio/*/*.wav"))
    print(f"Wrote {clips} clips and {page}")
    print(f"Open it with: open {page}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
