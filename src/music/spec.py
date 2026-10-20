"""Load and validate config/music_spec.yaml, the single home of every musical constant.

    from src.music.spec import load_spec

    spec = load_spec()
    spec.voice_of(38)                   # 'snare'; None for dropped notes
    spec.same_rhythm.steps_per_beat     # 12
    spec.ghost_threshold("hats")        # 40
    spec.is_decided("same_rhythm.steps_per_beat")  # False: still a spec default

Loading fails with SpecError on unknown or missing keys, notes mapped twice, voices that
don't line up across sections, and out-of-range values, so a typo in the YAML can't
silently change the data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "music_spec.yaml"
DROP = "drop"
AVP_LABELS = ("kd", "sd", "hhc", "hho")  # the AVP dataset's four sound classes
AVP_SUBSETS = ("personal", "fixed")
FLAM_KEEP = ("first", "loudest")

Range = tuple[float, float]


class SpecError(ValueError):
    """config/music_spec.yaml is malformed or inconsistent."""


@dataclass(frozen=True)
class DrumGroup:
    notes: tuple[int, ...]
    group: str  # a voice name or DROP


@dataclass(frozen=True)
class SameRhythm:
    steps_per_beat: int
    ghost_velocity_below: dict[str, int]  # one entry per voice
    flam_merge_ms: float
    flam_keep: str | None  # "first" or "loudest"; None until the spec decides
    swing_is_different: bool
    half_time_double_time_different: bool
    min_hits_per_loop: dict[str, int]


@dataclass(frozen=True)
class Loops:
    bars: int
    time_signatures: tuple[str, ...]
    start_on_downbeat: bool
    exclude_fills: bool
    start_at_first_bar_with: tuple[str, ...]  # voices; skips count-ins
    drop_last_bars: int  # skips endings
    performance_tempo_bpm: Range
    hop_bars: int | None  # None until the spec decides
    max_loops_per_performance: int | None
    remove_duplicates_within_performance: bool


@dataclass(frozen=True)
class Rendering:
    tempo_bpm: Range
    max_tempo_change_pct: float | None  # None until the spec decides
    loudness_normalize: bool
    keep_performer_velocities: bool
    kits: tuple[dict, ...]  # entry format gets defined when kits are chosen


@dataclass(frozen=True)
class Beatbox:
    avp_labels: tuple[str, ...]
    simultaneous_priority: tuple[str, ...]
    voice_ghost_notes: bool
    thin_16th_hats_to_8ths_prob: float
    voice_cymbals_and_toms: bool
    open_hat_sound: str
    avp_subset: str


@dataclass(frozen=True)
class Sloppiness:
    offset_ms: Range
    jitter_ms: Range
    tempo_drift_pct: Range
    swing_drift_per_bar: Range | None  # None until the spec gives a number
    drop_prob: dict[str, Range]
    snr_db: Range
    mic_eq: dict | None  # None until the spec gives numbers


@dataclass(frozen=True)
class Taps:
    low: tuple[str, ...]
    high: tuple[str, ...]
    simultaneous: str  # "low" or "high"


@dataclass(frozen=True)
class MusicSpec:
    version: str
    voices: tuple[str, ...]
    drum_map: dict[str, DrumGroup]
    same_rhythm: SameRhythm
    loops: Loops
    rendering: Rendering
    beatbox: Beatbox
    sloppiness: Sloppiness
    taps: Taps
    decided: frozenset[str]
    units: tuple[str, ...]  # every decision-sized key, e.g. "same_rhythm.flam_merge_ms"
    path: Path

    @property
    def note_to_group(self) -> dict[int, str]:
        return {n: g.group for g in self.drum_map.values() for n in g.notes}

    def voice_of(self, note: int) -> str | None:
        """Voice for a MIDI note, or None if the spec drops it.

        Raises KeyError for notes the spec doesn't mention, so unexpected notes are
        never dropped silently.
        """
        group = self.note_to_group.get(note)
        if group is None:
            raise KeyError(f"MIDI note {note} is not in drum_map ({self.path})")
        return None if group == DROP else group

    def ghost_threshold(self, voice: str) -> int:
        return self.same_rhythm.ghost_velocity_below[voice]

    def is_decided(self, key: str) -> bool:
        return any(key == d or key.startswith(d + ".") for d in self.decided)

    def undecided(self) -> list[str]:
        """Decision-sized keys still at their spec default or still empty (null)."""
        return [u for u in self.units if not self.is_decided(u)]


# --------------------------------------------------------------------------- parsing


def _keys(d: Any, expected: list[str], where: str) -> dict:
    if not isinstance(d, dict):
        raise SpecError(f"{where}: expected a mapping, got {type(d).__name__}")
    unknown = sorted(set(d) - set(expected))
    missing = sorted(set(expected) - set(d))
    if unknown:
        raise SpecError(f"{where}: unknown key(s) {unknown}")
    if missing:
        raise SpecError(f"{where}: missing key(s) {missing}")
    return d


def _bool(v: Any, where: str) -> bool:
    if not isinstance(v, bool):
        raise SpecError(f"{where}: expected true or false, got {v!r}")
    return v


def _int(v: Any, where: str, lo: int | None = None, hi: int | None = None) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise SpecError(f"{where}: expected an integer, got {v!r}")
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise SpecError(f"{where}: {v} is outside [{lo}, {hi}]")
    return v


def _num(v: Any, where: str, lo: float | None = None, hi: float | None = None) -> float:
    if isinstance(v, bool) or not isinstance(v, int | float):
        raise SpecError(f"{where}: expected a number, got {v!r}")
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise SpecError(f"{where}: {v} is outside [{lo}, {hi}]")
    return float(v)


def _range(v: Any, where: str, lo: float | None = None, hi: float | None = None) -> Range:
    if not isinstance(v, list | tuple) or len(v) != 2:
        raise SpecError(f"{where}: expected [low, high], got {v!r}")
    a, b = (_num(x, where, lo, hi) for x in v)
    if a > b:
        raise SpecError(f"{where}: low {a} is above high {b}")
    return (a, b)


def _choice(v: Any, options: tuple[str, ...], where: str) -> str:
    if v not in options:
        raise SpecError(f"{where}: {v!r} is not one of {list(options)}")
    return v


def _str_list(v: Any, where: str, options: tuple[str, ...] | None = None) -> tuple[str, ...]:
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise SpecError(f"{where}: expected a list of names, got {v!r}")
    if len(set(v)) != len(v):
        raise SpecError(f"{where}: repeated names in {v}")
    if options is not None:
        for x in v:
            _choice(x, options, where)
    return tuple(v)


def _per_voice(v: Any, voices: tuple[str, ...], where: str, parse) -> dict:
    d = _keys(v, list(voices), where)
    return {voice: parse(d[voice], f"{where}.{voice}") for voice in voices}


def _drum_map(raw: Any, voices: tuple[str, ...]) -> dict[str, DrumGroup]:
    if not isinstance(raw, dict) or not raw:
        raise SpecError("drum_map: expected a non-empty mapping")
    seen: dict[int, str] = {}
    out = {}
    for name, entry in raw.items():
        where = f"drum_map.{name}"
        e = _keys(entry, ["notes", "group"], where)
        if not isinstance(e["notes"], list) or not e["notes"]:
            raise SpecError(f"{where}.notes: expected a non-empty list of MIDI notes")
        notes = tuple(_int(n, f"{where}.notes", 0, 127) for n in e["notes"])
        for n in notes:
            if n in seen:
                raise SpecError(f"{where}: note {n} is already mapped by drum_map.{seen[n]}")
            seen[n] = name
        out[name] = DrumGroup(notes, _choice(e["group"], (*voices, DROP), f"{where}.group"))
    unused = [v for v in voices if not any(g.group == v for g in out.values())]
    if unused:
        raise SpecError(f"drum_map: no notes map to voice(s) {unused}")
    return out


def _exists(raw: dict, dotted: str) -> bool:
    node: Any = raw
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


SECTIONS = ["same_rhythm", "loops", "rendering", "beatbox", "sloppiness", "taps"]
TOP = ["version", "decided", "voices", "drum_map", *SECTIONS]


def parse_spec(raw: Any, path: Path = DEFAULT_PATH) -> MusicSpec:
    """Validate an already-loaded YAML mapping. load_spec() is the usual entry point."""
    raw = _keys(raw, TOP, "music_spec")
    voices = _str_list(raw["voices"], "voices")
    if not voices or DROP in voices:
        raise SpecError(f"voices: need at least one voice, and none named {DROP!r}")
    drum_map = _drum_map(raw["drum_map"], voices)

    s = _keys(
        raw["same_rhythm"],
        [f.name for f in SameRhythm.__dataclass_fields__.values()],
        "same_rhythm",
    )
    ghost = s["ghost_velocity_below"]
    if not isinstance(ghost, dict):
        ghost = {v: ghost for v in voices}
    min_hits = s["min_hits_per_loop"]
    if not isinstance(min_hits, dict):
        raise SpecError("same_rhythm.min_hits_per_loop: expected a mapping of voice to count")
    same_rhythm = SameRhythm(
        steps_per_beat=_int(s["steps_per_beat"], "same_rhythm.steps_per_beat", 1),
        ghost_velocity_below=_per_voice(
            ghost, voices, "same_rhythm.ghost_velocity_below", lambda x, w: _int(x, w, 0, 128)
        ),
        flam_merge_ms=_num(s["flam_merge_ms"], "same_rhythm.flam_merge_ms", 0),
        flam_keep=None
        if s["flam_keep"] is None
        else _choice(s["flam_keep"], FLAM_KEEP, "same_rhythm.flam_keep"),
        swing_is_different=_bool(s["swing_is_different"], "same_rhythm.swing_is_different"),
        half_time_double_time_different=_bool(
            s["half_time_double_time_different"], "same_rhythm.half_time_double_time_different"
        ),
        min_hits_per_loop={
            _choice(k, voices, "same_rhythm.min_hits_per_loop"): _int(
                n, f"same_rhythm.min_hits_per_loop.{k}", 0
            )
            for k, n in min_hits.items()
        },
    )

    lp = _keys(raw["loops"], [f.name for f in Loops.__dataclass_fields__.values()], "loops")
    sigs = _str_list(lp["time_signatures"], "loops.time_signatures")
    bad = [t for t in sigs if not re.fullmatch(r"\d+-\d+", t)]
    if not sigs or bad:
        raise SpecError(f"loops.time_signatures: use GMD's '4-4' format, got {list(sigs)}")
    max_loops = lp["max_loops_per_performance"]
    hop = lp["hop_bars"]
    loops = Loops(
        bars=_int(lp["bars"], "loops.bars", 1),
        time_signatures=sigs,
        start_on_downbeat=_bool(lp["start_on_downbeat"], "loops.start_on_downbeat"),
        exclude_fills=_bool(lp["exclude_fills"], "loops.exclude_fills"),
        start_at_first_bar_with=_str_list(
            lp["start_at_first_bar_with"], "loops.start_at_first_bar_with", voices
        ),
        drop_last_bars=_int(lp["drop_last_bars"], "loops.drop_last_bars", 0),
        performance_tempo_bpm=_range(
            lp["performance_tempo_bpm"], "loops.performance_tempo_bpm", 20, 400
        ),
        hop_bars=None if hop is None else _int(hop, "loops.hop_bars", 1),
        max_loops_per_performance=None
        if max_loops is None
        else _int(max_loops, "loops.max_loops_per_performance", 1),
        remove_duplicates_within_performance=_bool(
            lp["remove_duplicates_within_performance"],
            "loops.remove_duplicates_within_performance",
        ),
    )

    r = _keys(
        raw["rendering"], [f.name for f in Rendering.__dataclass_fields__.values()], "rendering"
    )
    if not isinstance(r["kits"], list) or not all(isinstance(k, dict) for k in r["kits"]):
        raise SpecError("rendering.kits: expected a list of kit entries")
    rendering = Rendering(
        tempo_bpm=_range(r["tempo_bpm"], "rendering.tempo_bpm", 20, 400),
        max_tempo_change_pct=None
        if r["max_tempo_change_pct"] is None
        else _num(r["max_tempo_change_pct"], "rendering.max_tempo_change_pct", 0, 100),
        loudness_normalize=_bool(r["loudness_normalize"], "rendering.loudness_normalize"),
        keep_performer_velocities=_bool(
            r["keep_performer_velocities"], "rendering.keep_performer_velocities"
        ),
        kits=tuple(r["kits"]),
    )

    b = _keys(raw["beatbox"], [f.name for f in Beatbox.__dataclass_fields__.values()], "beatbox")
    avp_labels = _str_list(b["avp_labels"], "beatbox.avp_labels", AVP_LABELS)
    priority = _str_list(b["simultaneous_priority"], "beatbox.simultaneous_priority", voices)
    if set(priority) != set(voices):
        raise SpecError(f"beatbox.simultaneous_priority: must list every voice {list(voices)}")
    beatbox = Beatbox(
        avp_labels=avp_labels,
        simultaneous_priority=priority,
        voice_ghost_notes=_bool(b["voice_ghost_notes"], "beatbox.voice_ghost_notes"),
        thin_16th_hats_to_8ths_prob=_num(
            b["thin_16th_hats_to_8ths_prob"], "beatbox.thin_16th_hats_to_8ths_prob", 0, 1
        ),
        voice_cymbals_and_toms=_bool(b["voice_cymbals_and_toms"], "beatbox.voice_cymbals_and_toms"),
        open_hat_sound=_choice(b["open_hat_sound"], avp_labels, "beatbox.open_hat_sound"),
        avp_subset=_choice(b["avp_subset"], AVP_SUBSETS, "beatbox.avp_subset"),
    )

    sl = _keys(
        raw["sloppiness"], [f.name for f in Sloppiness.__dataclass_fields__.values()], "sloppiness"
    )
    if sl["mic_eq"] is not None and not isinstance(sl["mic_eq"], dict):
        raise SpecError("sloppiness.mic_eq: expected null or a mapping of EQ settings")
    sloppiness = Sloppiness(
        offset_ms=_range(sl["offset_ms"], "sloppiness.offset_ms"),
        jitter_ms=_range(sl["jitter_ms"], "sloppiness.jitter_ms", 0),
        tempo_drift_pct=_range(sl["tempo_drift_pct"], "sloppiness.tempo_drift_pct", 0, 100),
        swing_drift_per_bar=None
        if sl["swing_drift_per_bar"] is None
        else _range(sl["swing_drift_per_bar"], "sloppiness.swing_drift_per_bar", 0),
        drop_prob=_per_voice(
            sl["drop_prob"], voices, "sloppiness.drop_prob", lambda x, w: _range(x, w, 0, 1)
        ),
        snr_db=_range(sl["snr_db"], "sloppiness.snr_db"),
        mic_eq=sl["mic_eq"],
    )

    t = _keys(raw["taps"], ["low", "high", "simultaneous"], "taps")
    low = _str_list(t["low"], "taps.low", voices)
    high = _str_list(t["high"], "taps.high", voices)
    if set(low) & set(high) or set(low) | set(high) != set(voices):
        raise SpecError(f"taps: low and high must split the voices {list(voices)} between them")
    taps = Taps(low, high, _choice(t["simultaneous"], ("low", "high"), "taps.simultaneous"))

    decided = _str_list(raw["decided"] or [], "decided")
    missing = [d for d in decided if not _exists(raw, d)]
    if missing:
        raise SpecError(f"decided: {missing} do not name keys in the spec")
    units = [f"drum_map.{k}" for k in raw["drum_map"]]
    units += [f"{sec}.{k}" for sec in SECTIONS for k in raw[sec]]

    return MusicSpec(
        version=str(raw["version"]),
        voices=voices,
        drum_map=drum_map,
        same_rhythm=same_rhythm,
        loops=loops,
        rendering=rendering,
        beatbox=beatbox,
        sloppiness=sloppiness,
        taps=taps,
        decided=frozenset(decided),
        units=tuple(units),
        path=path,
    )


def load_spec(path: str | Path | None = None) -> MusicSpec:
    """Read and validate the spec (default: config/music_spec.yaml)."""
    p = Path(path) if path is not None else DEFAULT_PATH
    with p.open() as f:
        raw = yaml.safe_load(f)
    return parse_spec(raw, p)
