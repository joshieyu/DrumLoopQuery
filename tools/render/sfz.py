"""Read SFZ drum programs: the part of the format the project's open kits use.

    program = read_program(Path("data/raw/kits/avl_black_pearl/.../Black_Pearl_2023_repack.sfz"))
    program.regions       # every note-triggered region, with inherited opcodes resolved
    program.cc_defaults   # CC values from <control> set_ccN / set_hdccN, normalized to 0..1
    program.unsupported   # opcodes that would change the sound but aren't implemented

Covered: #define and #include (several on one line), <control> / <global> / <master> /
<group> / <region> inheritance, <curve>, default_path, keyswitches (sw_*), and the
opcodes the Region fields hold. IGNORED opcodes don't change a drum one-shot rendered
offline (labels, GUI hints, MIDI channels). Sound-changing opcodes this reader doesn't
implement (pitch, filters, delays, LFOs, loops...) are counted in `program.unsupported`
when set to anything but a neutral value, and the sampler refuses such a program rather
than rendering it wrong. Any other name is unknown (usually a typo in the kit, such as
"offby"): players ignore those, so we do too, and count them in `program.unknown`.

Semantics follow sfzformat.com. Where the reference is silent, the choice is noted next
to the code; "ARIA" marks behaviour of Plogue's ARIA player, which these kits target.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

HEADERS = ("global", "master", "group", "region")  # inheritance, outermost first
SILENCE = "*silence"
_NOTE = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
_MAX_INCLUDE_DEPTH = 32

# Opcodes the Region fields implement.
IMPLEMENTED = re.compile(
    r"sample|key|lokey|hikey|lovel|hivel|(lo|hi)cc\d+|sw_last|sw_lolast|sw_hilast|"
    r"sw_lokey|sw_hikey|sw_default|seq_length|seq_position|lorand|hirand|trigger|"
    r"volume|(group|master|global)_volume|amplitude|pan|phase|amp_veltrack|amp_velcurve_\d+|"
    r"offset|end|ampeg_(attack|hold|decay|sustain|release)|group|off_by|off_mode|off_time|"
    r"polyphony|note_polyphony|lochan|hichan|loop_mode|"
    r"(amplitude|volume|gain|pan)_(on)?cc\d+|(amplitude|volume|gain|pan)_curvecc\d+"
)
# Opcodes that don't change an offline one-shot render.
IGNORED = re.compile(
    r"(region|group|master|global|sw|key)_label|label_cc\d+|off_shape|pitch_keycenter|"
    r"note_selfmask|rt_decay|output|image|hint_\w+|sw_vel|count|fil_type"
)
# Opcodes we don't implement that are harmless at these values.
NEUTRAL = {"tune": "0", "transpose": "0", "delay": "0", "width": "100", "position": "0"}
LOOP_MODES = ("no_loop", "one_shot")  # both play to the sample's end: we send no note-offs


class SfzError(ValueError):
    """The program can't be read, or uses something this reader doesn't implement."""


@dataclass(frozen=True)
class CCMod:
    """One CC modulation of a region parameter, e.g. amplitude_oncc7=100."""

    cc: int
    depth: float
    curve: int  # curve index; 0 (linear 0..1) unless *_curveccN says otherwise


@dataclass(frozen=True)
class Region:
    sample: str  # absolute path, or SILENCE
    lokey: int
    hikey: int
    lovel: int
    hivel: int
    cc_ranges: tuple[tuple[int, int, int], ...]  # (cc, lo, hi), inclusive, on 0..127
    sw_last: tuple[int, int] | None  # keyswitch range the region needs, inclusive
    seq_length: int
    seq_position: int
    lorand: float
    hirand: float
    volume_db: float  # volume + group_volume + master_volume + global_volume
    amplitude: float  # percent
    amplitude_cc: tuple[CCMod, ...]
    volume_cc: tuple[CCMod, ...]  # dB
    pan: float  # -100..100
    pan_cc: tuple[CCMod, ...]
    amp_veltrack: float  # percent
    amp_velcurve: tuple[tuple[int, float], ...]  # (velocity, gain 0..1), sorted
    phase_invert: bool
    offset: int  # first sample frame played
    end: int | None  # last frame played, inclusive; -1 means the region is silent
    ampeg_attack: float  # seconds
    ampeg_hold: float
    ampeg_decay: float
    ampeg_sustain: float  # percent
    ampeg_release: float
    group: int
    off_by: int | None
    off_mode: str  # "fast", "normal", or "time"
    off_time: float | None
    polyphony: int | None
    note_polyphony: int | None
    label: str  # for diagnostics

    @property
    def silent(self) -> bool:
        """Plays nothing, though it still takes part in choke groups."""
        return (
            self.sample == SILENCE
            or self.end == -1
            or (self.ampeg_sustain == 0 and self.ampeg_decay == 0)
        )

    def cc_allows(self, values: dict[int, float]) -> bool:
        """The region's CC conditions hold at these values (0..1; CCs not listed are 0)."""
        return all(lo <= values.get(cc, 0.0) * 127 <= hi for cc, lo, hi in self.cc_ranges)

    def matches(self, key: int, velocity: int, values: dict[int, float]) -> bool:
        """Key, velocity, and CC conditions (not keyswitch, round robin, or random)."""
        return (
            self.lokey <= key <= self.hikey
            and self.lovel <= velocity <= self.hivel
            and self.cc_allows(values)
        )


@dataclass(frozen=True)
class Curve:
    points: tuple[tuple[int, float], ...]  # (0..127, value), sorted

    def __call__(self, x: float) -> float:
        return _interp(x * 127, [p for p, _ in self.points], [v for _, v in self.points])


def builtin_curve(index: int, x: float) -> float:
    """Built-in curves 0-4 of sfzformat.com/headers/curve, for x in 0..1.

    The reference names curve 4 "Concave" but gives no formula; x squared is used. In
    these kits it only shapes CC7 (master volume), which scales every region the same.
    """
    if index == 0:
        return x
    if index == 1:
        return 2 * x - 1
    if index == 2:
        return 1 - x
    if index == 3:
        return 1 - 2 * x
    if index == 4:
        return x * x
    raise SfzError(f"built-in curve {index} is not implemented")


@dataclass(frozen=True)
class Program:
    path: Path
    regions: tuple[Region, ...]
    cc_defaults: dict[int, float]  # 0..1; CCs not listed start at 0
    sw_default: int | None  # the keyswitch in effect, since we never press one
    sw_keys: tuple[int, int] | None  # sw_lokey..sw_hikey: notes that switch, not sound
    curves: dict[int, Curve]
    unsupported: dict[str, int]  # opcode -> regions using it
    unknown: dict[str, int]

    def curve(self, index: int, x: float) -> float:
        if index in self.curves:
            return self.curves[index](x)
        return builtin_curve(index, x)

    def switch_allows(self, r: Region) -> bool:
        if r.sw_last is None:
            return True
        return self.sw_default is not None and r.sw_last[0] <= self.sw_default <= r.sw_last[1]

    def cc_values(self, overrides: dict[int, float] | None = None) -> dict[int, float]:
        return {**self.cc_defaults, **(overrides or {})}

    def candidates(self, key: int, velocity: int, values: dict[int, float]) -> list[int]:
        """Indexes of regions a note-on could play, before round robin and random."""
        return [
            i
            for i, r in enumerate(self.regions)
            if r.matches(key, velocity, values) and self.switch_allows(r)
        ]

    def keys(self, overrides: dict[int, float] | None = None) -> set[int]:
        """Notes with at least one audible region at these CC values, any velocity."""
        values = self.cc_values(overrides)
        out: set[int] = set()
        for r in self.regions:
            if not r.silent and self.switch_allows(r) and r.cc_allows(values):
                out.update(range(r.lokey, r.hikey + 1))
        if self.sw_keys:
            out -= set(range(self.sw_keys[0], self.sw_keys[1] + 1))
        return out

    def level(self, r: Region, velocity: int, values: dict[int, float]) -> tuple[float, float]:
        """(linear gain, pan -100..100) of one voice of region r, before its envelope.

        CC amplitude modulations multiply: the reference's amplitude_onccN "volume
        control which goes from silence to full volume" means amplitude_oncc7=100 at CC
        0 is silent, and several of them on one region multiply together (ARIA). Volume
        modulations add in dB, pan modulations add in pan units.
        """
        amp = r.amplitude / 100
        for m in r.amplitude_cc:
            amp *= m.depth / 100 * self.curve(m.curve, values.get(m.cc, 0.0))
        vol_db = r.volume_db + sum(
            m.depth * self.curve(m.curve, values.get(m.cc, 0.0)) for m in r.volume_cc
        )
        pan = r.pan + sum(m.depth * self.curve(m.curve, values.get(m.cc, 0.0)) for m in r.pan_cc)
        gain = velocity_gain(r, velocity) * amp * 10 ** (vol_db / 20)
        return (-gain if r.phase_invert else gain), min(max(pan, -100.0), 100.0)


def note_number(text: str) -> int:
    """MIDI note from "38" or a note name like "c#4" (c4 = 60)."""
    t = text.strip().lower()
    if re.fullmatch(r"-?\d+", t):
        return int(t)
    m = re.fullmatch(r"([a-g])(#|b)?(-?\d+)", t)
    if not m:
        raise SfzError(f"not a MIDI note: {text!r}")
    return _NOTE[m[1]] + {"#": 1, "b": -1, None: 0}[m[2]] + (int(m[3]) + 1) * 12


def velocity_gain(r: Region, velocity: int) -> float:
    """Linear gain from velocity: amp_velcurve points, else (vel/127)^2, then amp_veltrack.

    With amp_velcurve points, unlisted velocities interpolate linearly between (0, 0),
    the points, and (127, 1). amp_veltrack below 100 blends toward a flat 1 (sfizz/ARIA
    behaviour; the reference only defines the 100% case).
    """
    v = min(max(velocity, 0), 127)
    if r.amp_velcurve:
        pts = {0: 0.0, 127: 1.0, **dict(r.amp_velcurve)}
        xs = sorted(pts)
        curve = _interp(v, xs, [pts[x] for x in xs])
    else:
        curve = (v / 127) ** 2
    t = r.amp_veltrack / 100
    if t < 0:
        raise SfzError("negative amp_veltrack is not implemented")
    return (1 - t) + t * curve


def _interp(x: float, xs: list[int], ys: list[float]) -> float:
    if x <= xs[0]:
        return ys[0]
    for x0, y0, x1, y1 in zip(xs, ys, xs[1:], ys[1:], strict=False):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return ys[-1]


# --------------------------------------------------------------------------- reading


_DIRECTIVE = re.compile(r'#define\s+(\$\w+)\s+(\S+)|#include\s+"([^"]+)"')
_TOKEN = re.compile(r"<(\w+)>|([A-Za-z_$][A-Za-z0-9_$]*)=")


def _substitute(text: str, defines: dict[str, str]) -> str:
    for name in sorted(defines, key=len, reverse=True):
        text = text.replace(name, defines[name])
    return text


def _segments(path: Path, root: Path, defines: dict[str, str], depth: int = 0) -> Iterator[str]:
    """Program text with #define applied and #include expanded, a piece at a time.

    Include paths are relative to the top program's folder, as in sfizz and ARIA.
    """
    if depth > _MAX_INCLUDE_DEPTH:
        raise SfzError(f"#include nested deeper than {_MAX_INCLUDE_DEPTH} at {path}")
    text = re.sub(r"/\*.*?\*/", " ", path.read_text(errors="replace"), flags=re.S)
    for raw in text.splitlines():
        line = raw.split("//", 1)[0]
        pos = 0
        for m in _DIRECTIVE.finditer(line):
            yield _substitute(line[pos : m.start()], defines)
            pos = m.end()
            if m[1]:
                defines[m[1]] = m[2]
            else:
                include = root / _substitute(m[3], defines).replace("\\", "/")
                if not include.is_file():
                    raise SfzError(f"{path}: #include {m[3]!r} not found at {include}")
                yield from _segments(include, root, defines, depth + 1)
        yield _substitute(line[pos:], defines)


def _tokens(path: Path) -> Iterator[tuple[str | None, str, str]]:
    """(header, "", "") at each header, then (header, opcode, value) for its opcodes."""
    header = None
    for segment in _segments(path, path.parent, {}):
        found = list(_TOKEN.finditer(segment))
        for i, t in enumerate(found):
            if t[1]:
                header = t[1].lower()
                yield header, "", ""
            elif header is not None:
                end = found[i + 1].start() if i + 1 < len(found) else len(segment)
                yield header, t[2], segment[t.end() : end].strip()


def read_program(path: Path, sample_dir: Path | None = None) -> Program:
    """Parse one .sfz file. Sample paths start from `sample_dir` (default: its folder)."""
    path = Path(path)
    sample_root = Path(sample_dir) if sample_dir is not None else path.parent
    scopes: dict[str, dict[str, str]] = {h: {} for h in HEADERS}
    raw_regions: list[tuple[dict[str, str], str]] = []
    cc_defaults: dict[int, float] = {}
    curves: dict[int, Curve] = {}
    curve_ops: dict[str, str] = {}
    default_path = ""
    current: str | None = None

    def close(header: str | None) -> None:
        if header == "region":
            merged = {k: v for h in HEADERS for k, v in scopes[h].items()}
            raw_regions.append((merged, default_path))
        elif header == "curve":
            if "curve_index" not in curve_ops:
                raise SfzError(f"{path}: <curve> without curve_index")
            index = int(curve_ops.pop("curve_index"))
            curves[index] = _curve(curve_ops, path)
            curve_ops.clear()

    for header, op, value in _tokens(path):
        if not op:
            close(current)
            current = header
            if header in HEADERS:  # a header clears its own level and everything inside it
                for h in HEADERS[HEADERS.index(header) :]:
                    scopes[h] = {}
            continue
        if current == "curve":
            curve_ops[op] = value
        elif current == "control":
            if op == "default_path":
                default_path = value.replace("\\", "/")
            elif m := re.fullmatch(r"set_cc(\d+)", op):
                cc_defaults[int(m[1])] = float(value) / 127
            elif m := re.fullmatch(r"set_(?:hd|real)cc(\d+)", op):
                cc_defaults[int(m[1])] = float(value)
        elif current in HEADERS:
            scopes[current][op] = value
    close(current)

    regions: list[Region] = []
    unsupported: Counter[str] = Counter()
    unknown: Counter[str] = Counter()
    switches: list[tuple[int, int, str | None]] = []
    for ops, dpath in raw_regions:
        if any(re.fullmatch(r"on_(lo|hi)cc\d+", o) for o in ops):
            continue  # triggered by CC events, never by notes
        if ops.get("trigger", "attack") != "attack":
            continue  # release and legato triggers: we play one-shots and send no note-offs
        if int(ops.get("lochan", 1)) > 1:
            continue  # we play on MIDI channel 1
        for o, v in ops.items():
            if o in NEUTRAL or o == "pitch_keytrack":
                if not _neutral(o, v, ops):
                    unsupported[o] += 1
            elif o == "loop_mode":
                if v not in LOOP_MODES:
                    unsupported[f"loop_mode={v}"] += 1
            elif IMPLEMENTED.fullmatch(o) or IGNORED.fullmatch(o):
                pass
            elif _looks_real(o):
                unsupported[o] += 1
            else:
                unknown[o] += 1
        if "sw_lokey" in ops and "sw_hikey" in ops:
            switches.append((int(ops["sw_lokey"]), int(ops["sw_hikey"]), ops.get("sw_default")))
        regions.append(_region(ops, dpath, sample_root, path))

    return Program(
        path=path,
        regions=tuple(regions),
        cc_defaults=cc_defaults,
        sw_default=next((int(d) for _, _, d in switches if d is not None), None),
        sw_keys=(min(s[0] for s in switches), max(s[1] for s in switches)) if switches else None,
        curves=curves,
        unsupported=dict(unsupported),
        unknown=dict(unknown),
    )


# Families of real SFZ opcodes that change the sound and aren't implemented here.
_REAL = re.compile(
    r"(tune|pitch|transpose|delay|offset|cutoff|resonance|fil\d?|eq\d|width|position|"
    r"amp_random|pitch_random|pitch_veltrack|amp_keytrack|amp_keycenter|xf\w*|loop\w*|"
    r"(amp|pitch|fil)(eg|lfo)|ampeg|bend|sustain|trigger|on|lfo\d*|eg\d*|sw|seq|"
    r"(lo|hi)(bend|chanaft|polyaft|bpm|timer|prog|cc))(_\w+|\d+)?"
)


def _looks_real(op: str) -> bool:
    return bool(_REAL.fullmatch(op))


def _neutral(op: str, value: str, ops: dict[str, str]) -> bool:
    if op == "pitch_keytrack":  # only matters when one sample spans several keys
        lo, hi = ops.get("lokey", ops.get("key")), ops.get("hikey", ops.get("key"))
        return float(value) == 0 or (lo is not None and lo == hi)
    return float(value) == float(NEUTRAL[op])


def _curve(ops: dict[str, str], path: Path) -> Curve:
    points = {0: 0.0, 127: 1.0}
    for op, value in ops.items():
        m = re.fullmatch(r"v(\d{1,3})", op)
        if not m:
            raise SfzError(f"{path}: unknown <curve> opcode {op!r}")
        points[int(m[1])] = float(value)
    return Curve(tuple(sorted(points.items())))


def _mods(ops: dict[str, str], targets: tuple[str, ...]) -> tuple[CCMod, ...]:
    """CC modulations of one parameter, accepting the *_ccN and *_onccN spellings."""
    out = []
    for op, value in ops.items():
        m = re.fullmatch(r"(\w+?)_(?:on)?cc(\d+)", op)
        if not m or m[1] not in targets:
            continue
        cc = int(m[2])
        curve = next(
            (int(ops[f"{t}_curvecc{cc}"]) for t in targets if f"{t}_curvecc{cc}" in ops), 0
        )
        out.append(CCMod(cc, float(value), curve))
    return tuple(sorted(out, key=lambda c: c.cc))


def _region(ops: dict[str, str], dpath: str, sample_root: Path, path: Path) -> Region:
    def num(name: str, default: float) -> float:
        return float(ops[name]) if name in ops else default

    def opt_int(name: str) -> int | None:
        return int(float(ops[name])) if name in ops else None

    if "key" in ops:
        lokey = hikey = note_number(ops["key"])
    else:
        lokey = note_number(ops.get("lokey", "0"))
        hikey = note_number(ops.get("hikey", "127"))
    ccs = sorted({int(m[2]) for o in ops if (m := re.fullmatch(r"(lo|hi)cc(\d+)", o))})
    cc_ranges = tuple(
        (cc, int(float(ops.get(f"locc{cc}", 0))), int(float(ops.get(f"hicc{cc}", 127))))
        for cc in ccs
    )
    if "sw_last" in ops:
        sw_last: tuple[int, int] | None = (int(ops["sw_last"]),) * 2
    elif "sw_lolast" in ops or "sw_hilast" in ops:
        sw_last = (int(ops.get("sw_lolast", 0)), int(ops.get("sw_hilast", 127)))
    else:
        sw_last = None
    sample = ops.get("sample", "").replace("\\", "/").strip()
    if not sample:
        raise SfzError(f"{path}: a region has no sample")
    if sample == SILENCE:
        pass
    elif sample.startswith("*"):
        raise SfzError(f"{path}: generated sample {sample!r} is not implemented")
    else:
        sample = str((sample_root / dpath / sample).resolve())
    off_mode = ops.get("off_mode", "fast")
    if off_mode not in ("fast", "normal", "time"):
        raise SfzError(f"{path}: off_mode={off_mode!r} is not implemented")
    velcurve = sorted(
        (int(m[1]), float(v))
        for o, v in ops.items()
        if (m := re.fullmatch(r"amp_velcurve_(\d+)", o))
    )
    return Region(
        sample=sample,
        lokey=lokey,
        hikey=hikey,
        lovel=int(float(ops.get("lovel", 0))),
        hivel=int(float(ops.get("hivel", 127))),
        cc_ranges=cc_ranges,
        sw_last=sw_last,
        seq_length=max(1, int(float(ops.get("seq_length", 1)))),
        seq_position=int(float(ops.get("seq_position", 1))),
        lorand=num("lorand", 0.0),
        hirand=num("hirand", 1.0),
        volume_db=sum(num(f"{h}volume", 0.0) for h in ("", "group_", "master_", "global_")),
        amplitude=num("amplitude", 100.0),
        amplitude_cc=_mods(ops, ("amplitude",)),
        volume_cc=_mods(ops, ("volume", "gain")),
        pan=num("pan", 0.0),
        pan_cc=_mods(ops, ("pan",)),
        amp_veltrack=num("amp_veltrack", 100.0),
        amp_velcurve=tuple(velcurve),
        phase_invert=ops.get("phase", "normal") == "invert",
        offset=int(num("offset", 0)),
        end=opt_int("end"),
        ampeg_attack=num("ampeg_attack", 0.0),
        ampeg_hold=num("ampeg_hold", 0.0),
        ampeg_decay=num("ampeg_decay", 0.0),
        ampeg_sustain=num("ampeg_sustain", 100.0),
        ampeg_release=num("ampeg_release", 0.001),
        group=int(num("group", 0)),
        off_by=opt_int("off_by"),
        off_mode=off_mode,
        off_time=num("off_time", 0.006) if "off_time" in ops else None,
        polyphony=opt_int("polyphony"),
        note_polyphony=opt_int("note_polyphony"),
        label=ops.get("region_label") or ops.get("group_label") or Path(sample).name,
    )
