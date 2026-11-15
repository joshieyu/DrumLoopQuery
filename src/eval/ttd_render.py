from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from src.music.spec import MusicSpec, load_spec

from src.eval.audio_io import load_mono, resolve
from src.eval.evaluate import EvalError
from src.eval.onset_baseline import loop_beats
from src.eval.relevance import read_manifest
from src.eval.ttd_prep import LEFT_PITCHES, read_notes

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIO_DIR = REPO_ROOT / "data/datasets/taptamdrum/audio"
PEAK = 0.8
HAND_RULES = ("density", "left-low", "right-low")


def placeholder_knocks(sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    ## two stand-in desk knocks: a low thump and a higher tick. They are only a placeholder
    ## until the team's recorded knocks exist, so results on them must say so
    rng = np.random.default_rng(0)
    t = np.arange(int(0.12 * sample_rate)) / sample_rate
    low = np.sin(2 * np.pi * 110 * t) * np.exp(-t / 0.05)
    low += 0.3 * rng.normal(0, 1, len(t)) * np.exp(-t / 0.008)
    high = np.diff(rng.normal(0, 1, len(t) + 1)) * np.exp(-t / 0.015)
    high += 0.5 * np.sin(2 * np.pi * 2500 * t) * np.exp(-t / 0.01)
    return low / np.abs(low).max(), high / np.abs(high).max()


def load_knocks(low_path: Path, high_path: Path, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    knocks = []
    for path in (low_path, high_path):
        wave = load_mono(path, sample_rate)
        knocks.append(wave / np.abs(wave).max())
    return knocks[0], knocks[1]


def pick_low_hand(notes: list[tuple[int, int, float]], rule: str) -> str:
    if rule not in HAND_RULES:
        raise EvalError(f"hand rule must be one of {HAND_RULES}, got {rule!r}")
    if rule in ("left-low", "right-low"):
        return rule.split("-")[0]
    n_left = sum(pitch in LEFT_PITCHES for pitch, _, _ in notes)
    ## the sparser hand gets the low knock, the way a kick is sparser than snare plus hats
    return "left" if n_left <= len(notes) - n_left else "right"


def render_tap(
    notes: list[tuple[int, int, float]],
    tempo_bpm: float,
    lead_in_s: float,
    loop_s: float,
    sample_rate: int,
    low_hand: str,
    knocks: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    low, high = knocks
    length = round((lead_in_s + loop_s) * sample_rate)
    wave = np.zeros(length)
    for pitch, velocity, beat in notes:
        hand = "left" if pitch in LEFT_PITCHES else "right"
        knock = low if hand == low_hand else high
        start = round((lead_in_s + beat * 60.0 / tempo_bpm) * sample_rate)
        if 0 <= start < length:
            end = min(start + len(knock), length)
            wave[start:end] += knock[: end - start] * (velocity / 127.0)
    peak = np.abs(wave).max()
    if peak == 0:
        raise EvalError("a tap has no notes inside its loop")
    return (wave / peak * PEAK).astype(np.float32)


def library_lead_in(library_path: Path) -> float:
    ## every library clip starts with the same lead-in, and a tap gets the same one
    library = read_manifest(library_path, ("lead_in_s",))
    lead_ins = pd.to_numeric(library["lead_in_s"]).round(6).unique()
    if len(lead_ins) != 1:
        raise EvalError(f"library clips use different lead-ins: {sorted(lead_ins)}")
    return float(lead_ins[0])


def query_sample_rate(queries_path: Path) -> int:
    ## the library has clips at more than one rate, so taps follow the other queries instead
    queries = read_manifest(queries_path, ("sample_rate",))
    rates = pd.to_numeric(queries["sample_rate"]).unique()
    if len(rates) != 1:
        raise EvalError(f"queries use different sample rates: {sorted(rates)}")
    return int(rates[0])


def render_all(
    manifest: pd.DataFrame,
    spec: MusicSpec,
    lead_in_s: float,
    sample_rate: int,
    out_dir: Path,
    knocks: tuple[np.ndarray, np.ndarray],
    rule: str = "density",
) -> pd.DataFrame:
    import soundfile as sf

    beats = loop_beats(spec)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = manifest.copy()
    paths, durations, low_hands = [], [], []
    for row in result.itertuples():
        notes, tempo = read_notes(resolve(row.midi_path))
        low_hand = pick_low_hand(notes, rule)
        wave = render_tap(
            notes, tempo, lead_in_s, beats * 60.0 / tempo, sample_rate, low_hand, knocks
        )
        path = out_dir / f"{row.query_id}.wav"
        sf.write(path, wave, sample_rate)
        try:
            paths.append(path.relative_to(REPO_ROOT).as_posix())
        except ValueError:
            paths.append(str(path))
        durations.append(len(wave) / sample_rate)
        low_hands.append(low_hand)
    result["audio_path"] = paths
    result["duration_s"] = durations
    result["low_hand"] = low_hands
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Render TapTamDrum taps to audio.")
    parser.add_argument(
        "--manifest", type=Path, default=REPO_ROOT / "manifests/ttd_query_manifest.csv"
    )
    parser.add_argument(
        "--library", type=Path, default=REPO_ROOT / "manifests/library_manifest.csv"
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=REPO_ROOT / "manifests/query_manifest.csv",
        help="the synthetic query manifest, which sets the sample rate",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument(
        "--manifest-out",
        type=Path,
        help="write the rendered manifest here instead of over --manifest",
    )
    parser.add_argument("--hand-rule", choices=HAND_RULES, default="density")
    parser.add_argument(
        "--knocks",
        nargs=2,
        type=Path,
        metavar=("LOW_WAV", "HIGH_WAV"),
        help="recorded desk knocks; without them a placeholder pair is used",
    )
    args = parser.parse_args()
    try:
        spec = load_spec()
        lead_in_s = library_lead_in(args.library)
        sample_rate = query_sample_rate(args.queries)
        knocks = (
            load_knocks(args.knocks[0], args.knocks[1], sample_rate)
            if args.knocks
            else placeholder_knocks(sample_rate)
        )
        manifest = read_manifest(args.manifest, ("query_id", "midi_path"))
        rendered = render_all(
            manifest, spec, lead_in_s, sample_rate, args.out_dir, knocks, args.hand_rule
        )
    except (OSError, EvalError) as error:
        parser.exit(1, f"Rendering failed: {error}\n")
    manifest_out = args.manifest_out or args.manifest
    rendered.to_csv(manifest_out, index=False)
    print(f"Rendered {len(rendered)} taps to {args.out_dir.resolve()}")
    print(f"lead-in {lead_in_s:g} s, {sample_rate} Hz, hand rule {args.hand_rule}")
    print(f"low hand: {rendered['low_hand'].value_counts().to_dict()}")
    print("knocks: " + ("recorded" if args.knocks else "PLACEHOLDER, not the team's recordings"))
    print(f"Wrote {manifest_out.resolve()}")


if __name__ == "__main__":
    main()
