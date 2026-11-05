"""Download and unpack the datasets and drum kits used for exploration.

Archives go to data/downloads/[<group>/]<name>/, extracted contents to
data/raw/[<group>/]<name>/. Each archive is checked against the checksum the host
publishes, or at least the size it reports, before it is extracted.
data/raw/SOURCES.md records what was fetched, from where, when, and the credit each
license asks for.

Usage:
    python -m tools.download_datasets                 # default set (small downloads)
    python -m tools.download_datasets gmd avp         # just these
    python -m tools.download_datasets kits            # every entry in the "kits" group
    python -m tools.download_datasets --list
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Dataset:
    name: str
    title: str
    version: str
    url: str
    filename: str
    size_bytes: int
    license: str
    page: str
    sha256: str | None = None
    md5: str | None = None
    checksum_source: str = ""
    default: bool = True
    group: str = ""  # files go to data/{downloads,raw}/<group>/<name>/ when set
    archive: str = "zip"  # "zip", "tar" (any compression), or "file" (used as is)
    size_exact: bool = True  # False when the host can't say the size before download
    credit: str = ""  # the attribution the license asks for


ARCHIVE_KINDS = ("zip", "tar", "file")
KITS_CHECKED = "checked against the kit's official page on 2026-11-04"


def github_zip(repo: str, commit: str) -> str:
    """A GitHub repository snapshot, pinned to one commit."""
    return f"https://codeload.github.com/{repo}/zip/{commit}"


# Facts below were checked against the official pages on 2026-10-20 (datasets) and
# 2026-11-04 (kits). Hosts that publish no checksum get a size check, and GitHub
# snapshots are pinned to a commit.
DATASETS: dict[str, Dataset] = {
    d.name: d
    for d in [
        Dataset(
            name="gmd",
            title="Groove MIDI Dataset (MIDI only)",
            version="v1.0.0",
            url="https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip",
            filename="groove-v1.0.0-midionly.zip",
            size_bytes=3_260_318,
            license="CC BY 4.0",
            page="https://magenta.tensorflow.org/datasets/groove",
            md5="0b59b75f6048a60f60f0ffe2e1410d07",
            checksum_source="md5 from Google Cloud Storage (x-goog-hash); page lists none",
        ),
        Dataset(
            name="egmd_midi",
            title="Expanded Groove MIDI Dataset (MIDI only)",
            version="v1.0.0",
            url="https://storage.googleapis.com/magentadata/datasets/e-gmd/v1.0.0/e-gmd-v1.0.0-midi.zip",
            filename="e-gmd-v1.0.0-midi.zip",
            size_bytes=107_076_192,
            license="CC BY 4.0",
            page="https://magenta.tensorflow.org/datasets/e-gmd",
            sha256="5e70a6f4d760385a5e5ec986a2f02179d96f61181a920e592876b577a75844d3",
            checksum_source="sha256 published on the dataset page",
        ),
        Dataset(
            name="avp",
            title="Amateur Vocal Percussion Dataset",
            version="v4 (Zenodo record 5036529, 2021-06-28)",
            url="https://zenodo.org/records/5036529/files/AVP_Dataset.zip?download=1",
            filename="AVP_Dataset.zip",
            size_bytes=230_840_983,
            license="CC BY 4.0",
            page="https://zenodo.org/records/5036529",
            md5="a40f3f3bc9d5bc0f52ab9aa4c13a907e",
            checksum_source="md5 published on the Zenodo record",
        ),
        # ---- drum kits (spec section 4); none are in the default set
        Dataset(
            name="drskit",
            title="DRSKit, SFZ port (DrumGizmo)",
            version="SFZ port v1.0 with the 2020 stereo version, commit b014489",
            url=github_zip(
                "sfzinstruments/DrumGizmo.DRSKit", "b01448990a31f722542f1812e56159a97a6a2a82"
            ),
            filename="DrumGizmo.DRSKit-b014489.zip",
            size_bytes=753_659_000,  # repository size; GitHub builds the zip on request
            size_exact=False,
            license="CC BY 4.0",
            page="https://github.com/sfzinstruments/DrumGizmo.DRSKit",
            checksum_source="none published; pinned to commit b01448990a31",
            default=False,
            group="kits",
            credit=(
                "DRSKit by Lars Muldjord and Bent Bisballe Nyeng (DrumGizmo) with Jes Eiler "
                "of DRSDrums; SFZ mapping by kinwie"
            ),
        ),
        Dataset(
            name="muldjordkit",
            title="MuldjordKit, SFZ port (DrumGizmo)",
            version="SFZ port with the stereo version, commit fc16571",
            url=github_zip(
                "sfzinstruments/DrumGizmo.MuldjordKit", "fc165714974aa843125ce88597f341b31376e1d5"
            ),
            filename="DrumGizmo.MuldjordKit-fc16571.zip",
            size_bytes=345_134_000,  # repository size; GitHub builds the zip on request
            size_exact=False,
            license="CC BY 4.0",
            page="https://github.com/sfzinstruments/DrumGizmo.MuldjordKit",
            checksum_source="none published; pinned to commit fc165714974a",
            default=False,
            group="kits",
            credit="MuldjordKit by Lars Muldjord (DrumGizmo); SFZ mapping by kinwie",
        ),
        Dataset(
            name="salamander",
            title="Salamander Drumkit (original samples)",
            version="v1 (archive.org item SalamanderDrumkit)",
            url="https://archive.org/download/SalamanderDrumkit/salamanderDrumkit.tar.bz2",
            filename="salamanderDrumkit.tar.bz2",
            size_bytes=387_611_727,
            license="CC BY-SA 3.0; share-alike applies only to modified samples or new sample "
            "libraries, not to music made with them",
            page="https://archive.org/details/SalamanderDrumkit",
            md5="af8e2067668a7f438e7d981877fb771f",
            checksum_source="md5 in the archive.org item's file list",
            default=False,
            group="kits",
            archive="tar",
            credit="Salamander Drumkit by Alexander Holm",
        ),
        Dataset(
            name="salamander_sfz",
            title="Salamander Drumkit, kinwie's Sforzando mapping (SFZ files only)",
            version="endolith/Salamander-Drumkit commit 9e9c9fc",
            url=github_zip(
                "endolith/Salamander-Drumkit", "9e9c9fca9f26e1763435beca4170505d0fa32fb0"
            ),
            filename="Salamander-Drumkit-9e9c9fc.zip",
            size_bytes=40_000,  # repository size; GitHub builds the zip on request
            size_exact=False,
            license="mapping text only, no license stated; the samples are in 'salamander'",
            page="https://github.com/endolith/Salamander-Drumkit",
            checksum_source="none published; pinned to commit 9e9c9fca9f26",
            default=False,
            group="kits",
            credit="SFZ mapping by kinwie",
        ),
        Dataset(
            name="gogodze_phu_vol_ii",
            title="Gogodze Phu Vol II (Karoryfer)",
            version="v1.001",
            url="https://github.com/sfzinstruments/karoryfer.gogodze-phu-vol-ii/releases/"
            "download/v1.001/Karoryfer_Gogodze_Phu_vol_II.v1.001.zip",
            filename="Karoryfer_Gogodze_Phu_vol_II.v1.001.zip",
            size_bytes=139_517_148,
            license="CC0 1.0",
            page="https://github.com/sfzinstruments/karoryfer.gogodze-phu-vol-ii",
            checksum_source="none published; size matches the GitHub release",
            default=False,
            group="kits",
            credit="Gogodze Phu Vol II by Karoryfer Lecolds (CC0: credit optional)",
        ),
        *[
            Dataset(
                name=f"avl_{name}",
                title=f"AVL Drumkits: {title} (SFZ)",
                version=version,
                url=f"https://www.bandshed.net/sounds/AVLDrumkits_SFZ/{filename}",
                filename=filename,
                size_bytes=size,
                license="CC BY-SA 3.0",
                page="http://www.bandshed.net/avldrumkits/",
                checksum_source="none published; size matches the server's",
                default=False,
                group="kits",
                credit="AVL Drumkits by Glen MacArthur (bandshed.net)",
            )
            for name, title, version, filename, size in [
                (
                    "black_pearl",
                    "Black Pearl",
                    "2023 repack",
                    "BLACK_PEARL_2023_repack.zip",
                    14_176_289,
                ),
                (
                    "red_zeppelin",
                    "Red Zeppelin",
                    "2023 repack",
                    "RED_ZEPPELIN_2023_repack.zip",
                    17_853_157,
                ),
                ("blonde_bop", "Blonde Bop", "2023-09-05", "BLONDE_BOP_SFZ.zip", 18_894_879),
                (
                    "blonde_bop_hr",
                    "Blonde Bop HR",
                    "2023-09-05",
                    "BLONDE_BOP_HR_SFZ.zip",
                    16_892_964,
                ),
            ]
        ],
        Dataset(
            name="musescore_general",
            title="MuseScore_General.sf2 (General MIDI SoundFont with drum kits)",
            version="0.2.0",
            url="https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/"
            "MuseScore_General.sf2",
            filename="MuseScore_General.sf2",
            size_bytes=215_614_036,
            license="MIT",
            page="https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/",
            checksum_source="none published; size matches the server's",
            default=False,
            group="kits",
            archive="file",
            credit=(
                "MuseScore_General.sf2 by S. Christian Collins, from FluidR3 by Frank Wen and "
                "FluidR3Mono by Michael Cowgill; ship the MIT notice with any copy"
            ),
        ),
    ]
}


def file_digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path, ds: Dataset) -> str:
    """Check the archive against the published checksum and size. Returns its sha256."""
    size = path.stat().st_size
    if ds.size_exact and size != ds.size_bytes:
        raise ValueError(f"{path.name}: {size} bytes, expected {ds.size_bytes}")
    if ds.md5 and file_digest(path, "md5") != ds.md5:
        raise ValueError(f"{path.name}: md5 does not match the published value")
    sha256 = file_digest(path, "sha256")
    if ds.sha256 and sha256 != ds.sha256:
        raise ValueError(f"{path.name}: sha256 does not match the published value")
    return sha256


def fetch(url: str, dest: Path, expected_size: int) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "DrumLoopQuery-downloader"})
    with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as out:
        total = int(resp.headers.get("Content-Length") or expected_size)
        done, next_report = 0, 0.1
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if done / total >= next_report:
                print(f"  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", flush=True)
                next_report += 0.1
    tmp.rename(dest)


def dataset_dirs(ds: Dataset, data_dir: Path) -> tuple[Path, Path]:
    """(archive folder, extracted folder) for one dataset."""
    sub = Path(ds.group, ds.name) if ds.group else Path(ds.name)
    return data_dir / "downloads" / sub, data_dir / "raw" / sub


def extract(archive: Path, ds: Dataset, raw_dir: Path) -> None:
    """Unpack into raw_dir, refusing members that would land outside it."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    if ds.archive == "zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(raw_dir)  # extractall strips absolute paths and '..'
    elif ds.archive == "tar":
        with tarfile.open(archive) as tf:
            tf.extractall(raw_dir, filter="data")  # refuses links and paths outside raw_dir
    elif ds.archive == "file":
        shutil.copy2(archive, raw_dir / ds.filename)
    else:
        raise ValueError(f"{ds.name}: unknown archive kind {ds.archive!r}")


def download(ds: Dataset, data_dir: Path) -> dict:
    archive_dir, raw_dir = dataset_dirs(ds, data_dir)
    archive = archive_dir / ds.filename
    archive_dir.mkdir(parents=True, exist_ok=True)

    if archive.exists():
        print(f"[{ds.name}] archive already present, re-verifying")
    else:
        print(f"[{ds.name}] downloading {ds.size_bytes / 1e6:.1f} MB from {ds.url}")
        fetch(ds.url, archive, ds.size_bytes)
    sha256 = verify(archive, ds)
    print(f"[{ds.name}] checksum ok ({ds.checksum_source})")

    marker = raw_dir / ".extracted"
    if not marker.exists():
        extract(archive, ds, raw_dir)
        marker.write_text(sha256 + "\n")
        print(f"[{ds.name}] extracted to {raw_dir}")
    else:
        print(f"[{ds.name}] already extracted")

    return {
        "name": ds.name,
        "title": ds.title,
        "version": ds.version,
        "url": ds.url,
        "page": ds.page,
        "license": ds.license,
        "credit": ds.credit,
        "folder": raw_dir.relative_to(data_dir).as_posix(),
        "archive_sha256": sha256,
        "checksum_source": ds.checksum_source,
        "fetched_on": dt.date.today().isoformat(),
    }


def write_sources(data_dir: Path, records: list[dict]) -> None:
    state_path = data_dir / "raw" / "sources.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    for rec in records:
        # Keep the original fetch date when re-running on an existing archive.
        rec["fetched_on"] = state.get(rec["name"], rec)["fetched_on"]
        state[rec["name"]] = rec
    state_path.write_text(json.dumps(state, indent=2) + "\n")

    lines = ["# Raw dataset sources", "", "Generated by `tools/download_datasets.py`.", ""]
    for rec in state.values():
        lines += [
            f"## {rec['name']}: {rec['title']}",
            "",
            f"- Version: {rec['version']}",
            f"- License: {rec['license']}",
            *([f"- Credit: {rec['credit']}"] if rec.get("credit") else []),
            f"- Folder: data/{rec.get('folder', 'raw/' + rec['name'])}",
            f"- Page: {rec['page']}",
            f"- URL: {rec['url']}",
            f"- Archive sha256: `{rec['archive_sha256']}`",
            f"- Verified against: {rec['checksum_source']}",
            f"- Fetched on: {rec['fetched_on']}",
            "",
        ]
    (data_dir / "raw" / "SOURCES.md").write_text("\n".join(lines))


def resolve_names(names: list[str]) -> list[str]:
    """Dataset names, with group names (e.g. "kits") expanded; [] means the defaults."""
    if not names:
        return [d.name for d in DATASETS.values() if d.default]
    out: list[str] = []
    for n in names:
        members = [d.name for d in DATASETS.values() if d.group == n] if n not in DATASETS else [n]
        if not members:
            raise KeyError(n)
        out += [m for m in members if m not in out]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "names", nargs="*", help="datasets or groups to fetch (default: all defaults)"
    )
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--list", action="store_true", help="list known datasets and exit")
    args = parser.parse_args(argv)

    if args.list:
        for ds in DATASETS.values():
            approx = "" if ds.size_exact else "~"
            print(
                f"{ds.name:20s} {ds.group or '-':5s} {approx}{ds.size_bytes / 1e6:8.1f} MB  "
                f"{ds.title}, {ds.version}"
            )
        return 0

    try:
        names = resolve_names(args.names)
    except KeyError as e:
        parser.error(f"unknown dataset or group: {e.args[0]}")

    records = [download(DATASETS[n], args.data_dir) for n in names]
    write_sources(args.data_dir, records)
    print(f"Sources written to {(args.data_dir / 'raw' / 'SOURCES.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
