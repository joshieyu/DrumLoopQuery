"""Download and unpack the datasets used for exploration.

Archives go to data/downloads/<name>/, extracted contents to data/raw/<name>/.
Each archive is checked against the checksum the dataset's host publishes before
it is extracted. data/raw/SOURCES.md records what was fetched, from where, and when.

Usage:
    python -m tools.download_datasets                 # default set (small downloads)
    python -m tools.download_datasets gmd avp         # just these
    python -m tools.download_datasets --list
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
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


# Facts below were checked against the official pages on 2026-10-20.
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
    ]
}


def file_digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path, ds: Dataset) -> str:
    """Check the archive against the published checksum. Returns its sha256."""
    if ds.md5 and file_digest(path, "md5") != ds.md5:
        raise ValueError(f"{path.name}: md5 does not match the published value")
    sha256 = file_digest(path, "sha256")
    if ds.sha256 and sha256 != ds.sha256:
        raise ValueError(f"{path.name}: sha256 does not match the published value")
    return sha256


def fetch(url: str, dest: Path, expected_size: int) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "DrumLoopQuery-downloader"})
    with urllib.request.urlopen(req) as resp, tmp.open("wb") as out:
        total = int(resp.headers.get("Content-Length") or expected_size)
        done, next_report = 0, 0.1
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if done / total >= next_report:
                print(f"  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", flush=True)
                next_report += 0.1
    tmp.rename(dest)


def download(ds: Dataset, data_dir: Path) -> dict:
    archive_dir = data_dir / "downloads" / ds.name
    raw_dir = data_dir / "raw" / ds.name
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
        raw_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(raw_dir)  # extractall strips absolute paths and '..'
        marker.write_text(sha256 + "\n")
        print(f"[{ds.name}] extracted to {raw_dir.relative_to(REPO_ROOT)}")
    else:
        print(f"[{ds.name}] already extracted")

    return {
        "name": ds.name,
        "title": ds.title,
        "version": ds.version,
        "url": ds.url,
        "page": ds.page,
        "license": ds.license,
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
            f"- Page: {rec['page']}",
            f"- URL: {rec['url']}",
            f"- Archive sha256: `{rec['archive_sha256']}`",
            f"- Verified against: {rec['checksum_source']}",
            f"- Fetched on: {rec['fetched_on']}",
            "",
        ]
    (data_dir / "raw" / "SOURCES.md").write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("names", nargs="*", help="datasets to fetch (default: all defaults)")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--list", action="store_true", help="list known datasets and exit")
    args = parser.parse_args(argv)

    if args.list:
        for ds in DATASETS.values():
            print(f"{ds.name:10s} {ds.size_bytes / 1e6:8.1f} MB  {ds.title} {ds.version}")
        return 0

    names = args.names or [d.name for d in DATASETS.values() if d.default]
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        parser.error(f"unknown dataset(s): {', '.join(unknown)}")

    records = [download(DATASETS[n], args.data_dir) for n in names]
    write_sources(args.data_dir, records)
    print(f"Sources written to {(args.data_dir / 'raw' / 'SOURCES.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
