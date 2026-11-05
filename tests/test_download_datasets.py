"""Tests for the dataset and kit downloader (no network: `fetch` is replaced)."""

import io
import json
import tarfile
import zipfile
from dataclasses import replace

import pytest

from tools import download_datasets as dd


def fake_dataset(payload: bytes, **kw) -> dd.Dataset:
    base = dd.Dataset(
        name="toy",
        title="Toy kit",
        version="1",
        url="https://example.invalid/toy",
        filename="toy.bin",
        size_bytes=len(payload),
        license="CC0 1.0",
        page="https://example.invalid",
        group="kits",
        credit="Toy kit by nobody",
    )
    return replace(base, **kw)


@pytest.fixture
def serve(monkeypatch):
    """Make `fetch` write the given bytes instead of downloading."""

    def install(payload: bytes):
        monkeypatch.setattr(dd, "fetch", lambda url, dest, size: dest.write_bytes(payload))

    return install


def zip_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def tar_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:bz2") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_kits_are_a_group_outside_the_default_set():
    kits = dd.resolve_names(["kits"])
    assert "drskit" in kits and "musescore_general" in kits
    assert not set(kits) & set(dd.resolve_names([]))
    assert all(dd.DATASETS[k].group == "kits" for k in kits)
    assert dd.resolve_names(["gmd", "kits", "drskit"]) == ["gmd", *kits]
    with pytest.raises(KeyError):
        dd.resolve_names(["nope"])


def test_every_entry_is_well_formed():
    for ds in dd.DATASETS.values():
        assert ds.archive in dd.ARCHIVE_KINDS
        assert ds.url.startswith("https://"), ds.name
        assert ds.size_bytes > 0
        assert ds.license and ds.checksum_source


@pytest.mark.parametrize("kind,make", [("zip", zip_bytes), ("tar", tar_bytes)])
def test_archives_unpack_into_their_group_folder(tmp_path, serve, kind, make):
    payload = make({"Samples/kick.wav": b"RIFF", "kit.sfz": b"<region>"})
    serve(payload)
    rec = dd.download(fake_dataset(payload, archive=kind), tmp_path)
    raw = tmp_path / "raw" / "kits" / "toy"
    assert (raw / "Samples" / "kick.wav").read_bytes() == b"RIFF"
    assert (tmp_path / "downloads" / "kits" / "toy" / "toy.bin").exists()
    assert rec["folder"] == "raw/kits/toy" and rec["credit"] == "Toy kit by nobody"


def test_single_files_are_copied(tmp_path, serve):
    serve(b"sfbk")
    dd.download(fake_dataset(b"sfbk", archive="file", filename="toy.sf2"), tmp_path)
    assert (tmp_path / "raw" / "kits" / "toy" / "toy.sf2").read_bytes() == b"sfbk"


def test_tar_members_outside_the_folder_are_refused(tmp_path, serve):
    payload = tar_bytes({"../escape.txt": b"x"})
    serve(payload)
    with pytest.raises(tarfile.FilterError):
        dd.download(fake_dataset(payload, archive="tar"), tmp_path)
    assert not (tmp_path / "raw" / "kits" / "escape.txt").exists()


def test_size_and_checksum_mismatches_stop_before_extraction(tmp_path, serve):
    serve(b"abc")
    with pytest.raises(ValueError, match="bytes, expected"):
        dd.download(fake_dataset(b"abc", size_bytes=4), tmp_path)
    with pytest.raises(ValueError, match="md5"):
        dd.download(fake_dataset(b"abc", md5="0" * 32), tmp_path)
    assert not (tmp_path / "raw" / "kits" / "toy").exists()
    # An inexact size (GitHub snapshots) is only a progress estimate.
    payload = zip_bytes({"a.txt": b"a"})
    serve(payload)
    dd.download(fake_dataset(payload, name="snap", size_bytes=1, size_exact=False), tmp_path)
    assert (tmp_path / "raw" / "kits" / "snap" / "a.txt").exists()


def test_sources_list_the_credit_and_folder(tmp_path, serve):
    payload = zip_bytes({"a.txt": b"a"})
    serve(payload)
    (tmp_path / "raw").mkdir()
    old = {"name": "gmd", "title": "GMD", "version": "v1", "url": "u", "page": "p"}
    old |= {"license": "CC BY 4.0", "archive_sha256": "0", "checksum_source": "c"}
    (tmp_path / "raw" / "sources.json").write_text(json.dumps({"gmd": old | {"fetched_on": "d"}}))
    dd.write_sources(tmp_path, [dd.download(fake_dataset(payload), tmp_path)])
    text = (tmp_path / "raw" / "SOURCES.md").read_text()
    assert "- Credit: Toy kit by nobody" in text
    assert "- Folder: data/raw/kits/toy" in text and "- Folder: data/raw/gmd" in text
