import hashlib
import io
from pathlib import Path

import pytest

from jardik.dataset_manager.adapters.base import DatasetError
from jardik.dataset_manager.adapters.bdg2_download import MEDIA_BASE, RAW_BASE, fetch_bdg2

CONTENT = {
    "data/metadata/metadata.csv": b"building_id,site_id\nA,S\n",
    "data/weather/weather.csv": b"timestamp,site_id\n2016-01-01 00:00:00,S\n",
    "data/meters/cleaned/electricity_cleaned.csv": b"timestamp,A\n2016-01-01 00:00:00,1.0\n",
}


def _pointer(data: bytes, oid: str | None = None) -> bytes:
    sha = oid or hashlib.sha256(data).hexdigest()
    return (
        f"version https://git-lfs.github.com/spec/v1\noid sha256:{sha}\nsize {len(data)}\n".encode()
    )


class FakeServer:
    def __init__(self, corrupt: str | None = None) -> None:
        self.requests: list[str] = []
        self.corrupt = corrupt

    def __call__(self, url: str) -> io.BytesIO:
        self.requests.append(url)
        for path, data in CONTENT.items():
            if url == f"{RAW_BASE}/{path}":
                return io.BytesIO(_pointer(data))
            if url == f"{MEDIA_BASE}/{path}":
                return io.BytesIO(b"corrupted!" if path == self.corrupt else data)
        raise AssertionError(f"unexpected URL {url}")


def test_downloads_all_files_and_verifies_checksums(tmp_path: Path):
    server = FakeServer()
    files = fetch_bdg2(tmp_path, opener=server)
    assert {f["name"] for f in files} == {"metadata.csv", "weather.csv", "electricity_cleaned.csv"}
    assert (tmp_path / "metadata.csv").read_bytes() == CONTENT["data/metadata/metadata.csv"]
    for f in files:
        assert f["sha256"] == hashlib.sha256((tmp_path / f["name"]).read_bytes()).hexdigest()
        assert f["status"] == "downloaded"


def test_second_run_skips_files_that_already_match(tmp_path: Path):
    fetch_bdg2(tmp_path, opener=FakeServer())
    server = FakeServer()
    files = fetch_bdg2(tmp_path, opener=server)
    assert all(f["status"] == "verified" for f in files)
    assert all(url.startswith(RAW_BASE) for url in server.requests)  # pointers only, no bulk data


def test_checksum_mismatch_is_an_error_and_leaves_no_file(tmp_path: Path):
    bad = "data/weather/weather.csv"
    with pytest.raises(DatasetError, match="checksum"):
        fetch_bdg2(tmp_path, opener=FakeServer(corrupt=bad))
    assert not (tmp_path / "weather.csv").exists()
    assert not list(tmp_path.glob("*.part"))


def test_local_file_with_wrong_content_is_replaced(tmp_path: Path):
    (tmp_path / "metadata.csv").write_bytes(b"stale")
    files = fetch_bdg2(tmp_path, opener=FakeServer())
    assert (tmp_path / "metadata.csv").read_bytes() == CONTENT["data/metadata/metadata.csv"]
    assert {f["name"]: f["status"] for f in files}["metadata.csv"] == "downloaded"


def test_raw_variant_downloads_the_uncleaned_electricity_file(tmp_path: Path):
    content = dict(CONTENT)
    content["data/meters/raw/electricity.csv"] = b"timestamp,A\n2016-01-01 00:00:00,0.0\n"

    def server(url: str) -> io.BytesIO:
        for path, data in content.items():
            if url == f"{RAW_BASE}/{path}":
                return io.BytesIO(_pointer(data))
            if url == f"{MEDIA_BASE}/{path}":
                return io.BytesIO(data)
        raise AssertionError(url)

    files = fetch_bdg2(tmp_path, variant="raw", opener=server)
    assert "electricity.csv" in {f["name"] for f in files}


def test_a_non_lfs_pointer_is_rejected(tmp_path: Path):
    def server(url: str) -> io.BytesIO:
        return io.BytesIO(b"<html>not a pointer</html>")

    with pytest.raises(DatasetError, match="pointer"):
        fetch_bdg2(tmp_path, opener=server)
