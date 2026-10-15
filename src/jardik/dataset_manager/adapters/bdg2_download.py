"""Download Building Data Genome 2 files with integrity checks.

The repository stores its CSV files with Git LFS, so the plain GitHub "raw" URL returns a small
pointer file that carries the real file's SHA-256 and size. We read the pointer, download the
real file from the LFS media URL and refuse it unless its SHA-256 matches the pointer.
"""

from __future__ import annotations

import hashlib
import os
import re
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO, Literal

from jardik import __version__
from jardik.core.logging import get_logger
from jardik.dataset_manager.adapters.base import DatasetError

log = get_logger("dataset_manager.bdg2_download")

_REPO = "buds-lab/building-data-genome-project-2"
RAW_BASE = f"https://raw.githubusercontent.com/{_REPO}/master"
MEDIA_BASE = f"https://media.githubusercontent.com/media/{_REPO}/master"

Opener = Callable[[str], BinaryIO]
_POINTER = re.compile(
    rb"^version https://git-lfs\.github\.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize (\d+)\n"
)
_CHUNK = 1024 * 1024


def _paths(variant: Literal["cleaned", "raw"]) -> list[str]:
    electricity = (
        "data/meters/cleaned/electricity_cleaned.csv"
        if variant == "cleaned"
        else "data/meters/raw/electricity.csv"
    )
    return ["data/metadata/metadata.csv", "data/weather/weather.csv", electricity]


def _default_opener(url: str) -> BinaryIO:
    request = urllib.request.Request(url, headers={"User-Agent": f"jardik/{__version__}"})
    return urllib.request.urlopen(request, timeout=60)  # type: ignore[return-value]  # noqa: S310


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_pointer(repo_path: str, opener: Opener) -> tuple[str, int]:
    with opener(f"{RAW_BASE}/{repo_path}") as stream:
        head = stream.read(512)
    match = _POINTER.match(head)
    if match is None:
        raise DatasetError(
            f"{repo_path}: expected a Git LFS pointer file but got something else. "
            "The repository layout may have changed; refusing to download unverified data."
        )
    return match.group(1).decode(), int(match.group(2))


def fetch_bdg2(
    dest_dir: Path,
    variant: Literal["cleaned", "raw"] = "cleaned",
    opener: Opener | None = None,
) -> list[dict[str, Any]]:
    """Make sure the three BDG2 files are in `dest_dir`, verified by SHA-256.

    Returns one dict per file: name, path, sha256, size_bytes and status
    ('verified' when the local file already matched, 'downloaded' otherwise).
    """
    open_url = opener or _default_opener
    dest_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for repo_path in _paths(variant):
        name = Path(repo_path).name
        target = dest_dir / name
        expected_sha, expected_size = _read_pointer(repo_path, open_url)

        if target.is_file() and target.stat().st_size == expected_size:
            if _sha256_of(target) == expected_sha:
                log.info("%s already present and verified", name)
                results.append(_entry(target, expected_sha, "verified"))
                continue

        log.info("downloading %s (%d bytes)", name, expected_size)
        partial = target.with_suffix(target.suffix + ".part")
        digest = hashlib.sha256()
        try:
            with open_url(f"{MEDIA_BASE}/{repo_path}") as stream, partial.open("wb") as out:
                for chunk in iter(lambda s=stream: s.read(_CHUNK), b""):
                    digest.update(chunk)
                    out.write(chunk)
            if digest.hexdigest() != expected_sha:
                raise DatasetError(
                    f"{name}: checksum mismatch (expected {expected_sha[:12]}..., "
                    f"got {digest.hexdigest()[:12]}...). The download was discarded."
                )
            os.replace(partial, target)
        finally:
            if partial.exists():
                partial.unlink()
        results.append(_entry(target, expected_sha, "downloaded"))
    return results


def _entry(path: Path, sha256: str, status: str) -> dict[str, Any]:
    return {
        "name": path.name,
        "path": str(path),
        "sha256": sha256,
        "size_bytes": path.stat().st_size,
        "status": status,
    }
