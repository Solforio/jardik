"""On-disk cache of API responses.

Every provider call goes through `ResponseCache.get_or_fetch`. Reasons: reproducibility (the same
request returns the same data later), fewer API calls (free tiers have limits) and offline reruns.

Entries are JSON files named by a hash of (provider, endpoint, parameters). History requests never
expire; forecast requests pass `max_age` because a forecast gets stale.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from jardik.core.logging import get_logger

log = get_logger("providers.cache")

Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class CachedResponse:
    payload: Any
    fetched_at: datetime
    from_cache: bool


class ResponseCache:
    def __init__(self, cache_dir: Path, enabled: bool = True, clock: Clock = _utc_now) -> None:
        self.cache_dir = cache_dir
        self.enabled = enabled
        self._clock = clock
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(provider: str, endpoint: str, params: dict[str, Any]) -> str:
        canonical = json.dumps(
            {"provider": provider, "endpoint": endpoint, "params": params},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _path(self, provider: str, key: str) -> Path:
        return self.cache_dir / provider / f"{key}.json"

    def get_or_fetch(
        self,
        provider: str,
        endpoint: str,
        params: dict[str, Any],
        fetch: Callable[[], Any],
        max_age: timedelta | None = None,
    ) -> CachedResponse:
        """Return the cached response if present and fresh, otherwise call `fetch` and store it."""
        if not self.enabled:
            self.misses += 1
            return CachedResponse(fetch(), self._clock(), from_cache=False)

        key = self.make_key(provider, endpoint, params)
        path = self._path(provider, key)
        cached = self._read(path, max_age)
        if cached is not None:
            self.hits += 1
            log.debug("cache hit %s/%s", provider, key[:10])
            return cached

        self.misses += 1
        log.debug("cache miss %s/%s -> fetching", provider, key[:10])
        payload = fetch()
        fetched_at = self._clock()
        self._write(path, provider, endpoint, params, payload, fetched_at)
        return CachedResponse(payload, fetched_at, from_cache=False)

    def _read(self, path: Path, max_age: timedelta | None) -> CachedResponse | None:
        if not path.is_file():
            return None
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            fetched_at = datetime.fromisoformat(entry["fetched_at"])
            payload = entry["payload"]
        except (OSError, ValueError, KeyError, TypeError):
            log.warning("Ignoring unreadable cache entry %s", path.name)
            return None
        if max_age is not None and self._clock() - fetched_at > max_age:
            return None
        return CachedResponse(payload, fetched_at, from_cache=True)

    @staticmethod
    def _write(
        path: Path,
        provider: str,
        endpoint: str,
        params: dict[str, Any],
        payload: Any,
        fetched_at: datetime,
    ) -> None:
        entry = {
            "provider": provider,
            "endpoint": endpoint,
            "params": params,
            "fetched_at": fetched_at.isoformat(),
            "payload": payload,
        }
        text = json.dumps(entry, default=str)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)  # atomic: a crash never leaves a half-written entry
