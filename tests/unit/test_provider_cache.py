from datetime import UTC, datetime, timedelta
from pathlib import Path

from jardik.providers.cache import ResponseCache


class FakeClock:
    def __init__(self) -> None:
        self.now = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> None:
        self.now += timedelta(**kwargs)


class Counter:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> dict:
        self.calls += 1
        return {"n": self.calls}


def test_second_identical_request_is_served_from_cache(tmp_path: Path):
    cache, fetch = ResponseCache(tmp_path, clock=FakeClock()), Counter()
    first = cache.get_or_fetch("om", "archive", {"lat": 1, "lon": 2}, fetch)
    second = cache.get_or_fetch("om", "archive", {"lon": 2, "lat": 1}, fetch)  # order irrelevant
    assert fetch.calls == 1
    assert (first.from_cache, second.from_cache) == (False, True)
    assert second.payload == {"n": 1}
    assert (cache.hits, cache.misses) == (1, 1)


def test_different_params_or_provider_miss(tmp_path: Path):
    cache, fetch = ResponseCache(tmp_path), Counter()
    cache.get_or_fetch("om", "archive", {"lat": 1}, fetch)
    cache.get_or_fetch("om", "archive", {"lat": 2}, fetch)
    cache.get_or_fetch("other", "archive", {"lat": 1}, fetch)
    cache.get_or_fetch("om", "forecast", {"lat": 1}, fetch)
    assert fetch.calls == 4


def test_cache_persists_across_instances(tmp_path: Path):
    fetch = Counter()
    ResponseCache(tmp_path).get_or_fetch("om", "archive", {"a": 1}, fetch)
    again = ResponseCache(tmp_path).get_or_fetch("om", "archive", {"a": 1}, fetch)
    assert fetch.calls == 1 and again.from_cache


def test_max_age_expires_entries(tmp_path: Path):
    clock, fetch = FakeClock(), Counter()
    cache = ResponseCache(tmp_path, clock=clock)
    ttl = timedelta(hours=3)
    cache.get_or_fetch("om", "forecast", {"a": 1}, fetch, max_age=ttl)
    clock.advance(hours=2)
    assert cache.get_or_fetch("om", "forecast", {"a": 1}, fetch, max_age=ttl).from_cache
    clock.advance(hours=2)
    refreshed = cache.get_or_fetch("om", "forecast", {"a": 1}, fetch, max_age=ttl)
    assert not refreshed.from_cache and fetch.calls == 2
    assert refreshed.fetched_at == clock.now


def test_no_max_age_never_expires(tmp_path: Path):
    clock, fetch = FakeClock(), Counter()
    cache = ResponseCache(tmp_path, clock=clock)
    cache.get_or_fetch("om", "archive", {"a": 1}, fetch)
    clock.advance(days=3650)
    assert cache.get_or_fetch("om", "archive", {"a": 1}, fetch).from_cache


def test_disabled_cache_always_fetches_and_writes_nothing(tmp_path: Path):
    cache, fetch = ResponseCache(tmp_path, enabled=False), Counter()
    cache.get_or_fetch("om", "archive", {"a": 1}, fetch)
    cache.get_or_fetch("om", "archive", {"a": 1}, fetch)
    assert fetch.calls == 2
    assert not any(tmp_path.rglob("*.json"))


def test_corrupted_entry_is_refetched(tmp_path: Path):
    cache, fetch = ResponseCache(tmp_path), Counter()
    cache.get_or_fetch("om", "archive", {"a": 1}, fetch)
    for file in tmp_path.rglob("*.json"):
        file.write_text("{not json", encoding="utf-8")
    result = cache.get_or_fetch("om", "archive", {"a": 1}, fetch)
    assert fetch.calls == 2 and not result.from_cache


def test_failed_fetch_is_not_cached(tmp_path: Path):
    cache = ResponseCache(tmp_path)

    def boom() -> dict:
        raise RuntimeError("network down")

    try:
        cache.get_or_fetch("om", "archive", {"a": 1}, boom)
    except RuntimeError:
        pass
    assert not any(tmp_path.rglob("*.json"))
    assert cache.get_or_fetch("om", "archive", {"a": 1}, Counter()).payload == {"n": 1}
