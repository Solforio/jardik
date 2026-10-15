"""Minimal HTTP JSON client (standard library only)."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

from jardik import __version__
from jardik.providers.base import ProviderError

FetchJson = Callable[[str, dict[str, Any]], dict[str, Any]]


def make_fetch_json(timeout_s: float = 30.0) -> FetchJson:
    """Return a function `(url, params) -> parsed JSON` with a timeout and clear errors."""

    def fetch_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
        full_url = f"{url}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(full_url, headers={"User-Agent": f"jardik/{__version__}"})
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310
                return json.loads(response.read().decode("utf-8"))  # type: ignore[no-any-return]
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:300]
            raise ProviderError(f"HTTP {exc.code} from {url}: {body}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ProviderError(f"Request to {url} failed: {exc}") from exc

    return fetch_json
