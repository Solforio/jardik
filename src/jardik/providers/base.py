"""Errors shared by all providers."""

from __future__ import annotations


class ProviderError(RuntimeError):
    """A provider could not deliver valid data."""


class ProviderConfigError(ProviderError):
    """The provider is not configured correctly."""


class ProviderNotSupportedError(ProviderError):
    """The provider does not offer the requested capability (e.g. forecasts)."""
