"""Light-weight DataFrame contract engine.

A `FrameContract` lists columns with a kind, nullability and optional allowed values. Violations
are collected and reported together (typed `ContractError`); nothing is repaired silently.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd
from pandas.api import types as ptypes

Kind = Literal["str", "float", "int", "bool", "datetime_utc", "datetime_tz", "datetime_naive"]


class ContractError(ValueError):
    """A DataFrame does not satisfy its contract."""

    def __init__(self, contract: str, violations: list[str]) -> None:
        self.contract = contract
        self.violations = violations
        shown = "\n  - ".join(violations[:20])
        more = f"\n  ... and {len(violations) - 20} more" if len(violations) > 20 else ""
        super().__init__(f"Contract '{contract}' violated:\n  - {shown}{more}")


@dataclass(frozen=True)
class Column:
    name: str
    kind: Kind
    nullable: bool = False
    required: bool = True
    allowed: frozenset[str] | None = None


@dataclass(frozen=True)
class FrameContract:
    name: str
    columns: tuple[Column, ...]
    unique_key: tuple[str, ...] = ()
    extra_checks: tuple[Callable[[pd.DataFrame], list[str]], ...] = field(default_factory=tuple)

    def column(self, name: str) -> Column:
        for col in self.columns:
            if col.name == name:
                return col
        raise KeyError(name)


def _kind_ok(series: pd.Series, kind: Kind) -> bool:
    if kind == "str":
        return bool(ptypes.is_string_dtype(series) or ptypes.is_object_dtype(series))
    if kind == "float":
        return bool(ptypes.is_numeric_dtype(series) and not ptypes.is_bool_dtype(series))
    if kind == "int":
        return bool(ptypes.is_integer_dtype(series))
    if kind == "bool":
        return bool(ptypes.is_bool_dtype(series))
    if kind == "datetime_utc":
        return isinstance(series.dtype, pd.DatetimeTZDtype) and str(series.dt.tz) == "UTC"
    if kind == "datetime_tz":
        return isinstance(series.dtype, pd.DatetimeTZDtype)
    if kind == "datetime_naive":
        return bool(
            ptypes.is_datetime64_dtype(series) and not isinstance(series.dtype, pd.DatetimeTZDtype)
        )
    return False


def collect_violations(df: pd.DataFrame, contract: FrameContract) -> list[str]:
    """Return every violation as a readable string (empty list means valid)."""
    problems: list[str] = []
    for col in contract.columns:
        if col.name not in df.columns:
            if col.required:
                problems.append(f"missing required column '{col.name}'")
            continue
        series = df[col.name]
        if not _kind_ok(series, col.kind):
            problems.append(f"column '{col.name}' has dtype {series.dtype}, expected {col.kind}")
            continue
        if not col.nullable and series.isna().any():
            problems.append(f"column '{col.name}' has {int(series.isna().sum())} null value(s)")
        if col.allowed is not None:
            bad = set(series.dropna().astype(str).unique()) - col.allowed
            if bad:
                problems.append(f"column '{col.name}' has disallowed values: {sorted(bad)[:5]}")

    # Optional key columns that are absent are simply not part of the key.
    key = [k for k in contract.unique_key if k in df.columns]
    if key:
        dup = int(df.duplicated(subset=key).sum())
        if dup:
            problems.append(f"{dup} duplicated row(s) on key {key}")

    if not problems:  # extra checks assume the basic structure is sound
        for check in contract.extra_checks:
            problems.extend(check(df))
    return problems


def validate_frame(df: pd.DataFrame, contract: FrameContract) -> None:
    """Raise `ContractError` listing all violations, or return None if the frame is valid."""
    problems = collect_violations(df, contract)
    if problems:
        raise ContractError(contract.name, problems)
