"""Explicit unit registry and conversions.

Rule: units are always declared by the caller and never guessed. A unit that is not in the
registry raises `UnknownUnitError`. Names are case sensitive on purpose ('MW' is not 'mW').

Conversion model: base_value = value * factor + offset. Every quantity has one canonical
unit (see CANONICAL_UNITS) that is used internally and in all stored tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

import numpy as np

T = TypeVar("T")


class UnitError(ValueError):
    """Base class for unit problems."""


class UnknownUnitError(UnitError):
    """The unit is not in the registry."""


class IncompatibleUnitError(UnitError):
    """Source and target units measure different quantities."""


@dataclass(frozen=True)
class Unit:
    name: str
    quantity: str
    factor: float  # multiply by this to reach the canonical unit
    offset: float = 0.0  # then add this


CANONICAL_UNITS: dict[str, str] = {
    "energy": "kWh",
    "power": "kW",
    "temperature": "degC",
    "speed": "m/s",
    "pressure": "hPa",
    "irradiance": "W/m2",
    "percent": "%",
    "angle": "deg",
    "precipitation": "mm",
}

_F_TO_C = 5.0 / 9.0

_UNIT_LIST: tuple[Unit, ...] = (
    # energy (canonical kWh)
    Unit("Wh", "energy", 1e-3),
    Unit("kWh", "energy", 1.0),
    Unit("MWh", "energy", 1e3),
    # power (canonical kW)
    Unit("W", "power", 1e-3),
    Unit("kW", "power", 1.0),
    Unit("MW", "power", 1e3),
    # temperature (canonical degC)
    Unit("degC", "temperature", 1.0),
    Unit("K", "temperature", 1.0, -273.15),
    Unit("degF", "temperature", _F_TO_C, -32.0 * _F_TO_C),
    # speed (canonical m/s)
    Unit("m/s", "speed", 1.0),
    Unit("km/h", "speed", 1.0 / 3.6),
    Unit("kn", "speed", 1852.0 / 3600.0),
    # pressure (canonical hPa)
    Unit("hPa", "pressure", 1.0),
    Unit("Pa", "pressure", 0.01),
    Unit("kPa", "pressure", 10.0),
    Unit("bar", "pressure", 1000.0),
    # irradiance (canonical W/m2)
    Unit("W/m2", "irradiance", 1.0),
    Unit("kW/m2", "irradiance", 1e3),
    # percent (canonical %)
    Unit("%", "percent", 1.0),
    Unit("fraction", "percent", 100.0),
    # angle (canonical deg)
    Unit("deg", "angle", 1.0),
    Unit("rad", "angle", 180.0 / np.pi),
    # precipitation (canonical mm per interval)
    Unit("mm", "precipitation", 1.0),
    Unit("in", "precipitation", 25.4),
)

# Only unambiguous spellings. Anything else must be declared with a registry name.
_ALIASES: dict[str, str] = {
    "°C": "degC",
    "°F": "degF",
    "W/m^2": "W/m2",
    "W/m²": "W/m2",
    "W m-2": "W/m2",
    "kW/m^2": "kW/m2",
    "kW/m²": "kW/m2",
    "°": "deg",
}

_REGISTRY: dict[str, Unit] = {unit.name: unit for unit in _UNIT_LIST}


def get_unit(name: str) -> Unit:
    """Look up a unit by registry name or accepted alias. Never guesses."""
    key = _ALIASES.get(name, name)
    try:
        return _REGISTRY[key]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY))
        raise UnknownUnitError(f"Unknown unit {name!r}. Declare one of: {known}") from None


def canonical_unit(quantity: str) -> str:
    try:
        return CANONICAL_UNITS[quantity]
    except KeyError:
        raise UnitError(f"Unknown quantity {quantity!r}") from None


def convert(values: T, from_unit: str, to_unit: str) -> T:
    """Convert a scalar, array or pandas object between two units of the same quantity."""
    src, dst = get_unit(from_unit), get_unit(to_unit)
    if src.quantity != dst.quantity:
        raise IncompatibleUnitError(
            f"Cannot convert {src.name} ({src.quantity}) to {dst.name} ({dst.quantity})"
        )
    if src == dst:
        return values
    data = np.asarray(values) if isinstance(values, (list, tuple)) else values
    base = data * src.factor + src.offset
    return (base - dst.offset) / dst.factor  # type: ignore[no-any-return]


def to_canonical(values: T, from_unit: str) -> tuple[T, str]:
    """Convert to the canonical unit of the same quantity. Returns (values, canonical_name)."""
    target = canonical_unit(get_unit(from_unit).quantity)
    return convert(values, from_unit, target), target


def mean_power_to_energy_kwh(mean_power: T, power_unit: str, interval_s: int) -> T:
    """Energy in kWh delivered by a constant mean power over an interval."""
    if interval_s <= 0:
        raise UnitError("interval_s must be positive")
    if get_unit(power_unit).quantity != "power":
        raise IncompatibleUnitError(f"{power_unit} is not a power unit")
    kw = convert(mean_power, power_unit, "kW")
    return kw * (interval_s / 3600.0)  # type: ignore[operator, no-any-return]


def energy_to_mean_power_kw(energy: T, energy_unit: str, interval_s: int) -> T:
    """Mean power in kW over an interval that consumed the given energy."""
    if interval_s <= 0:
        raise UnitError("interval_s must be positive")
    if get_unit(energy_unit).quantity != "energy":
        raise IncompatibleUnitError(f"{energy_unit} is not an energy unit")
    kwh = convert(energy, energy_unit, "kWh")
    return kwh / (interval_s / 3600.0)  # type: ignore[operator, no-any-return]
