import numpy as np
import pandas as pd
import pytest

from jardik.core.units import (
    IncompatibleUnitError,
    UnknownUnitError,
    canonical_unit,
    convert,
    energy_to_mean_power_kw,
    mean_power_to_energy_kwh,
    to_canonical,
)


@pytest.mark.parametrize(
    ("value", "src", "dst", "expected"),
    [
        (1500.0, "Wh", "kWh", 1.5),
        (2.0, "MWh", "kWh", 2000.0),
        (1.0, "kWh", "Wh", 1000.0),
        (500.0, "W", "kW", 0.5),
        (212.0, "degF", "degC", 100.0),
        (0.0, "degC", "K", 273.15),
        (273.15, "K", "degC", 0.0),
        (36.0, "km/h", "m/s", 10.0),
        (101325.0, "Pa", "hPa", 1013.25),
        (0.5, "fraction", "%", 50.0),
        (1.0, "kW/m2", "W/m2", 1000.0),
    ],
)
def test_scalar_conversions(value, src, dst, expected):
    assert convert(value, src, dst) == pytest.approx(expected)


def test_series_and_array_conversion():
    series = pd.Series([1000.0, 2000.0])
    assert convert(series, "Wh", "kWh").tolist() == [1.0, 2.0]
    assert convert(np.array([1.0, 2.0]), "kWh", "Wh").tolist() == [1000.0, 2000.0]
    assert convert([1000.0], "Wh", "kWh").tolist() == [1.0]


def test_same_unit_is_identity():
    series = pd.Series([1.0, 2.0])
    assert convert(series, "kWh", "kWh") is series


def test_unknown_unit_is_never_guessed():
    with pytest.raises(UnknownUnitError):
        convert(1.0, "kwh", "kWh")  # case sensitive on purpose
    with pytest.raises(UnknownUnitError):
        convert(1.0, "mW", "kW")  # 'mW' is not 'MW'


def test_incompatible_quantities_raise():
    with pytest.raises(IncompatibleUnitError):
        convert(1.0, "kW", "kWh")


def test_alias_is_accepted():
    assert convert(100.0, "°C", "K") == pytest.approx(373.15)


def test_to_canonical_returns_canonical_name():
    value, unit = to_canonical(2000.0, "Wh")
    assert (value, unit) == (2.0, "kWh")
    assert canonical_unit("power") == "kW"


def test_power_energy_relation():
    assert mean_power_to_energy_kwh(2.0, "kW", 3600) == pytest.approx(2.0)
    assert mean_power_to_energy_kwh(2000.0, "W", 900) == pytest.approx(0.5)
    assert energy_to_mean_power_kw(0.5, "kWh", 900) == pytest.approx(2.0)
    with pytest.raises(IncompatibleUnitError):
        mean_power_to_energy_kwh(1.0, "kWh", 3600)
    with pytest.raises(ValueError):
        energy_to_mean_power_kw(1.0, "kWh", 0)
