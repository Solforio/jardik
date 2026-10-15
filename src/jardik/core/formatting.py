"""Number formatting for user-facing reports and exports (Spanish convention).

Spanish format: period for thousands, comma for decimals (1.234.567,89).
Internal files and Parquet keep plain numeric types; only presentation uses this.
"""

from __future__ import annotations

import math

_MISSING = "n/a"


def format_number_es(value: float | int | None, decimals: int = 2) -> str:
    """Format a number as 1.234.567,89. None and NaN become 'n/a'."""
    if value is None:
        return _MISSING
    number = float(value)
    if math.isnan(number):
        return _MISSING
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    text = f"{number:,.{decimals}f}"
    # Swap separators through a placeholder so the two replacements do not collide.
    return text.replace(",", "\0").replace(".", ",").replace("\0", ".")


def format_percent_es(fraction: float | None, decimals: int = 1) -> str:
    """Format a fraction (0.153) as a Spanish percentage ('15,3 %')."""
    if fraction is None or math.isnan(float(fraction)):
        return _MISSING
    return f"{format_number_es(float(fraction) * 100, decimals)} %"
