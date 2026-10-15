"""Small building blocks shared by the PDF reports (extraction report, forecast report).

Numbers in reports follow the Spanish convention (1.234,5); see `jardik.core.formatting`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Table, TableStyle

from jardik.core.formatting import format_number_es

_STYLES = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=_STYLES["BodyText"], fontSize=9, leading=12)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=7.5, leading=9.5)
H1 = ParagraphStyle("h1", parent=_STYLES["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4)
TITLE = ParagraphStyle("title", parent=_STYLES["Title"], fontSize=18, spaceAfter=6)
BULLET = ParagraphStyle("bullet", parent=BODY, leftIndent=10, bulletIndent=0)
PAGE_WIDTH_MM = 172  # usable width with the margins below


def para(text: object, style: ParagraphStyle = BODY) -> Paragraph:
    """A paragraph of plain text (special characters are escaped)."""
    return Paragraph(escape(str(text)), style)


def bullets(items: list[str]) -> list[Paragraph]:
    return [Paragraph(escape(item), BULLET, bulletText="-") for item in items]


def table(
    rows: list[list[object]], widths: list[float], header: bool = True, font: float = 7.5
) -> Table:
    """A grid table; widths are in mm and the first row is the header."""
    style = ParagraphStyle("cell", parent=SMALL, fontSize=font, leading=font + 2)
    cells = [[Paragraph(escape(str(c)), style) for c in row] for row in rows]
    result = Table(cells, colWidths=[w * mm for w in widths], repeatRows=1 if header else 0)
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        commands.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")))
    result.setStyle(TableStyle(commands))
    return result


def kv(pairs: list[tuple[str, object]]) -> Table:
    """A two-column key/value table without header."""
    return table([[k, v] for k, v in pairs], [48, 124], header=False, font=8)


def image(path: Path, width_mm: float = PAGE_WIDTH_MM) -> Image:
    """A PNG scaled to a width, keeping its aspect ratio."""
    from reportlab.lib.utils import ImageReader

    w, h = ImageReader(str(path)).getSize()
    return Image(str(path), width=width_mm * mm, height=width_mm * mm * h / w)


def num(value: object, decimals: int = 0) -> str:
    return format_number_es(value, decimals)  # type: ignore[arg-type]


def build_document(path: Path, story: list[Any], title: str) -> None:
    """Write the PDF. Deterministic: the same story gives byte-identical output."""
    previous = rl_config.invariant
    rl_config.invariant = 1
    try:
        doc = SimpleDocTemplate(
            str(path),
            pagesize=A4,
            leftMargin=19 * mm,
            rightMargin=19 * mm,
            topMargin=16 * mm,
            bottomMargin=16 * mm,
            title=title,
            author="Jardik",
        )
        doc.build(story)
    finally:
        rl_config.invariant = previous
