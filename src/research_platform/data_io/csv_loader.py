"""CSV loading utilities."""

import csv
from io import BytesIO, StringIO
from pathlib import Path

import pandas as pd

from research_platform.utils.errors import DataLoadError

# Разделители, которые инструмент умеет распознавать автоматически.
_SUPPORTED_DELIMITERS = ",;"
_SAMPLE_SIZE = 16384


def _detect_delimiter(sample: str) -> str:
    """Detect the CSV field delimiter (``,`` or ``;``) from a text sample."""
    if not sample.strip():
        return ","
    try:
        return csv.Sniffer().sniff(sample, delimiters=_SUPPORTED_DELIMITERS).delimiter
    except csv.Error:
        # Fallback: pick whichever candidate appears more often in the header line.
        first_line = sample.splitlines()[0]
        return ";" if first_line.count(";") > first_line.count(",") else ","


def _read_text(source: str | Path | BytesIO) -> str:
    """Read the raw CSV text, decoding bytes with BOM-aware UTF-8."""
    if isinstance(source, BytesIO):
        source.seek(0)
        raw = source.read()
        return raw.decode("utf-8-sig", errors="replace") if isinstance(raw, bytes) else raw
    return Path(source).read_text(encoding="utf-8-sig", errors="replace")


def load_csv(source: str | Path | BytesIO) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame.

    Field delimiter is auto-detected: both comma (``,``) and semicolon (``;``)
    separated files are supported.

    Args:
        source: File path or uploaded bytes buffer.

    Returns:
        Parsed DataFrame.

    Raises:
        DataLoadError: If parsing fails.
    """
    try:
        text = _read_text(source)
        sep = _detect_delimiter(text[:_SAMPLE_SIZE])
        df = pd.read_csv(StringIO(text), sep=sep)
    except Exception as exc:
        raise DataLoadError(f"Failed to read CSV: {exc}") from exc

    if df.empty:
        raise DataLoadError("CSV is empty.")
    if df.shape[1] < 2:
        raise DataLoadError(
            "CSV распознан как один столбец — проверьте разделитель (поддерживаются ',' и ';')."
        )
    return df
