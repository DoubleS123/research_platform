from io import BytesIO

import pytest

from research_platform.data_io.csv_loader import load_csv
from research_platform.utils.errors import DataLoadError


def _buf(text: str) -> BytesIO:
    return BytesIO(text.encode("utf-8"))


def test_load_comma_separated() -> None:
    df = load_csv(_buf("date,metric\n2025-01-01,10\n2025-01-02,12\n"))
    assert list(df.columns) == ["date", "metric"]
    assert df.shape == (2, 2)


def test_load_semicolon_separated() -> None:
    df = load_csv(_buf("date;metric\n2025-01-01;10\n2025-01-02;12\n"))
    assert list(df.columns) == ["date", "metric"]
    assert df.shape == (2, 2)


def test_load_handles_utf8_bom() -> None:
    df = load_csv(BytesIO("﻿a;b\n1;2\n".encode("utf-8")))
    assert list(df.columns) == ["a", "b"]


def test_empty_csv_raises() -> None:
    with pytest.raises(DataLoadError):
        load_csv(_buf("date,metric\n"))
