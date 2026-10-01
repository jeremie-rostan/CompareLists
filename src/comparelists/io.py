"""CSV helpers: read lists, expose columns, write match reports."""

from __future__ import annotations

from typing import Optional

import pandas as pd


def read_csv(path: str, delimiter: Optional[str] = None) -> pd.DataFrame:
    """Read a CSV (or TSV) into a DataFrame with every column as text."""
    if delimiter is None:
        delimiter = "\t" if path.lower().endswith((".tsv", ".tab")) else ","
    return pd.read_csv(path, sep=delimiter, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def list_columns(path: str, delimiter: Optional[str] = None) -> list[str]:
    return list(read_csv(path, delimiter).columns)


def column_values(path: str, column: str, delimiter: Optional[str] = None) -> list[str]:
    frame = read_csv(path, delimiter)
    if column not in frame.columns:
        raise KeyError(f"column {column!r} not found; available: {list(frame.columns)}")
    return [str(v) for v in frame[column].tolist()]


def write_results(path: str, rows: list[dict], delimiter: str = ",") -> None:
    pd.DataFrame(rows).to_csv(path, sep=delimiter, index=False, encoding="utf-8-sig")
