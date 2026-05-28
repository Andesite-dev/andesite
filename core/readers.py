import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from itertools import islice
from pathlib import Path
from typing import Any, Self, Union

import dask.dataframe as dd
import h5py
import numpy as np
import pandas as pd
import polars as pl
from loguru import logger

# (magic_bytes, human-readable description)
_BINARY_SIGNATURES: list[tuple[bytes, str]] = [
    (b"\x4d\x5a", "Windows PE executable"),  # MZ header (EXE/DLL)
    (b"\x7fELF", "ELF binary"),  # Linux ELF
    (b"\x50\x4b\x03\x04", "ZIP archive"),  # ZIP/XLSX/DOCX
    (b"\x25\x50\x44\x46", "PDF document"),  # %PDF
    (b"\x89\x50\x4e\x47", "PNG image"),  # PNG
    (b"\xff\xd8\xff", "JPEG image"),  # JPEG
    (b"\x47\x49\x46\x38", "GIF image"),  # GIF8
    (b"\x52\x61\x72\x21", "RAR archive"),  # Rar!
    (b"\x1f\x8b", "GZIP compressed data"),  # GZIP
    (b"\xca\xfe\xba\xbe", "Java class file"),  # Java .class
    (b"\xfe\xed\xfa\xce", "Mach-O binary"),  # macOS 32-bit
    (b"\xfe\xed\xfa\xcf", "Mach-O binary 64-bit"),  # macOS 64-bit
]

# HDF5 signature — valid binary that AndesiteDatafile can open directly
_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


class AndesiteUnableToReadFileError(Exception):
    def __init__(self, message: str = "") -> None:
        super().__init__(message)


@dataclass
class FileValidationResult:
    is_valid: bool
    error: str = ""
    non_numeric_columns: list[str] = field(default_factory=list)


def grab_n_cols(datafile: str | Path) -> np.int32 | None:
    datafile = Path(datafile)
    with datafile.open() as file:
        for line in islice(file, 1, 2):
            n_cols = line.strip().split()
            return np.int32(n_cols[0])


def grab_col_names(datafile: str | Path) -> list | None:
    datafile = Path(datafile)
    n_cols = grab_n_cols(datafile)
    cols = []
    with datafile.open() as file:
        for line in islice(file, 2, n_cols + 2):
            cols.append(line.strip())
    return cols


datafile_readers: dict[str, Callable[[str], pl.LazyFrame]] = {}


def register_datafile_reader(
    format: str,
) -> Callable[[Callable[[str], pl.LazyFrame]], Callable[[str], pl.LazyFrame]]:
    if not format:
        logger.error("register_datafile_reader called with empty format string")
        raise ValueError("Format cannot be empty")

    def decorator(fn: Callable[[str], pl.LazyFrame]) -> Callable[[str], pl.LazyFrame]:
        if format in datafile_readers:
            logger.error(f"Datafile reader format '{format}' is already registered")
            raise ValueError(f"Format '{format}' already registered")

        @wraps(fn)
        def wrapper(source: str) -> pl.LazyFrame:
            return fn(source)

        datafile_readers[format] = wrapper
        return wrapper

    return decorator


@register_datafile_reader("fixedgslib")
def read_gslib(source: str) -> pl.LazyFrame:
    col_names = grab_col_names(source)
    columns = [c.strip().replace(" ", "") for c in col_names]
    n_cols = len(col_names)

    dataframe = pl.scan_csv(
        source=source,
        separator=" ",
        skip_rows=n_cols + 2,
        new_columns=columns,
        has_header=False,
    )

    return dataframe


@register_datafile_reader("gslib")
def read_raw_gslib(source: str) -> dd.DataFrame:
    col_names = grab_col_names(source)
    columns = [c.strip().replace(" ", "") for c in col_names]
    n_cols = len(col_names)

    dataframe = dd.read_csv(source, delimiter=r"\s+", skiprows=n_cols + 2, names=columns)

    return dataframe


@register_datafile_reader("csv")
def read_csv(source: str) -> pl.LazyFrame:
    return pl.scan_csv(source=source, separator=",")


def _read_hdf5_source(source: str) -> pl.LazyFrame:
    with h5py.File(source, "r") as f:
        data = {key: f[key][()] for key in f.keys() if isinstance(f[key], h5py.Dataset)}
    return pl.DataFrame(data).lazy()


@register_datafile_reader("h5")
def read_h5(source: str) -> pl.LazyFrame:
    return _read_hdf5_source(source)


@register_datafile_reader("hdf5")
def read_hdf5(source: str) -> pl.LazyFrame:
    return _read_hdf5_source(source)


class AndesiteDatafile:
    def __init__(self, source: str) -> None:
        self.source = source
        logger.debug(f"Reading the file: {self.source}")
        self._check_content()

    def __repr__(self) -> str:
        return f"<AndesiteDatafile(source={self.source!r})>"

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    @staticmethod
    def datafile_reader_formats() -> list[str]:
        return list(datafile_readers.keys())

    def _check_content(self) -> None:
        source_path = Path(self.source)

        if ".." in source_path.name:
            logger.error(f"Path traversal detected in filename: {source_path.name}")
            raise AndesiteUnableToReadFileError(
                f"Unsafe filename: path traversal detected in '{source_path.name}'"
            )

        try:
            content = source_path.read_bytes()
        except OSError as exc:
            logger.error(f"Cannot read file '{self.source}': {exc}")
            raise AndesiteUnableToReadFileError(
                f"Cannot read file '{self.source}': {exc}"
            ) from exc

        if len(content) == 0:
            logger.error(f"File is empty: {self.source}")
            raise AndesiteUnableToReadFileError("File is empty")

        # HDF5 is a valid binary format — skip text checks entirely
        if content[:8] == _HDF5_SIGNATURE:
            logger.debug(f"HDF5 file detected, skipping text checks: {self.source}")
            return

        for signature, description in _BINARY_SIGNATURES:
            if content.startswith(signature):
                logger.error(
                    f"File '{self.source}' matches binary signature for {description}"
                )
                raise AndesiteUnableToReadFileError(
                    f"File appears to be a {description}, not a data file"
                )

        sample = content[:512]
        for encoding in ("utf-8", "latin-1"):
            try:
                sample.decode(encoding)
                logger.debug(f"Content check passed ({encoding}) for: {self.source}")
                return
            except UnicodeDecodeError:
                continue

        logger.error(f"File '{self.source}' cannot be decoded as text")
        raise AndesiteUnableToReadFileError(
            "File appears to contain binary data and cannot be read as text"
        )

    def validate_format(self, format: str) -> FileValidationResult:
        if format.lower() in ["gslib", "fixedgslib"]:
            return self._validate_gslib_format()
        if format.lower() == "csv":
            return self._validate_csv_format()
        if format.lower() in ["h5", "hdf5"]:
            return self._validate_hdf5_format()
        return FileValidationResult(
            is_valid=False,
            error=f"No format validator available for '{format}'",
        )

    def _validate_gslib_format(self) -> FileValidationResult:
        try:
            with Path(self.source).open(encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
        except OSError as exc:
            return FileValidationResult(is_valid=False, error=str(exc))

        if len(lines) < 2:
            return FileValidationResult(
                is_valid=False, error="File has less than 2 lines; invalid GSLIB format"
            )

        try:
            n_cols = int(lines[1].strip())
            if n_cols <= 0:
                raise ValueError
        except ValueError:
            return FileValidationResult(
                is_valid=False,
                error=(
                    "Line 2 must be a positive integer (column count); "
                    f"got '{lines[1].strip()}'"
                ),
            )

        header_end = n_cols + 2
        if len(lines) < header_end:
            return FileValidationResult(
                is_valid=False,
                error=(
                    f"Expected {n_cols} column name lines but file has only "
                    f"{len(lines) - 2}"
                ),
            )

        col_names = [lines[i].strip() for i in range(2, header_end)]

        non_numeric: list[str] = []
        for line in lines[header_end : header_end + 20]:
            stripped = line.strip()
            if not stripped:
                continue
            tokens = stripped.split()
            if len(tokens) != n_cols:
                return FileValidationResult(
                    is_valid=False,
                    error=(
                        f"Data row has {len(tokens)} values but expected {n_cols}; "
                        f"row: '{stripped}'"
                    ),
                )
            for idx, token in enumerate(tokens):
                try:
                    float(token)
                except ValueError:
                    col = col_names[idx] if idx < len(col_names) else str(idx)
                    if col not in non_numeric:
                        non_numeric.append(col)

        return FileValidationResult(is_valid=True, non_numeric_columns=non_numeric)

    def _validate_csv_format(self) -> FileValidationResult:
        try:
            with Path(self.source).open(encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
        except OSError as exc:
            return FileValidationResult(is_valid=False, error=str(exc))

        if not lines:
            return FileValidationResult(is_valid=False, error="File is empty")

        header = lines[0].rstrip("\n")
        if "," not in header:
            return FileValidationResult(
                is_valid=False,
                error="Header line contains no commas; wrong delimiter or not a CSV file",
            )

        col_names = [c.strip() for c in header.split(",")]

        non_numeric: list[str] = []
        for line in lines[1:21]:
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            values = stripped.split(",")
            for idx, raw in enumerate(values):
                val = raw.strip()
                col = col_names[idx] if idx < len(col_names) else str(idx)
                if val == "":
                    if col not in non_numeric:
                        non_numeric.append(col)
                    continue
                try:
                    float(val)
                except ValueError:
                    if col not in non_numeric:
                        non_numeric.append(col)

        return FileValidationResult(is_valid=True, non_numeric_columns=non_numeric)

    def _validate_hdf5_format(self) -> FileValidationResult:
        try:
            with h5py.File(self.source, "r") as f:
                keys = list(f.keys())
            if not keys:
                return FileValidationResult(is_valid=False, error="HDF5 file has no datasets")
            return FileValidationResult(is_valid=True)
        except OSError as exc:
            return FileValidationResult(is_valid=False, error=str(exc))

    def read(self, format: str) -> pl.LazyFrame:
        reader = datafile_readers.get(format)
        if not reader:
            logger.error(
                f"No reader for format {format}. Available: \n{list(datafile_readers.keys())}",
            )
            raise AndesiteUnableToReadFileError(
                f"No reader found for format '{format}'. Available: {list(datafile_readers.keys())}"
            )

        validation = self.validate_format(format)
        if not validation.is_valid:
            logger.error(
                "File '%s' failed format validation for '%s': %s",
                self.source,
                format,
                validation.error,
            )
            raise AndesiteUnableToReadFileError(
                f"File does not match {format} format: {validation.error}"
            )

        return reader(self.source)


# ── Export functions ───────────────────────────────────────────────────────────

def _to_polars(df: Union[pd.DataFrame, pl.DataFrame, dd.DataFrame]) -> pl.DataFrame:
    if isinstance(df, dd.DataFrame):
        return pl.DataFrame(df.compute())
    if isinstance(df, pd.DataFrame):
        return pl.DataFrame(df)
    return df


def dataframe_to_gslib(
    df: Union[pd.DataFrame, pl.DataFrame, dd.DataFrame],
    output_filename: str,
) -> None:
    """Export DataFrame to Geo-EAS (GSLIB) format."""
    dataframe = _to_polars(df)
    cols = dataframe.columns
    header = f"{output_filename}\n{len(cols)}\n" + "\n".join(cols)
    first_line = " ".join(cols)
    size_first_line = len(first_line.encode("utf-8"))
    size_header = len(header.encode("utf-8"))
    dataframe = dataframe.rename(
        {f"{cols[-1]}": f"{cols[-1]}" + "_" * (size_header - size_first_line + len(cols) + 1)}
    )
    for c in dataframe.columns:
        if dataframe[c].dtype in pl.NUMERIC_DTYPES:
            dataframe = dataframe.with_columns(pl.col(c).fill_null(-999).alias(c))
        else:
            dataframe = dataframe.with_columns(pl.col(c).fill_null("NONE").alias(c))
    dataframe.write_csv(output_filename, separator=" ")
    time.sleep(0.01)
    with open(output_filename, "r+", encoding="utf-8", errors="replace") as f:
        line = next(f)
        f.seek(0)
        f.write(line.replace(line, header))
    logger.debug(f"Andesite export → GSLIB: {output_filename}")


def dataframe_to_csv(
    df: Union[pd.DataFrame, pl.DataFrame, dd.DataFrame],
    output_filename: str,
) -> None:
    """Export DataFrame to CSV."""
    frame = _to_polars(df)
    frame.write_csv(output_filename)
    logger.debug(f"Andesite export → CSV: {output_filename}")


def dataframe_to_h5(
    df: Union[pd.DataFrame, pl.DataFrame, dd.DataFrame],
    output_filename: str,
    key: str = "data",
) -> None:
    """Export DataFrame to HDF5 (.h5 / .hdf5)."""
    frame = _to_polars(df)
    with h5py.File(output_filename, "w") as f:
        grp = f.create_group(key)
        for col in frame.columns:
            grp.create_dataset(col, data=frame[col].to_numpy())
    logger.debug(f"Andesite export → HDF5: {output_filename} (key={key!r})")
