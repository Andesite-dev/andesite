from os import PathLike
import os
import tempfile
import time
from typing import Union, Sequence
from itertools import islice
import dask.dataframe as dd
import polars as pl
import numpy as np
import pandas as pd

def grab_n_cols(
    datafile: Union[str, "PathLike[str]"]
) -> int:
    """Reads the number of columns from the second line of a GSLIB file.

    Parameters
    ----------
    datafile : PathLike or str
        Path to the input data file.

    Returns
    -------
    int
        Number of columns in the file.
    """
    # TODO examples
    with open(datafile) as file:
        for line in islice(file, 1, 2):
            n_cols = line.strip().split()
            return np.int32(n_cols[0])
        
def grab_col_names(
    datafile: Union[str, "PathLike[str]"]
) -> Sequence[str]:
    """Reads the number of columns from the second line of a GSLIB file.

    Parameters
    ----------
    datafile : str
        Path to the input data file.

    Returns
    -------
    int
        Number of columns in the file.
    """
    # TODO examples
    n_cols = grab_n_cols(datafile)
    cols = []
    with open(datafile) as file:
        for line in islice(file, 2, n_cols + 2):
            cols.append(line.strip())
    return cols

def read_file_from_gslib(
    datafile: Union[str, "PathLike[str]"]
):
    """Reads a file in GSLIB format into a Dask DataFrame.

    Parameters
    ----------
    datafile : str
        Path to the input data file.
    to_csv : bool, optional
        If True, converts the DataFrame to a CSV file (default is False).

    Returns
    -------
    dask.dataframe.DataFrame or None
        The Dask DataFrame representing the GSLIB file if `to_csv` is False.
        If `to_csv` is True, writes a CSV file and returns None.
    """
    # TODO examples
    # Collect the columns into a list
    col_names = grab_col_names(datafile)
    # Shrink the column names
    columns = [c.strip().replace(' ', '') for c in col_names]
    n_cols = len(col_names)
    dataframe = dd.read_csv(datafile, delimiter=r"\s+", skiprows=n_cols+2, names=columns)
    return dataframe

def dataframe_to_gslib(df: Union[pd.DataFrame, pl.DataFrame], output_filename: str):
    """
    Converts a Pandas or Polars DataFrame to a GSLIB-formatted file.

    Parameters
    ----------
    df : Union[pd.DataFrame, pl.DataFrame]
        Input DataFrame (Pandas or Polars) to be converted.
    output_filename : str
        Output file name for the GSLIB-formatted file.

    Returns
    -------
    None
        This function does not return a value.

    Notes
    -----
    The GSLIB-formatted file will have a header containing the output filename,
    the number of columns, and the column names. Numeric columns with NaN values
    will be replaced with -999, and non-numeric columns with NaN values will be
    replaced with 'NONE'.
    """
    # TODO examples
    dataframe = pl.DataFrame(df)
    cols = dataframe.columns
    header = f"{output_filename}\n{len(cols)}\n" + "\n".join(cols)
    first_line = " ".join(cols)
    size_first_line = len(first_line.encode("utf-8"))
    size_header = len(header.encode("utf-8"))
    dataframe = dataframe.rename({f'{cols[-1]}': f'{cols[-1]}' + '_'*(size_header - size_first_line + len(cols) + 1)})
    for c in dataframe.columns:
        if dataframe[c].dtype() in pl.NUMERIC_DTYPES:
            dataframe = dataframe.with_columns(pl.col(c).fill_null(-999).alias(c))
        else:
            dataframe = dataframe.with_columns(pl.col(c).fill_null('NONE').alias(c))
    dataframe.write_csv(output_filename, separator=' ')
    time.sleep(0.01)
    with open(output_filename, 'r+') as f:
        line = next(f)
        f.seek(0)
        f.write(line.replace(line, header))
    return

def check_nan_values(dataframe):
    for c in dataframe.columns:
        null_values_count = dataframe[dataframe[c].isna()].shape[0]
        percentage_null = null_values_count*100/(dataframe.shape[0])
        print(f'In the column {c} there are {null_values_count} null values ({percentage_null:.2f})%')

def grab_index_coordinates(drillholes_datafile, coord_names):
    dframe = read_file_from_gslib(drillholes_datafile)
    variables = dframe.columns
    ix = variables.get_loc(coord_names[0]) + 1
    iy = variables.get_loc(coord_names[1]) + 1
    iz = variables.get_loc(coord_names[2]) + 1
    return ix, iy, iz

def grab_index_target(drillholes_datafile, target):
    dframe = read_file_from_gslib(drillholes_datafile)
    variables = dframe.columns
    igrade = variables.get_loc(target) + 1
    return igrade

def remove_categorical_columns(filepath):
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith('.hdf5'):
        df = pd.read_hdf(filepath)
    else:
        df = read_file_from_gslib(filepath)
    not_obj_df = df.select_dtypes(exclude=['object'])
    return not_obj_df

def transform_datafile_to_gslib(filepath):
    """Transform a datafile in diferent formats, into a gslib file, save it on a Temp file

    Parameters
    ----------
    filepath : str
        Datafile path to be transformed into GSLIB format

    Returns
    -------
    str
        After transform the datafile into GSLIB format, it return the path where this file was saved
    """
    base_filename = os.path.splitext(os.path.basename(filepath))[0]
    temp_gslib_datafile_path = os.path.join(tempfile.gettempdir(), f'{base_filename}.dat')
    df = remove_categorical_columns(filepath)
    dataframe_to_gslib(df, temp_gslib_datafile_path)
    return temp_gslib_datafile_path