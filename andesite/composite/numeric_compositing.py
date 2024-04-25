import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from icecream import ic

def one_drill_numeric(dataframe, hole_id_col: str, hole_id: str, from_col: str, to_col, var_col: str, comp_length):
    """
    Given a pandas DataFrame `dataframe`, composites the data for a single hole ID based on a compositing length `comp_length`, for a single variable

    Parameters:
    ----------
    dataframe : pd.DataFrame
        Dataset that contains information about Hole ID, head, tail drills and features asociated
    hole_id : str
        Tag that representes the Hole ID
    from_col : str
        Name of the column that has information about FROM
    to_col : str
        Name of the column that has information about TO
    var_col : str
        Name of the column that information about de target variable to composite
    comp_length : int
        length of the composite to regularized

    Returns:
    --------
    composite : pd.DataFrame
        Returns a pandas dataframe of composites for the given hole ID with columns 'FROM', 'TO', and the variable of interest.

    Example:
    -------
    >>> TODO
    """
    df = dataframe.copy()

    if var_col in df.columns.tolist():
        if not is_numeric_dtype(df[var_col]):
            raise Exception(f'The column <{var_col}> must be numeric')
    else:
        raise Exception(f'{var_col} column not found in DataFrame')

    # Extract the data for the given hole ID from the dataframe df
    df.index = df[hole_id_col]
    data_frame = df.loc[hole_id]
    # Check if the hole ID has just a single value (pandas Series) and then convert it to a numpy array
    if isinstance(data_frame, pd.Series):
        drill_from = np.array([data_frame[from_col]])
        drill_to = np.array([data_frame[to_col]])
        drill_variable = np.array([data_frame[var_col]])
    # In other case the data is converted from a column to a numpy array
    else:
        drill_from = data_frame[from_col].to_numpy()
        drill_to = data_frame[to_col].to_numpy()
        drill_variable = data_frame[var_col].to_numpy()
    # 'n_comp' is the amount of composite segments of this hole ID
    n_comp = np.ceil(drill_to[-1]/comp_length).astype(np.int32)
    # Create numpy arrays for the compositing intervals
    comp_from = np.arange(0, n_comp*comp_length, comp_length, dtype=np.float32)
    comp_to = np.arange(comp_length, n_comp*comp_length + comp_length, comp_length, dtype=np.float32)
    comp_len = np.zeros(n_comp, dtype=np.float32)
    # Variable of interest to composite
    comp_var = np.full_like(comp_len, np.nan)
    # Cumulate values to sum that integrate the composite
    comp_acum = np.zeros(n_comp, dtype=np.float32)
    # Lengths that represent portions of the composite
    lengths = np.zeros(len(drill_to), dtype=np.float32)
    # numerator of average weights of every composite interval
    cumulate_sum = np.zeros(len(drill_to), dtype=np.float32)
    for icomp in range(n_comp):
        # Start and End of this composite interval
        comp_start = comp_from[icomp]
        comp_end = comp_to[icomp]
        # Reset the lengths and the cumulate sum to 0 for every composite interval
        lengths[:] = 0
        cumulate_sum[:] = 0
        # Determine which drill intervals overlap with the composite interval with this boolean mask
        in_range = (drill_from < comp_end) & (drill_to > comp_start)
        # Calculate the overlap between each drill interval and the composite interval
        overlap = np.minimum(drill_to, comp_end) - np.maximum(drill_from, comp_start)
        # Only the possitive values are importante to the weighted average
        # We set the negative values to 0
        overlap[overlap < 0] = 0
        # Assign the associated lengths and cumulative sums to the appropriate indices for overlapping drill intervals
        lengths[in_range] = overlap[in_range]
        cumulate_sum[in_range] = overlap[in_range] * drill_variable[in_range]
        comp_len[icomp] = np.nansum(lengths)
        comp_acum[icomp] = np.nansum(cumulate_sum)
        comp_var[icomp] = np.nan if np.isnan(drill_variable[in_range]).all() else comp_acum[icomp]/comp_len[icomp]

    return pd.DataFrame({
        f'{hole_id_col}': hole_id,
        'from': comp_from,
        'to': comp_to,
        f'{var_col}': np.round(comp_var, 3)
    }, index=np.repeat(hole_id, n_comp))

def one_drill_numeric_multi(dataframe, hole_id_col: str, hole_id, from_col, to_col, var_cols, comp_length):
    """
    Given a pandas DataFrame `dataframe`, composites the data for a single hole ID based on a compositing length `comp_length`, for multiple variables

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataset that contains information about Hole ID, head, tail drills and features asociated
    hole_id : str
        Tag that representes the Hole ID
    from_col : str
        Name of the column that has information about FROM
    to_col : str
        Name of the column that has information about TO
    var_cols : list-like, 1D-array
        Name of the column that information about de target variable to composite
    comp_length : int
        length of the composite to regularized

    Returns
    -------
    composite : pd.DataFrame
        Returns a pandas dataframe of composites for the given hole ID with columns 'FROM', 'TO', and the variable of interest.

    Raises
    ------
    TypeError
        _description_
    Exception
        _description_
    Exception
        _description_
    """
    df = dataframe.copy()

    if isinstance(var_cols, str):
        raise TypeError('The parameter `var_cols` must be <List> not <str>')
    if (len(var_cols) == 1 or len(np.unique(var_cols)) == 1):
        raise Exception('The aproppiate function is `one_drill_numeric`')
    for c in var_cols:
        if c in df.columns:
            if not is_numeric_dtype(df[c]):
                raise Exception(f'The column <{c}> must be numeric')
        else:
            raise Exception(f'{c} column not found in DataFrame')

    df.index = df[hole_id_col]
    data_frame = df.loc[hole_id]
    if isinstance(data_frame, pd.Series):
        drill_from = np.array([data_frame[from_col]])
        drill_to = np.array([data_frame[to_col]])
        drill_variables = np.array([data_frame[var_cols]], dtype=np.float32)
    else:
        drill_from = data_frame[from_col].to_numpy()
        drill_to = data_frame[to_col].to_numpy()
        drill_variables = data_frame[var_cols].to_numpy()
    n_comp = np.ceil(drill_to[-1]/comp_length).astype(np.int32)
    comp_from = np.arange(0, n_comp*comp_length, comp_length, dtype=np.float32)
    comp_to = np.arange(comp_length, n_comp*comp_length + comp_length, comp_length, dtype=np.float32)
    comp_len = np.zeros(n_comp, dtype=np.float32)
    comp_var = {col: np.full_like(comp_len, np.nan) for col in var_cols}
    comp_acum = np.zeros(n_comp, dtype=np.object0)
    lengths = np.zeros(len(drill_to), dtype=np.float32)
    cumulate_sum = np.zeros((len(drill_to), len(var_cols)), dtype=np.float32)
    results = {f'{hole_id_col}': hole_id, f'from': comp_from, f'to': comp_to}
    for icomp in range(n_comp):
        comp_start = comp_from[icomp]
        comp_end = comp_to[icomp]
        lengths[:] = 0
        cumulate_sum[:] = 0
        in_range = (drill_from < comp_end) & (drill_to > comp_start)
        overlap = np.minimum(drill_to, comp_end) - np.maximum(drill_from, comp_start)
        overlap[overlap < 0] = 0
        lengths[in_range] = overlap[in_range]
        cumulate_sum[in_range] = overlap[in_range, np.newaxis] * drill_variables[in_range]
        comp_len[icomp] = np.nansum(lengths)
        comp_acum[icomp] = np.nansum(cumulate_sum, axis=0)
        for i, col in enumerate(var_cols):
            comp_var[col][icomp] = np.nan if np.isnan(drill_variables[in_range, i]).all() else comp_acum[icomp][i]/comp_len[icomp]
    results.update(comp_var)

    return pd.DataFrame(results, index=np.repeat(hole_id, n_comp))

