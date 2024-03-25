
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from utils.manipulations import agg_by_categoric, find_most_relevant_cat


def one_drill_categoric(dataframe: pd.DataFrame, hole_id_col: str, hole_id: str, from_col: str, to_col: str, var_col: str, comp_length: float) -> pd.DataFrame:
    """
    Given a pandas dataframe `dataframe`, composites the data for a single hole ID `hole_id` based on a compositing length `comp_length`.

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
	var_col : str
		Name of the column that has information about the target variable to composite
	comp_length : int-float
		length of the composite to regularized

    Returns
    -------
    pd.DataFrame
        Returns a pandas dataframe of composites for the given hole ID with columns 'FROM', 'TO', and the variable of interest.

    Examples
    -------
    >>> TODO
    """
    df = dataframe.copy()

    if is_numeric_dtype(df[var_col]):
        raise TypeError('var_col must be categorical')
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
    # Composite length
    comp_len = np.zeros(n_comp, dtype=np.float32)
    # Variable of interest to composite
    comp_var = np.full_like(comp_len, np.nan, dtype=np.object0)
    # Cumulate values to sum that integrate the composite
    comp_acum = np.zeros(n_comp, dtype=np.float32)
    # Lengths that represent porcions of the composite
    lengths = np.zeros_like(drill_to, dtype=np.float32)
    vals = np.array([lengths, lengths], dtype=np.object0).reshape(-1, 2)

    for icomp in range(n_comp):
        # Start and End of this composite interval
        comp_start = comp_from[icomp]
        comp_end = comp_to[icomp]
        # Reset the lengths and the cumulate sum to 0 for every composite interval
        vals[:] = 0
        # Determine which drill intervals overlap with the composite interval with this boolean mask
        in_range = (drill_from < comp_end) & (drill_to > comp_start)
        # Determine the length of each drill interval
        drill_lenght = drill_to - drill_from
        # Calculate the overlap between each drill interval and the composite interval
        overlap = np.minimum(drill_to, comp_end) - np.maximum(drill_from, comp_start)
        vals[in_range] = np.stack((drill_variable[in_range], overlap[in_range].astype(np.object0)), axis=-1)
        arr = agg_by_categoric(vals[in_range])
        comp_var[icomp] = find_most_relevant_cat(arr)

    return pd.DataFrame({
        f'{hole_id_col}': hole_id,
        f'from': comp_from,
        f'to': comp_to,
        f'{var_col}': comp_var
    }, index=np.repeat(hole_id, n_comp))


def one_drill_categoric_multi(dataframe, hole_id_col: str, hole_id, from_col, to_col, var_cols, comp_length):
    """
    Given a pandas dataframe `dataframe`, composites the data for a single hole ID `hole_id` based on a compositing length `comp_length`.

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
	var_cols : list
		Name of the columns that has information about the target variables to composite
	comp_length : int-float
		length of the composite to regularized

    Returns
    -------
    pd.DataFrame
        Returns a pandas dataframe of composites for the given hole ID with columns 'FROM', 'TO', and the variable of interest.

    Examples
    -------
    >>> TODO
    """
    df = dataframe.copy()
    if len(var_cols) == 1 or len(np.unique(var_cols)) == 1:
        assert Exception('The aproppiate function is `one_drill_categoric`')
    df.index = df[hole_id_col]
    data_frame = df.loc[hole_id]
    if isinstance(data_frame, pd.Series):
        drill_from = np.array([data_frame[from_col]])
        drill_to = np.array([data_frame[to_col]])
        drill_variables = np.array([[data_frame[col]] for col in var_cols])
    else:
        drill_from = data_frame[from_col].to_numpy()
        drill_to = data_frame[to_col].to_numpy()
        drill_variables = np.array([data_frame[col].to_numpy() for col in var_cols])
    n_comp = np.ceil(drill_to[-1] / comp_length).astype(np.int32)
    comp_from = np.arange(0, n_comp*comp_length, comp_length, dtype=np.float32)
    comp_to = np.arange(comp_length, n_comp*comp_length + comp_length, comp_length, dtype=np.float32)
    comp_vars = np.full((len(var_cols), n_comp), np.nan, dtype=np.object0)
    for icomp in range(n_comp):
        comp_start = comp_from[icomp]
        comp_end = comp_to[icomp]
        in_range = (drill_from < comp_end) & (drill_to > comp_start)
        overlap = np.minimum(drill_to, comp_end) - np.maximum(drill_from, comp_start)
        vals = np.zeros((len(var_cols)), dtype=np.object0)
        for i in range(len(var_cols)):
          vals[i] = np.stack([drill_variables[i, in_range], overlap[in_range]], axis=-1)
          arr = agg_by_categoric(vals[i])
          comp_vars[i, icomp] = np.nan if len(arr) == 0 else arr[np.argmax(arr[:, 1]), 0]

    data_dict = {f'{hole_id_col}': hole_id, f'from': comp_from, f'to': comp_to}
    for i, col in enumerate(var_cols):
        data_dict[col] = comp_vars[i]
    return pd.DataFrame(data_dict, index=np.repeat(hole_id, n_comp))