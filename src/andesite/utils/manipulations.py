from typing import Sequence, Union
import re
import numpy as np
import pandas as pd
from utils.log import andes_logger
from itertools import islice
import dask.dataframe as dd
from os import PathLike

def globalize_backslashes(text):
    """
    Converts all backslashes into forward slashes in a given string.
    """
    return text.replace('\\', '/')

def find_most_relevant_cat(
    arr: list
) -> str:
    """
    Find the most relevant categorical value from a list, respect to the length
    representation

    Parameters
    ----------
    arr : list
        list-like or array-like of shape (n, 2)

    Returns
    -------
    str
        String value of the most representative feature on this array

    Example
    -------
    >>> exm_list = np.array([['A', 7],
                             ['B', 3]])
    >>> find_most_relevant_cat(exm_list)
    >>> A
    """
    # Returns NaN if there is no values inside the array
    if len(arr) == 0:
        return np.nan
    # Replace the type of the data passed, to accept strings and float values
    array = np.array(arr, dtype=np.object0)
    # Find the major value in the second column and store the first column
    result = array[np.argmax(array[:, 1]), 0]
    return result


def agg_by_categoric(
    data: np.array
) -> np.array:
    """
    This function takes the lengths of the diferent features inside an array
    and add the lengths of the duplicate values

    Parameters
    ----------
    data : np.array
        list-like or array-like of shape (n, 2)

    Returns
    -------
    np.array
        list-like or array-like of shape (n, 2) with no duplicated values on
        the first column
    Example
    -------
    >>> exm_array = np.array([['A', 7],
                             ['B', 3],
                             ['B', 12],
                             ['A', 4]])
    >>> agg_by_categoric(exm_array)
    >>> [['A', 11],
         ['B', 15]]
    """
    # First we make sure that the second column it was not passed as type <str>
    data = data.astype(np.object0)
    data[:, 1] = np.float32(data[:, 1])

    # Replace np.nan with 'None' value if any
    for i in range(data.shape[0]):
        if isinstance(data[i, 0], float) and np.isnan(data[i, 0]):
            data[i, 0] = 'None'

    # Check if there is there is duplicate values
    unique_categoric = np.unique(data[:, 0])
    if len(unique_categoric) == data.shape[0]:
        return data

    # Create an empty array with the output shape, to fill the data
    data_transformed = np.zeros((unique_categoric.shape[0], 2), dtype=np.object0)
    for i, val in enumerate(unique_categoric):
        # Take only the data containing "val"
        mask = (data[:, 0] == val)
        # Put in the first column the feature "val"
        data_transformed[i, 0] = val
        # Insert in the second column the sum of the lengths
        data_transformed[i, 1] = np.sum(data[mask, 1])
    # Return the data filled with the non-duplicate values
    return data_transformed

def find_pattern_on_list(
    items: Union[Sequence[str], str], 
    word_keys: Sequence[str]
) -> str:
    """Finds if a variable match a certain bag of word keys. Most common use for
       finding coordinate columns on a dataset

    Parameters
    ----------
    items : str or list
        items where match needs to be found.
    word_key : list
        bag of words to match with the variable

    Returns
    -------
    str
        String variable where the match was found, also can be empty string

    Example
    -------
    variable_columns = [xcoord, ycoord, zcoord, cut, density]
    possible_x_columns = [x, xm, y_center, x_dim, midx]
    match = find_pattern_on_list(variable columns, possible_x_columns)
    print(match)
    >>> xcoord
    """
    if isinstance(word_keys, str):
        word_keys = [word_keys]
    if isinstance(items, str):
        items = [items]
    patterns = []
    for word_key in word_keys:
        # If the word_key is a single character, match it exactly (case insensitive)
        if len(word_key) == 1:
            patterns.append(r'(?i)\b{}\b'.format(re.escape(word_key)))
        else:
            # If the word_key has more than one character, create the pattern as before
            key = ''
            for w in word_key:
                key += f"[{w.upper()}{w.lower()}]"
            patterns.append(r'(?i){}'.format(key))

    matching_items = []
    for pattern in patterns:
        for string in items:
            if re.search(pattern, string):
                matching_items.append(string)

    if len(matching_items) > 1:
        andes_logger.debug(f'Warning: More than one item matches the pattern \'{word_keys}\'')
        andes_logger.debug(f'Conflicts with {matching_items}')
    return matching_items[0] if len(matching_items) > 0 else ""