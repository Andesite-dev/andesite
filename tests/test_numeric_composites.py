# unit test using pytest
from os.path import join, abspath
import numpy as np
import pandas as pd
import sys
import pytest

# sys.path.append('..')
from andesite.composite.numeric_compositing import one_drill_numeric, one_drill_numeric_multi

@pytest.fixture(scope='module')
def drillholes_init():
    drillholes_path = join('data', 'parker', 'Assay.csv')
    return pd.read_csv(drillholes_path)

def test_simple_composite(drillholes_init):
    # composite of drillhole OTD002 for Cu_pct at 2 meters
    # This sample has 98.9 m, so the total number of composite samples is 50, and the min value is greater than 0
    composited_df = one_drill_numeric(drillholes_init, 'HOLEID', 'OTD002', 'SAMPFROM', 'SAMPTO', 'Cu_pct', 2)
    assert len(composited_df) == 50
    assert np.min(composited_df['Cu_pct']) >= 0

def test_check_nan_drills(drillholes_init):
    # This function checks if this hole has all is values set as null, because this entire hole has no Cu
    composited_df = one_drill_numeric(drillholes_init, 'HOLEID', 'UGD061', 'SAMPFROM', 'SAMPTO', 'Cu_pct', 2)
    assert composited_df['Cu_pct'].isnull().all() == True

def test_only_one_column_passed_list(drillholes_init):
    # Here we catch an Exception if this function is used incorrectly
    # We expect tu use this function with multiple columns, but we passed only one
    with pytest.raises(Exception, match='The aproppiate function is `one_drill_numeric`'):
        one_drill_numeric_multi(drillholes_init, 'HOLEID', 'UGD061', 'SAMPFROM', 'SAMPTO', ['Cu_pct'], 2)

def test_only_one_column_passed_str(drillholes_init):
    # This function is used incorrectly because we expect to recieve a list of columns
    # But a single columns is passed as a string value `Cu_pct`
    with pytest.raises(TypeError, match='The parameter `var_cols` must be <List> not <str>'):
        one_drill_numeric_multi(drillholes_init, 'HOLEID', 'OTD1155', 'SAMPFROM', 'SAMPTO', 'Cu_pct', 2)

def test_column_not_numeric_onedrill(drillholes_init):
    # This function is used incorrectly because a not numeric col is passed to the function
    # Only numeric columns can be passed on the 'var_cols' parameter
    with pytest.raises(Exception, match='The column <SAMPLETYPE> must be numeric'):
        one_drill_numeric(drillholes_init, 'HOLEID', 'OTD1155', 'SAMPFROM', 'SAMPTO', 'SAMPLETYPE', 2)

def test_column_not_numeric_multivar(drillholes_init):
    # This function is used incorrectly because a not numeric col is passed to the function
    # Only numeric columns can be passed on the 'var_cols' parameter
    with pytest.raises(Exception, match='The column <SAMPLETYPE> must be numeric'):
        one_drill_numeric_multi(drillholes_init, 'HOLEID', 'OTD1155', 'SAMPFROM', 'SAMPTO', ['Cu_pct', 'SAMPLETYPE'], 2)

def test_column_not_found_onedrill(drillholes_init):
    # This Exception is raised if the column passed in the `var_col` parameter is not in
    # the original dataframe `df`
    with pytest.raises(Exception, match='Cu column not found in DataFrame'):
        one_drill_numeric(drillholes_init, 'HOLEID', 'OTD1155', 'SAMPFROM', 'SAMPTO', 'Cu', 2)

def test_column_not_found_multivar(drillholes_init):
    # This Exception is raised if a column passed in the `var_cols` parameter is not in
    # the original dataframe `df`
    with pytest.raises(Exception, match='Cu column not found in DataFrame'):
        one_drill_numeric_multi(drillholes_init, 'HOLEID', 'OTD1155', 'SAMPFROM', 'SAMPTO', ['Cu_pct', 'Au_ppm', 'Cu'], 2)
