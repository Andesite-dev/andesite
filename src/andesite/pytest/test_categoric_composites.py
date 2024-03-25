# unit test using pytest
from os.path import join, abspath
import numpy as np
import shutil
import pandas as pd
import sys
import pytest
# Uncomment this line if you have issues with andes module
sys.path.append('..')


from composite.categoric_compositing import one_drill_categoric

@pytest.fixture(scope='module')
def drillholes_init():
    drillholes_path = join('data', 'parker', 'Assay.csv')
    return pd.read_csv(drillholes_path)

def column_not_categoric_onedrill(drillholes_init):
    # Raise and Exception if the columns passed in the parameter var_col is not
    # categoric (not <str>)
    with pytest.raises(Exception, match='var_col must be categorical'):
        one_drill_categoric(drillholes_init, 'OTD1155', 'SAMPFROM', 'SAMPTO', 'Cu_pct', 2)