from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from andesite.composite.compositing import Assay


@pytest.fixture(scope="module")
def assay():
    df = pl.DataFrame({
        "HOLEID": ["DH001"] * 5,
        "FROM": [0.0, 2.0, 4.0, 6.0, 8.0],
        "TO": [2.0, 4.0, 6.0, 8.0, 10.0],
        "CuGrade": [0.5, 0.8, 1.2, 0.3, 0.9],
        "Rockcode": [1, 1, 2, 2, 1],
    })
    return Assay(df, dhid="HOLEID", from_col="FROM", to_col="TO", target_variables=["CuGrade", "Rockcode"])


def test_numeric_regularization_shape(assay):
    result = assay.numeric_multivar_regularization("DH001", comp_length=5)
    assert len(result) == 2


def test_numeric_regularization_columns(assay):
    result = assay.numeric_multivar_regularization("DH001", comp_length=5)
    assert "FROM" in result.columns
    assert "TO" in result.columns
    assert "CuGrade" in result.columns


def test_numeric_regularization_weighted_mean(assay):
    result = assay.numeric_multivar_regularization("DH001", comp_length=5)
    assert result["CuGrade"][0] == pytest.approx(0.76, abs=1e-3)
    assert result["CuGrade"][1] == pytest.approx(0.72, abs=1e-3)


def test_numeric_regularization_no_nulls_in_fully_covered_interval(assay):
    result = assay.numeric_multivar_regularization("DH001", comp_length=5)
    assert not np.isnan(result["CuGrade"][0])
    assert not np.isnan(result["CuGrade"][1])
