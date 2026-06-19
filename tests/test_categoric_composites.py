from __future__ import annotations

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


def test_categoric_regularization_shape(assay):
    result = assay.categoric_multivar_regularization("DH001", comp_length=5)
    assert len(result) == 2


def test_categoric_regularization_columns(assay):
    result = assay.categoric_multivar_regularization("DH001", comp_length=5)
    assert "FROM" in result.columns
    assert "TO" in result.columns
    assert "Rockcode" in result.columns


def test_categoric_majority_vote(assay):
    result = assay.categoric_multivar_regularization("DH001", comp_length=5)
    assert result["Rockcode"][0] == "1"
    assert result["Rockcode"][1] == "2"
