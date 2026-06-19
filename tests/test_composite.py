from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

from andesite.composite.compositing import Assay, Collar, DrillholesCampaign, Survey


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


@pytest.fixture(scope="module")
def survey():
    df = pl.DataFrame({
        "HOLEID": ["DH001"],
        "DEPTH": [0.0],
        "AZ": [0.0],
        "DIP": [-90.0],
    })
    return Survey(df, dhid="HOLEID", depth_col="DEPTH", azimuth_col="AZ", dip_col="DIP")


@pytest.fixture(scope="module")
def collar():
    df = pl.DataFrame({
        "HOLEID": ["DH001"],
        "X": [1000.0],
        "Y": [2000.0],
        "Z": [500.0],
        "LENGTH": [10.0],
    })
    return Collar(df, dhid="HOLEID", east_col="X", north_col="Y", elev_col="Z", length_col="LENGTH")


@pytest.fixture(scope="module")
def campaign(assay, survey, collar):
    return DrillholesCampaign(assay, survey, collar)


def test_composite_returns_polars(campaign):
    result = campaign.composite(comp_length=5)
    assert isinstance(result, pl.DataFrame)


def test_composite_has_coordinate_columns(campaign):
    result = campaign.composite(comp_length=5)
    assert "x" in result.columns
    assert "y" in result.columns
    assert "z" in result.columns


def test_composite_row_count(campaign):
    result = campaign.composite(comp_length=5)
    assert len(result) == 2


def test_composite_returns_pandas(campaign):
    result = campaign.composite(comp_length=5, return_dtype="pandas")
    assert isinstance(result, pd.DataFrame)
