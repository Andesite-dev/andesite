"""
First-approach integration tests for variography plotting.

Covers:
  - Experimental single variogram (runs GSLIB gamv_OpenMP.exe)
  - Experimental elipsoid (triple-direction, 3 orthogonal directions) — uses synthetic
    VariogramDatafile because the flat Cerro Blanco dataset (Cota range ~0.6 m) cannot
    produce valid output for the vertical 3rd direction.  The plotting code is fully exercised.
  - Variogram modeling (single direction)
  - VarMap polar plot  (vm.plot)
  - VarMap contour plot (vm.plot_contour)
  - VarMap heatmap plot (vm.plot_heatmap)

Skipped: Correlogram, DDH variogram.

Each test saves 3 HTML variants (Andes, Nevado dark, Nevado light) to tests/outputs/.
Module-scoped fixtures ensure the GSLIB exe runs only once per test session.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from andesite.variography.experimental import Variogram, VariogramDatafile
from andesite.variography.maps import VarMap
from andesite.variography.modeling import VariogramModeling
from andesite.visualizations.base_plotly import AndesPlotlyTheme
from andesite.visualizations.plotly_plots_nevado import apply_nevado_theme

DATA_PATH = str(Path(__file__).parent / "data" / "sondajes_cerro_blanco_final.csv")
OUTPUT_DIR = Path(__file__).parent / "outputs"

_COORDS = ["Este", "Norte", "Cota"]
_GRADE = "AuGrade"

_VARIO_PARAMS = {
    "azimuth": 0,
    "azimuth_tolerance": 90,
    "dip": 0,
    "dip_tolerance": 90,
    "lag_count": 8,
    "lag_size": 30,
    "lag_tolerance": 15,
    "horizontal_bandwidth": 999999,
    "vertical_bandwidth": 999999,
}

_MODEL_PARAMS = {
    "nugget": 0.1,
    "structures": [
        {
            "type": "Spherical",
            "sill": 0.7,
            "angles": (0, 0, 0),
            "ranges": (200, 200, 200),
        }
    ],
}

_THEMES = [("andes", None), ("nevado_dark", True), ("nevado_light", False)]


def _apply_theme(fig, is_dark, **nevado_kwargs) -> None:
    if is_dark is None:
        AndesPlotlyTheme.apply_common_layout(fig)
    else:
        apply_nevado_theme(fig, is_dark=is_dark, **nevado_kwargs)


@pytest.fixture(scope="module", autouse=True)
def ensure_output_dir():
    OUTPUT_DIR.mkdir(exist_ok=True)


@pytest.fixture(scope="module")
def single_vario_fixture():
    vario = Variogram(DATA_PATH, _COORDS, _GRADE, _VARIO_PARAMS)
    vario_file = vario.single_semivariogram()
    return vario, vario_file


@pytest.fixture(scope="module")
def elipsoid_vario_fixture(single_vario_fixture):
    """Synthetic elipsoid VariogramDatafile.

    The Cerro Blanco sample data is nearly flat (Cota range < 1 m), so the vertical
    3rd direction of a real elipsoid gamv run produces no output.  We build a realistic
    synthetic DataFrame that exercises Variogram.plot() in elipsoid mode (shape[1] > 7).
    """
    vario, _ = single_vario_fixture
    n_lags = 8
    steps = [_VARIO_PARAMS["lag_size"] * (i + 1) for i in range(n_lags)]
    pairs = [max(60 - 6 * i, 4) for i in range(n_lags)]

    df = pd.concat(
        [
            pd.DataFrame(
                {
                    "steps_dir1": steps,
                    "gamma_dir1": [0.05 + 0.07 * i for i in range(n_lags)],
                    "pairs_dir1": pairs,
                }
            ),
            pd.DataFrame(
                {
                    "steps_dir2": steps,
                    "gamma_dir2": [0.04 + 0.06 * i for i in range(n_lags)],
                    "pairs_dir2": pairs,
                }
            ),
            pd.DataFrame(
                {
                    "steps_dir3": steps,
                    "gamma_dir3": [0.06 + 0.05 * i for i in range(n_lags)],
                    "pairs_dir3": pairs,
                }
            ),
        ],
        axis=1,
    )

    params = dict(vario.metadata)
    params.update(
        {
            "vartype": "elipsoid",
            "directions": [[0, 0], [90, 0], [0, 90]],
            "norm": False,
        }
    )
    vario_file = VariogramDatafile(variogram=df, parameters=params)
    return vario, vario_file


@pytest.fixture(scope="module")
def varmap_fixture():
    vm = VarMap(
        DATA_PATH,
        _COORDS,
        _GRADE,
        plane="XY",
        n_directions=8,
        lag_count=5,
        lag_size=50,
        legend_type="continuous",
    )
    varmap_df = vm.calculate()
    return vm, varmap_df


def test_experimental_single_variogram(single_vario_fixture) -> None:
    vario, vario_file = single_vario_fixture
    df = vario_file.load()

    assert len(df.columns) >= 3, "Expected at least steps/gamma/pairs columns"

    for theme_name, is_dark in _THEMES:
        fig = vario.plot(df)
        assert fig is not None
        assert len(fig.data) >= 1
        _apply_theme(fig, is_dark, title=f"Experimental Variogram — {theme_name}")
        fig.write_html(OUTPUT_DIR / f"variogram_single_{theme_name}.html")


def test_experimental_elipsoid_variogram(elipsoid_vario_fixture) -> None:
    vario, vario_file = elipsoid_vario_fixture
    df = vario_file.load()

    assert len(df.columns) >= 9, "Expected steps/gamma/pairs for 3 directions"

    for theme_name, is_dark in _THEMES:
        fig = vario.plot(df)
        assert fig is not None
        assert len(fig.data) >= 3, "Expected one trace per orthogonal direction"
        _apply_theme(fig, is_dark, title=f"Elipsoid Variogram (3 directions) — {theme_name}")
        fig.write_html(OUTPUT_DIR / f"variogram_elipsoid_{theme_name}.html")


def test_variogram_modeling(single_vario_fixture) -> None:
    _, vario_file = single_vario_fixture

    modeler = VariogramModeling(vario_file, _MODEL_PARAMS)
    model_df = modeler.modeling()
    assert model_df is not None

    for theme_name, is_dark in _THEMES:
        fig = modeler.plot()
        assert fig is not None
        assert len(fig.data) >= 2, "Expected model line + experimental scatter"
        _apply_theme(fig, is_dark, title=f"Variogram Modeling — {theme_name}")
        fig.write_html(OUTPUT_DIR / f"variogram_modeling_{theme_name}.html")


def test_varmap_plot(varmap_fixture) -> None:
    vm, varmap_df = varmap_fixture

    for theme_name, is_dark in _THEMES:
        fig = vm.plot(varmap_df)
        assert fig is not None
        _apply_theme(fig, is_dark)
        fig.write_html(OUTPUT_DIR / f"varmap_polar_{theme_name}.html")


def test_varmap_contour(varmap_fixture) -> None:
    vm, varmap_df = varmap_fixture

    for theme_name, is_dark in _THEMES:
        fig = vm.plot_contour(varmap_df)
        assert fig is not None
        _apply_theme(fig, is_dark, title=f"VarMap Contour — {theme_name}")
        fig.write_html(OUTPUT_DIR / f"varmap_contour_{theme_name}.html")


def test_varmap_heatmap(varmap_fixture) -> None:
    vm, varmap_df = varmap_fixture

    for theme_name, is_dark in _THEMES:
        fig = vm.plot_heatmap(varmap_df)
        assert fig is not None
        _apply_theme(fig, is_dark, title=f"VarMap Heatmap — {theme_name}")
        fig.write_html(OUTPUT_DIR / f"varmap_heatmap_{theme_name}.html")
