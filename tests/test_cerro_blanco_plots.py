"""
Focused plots for sondajes_cerro_blanco_final.csv.

Generates 6 HTML files in tests/outputs/ (Nevado light theme, 425×360 px):
  cb_histogram_UG1_light.html
  cb_cpplt_UG1_light.html
  cb_boxplot_UG1_light.html
  cb_histogram_NS1_light.html
  cb_cpplt_NS1_light.html
  cb_boxplot_NS1_light.html
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
from scipy.stats import gaussian_kde

from andesite.visualizations.base_plotly import NevadoPlotlyTheme
from andesite.visualizations.cpplt_plotly import apply_cpplt_layout, build_cpplt_figure
from andesite.visualizations.histogram_plotly import apply_histogram_layout, build_histogram_figure
from andesite.visualizations.local_props_plotly import apply_local_props_layout, build_local_props_figure

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "outputs"

_NULL_THRESHOLD = -98.0
_THEME = NevadoPlotlyTheme.light
_WIDTH = 425
_HEIGHT = 360
_CONFIG = {"displayModeBar": False, "scrollZoom": True, "displaylogo": False}


@pytest.fixture(scope="module", autouse=True)
def ensure_output_dir():
    OUTPUT_DIR.mkdir(exist_ok=True)


@pytest.fixture(scope="module")
def df() -> pl.DataFrame:
    raw = pl.read_csv(DATA_DIR / "sondajes_cerro_blanco_final.csv")
    numeric_cols = [
        c for c in raw.columns
        if raw[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
    ]
    return raw.with_columns(
        [
            pl.when(pl.col(c) <= _NULL_THRESHOLD).then(None).otherwise(pl.col(c)).alias(c)
            for c in numeric_cols
        ]
    )


def _vals(df: pl.DataFrame, col: str) -> np.ndarray:
    return df[col].drop_nulls().to_numpy().astype(float)


def _run_plots(df: pl.DataFrame, col: str, label: str) -> None:
    vals = _vals(df, col)
    assert len(vals) > 5, f"Too few valid values for {col}"

    # --- Histogram ---
    counts, edges = np.histogram(vals, bins=15)
    hist_data = [
        (int(c), float(lo), float(hi))
        for c, lo, hi in zip(counts, edges[:-1], edges[1:])
    ]
    tables = [pd.DataFrame({"min": [float(vals.min())], "max": [float(vals.max())]})]
    x_kde = np.linspace(float(vals.min()), float(vals.max()), 200)
    kde_result = (x_kde, gaussian_kde(vals)(x_kde))
    fig = build_histogram_figure(hist_data, tables, None, kde_result, theme=_THEME)
    assert fig is not None
    _THEME.apply_common_layout(fig, width=_WIDTH, height=_HEIGHT)
    apply_histogram_layout(fig, plot_title="", variable=col)
    fig.write_html(str(OUTPUT_DIR / f"cb_histogram_{label}_light.html"), config=_CONFIG)

    # --- Cumulative Frequency ---
    sorted_vals = np.sort(vals)
    probs = np.linspace(0, 100, len(sorted_vals))
    series_list = [(list(zip(sorted_vals.tolist(), probs.tolist())), col)]
    fig = build_cpplt_figure(series_list, theme=_THEME)
    assert fig is not None
    apply_cpplt_layout(fig, plot_title="", x_title=col)
    _THEME.apply_common_layout(fig, width=_WIDTH, height=_HEIGHT)
    fig.write_html(str(OUTPUT_DIR / f"cb_cpplt_{label}_light.html"), config=_CONFIG)

    # --- Boxplot ---
    results = ([col], [vals])
    fig = build_local_props_figure(results, theme=_THEME)
    assert fig is not None
    apply_local_props_layout(fig, plot_title="", variable=col)
    _THEME.apply_common_layout(fig, width=_WIDTH, height=_HEIGHT)
    fig.write_html(str(OUTPUT_DIR / f"cb_boxplot_{label}_light.html"), config=_CONFIG)


def test_cerro_blanco_au_ug1(df: pl.DataFrame) -> None:
    _run_plots(df, "AuGrade-UG1", "UG1")


def test_cerro_blanco_au_ns1(df: pl.DataFrame) -> None:
    _run_plots(df, "AuGrade-NS-1", "NS1")
