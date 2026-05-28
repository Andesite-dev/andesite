"""
Example-based visualization tests using samples_cerro_blanco_au.csv.

One test per plot type. Each test produces 3 HTML files in tests/outputs/:
  - <name>_andes.html       — Andes palette + AndesPlotlyTheme layout
  - <name>_nevado_dark.html — Nevado dark palette + dark layout
  - <name>_nevado_light.html — Nevado light palette + light layout

Dark HTML files embed a dark body background so they render visibly dark
in a standalone browser (the Nevado theme uses transparent backgrounds
designed for web-app CSS, so we wrap the HTML here for testing).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import pytest
from scipy.stats import gaussian_kde

from andesite.visualizations.base_plotly import AndesPlotlyTheme, NevadoPlotlyTheme
from andesite.visualizations.barchart_plotly import (
    apply_barchart_layout,
    build_barchart_figure,
)
from andesite.visualizations.cell_declustering_plotly import (
    apply_cell_declustering_layout,
    build_cell_declustering_figure,
)
from andesite.visualizations.cond_means_plotly import (
    apply_cond_means_layout,
    build_cond_means_figure,
)
from andesite.visualizations.contact_analysis_plotly import (
    apply_contact_analysis_layout,
    build_contact_analysis_figure,
)
from andesite.visualizations.cpplt_plotly import apply_cpplt_layout, build_cpplt_figure
from andesite.visualizations.correlation_matrix_plotly import (
    apply_correlation_layout,
    build_correlation_figure,
)
from andesite.visualizations.ddh_variogram_plotly import (
    apply_ddh_variogram_layout,
    build_ddh_variogram_figure,
)
from andesite.visualizations.histogram_plotly import (
    apply_histogram_layout,
    build_histogram_figure,
)
from andesite.visualizations.local_props_plotly import (
    apply_local_props_layout,
    build_local_props_figure,
)
from andesite.visualizations.pp_plotly import apply_pp_layout, build_pp_figure
from andesite.visualizations.proportional_effect_plotly import (
    apply_proportional_effect_layout,
    build_proportional_effect_figure,
)
from andesite.visualizations.prop_grade_curve_plotly import (
    apply_prop_grade_curve_layout,
    build_prop_grade_curve_figure,
)
from andesite.visualizations.qq_plotly import apply_qq_layout, build_qq_figure
from andesite.visualizations.scatter_mean_std_plotly import (
    apply_scatter_mean_std_layout,
    build_scatter_mean_std_figure,
)
from andesite.visualizations.scatter_plotly import (
    apply_same_scale_scatter_layout,
    apply_scatter_layout,
    build_same_scale_scatter_figure,
    build_scatter_figure,
)
from andesite.visualizations.stacked_histogram_plotly import (
    apply_category_histogram_layout,
    build_category_histogram_figure,
)
from andesite.visualizations.swath_plotly import apply_swath_layout, build_swath_figure
from andesite.visualizations.ton_grade_curve_plotly import (
    apply_ton_grade_curve_layout,
    build_ton_grade_curve_figure,
)
from andesite.visualizations.transfer_function_plotly import (
    apply_transfer_function_layout,
    build_transfer_function_figure,
)
from andesite.visualizations.variogram_plotly import (
    apply_variogram_layout,
    build_lmc_figure,
    build_lmc_figures,
    build_variogram_figure,
)
from andesite.visualizations.violin_plotly import apply_violin_layout, build_violin_figure

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "outputs"

_NULL_THRESHOLD = -90
_THEMES = [
    ("andes",        AndesPlotlyTheme),
    ("nevado_dark",  NevadoPlotlyTheme.dark),
    ("nevado_light", NevadoPlotlyTheme.light),
]


def write_themed_html(fig: go.Figure, path: Path, theme_name: str) -> None:
    """Write HTML, injecting a dark body background for dark-theme outputs."""
    if "dark" in theme_name:
        html = fig.to_html(full_html=True, include_plotlyjs="cdn")
        html = html.replace("<body>", '<body style="background-color:#1e1e2e;margin:0">')
        path.write_text(html, encoding="utf-8")
    else:
        fig.write_html(str(path))


@pytest.fixture(scope="module", autouse=True)
def ensure_output_dir():
    OUTPUT_DIR.mkdir(exist_ok=True)


@pytest.fixture(scope="module")
def df() -> pl.DataFrame:
    raw = pl.read_csv(DATA_DIR / "samples_cerro_blanco_au.csv")
    numeric_cols = [
        c for c in raw.columns
        if raw[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
    ]
    return raw.with_columns(
        [
            pl.when(pl.col(c) < _NULL_THRESHOLD).then(None).otherwise(pl.col(c)).alias(c)
            for c in numeric_cols
        ]
    )


# ---------------------------------------------------------------------------
# histogram
# ---------------------------------------------------------------------------

def test_histogram(df: pl.DataFrame) -> None:
    values = df["AuGrade"].drop_nulls().to_numpy()
    counts, edges = np.histogram(values, bins=15)
    hist_data = [
        (int(c), float(lo), float(hi))
        for c, lo, hi in zip(counts, edges[:-1], edges[1:])
    ]
    tables = [pd.DataFrame({"min": [float(values.min())], "max": [float(values.max())]})]
    x_kde = np.linspace(float(values.min()), float(values.max()), 200)
    kde_result = (x_kde, gaussian_kde(values)(x_kde))

    for theme_name, theme_cls in _THEMES:
        fig = build_histogram_figure(hist_data, tables, None, kde_result, plot_title="Au Grade", theme=theme_cls)
        assert fig is not None
        theme_cls.apply_common_layout(fig)
        apply_histogram_layout(fig, plot_title="Au Grade — Cerro Blanco", variable="Au Grade (g/t)")
        write_themed_html(fig, OUTPUT_DIR / f"histogram_{theme_name}.html", theme_name)
        assert len(fig.data) >= 1


# ---------------------------------------------------------------------------
# scatter
# ---------------------------------------------------------------------------

def test_scatter(df: pl.DataFrame) -> None:
    subset = df.select(["CuGrade", "AuGrade"]).drop_nulls()
    points = np.column_stack([subset["CuGrade"].to_numpy(), subset["AuGrade"].to_numpy()])
    results = (None, (points, None))

    for theme_name, theme_cls in _THEMES:
        fig = build_scatter_figure(results, x_title="Cu Grade (%)", y_title="Au Grade (g/t)", plot_title="Cu vs Au", theme=theme_cls)
        assert fig is not None
        apply_scatter_layout(fig, plot_title="Cu vs Au — Cerro Blanco", x_title="Cu Grade (%)", y_title="Au Grade (g/t)")
        theme_cls.apply_common_layout(fig)
        write_themed_html(fig, OUTPUT_DIR / f"scatter_{theme_name}.html", theme_name)
        assert len(fig.data) >= 1


def test_scatter_categorical(df: pl.DataFrame) -> None:
    top_rocks = df["Rockcode"].drop_nulls().unique().sort().to_list()[:4]
    per_cat: dict = {}
    for rock in top_rocks:
        sub = df.filter(pl.col("Rockcode") == rock).select(["CuGrade", "AuGrade"]).drop_nulls()
        if len(sub) > 1:
            per_cat[str(int(rock))] = (sub["CuGrade"].to_numpy(), sub["AuGrade"].to_numpy())

    cat_data = {"__category_mode__": True, "per_cat": per_cat}
    results = (None, cat_data)

    for theme_name, theme_cls in _THEMES:
        fig = build_scatter_figure(
            results,
            x_title="Cu Grade (%)",
            y_title="Au Grade (g/t)",
            category_name="Rockcode",
            theme=theme_cls,
        )
        assert fig is not None
        apply_scatter_layout(fig, plot_title="Cu vs Au by Rockcode", x_title="Cu Grade (%)", y_title="Au Grade (g/t)")
        theme_cls.apply_common_layout(fig)
        write_themed_html(fig, OUTPUT_DIR / f"scatter_categorical_{theme_name}.html", theme_name)
        assert len(fig.data) >= 1


def test_same_scale_scatter(df: pl.DataFrame) -> None:
    subset = df.select(["CuGrade", "AuGrade"]).drop_nulls()
    points = np.column_stack([subset["CuGrade"].to_numpy(), subset["AuGrade"].to_numpy()])
    results = (None, (points, None))

    for theme_name, theme_cls in _THEMES:
        fig = build_same_scale_scatter_figure(results, x_title="Cu Grade (%)", y_title="Au Grade (g/t)", theme=theme_cls)
        assert fig is not None
        apply_same_scale_scatter_layout(fig, plot_title="Cu vs Au (same scale)", x_title="Cu Grade (%)", y_title="Au Grade (g/t)")
        theme_cls.apply_common_layout(fig)
        write_themed_html(fig, OUTPUT_DIR / f"same_scale_scatter_{theme_name}.html", theme_name)
        assert len(fig.data) >= 1


# ---------------------------------------------------------------------------
# violin
# ---------------------------------------------------------------------------

def test_violin(df: pl.DataFrame) -> None:
    top_rocks = df["Rockcode"].drop_nulls().unique().sort().to_list()[:4]
    groups: dict = {}
    for rock in top_rocks:
        vals = df.filter(pl.col("Rockcode") == rock)["AuGrade"].drop_nulls().to_numpy()
        if len(vals) > 2:
            groups[str(int(rock))] = vals

    assert groups, "No groups built for violin test"

    for theme_name, theme_cls in _THEMES:
        fig = build_violin_figure(groups, box_visible=True, points="outliers", theme=theme_cls)
        assert fig is not None
        apply_violin_layout(fig, plot_title="Au Grade by Rockcode", variable="Au Grade (g/t)", categories="Rockcode")
        theme_cls.apply_common_layout(fig)
        write_themed_html(fig, OUTPUT_DIR / f"violin_{theme_name}.html", theme_name)
        assert len(fig.data) >= 1


# ---------------------------------------------------------------------------
# stacked histogram
# ---------------------------------------------------------------------------

def test_stacked_histogram(df: pl.DataFrame) -> None:
    top_rocks = df["Rockcode"].drop_nulls().unique().sort().to_list()[:3]
    all_vals = df["AuGrade"].drop_nulls().to_numpy()
    _, bin_edges = np.histogram(all_vals, bins=12)
    x_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = float(bin_edges[1] - bin_edges[0])

    per_cat: dict = {}
    kde_data: dict = {}
    for rock in top_rocks:
        vals = df.filter(pl.col("Rockcode") == rock)["AuGrade"].drop_nulls().to_numpy()
        if len(vals) < 3:
            continue
        counts, _ = np.histogram(vals, bins=bin_edges)
        per_cat[str(int(rock))] = counts
        x_kde = np.linspace(float(bin_edges[0]), float(bin_edges[-1]), 200)
        kde_data[str(int(rock))] = (x_kde, gaussian_kde(vals)(x_kde) * float(np.sum(counts)) * bin_width)

    assert per_cat, "No categories built for stacked histogram test"

    for theme_name, theme_cls in _THEMES:
        fig = build_category_histogram_figure(bin_edges, x_centers, per_cat, kde_data, "AuGrade", "Rockcode", theme=theme_cls)
        assert fig is not None
        apply_category_histogram_layout(fig, plot_title="Au Grade by Rockcode", variable="Au Grade (g/t)")
        theme_cls.apply_common_layout(fig)
        write_themed_html(fig, OUTPUT_DIR / f"stacked_histogram_{theme_name}.html", theme_name)
        assert len(fig.data) >= 1


# ---------------------------------------------------------------------------
# correlation matrix
# ---------------------------------------------------------------------------

def test_correlation_matrix(df: pl.DataFrame) -> None:
    cols = ["CuGrade", "AuGrade", "Cu-NS"]
    pandas_df = df.select(cols).to_pandas()
    corr_matrix = pandas_df.corr()

    for theme_name, theme_cls in _THEMES:
        fig = build_correlation_figure(corr_matrix, theme=theme_cls)
        assert fig is not None
        apply_correlation_layout(fig)
        theme_cls.apply_common_layout(fig)
        write_themed_html(fig, OUTPUT_DIR / f"correlation_matrix_{theme_name}.html", theme_name)
        assert len(fig.data) >= 1


# ---------------------------------------------------------------------------
# scatter mean/std
# ---------------------------------------------------------------------------

def test_scatter_mean_std(df: pl.DataFrame) -> None:
    top_rocks = df["Rockcode"].drop_nulls().unique().sort().to_list()[:5]
    results = []
    for rock in top_rocks:
        vals = df.filter(pl.col("Rockcode") == rock)["AuGrade"].drop_nulls().to_numpy()
        if len(vals) > 2:
            results.append(((float(np.mean(vals)), float(np.std(vals))), str(int(rock))))

    assert results, "No results built for scatter_mean_std test"

    for theme_name, theme_cls in _THEMES:
        fig = build_scatter_mean_std_figure(results, category_name="Rockcode", theme=theme_cls)
        assert fig is not None
        apply_scatter_mean_std_layout(
            fig,
            plot_title="Mean vs Std Dev of Au Grade by Rockcode",
            x_title="Mean (g/t)",
            y_title="Std Dev (g/t)",
        )
        theme_cls.apply_common_layout(fig)
        write_themed_html(fig, OUTPUT_DIR / f"scatter_mean_std_{theme_name}.html", theme_name)
        assert len(fig.data) >= 1


# ---------------------------------------------------------------------------
# barchart
# ---------------------------------------------------------------------------

def test_barchart(df: pl.DataFrame) -> None:
    top_rocks = df["Rockcode"].drop_nulls().unique().sort().to_list()[:6]
    categories = [str(int(r)) for r in top_rocks]
    values = [int(df.filter(pl.col("Rockcode") == r).height) for r in top_rocks]

    for theme_name, theme_cls in _THEMES:
        fig = build_barchart_figure(categories, values, theme=theme_cls)
        assert fig is not None
        apply_barchart_layout(fig, plot_title="Sample count by Rockcode", variable="Rockcode")
        theme_cls.apply_common_layout(fig)
        write_themed_html(fig, OUTPUT_DIR / f"barchart_{theme_name}.html", theme_name)
        assert len(fig.data) >= 1


# ---------------------------------------------------------------------------
# cell declustering
# ---------------------------------------------------------------------------

def test_cell_declustering() -> None:
    cell_sizes = np.linspace(10, 100, 15)
    means = 2.5 - 0.8 * np.exp(-cell_sizes / 40) + np.random.default_rng(0).normal(0, 0.05, 15)
    pairs = list(zip(cell_sizes.tolist(), means.tolist()))

    for theme_name, theme_cls in _THEMES:
        fig = build_cell_declustering_figure(pairs, theme=theme_cls)
        assert fig is not None
        apply_cell_declustering_layout(fig, plot_title="Cell Declustering — Au Grade", x_title="Cell size (m)", y_title="Declustered mean (g/t)")
        theme_cls.apply_common_layout(fig)
        write_themed_html(fig, OUTPUT_DIR / f"cell_declustering_{theme_name}.html", theme_name)
        assert len(fig.data) >= 1


# ---------------------------------------------------------------------------
# conditional means
# ---------------------------------------------------------------------------

def test_cond_means(df: pl.DataFrame) -> None:
    subset = df.select(["CuGrade", "AuGrade"]).drop_nulls()
    xs = subset["CuGrade"].to_numpy()
    ys = subset["AuGrade"].to_numpy()
    n_bins = 8
    bin_edges = np.percentile(xs, np.linspace(0, 100, n_bins + 1))
    pairs = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (xs >= lo) & (xs < hi)
        if mask.sum() > 0:
            pairs.append((float(xs[mask].mean()), float(ys[mask].mean())))

    # no-category mode: (anything, pairs_list)
    results_single = (None, pairs)

    # category mode: list of ((anything, pairs), category_str)
    rng = np.random.default_rng(1)
    top_rocks = df["Rockcode"].drop_nulls().unique().sort().to_list()[:3]
    results_cat = []
    for rock in top_rocks:
        sub = df.filter(pl.col("Rockcode") == rock).select(["CuGrade", "AuGrade"]).drop_nulls()
        if len(sub) < 5:
            continue
        rxs = sub["CuGrade"].to_numpy()
        rys = sub["AuGrade"].to_numpy()
        bedges = np.percentile(rxs, np.linspace(0, 100, 6))
        rpairs = []
        for lo, hi in zip(bedges[:-1], bedges[1:]):
            mask = (rxs >= lo) & (rxs < hi)
            if mask.sum() > 0:
                rpairs.append((float(rxs[mask].mean()), float(rys[mask].mean())))
        if rpairs:
            results_cat.append(((None, rpairs), str(int(rock))))

    for theme_name, theme_cls in _THEMES:
        fig = build_cond_means_figure(results_single, theme=theme_cls)
        assert fig is not None
        apply_cond_means_layout(fig, plot_title="Conditional Means — Au on Cu", x_title="Cu Grade (%)", y_title="Au Grade (g/t)")
        theme_cls.apply_common_layout(fig)
        write_themed_html(fig, OUTPUT_DIR / f"cond_means_{theme_name}.html", theme_name)
        assert len(fig.data) >= 1

    if results_cat:
        for theme_name, theme_cls in _THEMES:
            fig = build_cond_means_figure(results_cat, theme=theme_cls)
            assert fig is not None
            apply_cond_means_layout(fig, plot_title="Conditional Means by Rockcode", x_title="Cu Grade (%)", y_title="Au Grade (g/t)")
            theme_cls.apply_common_layout(fig)
            write_themed_html(fig, OUTPUT_DIR / f"cond_means_cat_{theme_name}.html", theme_name)
            assert len(fig.data) >= 1


# ---------------------------------------------------------------------------
# contact analysis — synthetic correlogram-shaped DataFrame
# ---------------------------------------------------------------------------

def test_contact_analysis() -> None:
    rng = np.random.default_rng(42)
    steps = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0])
    head_vals = 0.55 - 0.02 * np.arange(10) + rng.normal(0, 0.02, 10)
    tail_vals = 0.65 - 0.025 * np.arange(10) + rng.normal(0, 0.02, 10)

    contact_df = pd.DataFrame({
        "steps": steps,
        "head_UG2_Au": head_vals,
        "tail_UG1_Au": tail_vals,
    })

    for theme_name, theme_cls in _THEMES:
        fig = build_contact_analysis_figure(contact_df, theme=theme_cls)
        assert fig is not None
        apply_contact_analysis_layout(fig, plot_title="Contact Analysis — Au Grade", y_title="Au Grade (g/t)")
        theme_cls.apply_common_layout(fig)
        write_themed_html(fig, OUTPUT_DIR / f"contact_analysis_{theme_name}.html", theme_name)
        assert len(fig.data) >= 1


# ---------------------------------------------------------------------------
# cumulative probability plot
# ---------------------------------------------------------------------------

def test_cpplt(df: pl.DataFrame) -> None:
    top_rocks = df["Rockcode"].drop_nulls().unique().sort().to_list()[:3]
    series_list = []
    for rock in top_rocks:
        vals = df.filter(pl.col("Rockcode") == rock)["AuGrade"].drop_nulls().sort().to_numpy()
        if len(vals) < 5:
            continue
        probs = np.linspace(0, 100, len(vals))
        pairs = list(zip(vals.tolist(), probs.tolist()))
        series_list.append((pairs, str(int(rock))))

    assert series_list, "No series for cpplt test"

    for theme_name, theme_cls in _THEMES:
        fig = build_cpplt_figure(series_list, theme=theme_cls)
        assert fig is not None
        apply_cpplt_layout(fig, plot_title="Cumulative Probability — Au Grade", x_title="Au Grade (g/t)")
        theme_cls.apply_common_layout(fig)
        write_themed_html(fig, OUTPUT_DIR / f"cpplt_{theme_name}.html", theme_name)
        assert len(fig.data) >= 1


# ---------------------------------------------------------------------------
# downhole variogram
# ---------------------------------------------------------------------------

def test_ddh_variogram() -> None:
    lags = np.array([2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0])
    npairs = np.array([120, 110, 95, 80, 65, 50, 35, 20], dtype=float)
    gamma = np.array([0.20, 0.38, 0.55, 0.68, 0.76, 0.82, 0.86, 0.88])
    variance = 0.90

    for theme_name, theme_cls in _THEMES:
        fig = build_ddh_variogram_figure((npairs, lags, gamma, variance), theme=theme_cls)
        assert fig is not None
        apply_ddh_variogram_layout(fig, plot_title="Downhole Variogram — Au Grade", x_title="Lag (m)", y_title="γ (h)")
        theme_cls.apply_common_layout(fig)
        write_themed_html(fig, OUTPUT_DIR / f"ddh_variogram_{theme_name}.html", theme_name)
        assert len(fig.data) >= 1


# ---------------------------------------------------------------------------
# local proportions (box plots)
# ---------------------------------------------------------------------------

def test_local_props(df: pl.DataFrame) -> None:
    top_rocks = df["Rockcode"].drop_nulls().unique().sort().to_list()[:4]
    categories = []
    data = []
    for rock in top_rocks:
        vals = df.filter(pl.col("Rockcode") == rock)["AuGrade"].drop_nulls().to_numpy()
        if len(vals) > 5:
            categories.append(str(int(rock)))
            data.append(vals)

    assert categories, "No groups for local_props test"
    results = (categories, data)

    for theme_name, theme_cls in _THEMES:
        fig = build_local_props_figure(results, theme=theme_cls)
        assert fig is not None
        apply_local_props_layout(fig, plot_title="Au Grade by Rockcode", variable="Au Grade (g/t)", categories="Rockcode")
        theme_cls.apply_common_layout(fig)
        write_themed_html(fig, OUTPUT_DIR / f"local_props_{theme_name}.html", theme_name)
        assert len(fig.data) >= 1


# ---------------------------------------------------------------------------
# probability-probability plot
# ---------------------------------------------------------------------------

def test_pp(df: pl.DataFrame) -> None:
    vals = df["AuGrade"].drop_nulls().sort().to_numpy()
    n = len(vals)
    empirical = np.linspace(0, 1, n)
    theoretical = np.sort(np.random.default_rng(7).normal(np.mean(vals), np.std(vals), n))
    theoretical_cdf = np.searchsorted(theoretical, vals) / n
    pairs = list(zip(theoretical_cdf.tolist(), empirical.tolist()))

    for theme_name, theme_cls in _THEMES:
        fig = build_pp_figure(pairs, plot_title="PP — Au Grade", theme=theme_cls)
        assert fig is not None
        apply_pp_layout(fig, plot_title="PP Plot — Au Grade", x_title="Theoretical", y_title="Empirical")
        theme_cls.apply_common_layout(fig)
        write_themed_html(fig, OUTPUT_DIR / f"pp_{theme_name}.html", theme_name)
        assert len(fig.data) >= 1


# ---------------------------------------------------------------------------
# proportional effect
# ---------------------------------------------------------------------------

def test_proportional_effect(df: pl.DataFrame) -> None:
    subset = df.select(["CuGrade", "AuGrade"]).drop_nulls()
    xs = subset["CuGrade"].to_numpy()
    ys = subset["AuGrade"].to_numpy()
    n_bins = 10
    bin_edges = np.percentile(xs, np.linspace(0, 100, n_bins + 1))
    rows = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (xs >= lo) & (xs < hi)
        if mask.sum() > 3:
            rows.append([float(ys[mask].mean()), float(ys[mask].std()), float(mask.sum())])

    points = np.array(rows)

    for theme_name, theme_cls in _THEMES:
        fig = build_proportional_effect_figure(points, theme=theme_cls)
        assert fig is not None
        apply_proportional_effect_layout(fig, plot_title="Proportional Effect — Au Grade", x_title="Local mean (g/t)", y_title="Local std dev (g/t)")
        theme_cls.apply_common_layout(fig)
        write_themed_html(fig, OUTPUT_DIR / f"proportional_effect_{theme_name}.html", theme_name)
        assert len(fig.data) >= 1


# ---------------------------------------------------------------------------
# proportion / grade curve
# ---------------------------------------------------------------------------

def test_prop_grade_curve(df: pl.DataFrame) -> None:
    vals = df["AuGrade"].drop_nulls().to_numpy()
    cutoffs = np.linspace(float(vals.min()), float(np.percentile(vals, 95)), 30)
    proportions = np.array([float((vals >= c).mean()) for c in cutoffs])
    grades = np.array([float(vals[vals >= c].mean()) if (vals >= c).any() else 0.0 for c in cutoffs])
    data = np.column_stack([cutoffs, proportions, grades])

    for theme_name, theme_cls in _THEMES:
        fig = build_prop_grade_curve_figure(data, theme=theme_cls)
        assert fig is not None
        apply_prop_grade_curve_layout(fig, plot_title="Proportion / Grade Curve — Au", variable="Au Grade (g/t)")
        theme_cls.apply_common_layout(fig)
        write_themed_html(fig, OUTPUT_DIR / f"prop_grade_curve_{theme_name}.html", theme_name)
        assert len(fig.data) >= 1


# ---------------------------------------------------------------------------
# QQ plot
# ---------------------------------------------------------------------------

def test_qq(df: pl.DataFrame) -> None:
    vals = df["AuGrade"].drop_nulls().sort().to_numpy()
    n = len(vals)
    theoretical = np.sort(np.random.default_rng(3).normal(np.mean(vals), np.std(vals), n))
    points = np.column_stack([theoretical, vals])

    for theme_name, theme_cls in _THEMES:
        fig = build_qq_figure(points, plot_title="QQ — Au Grade", theme=theme_cls)
        assert fig is not None
        apply_qq_layout(fig, plot_title="QQ Plot — Au Grade", x_title="Theoretical quantiles", y_title="Sample quantiles")
        theme_cls.apply_common_layout(fig)
        write_themed_html(fig, OUTPUT_DIR / f"qq_{theme_name}.html", theme_name)
        assert len(fig.data) >= 1


# ---------------------------------------------------------------------------
# swath plot (pass-through)
# ---------------------------------------------------------------------------

def test_swath(df: pl.DataFrame) -> None:
    top_rocks = df["Rockcode"].drop_nulls().unique().sort().to_list()[:2]
    synthetic_fig = go.Figure()
    xs = np.linspace(0, 500, 20)
    for i, rock in enumerate(top_rocks):
        rng = np.random.default_rng(i)
        synthetic_fig.add_trace(go.Scatter(x=xs.tolist(), y=(1.5 + rng.normal(0, 0.15, 20)).tolist(), mode="lines+markers", name=str(int(rock))))

    for theme_name, theme_cls in _THEMES:
        fig = build_swath_figure(synthetic_fig, theme=theme_cls)
        assert fig is not None
        apply_swath_layout(fig, plot_title="Swath Plot — Au Grade")
        theme_cls.apply_common_layout(fig)
        write_themed_html(fig, OUTPUT_DIR / f"swath_{theme_name}.html", theme_name)
        assert len(fig.data) >= 1


# ---------------------------------------------------------------------------
# tonnage / grade curve
# ---------------------------------------------------------------------------

def test_ton_grade_curve(df: pl.DataFrame) -> None:
    vals = df["AuGrade"].drop_nulls().to_numpy()
    total = len(vals)
    cutoffs = np.linspace(float(vals.min()), float(np.percentile(vals, 95)), 30)
    tonnages = np.array([float((vals >= c).sum()) / total for c in cutoffs])
    grades = np.array([float(vals[vals >= c].mean()) if (vals >= c).any() else 0.0 for c in cutoffs])
    data = np.column_stack([cutoffs, tonnages, grades])

    for theme_name, theme_cls in _THEMES:
        fig = build_ton_grade_curve_figure(data, theme=theme_cls)
        assert fig is not None
        apply_ton_grade_curve_layout(fig, plot_title="Tonnage / Grade Curve — Au", variable="Au Grade (g/t)")
        theme_cls.apply_common_layout(fig)
        write_themed_html(fig, OUTPUT_DIR / f"ton_grade_curve_{theme_name}.html", theme_name)
        assert len(fig.data) >= 1


# ---------------------------------------------------------------------------
# transfer function
# ---------------------------------------------------------------------------

def test_transfer_function(df: pl.DataFrame) -> None:
    top_rocks = df["Rockcode"].drop_nulls().unique().sort().to_list()[:3]
    bins = np.linspace(0, 5, 25)
    func: dict = {}
    for rock in top_rocks:
        vals = df.filter(pl.col("Rockcode") == rock)["AuGrade"].drop_nulls().to_numpy()
        if len(vals) > 5:
            func[str(int(rock))] = np.array([float((vals >= c).mean()) for c in bins])

    assert func, "No categories for transfer_function test"

    for theme_name, theme_cls in _THEMES:
        fig = build_transfer_function_figure(func, bins, plot_title="Transfer Function — Au", theme=theme_cls)
        assert fig is not None
        apply_transfer_function_layout(fig, plot_title="Transfer Function — Au Grade", x_title="Cutoff (g/t)", y_title="Proportion above cutoff")
        theme_cls.apply_common_layout(fig)
        write_themed_html(fig, OUTPUT_DIR / f"transfer_function_{theme_name}.html", theme_name)
        assert len(fig.data) >= 1


# ---------------------------------------------------------------------------
# variogram (experimental only)
# ---------------------------------------------------------------------------

def test_variogram() -> None:
    rng = np.random.default_rng(5)
    n_lags = 10
    lags = np.linspace(5, 100, n_lags)
    npairs_dir1 = (np.linspace(200, 50, n_lags) + rng.normal(0, 5, n_lags)).clip(10)
    npairs_dir2 = (np.linspace(180, 45, n_lags) + rng.normal(0, 5, n_lags)).clip(10)
    gamma_dir1 = 1 - np.exp(-lags / 40) + rng.normal(0, 0.02, n_lags)
    gamma_dir2 = 1 - np.exp(-lags / 60) + rng.normal(0, 0.02, n_lags)

    exp_npairs = [npairs_dir1, npairs_dir2]
    exp_lags = [lags, lags]
    exp_values = [gamma_dir1, gamma_dir2]
    variance = 1.0

    directions = [
        {"azimuth": 0, "dip": 0},
        {"azimuth": 90, "dip": 0},
    ]

    results = (exp_npairs, exp_lags, exp_values, variance)

    for theme_name, theme_cls in _THEMES:
        fig = build_variogram_figure(results, directions, theme=theme_cls)
        assert fig is not None
        apply_variogram_layout(fig, plot_title="Experimental Variogram — Au Grade")
        theme_cls.apply_common_layout(fig)
        write_themed_html(fig, OUTPUT_DIR / f"variogram_{theme_name}.html", theme_name)
        assert len(fig.data) >= 1


# ---------------------------------------------------------------------------
# LMC variogram (single component)
# ---------------------------------------------------------------------------

def test_lmc() -> None:
    rng = np.random.default_rng(6)
    n_lags = 10
    lags = np.linspace(5, 100, n_lags)
    npairs = (np.linspace(200, 50, n_lags) + rng.normal(0, 5, n_lags)).clip(10)
    gamma = 1 - np.exp(-lags / 40) + rng.normal(0, 0.02, n_lags)

    results = ([npairs], [lags], [gamma])
    directions = [{"azimuth": 0, "dip": 0}]

    for theme_name, theme_cls in _THEMES:
        fig = build_lmc_figure(results, directions, theme=theme_cls)
        assert fig is not None
        apply_variogram_layout(fig, plot_title="LMC Variogram — V1")
        theme_cls.apply_common_layout(fig)
        write_themed_html(fig, OUTPUT_DIR / f"lmc_{theme_name}.html", theme_name)
        assert len(fig.data) >= 1


# ---------------------------------------------------------------------------
# LMC figures (dict of 3 components)
# ---------------------------------------------------------------------------

def test_lmc_figures() -> None:
    rng = np.random.default_rng(9)
    n_lags = 10
    lags = np.linspace(5, 100, n_lags)

    def _make_results(seed: int):
        r = np.random.default_rng(seed)
        npairs = (np.linspace(200, 50, n_lags) + r.normal(0, 5, n_lags)).clip(10)
        gamma = 1 - np.exp(-lags / (30 + seed * 10)) + r.normal(0, 0.02, n_lags)
        return ([npairs], [lags], [gamma])

    lmc_results = {"V1": _make_results(0), "V2": _make_results(1), "V1xV2": _make_results(2)}
    lmc_directions = {k: [{"azimuth": 0, "dip": 0}] for k in lmc_results}

    for theme_name, theme_cls in _THEMES:
        figs = build_lmc_figures(lmc_results, lmc_directions, theme=theme_cls)
        assert isinstance(figs, dict)
        for key, fig in figs.items():
            assert fig is not None, f"LMC figure for {key} is None"
            apply_variogram_layout(fig, plot_title=f"LMC — {key}")
            theme_cls.apply_common_layout(fig)
            write_themed_html(fig, OUTPUT_DIR / f"lmc_{key.lower().replace('x','x')}_{theme_name}.html", theme_name)
            assert len(fig.data) >= 1
