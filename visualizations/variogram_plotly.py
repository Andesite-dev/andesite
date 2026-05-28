from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .base_plotly import PALETTE_HEX, AndesPlotlyTheme

_VARIANCE_COLOR = "#E07B39"
_EXP_LINE_WIDTH = 2.5
_MODEL_LINE_WIDTH = 3.5
_VARIANCE_LINE_WIDTH = 3.5
_MIN_DOT = 3
_MAX_DOT = 12


def _add_experimental_traces(fig, exp_npairs, exp_lags, exp_values, directions=None, palette=None):
    n_dirs = len(exp_npairs)
    all_npairs = np.concatenate([np.asarray(exp_npairs[i]) for i in range(n_dirs)])
    min_np = float(np.nanmin(all_npairs)) if len(all_npairs) else 0.0
    max_np = float(np.nanmax(all_npairs)) if len(all_npairs) else 1.0

    _palette = palette if palette is not None else PALETTE_HEX
    for i in range(n_dirs):
        lags = np.asarray(exp_lags[i], dtype=float)
        vals = np.asarray(exp_values[i], dtype=float)
        npairs = np.asarray(exp_npairs[i], dtype=float)

        valid = ~(np.isnan(lags) | np.isnan(vals))
        lags, vals, npairs = lags[valid], vals[valid], npairs[valid]
        if len(lags) == 0:
            continue

        sizes = (
            np.interp(npairs, [min_np, max_np], [_MIN_DOT, _MAX_DOT]).tolist()
            if max_np > min_np
            else [float(_MIN_DOT)] * len(npairs)
        )
        color = _palette[i % len(_palette)]

        if directions is not None and i < len(directions):
            name = "Azm:{}°/ Dip:{}°".format(
                int(directions[i]["azimuth"]), int(directions[i]["dip"])
            )
        else:
            name = f"Dir {i + 1}"

        fig.add_trace(
            go.Scatter(
                x=lags.tolist(),
                y=vals.tolist(),
                mode="markers+lines",
                name=name,
                marker={"color": color, "size": sizes},
                line={"color": color, "width": _EXP_LINE_WIDTH, "dash": "dash"},
            )
        )


def _add_model_traces(fig, exp_lags, model_lags, model_values, angles=None, palette=None):
    max_lag = max(
        float(np.nanmax(np.asarray(exp_lags[i])))
        for i in range(len(exp_lags))
        if len(exp_lags[i]) > 0
    )
    _palette = palette if palette is not None else PALETTE_HEX
    for i, (ml, mv) in enumerate(zip(model_lags, model_values)):
        ml_arr = np.asarray(ml)
        mv_arr = np.asarray(mv)
        sel = ml_arr <= max_lag
        color = _palette[i % len(_palette)]
        if angles is not None and i < len(angles):
            name = "Dir{} (Azm:{}°/ Dip:{}°)".format(
                i + 1, int(angles[i][0]), int(angles[i][1])
            )
        else:
            name = f"Model Dir {i + 1}"
        fig.add_trace(
            go.Scatter(
                x=ml_arr[sel].tolist(),
                y=mv_arr[sel].tolist(),
                mode="lines",
                name=name,
                line={"color": color, "width": _MODEL_LINE_WIDTH},
            )
        )


def _add_variance_trace(fig, variance, max_lag, n_points=100):
    lags = np.linspace(0, max_lag, n_points)
    fig.add_trace(
        go.Scatter(
            x=lags.tolist(),
            y=[float(variance)] * n_points,
            mode="lines",
            name="Variance",
            line={"color": _VARIANCE_COLOR, "width": _VARIANCE_LINE_WIDTH, "dash": "dot"},
            hoverinfo="skip",
        )
    )


def build_variogram_figure(
    results,
    directions,
    *,
    theme: type[AndesPlotlyTheme] = AndesPlotlyTheme,
) -> go.Figure | None:
    """Build a univariate experimental (and optionally modeled) variogram.

    results:
      4-tuple (exp_npairs, exp_lags, exp_values, variance) — experimental only.
      7-tuple (exp_npairs, exp_lags, exp_values, model_lags, model_values, angles, variance)
              — experimental + model.

    directions: list of dicts with 'azimuth' and 'dip' keys (used for legend labels).
    """
    if results is None:
        return None

    exp_npairs, exp_lags, exp_values = results[0], results[1], results[2]
    if exp_npairs is None or len(exp_npairs) == 0:
        return None

    fig = go.Figure()
    _add_experimental_traces(fig, exp_npairs, exp_lags, exp_values, directions, palette=theme.palette)

    if len(results) == 7:
        model_lags, model_values, angles, variance = results[3], results[4], results[5], results[6]
        _add_model_traces(fig, exp_lags, model_lags, model_values, angles, palette=theme.palette)
    else:
        variance = results[3]

    all_lags = np.concatenate([np.asarray(exp_lags[i]) for i in range(len(exp_lags))])
    max_lag = float(np.nanmax(all_lags)) if len(all_lags) else 1.0
    _add_variance_trace(fig, variance, max_lag)

    return fig


def apply_variogram_layout(
    fig,
    *,
    plot_title: str = "",
    x_title: str = "Lag distance",
    y_title: str = "γ (h)",
) -> None:
    AndesPlotlyTheme.apply_common_layout(fig)
    fig.update_layout(
        title_text=plot_title,
        hovermode="closest",
        template="plotly_white",
    )
    fig.update_xaxes(title_text=x_title, showgrid=True, rangemode="tozero")
    fig.update_yaxes(title_text=y_title, rangemode="tozero")


def build_lmc_figure(
    results,
    directions,
    *,
    theme: type[AndesPlotlyTheme] = AndesPlotlyTheme,
) -> go.Figure | None:
    """Build a single LMC component variogram (no variance line).

    results:
      3-tuple (exp_npairs, exp_lags, exp_values) — experimental only.
      5-tuple (exp_npairs, exp_lags, exp_values, model_lags, model_values) — with model.

    directions: list of direction dicts for this LMC component.
    """
    if results is None:
        return None

    exp_npairs, exp_lags, exp_values = results[0], results[1], results[2]
    if exp_npairs is None or len(exp_npairs) == 0:
        return None

    fig = go.Figure()
    _add_experimental_traces(fig, exp_npairs, exp_lags, exp_values, directions, palette=theme.palette)

    if len(results) == 5:
        model_lags, model_values = results[3], results[4]
        _add_model_traces(fig, exp_lags, model_lags, model_values, palette=theme.palette)

    return fig


def build_lmc_figures(
    lmc_results: dict,
    lmc_directions: dict,
    *,
    theme: type[AndesPlotlyTheme] = AndesPlotlyTheme,
) -> dict[str, go.Figure | None]:
    """Build all three LMC component figures.

    lmc_results: {"V1": results, "V2": results, "V1xV2": results}.
    lmc_directions: {"V1": directions, "V2": directions, "V1xV2": directions}.
    Returns: {"V1": fig, "V2": fig, "V1xV2": fig}.
    """
    return {
        key: build_lmc_figure(lmc_results.get(key), lmc_directions.get(key, []), theme=theme)
        for key in ("V1", "V2", "V1xV2")
    }
