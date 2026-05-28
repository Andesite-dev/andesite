from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .base_plotly import AndesPlotlyTheme

_VARIANCE_COLOR = "#E07B39"
_EXP_LINE_WIDTH = 2.5
_VARIANCE_LINE_WIDTH = 3.5
_MIN_DOT = 3
_MAX_DOT = 12


def build_ddh_variogram_figure(
    results,
    *,
    theme: type[AndesPlotlyTheme] = AndesPlotlyTheme,
) -> go.Figure | None:
    """Build a downhole variogram with variable dot sizes and a variance reference line.

    results: (exp_npairs, exp_lags, exp_values, variance)
      exp_npairs: 1-D array of pair counts per lag.
      exp_lags:   1-D array of lag distances.
      exp_values: 1-D array of gamma values.
      variance:   scalar — plotted as horizontal reference line.
    """
    if results is None:
        return None

    exp_npairs, exp_lags, exp_values, variance = results

    exp_npairs = np.asarray(exp_npairs, dtype=float)
    exp_lags = np.asarray(exp_lags, dtype=float)
    exp_values = np.asarray(exp_values, dtype=float)

    if len(exp_lags) == 0:
        return None

    max_lag = float(np.nanmax(exp_lags))
    if max_lag == 0:
        return None

    min_np = float(np.nanmin(exp_npairs))
    max_np = float(np.nanmax(exp_npairs))
    sizes = (
        np.interp(exp_npairs, [min_np, max_np], [_MIN_DOT, _MAX_DOT]).tolist()
        if max_np > min_np
        else [float(_MIN_DOT)] * len(exp_npairs)
    )

    color = theme.palette[0]
    n_var = 100
    var_lags = np.linspace(0, max_lag, n_var)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=exp_lags.tolist(),
            y=exp_values.tolist(),
            mode="markers+lines",
            name="DDH Variogram",
            marker={"color": color, "size": sizes},
            line={"color": color, "width": _EXP_LINE_WIDTH, "dash": "dash"},
            hovertemplate="Lag: %{x:.3f}<br>γ: %{y:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=var_lags.tolist(),
            y=[float(variance)] * n_var,
            mode="lines",
            name="Variance",
            line={"color": _VARIANCE_COLOR, "width": _VARIANCE_LINE_WIDTH, "dash": "dot"},
            hoverinfo="skip",
        )
    )
    return fig


def apply_ddh_variogram_layout(
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
