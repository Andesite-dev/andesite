from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .base_plotly import AndesPlotlyTheme


def build_transfer_function_figure(
    function,
    bins,
    *,
    plot_title: str = "",
    theme: type[AndesPlotlyTheme] = AndesPlotlyTheme,
) -> go.Figure | None:
    """Build a transfer function (tonnage or mean-grade) curve.

    function: dict[str, array] (per-category curves) or array (single curve).
    bins: 1-D array of cutoff values shared across all curves.
    """
    if function is None or bins is None:
        return None

    bins_arr = np.asarray(bins, dtype=float)
    fig = go.Figure()

    if isinstance(function, dict):
        if not function:
            return None
        for i, (category, values) in enumerate(function.items()):
            color = theme.palette[i % len(theme.palette)]
            fig.add_trace(
                go.Scatter(
                    x=list(bins_arr),
                    y=list(np.asarray(values, dtype=float)),
                    mode="lines",
                    name=str(category),
                    line={"color": color, "width": 2},
                    hovertemplate="Cutoff: %{x:.3f}<br>Value: %{y:.4f}<extra></extra>",
                )
            )
    else:
        values = np.asarray(function, dtype=float)
        color = theme.palette[0]
        fig.add_trace(
            go.Scatter(
                x=list(bins_arr),
                y=list(values),
                mode="lines",
                name=plot_title or "Function",
                line={"color": color, "width": 2},
                showlegend=False,
                hovertemplate="Cutoff: %{x:.3f}<br>Value: %{y:.4f}<extra></extra>",
            )
        )

    return fig


def apply_transfer_function_layout(
    fig,
    *,
    plot_title: str = "",
    x_title: str = "Cutoff",
    y_title: str = "",
) -> None:
    AndesPlotlyTheme.apply_common_layout(fig)
    fig.update_layout(
        title_text=plot_title,
        hovermode="closest",
        template="plotly_white",
    )
    fig.update_xaxes(title_text=x_title)
    fig.update_yaxes(title_text=y_title)
