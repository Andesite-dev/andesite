from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .base_plotly import AndesPlotlyTheme


def build_proportional_effect_figure(
    points,
    *,
    theme: type[AndesPlotlyTheme] = AndesPlotlyTheme,
) -> go.Figure | None:
    """Build a proportional effect scatter plot.

    points: ndarray shape (n, 3) — columns are [local_mean, local_std_dev, n_neighbors].
    Only the first two columns are plotted.
    """
    if points is None or len(points) == 0:
        return None

    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 2:
        return None

    x_vals = points[:, 0]
    y_vals = points[:, 1]
    color = theme.palette[0]

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=x_vals,
            y=y_vals,
            mode="markers",
            marker={
                "color": theme.palette_rgba(color, theme.scatter_marker_opacity),
                "size": theme.scatter_marker_size,
                "line": {"color": color, "width": theme.scatter_marker_border_width},
            },
            showlegend=False,
            hovertemplate="Mean: %{x:.4f}<br>Std dev: %{y:.4f}<extra></extra>",
        )
    )
    return fig


def apply_proportional_effect_layout(
    fig,
    *,
    plot_title: str = "",
    x_title: str = "Local mean",
    y_title: str = "Local standard deviation",
) -> None:
    AndesPlotlyTheme.apply_common_layout(fig, show_legend=False)
    fig.update_layout(
        title_text=plot_title,
        hovermode="closest",
        template="plotly_white",
    )
    fig.update_xaxes(title_text=x_title)
    fig.update_yaxes(title_text=y_title)
