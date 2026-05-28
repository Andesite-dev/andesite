from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .base_plotly import AndesPlotlyTheme

_REF_LINE_COLOR = "#888888"


def build_qq_figure(
    points,
    *,
    plot_title: str = "",
    theme: type[AndesPlotlyTheme] = AndesPlotlyTheme,
) -> go.Figure | None:
    """Build a Q-Q plot with a 1:1 reference line.

    points: array-like of shape (n, 2) — (theoretical_quantile, sample_quantile) pairs.
    """
    if points is None or len(points) == 0:
        return None

    pts = np.asarray(points, dtype=float)
    min_value = float(np.min(pts))
    if 0 <= min_value < 1:
        min_value = 0.0
    max_value = float(np.max(pts))
    max_value += (max_value - min_value) / 100.0

    color = theme.palette[0]
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(pts[:, 0]),
            y=list(pts[:, 1]),
            mode="markers",
            name=plot_title or "QQ",
            marker={
                "color": theme.palette_rgba(color, theme.scatter_marker_opacity),
                "size": theme.scatter_marker_size,
                "line": {"color": color, "width": theme.scatter_marker_border_width},
            },
            showlegend=False,
            hovertemplate="Theoretical: %{x:.3f}<br>Sample: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[min_value, max_value],
            y=[min_value, max_value],
            mode="lines",
            name="1:1 line",
            line={"color": _REF_LINE_COLOR, "width": 2, "dash": "dash"},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    return fig


def apply_qq_layout(
    fig,
    *,
    plot_title: str = "",
    x_title: str = "Theoretical quantiles",
    y_title: str = "Sample quantiles",
) -> None:
    AndesPlotlyTheme.apply_common_layout(fig, show_legend=False)
    fig.update_layout(
        title_text=plot_title,
        hovermode="closest",
        template="plotly_white",
        yaxis={"scaleanchor": "x", "scaleratio": 1},
    )
    fig.update_xaxes(title_text=x_title)
    fig.update_yaxes(title_text=y_title)
