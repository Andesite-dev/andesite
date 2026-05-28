from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .base_plotly import AndesPlotlyTheme


def build_cpplt_figure(
    series_list,
    *,
    log_scale: bool = False,
    x_range: tuple[float, float] | None = None,
    theme: type[AndesPlotlyTheme] = AndesPlotlyTheme,
) -> go.Figure | None:
    """Build a Cumulative Probability Plot.

    series_list: list of (pairs, category_label) — output of CppltCalculation,
                 i.e. results2[0] from eda_calculate.
    pairs: list of (x, y) float tuples (finite values only).
    """
    if not series_list:
        return None

    fig = go.Figure()
    for i, (pairs, category) in enumerate(series_list):
        valid = [
            (float(p[0]), float(p[1]))
            for p in pairs
            if len(p) == 2 and np.isfinite(p[0]) and np.isfinite(p[1])
        ]
        if not valid:
            continue
        xs, ys = zip(*valid)
        color = theme.palette[i % len(theme.palette)]
        fig.add_trace(
            go.Scatter(
                x=list(xs),
                y=list(ys),
                mode="markers",
                name=str(category),
                marker={
                    "color": theme.palette_rgba(color, theme.scatter_marker_opacity),
                    "size": 5,
                    "line": {"color": color, "width": 1},
                },
            )
        )

    if x_range is not None:
        x_axis_kwargs: dict = {"range": [float(x_range[0]), float(x_range[1])]}
        if log_scale:
            x_axis_kwargs["type"] = "log"
        fig.update_xaxes(**x_axis_kwargs)
    elif log_scale:
        fig.update_xaxes(type="log")

    return fig


def apply_cpplt_layout(
    fig,
    *,
    plot_title: str = "",
    x_title: str = "",
    y_title: str = "Cumulative probability (%)",
) -> None:
    AndesPlotlyTheme.apply_common_layout(fig)
    fig.update_layout(
        title_text=plot_title,
        hovermode="closest",
        template="plotly_white",
    )
    fig.update_xaxes(title_text=x_title, showgrid=True)
    fig.update_yaxes(title_text=y_title, range=[0, 100])
