from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .base_plotly import AndesPlotlyTheme


def build_prop_grade_curve_figure(
    data,
    *,
    theme: type[AndesPlotlyTheme] = AndesPlotlyTheme,
) -> go.Figure | None:
    """Build a proportion / average grade curve with dual Y-axis.

    data: ndarray shape (n, 3):
      col 0 — cutoff values
      col 1 — proportions (left Y-axis)
      col 2 — average grades (right Y-axis)
    """
    if data is None or len(data) == 0:
        return None

    data = np.asarray(data)
    cutoffs = data[:, 0]
    proportions = data[:, 1]
    grades = data[:, 2]

    color_prop = theme.palette[0]
    color_grade = theme.palette[1]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=list(cutoffs),
            y=list(proportions),
            mode="lines",
            name="Proportions",
            line={"color": color_prop, "width": 2},
            hovertemplate="Cutoff: %{x:.3f}<br>Proportion: %{y:.4f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=list(cutoffs),
            y=list(grades),
            mode="lines",
            name="Av. grades",
            line={"color": color_grade, "width": 2},
            hovertemplate="Cutoff: %{x:.3f}<br>Grade: %{y:.4f}<extra></extra>",
        ),
        secondary_y=True,
    )

    return fig


def apply_prop_grade_curve_layout(
    fig,
    *,
    plot_title: str = "",
    variable: str = "",
    x_title: str = "Cutoff",
) -> None:
    AndesPlotlyTheme.apply_common_layout(fig)
    fig.update_layout(
        title_text=plot_title,
        hovermode="closest",
        template="plotly_white",
    )
    fig.update_xaxes(title_text=x_title)
    fig.update_yaxes(title_text="Proportion", secondary_y=False)
    fig.update_yaxes(title_text=f"Average {variable}" if variable else "Average grade", secondary_y=True)
