from __future__ import annotations

import plotly.graph_objects as go

from .base_plotly import AndesPlotlyTheme


def build_cell_declustering_figure(
    pairs,
    *,
    theme: type[AndesPlotlyTheme] = AndesPlotlyTheme,
) -> go.Figure | None:
    if not pairs:
        return None

    xs = [float(p[0]) for p in pairs]
    ys = [float(p[1]) for p in pairs]
    color = theme.palette[0]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            marker={
                "color": color,
                "size": 5,
            },
            line={"color": color, "width": 2},
            showlegend=False,
            hovertemplate="Cell size: %{x:.3f}<br>Mean: %{y:.4f}<extra></extra>",
        )
    )
    return fig


def apply_cell_declustering_layout(
    fig,
    *,
    plot_title: str = "",
    x_title: str = "Cell size",
    y_title: str = "Declustered mean",
) -> None:
    AndesPlotlyTheme.apply_common_layout(fig, show_legend=False)
    fig.update_layout(title_text=plot_title, template="plotly_white")
    fig.update_xaxes(title_text=x_title, showgrid=True)
    fig.update_yaxes(title_text=y_title)
