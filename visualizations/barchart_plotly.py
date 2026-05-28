from __future__ import annotations

import plotly.graph_objects as go

from .base_plotly import AndesPlotlyTheme


def build_barchart_figure(
    categories,
    values,
    stats_df=None,
    *,
    percentage: bool = False,
    theme: type[AndesPlotlyTheme] = AndesPlotlyTheme,
) -> go.Figure | None:
    if categories is None or values is None or len(categories) == 0:
        return None

    color = theme.palette[0]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[str(c) for c in categories],
            y=list(values),
            marker={
                "color": theme.palette_rgba(color, theme.bar_color_base_opacity),
                "line": {
                    "color": theme.palette_rgba(color, theme.bar_color_border_opacity),
                    "width": theme.bar_color_border_width,
                },
            },
            hovertemplate=(
                "<b>%{x}</b><br>"
                + ("Percentage: %{y:.2f} %<extra></extra>" if percentage else "Frequency: %{y}<extra></extra>")
            ),
            showlegend=False,
        )
    )
    return fig


def apply_barchart_layout(
    fig,
    *,
    plot_title: str = "",
    variable: str = "",
    percentage: bool = False,
) -> None:
    AndesPlotlyTheme.apply_common_layout(fig, show_legend=False)
    y_title = "Percentage (%)" if percentage else "Frequency"
    fig.update_layout(
        title_text=plot_title,
        bargap=0.15,
        template="plotly_white",
    )
    fig.update_xaxes(title_text=variable)
    fig.update_yaxes(title_text=y_title)
