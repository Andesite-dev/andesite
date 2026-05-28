from __future__ import annotations

import plotly.graph_objects as go

from .base_plotly import AndesPlotlyTheme


def build_local_props_figure(
    results,
    *,
    theme: type[AndesPlotlyTheme] = AndesPlotlyTheme,
) -> go.Figure | None:
    """Build box plots for local proportions / spatial statistics.

    results: tuple (categories, data, ...) from LocalProportionsCalculation.
      categories: list of label strings.
      data: list of arrays (one per category).
    """
    if results is None:
        return None

    categories, data = results[0], results[1]
    if not categories:
        return None

    fig = go.Figure()
    show_legend = len(categories) > 1

    for i, (label, values) in enumerate(zip(categories, data)):
        color = theme.palette[i % len(theme.palette)]
        fig.add_trace(
            go.Box(
                y=list(values),
                name=str(label),
                marker={
                    "color": theme.palette_rgba(color, theme.violin_outlier_color_opacity),
                    "line": {
                        "color": color,
                        "width": theme.violin_outlier_border_width,
                    },
                    "size": 4,
                },
                line={
                    "color": theme.palette_rgba(color, theme.violin_color_border_opacity),
                    "width": theme.violin_box_color_border_width,
                },
                fillcolor=theme.palette_rgba(color, theme.violin_color_base_opacity),
                boxmean=True,
                showlegend=show_legend,
            )
        )

    return fig


def apply_local_props_layout(
    fig,
    *,
    plot_title: str = "",
    variable: str = "",
    categories: str = "",
) -> None:
    AndesPlotlyTheme.apply_common_layout(fig)
    if not plot_title and variable:
        if categories:
            plot_title = f"{variable} by {categories}"
        else:
            plot_title = variable
    fig.update_layout(
        title_text=plot_title,
        template="plotly_white",
        boxgap=0.1,
        boxgroupgap=0.05,
    )
    fig.update_xaxes(title_text=categories or "")
    fig.update_yaxes(title_text=variable)
