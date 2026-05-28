from __future__ import annotations

import plotly.graph_objects as go

from .base_plotly import AndesPlotlyTheme


def build_violin_figure(
    groups: dict,
    *,
    box_visible: bool = False,
    points: str = "outliers",
    theme: type[AndesPlotlyTheme] = AndesPlotlyTheme,
) -> go.Figure | None:
    if not groups:
        return None

    points_arg = False if points == "none" else points

    fig = go.Figure()

    for i, (cat_str, values) in enumerate(groups.items()):
        hex_color = theme.palette[i % len(theme.palette)]
        fig.add_trace(
            go.Violin(
                uid=f"violin_{cat_str}",
                y=values,
                name=str(cat_str),
                box={
                    "visible": box_visible,
                    "line": {
                        "color": theme.palette_rgba(hex_color, theme.violin_box_color_border_opacity),
                        "width": theme.violin_box_color_border_width,
                    },
                },
                points=points_arg,
                meanline_visible=True,
                line={
                    "color": theme.palette_rgba(hex_color, theme.violin_color_border_opacity),
                    "width": theme.violin_color_border_width,
                },
                fillcolor=theme.palette_rgba(hex_color, theme.violin_color_base_opacity),
                marker={
                    "color": hex_color,
                    "line": {"color": "#1a1a1a", "width": theme.violin_outlier_border_width},
                },
                hovertemplate=(f"<b>{cat_str}</b><br><b>Value</b>: %{{y:.3f}}<extra></extra>"),
            )
        )

    return fig


def apply_violin_layout(
    fig,
    *,
    plot_title: str = "",
    variable: str = "",
    categories: str = "",
) -> None:
    title = plot_title or (f"{variable} by {categories}" if categories else variable)
    fig.update_layout(
        title_text=title,
        yaxis_title=variable,
        xaxis_title=categories or "",
        violingap=0.1,
        violingroupgap=0.05,
        uirevision="violin",
    )
    fig.update_xaxes(uirevision=f"violin-x:{categories}")
    fig.update_yaxes(uirevision=f"violin-y:{variable}")
