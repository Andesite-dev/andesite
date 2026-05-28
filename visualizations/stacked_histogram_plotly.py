from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .base_plotly import AndesPlotlyTheme


def build_category_histogram_figure(
    bin_edges,
    x_centers,
    per_cat: dict,
    kde_data: dict,
    variable: str,
    category: str,
    *,
    barmode: str = "stack",
    theme: type[AndesPlotlyTheme] = AndesPlotlyTheme,
) -> go.Figure | None:
    if not per_cat:
        return None

    bin_width = float(bin_edges[1] - bin_edges[0])
    n_cats = len(per_cat)

    if barmode == "group":
        bar_width = bin_width * 0.9 / n_cats if n_cats > 0 else bin_width * 0.95
    else:
        bar_width = bin_width * 0.95

    fig = go.Figure()

    for i, (cat_str, counts) in enumerate(per_cat.items()):
        hex_color = theme.palette[i % len(theme.palette)]
        fig.add_trace(
            go.Bar(
                x=x_centers,
                y=counts,
                name=cat_str,
                width=bar_width,
                marker={
                    "color": theme.palette_rgba(hex_color, theme.bar_color_base_opacity),
                    "line": {
                        "color": theme.palette_rgba(hex_color, theme.bar_color_border_opacity),
                        "width": theme.bar_color_border_width,
                    },
                },
                hovertemplate=(
                    f"<b>{category}</b>: {cat_str}<br>"
                    f"<b>{variable}</b>: %{{x:.3f}}<br>"
                    "<b>Frecuency</b>: %{y}<extra></extra>"
                ),
            )
        )

    for cat_str, (x_kde, y_kde) in kde_data.items():
        cat_index = list(per_cat.keys()).index(cat_str)
        hex_color = theme.palette[cat_index % len(theme.palette)]
        fig.add_trace(
            go.Scatter(
                x=x_kde,
                y=y_kde,
                mode="lines",
                name=f"kde {cat_str}",
                line={
                    "color": theme.palette_rgba(hex_color, 1.0),
                    "width": theme.bar_kde_line_width,
                },
                hoverinfo="skip",
            )
        )

    return fig


def apply_category_histogram_layout(
    fig,
    *,
    plot_title: str = "",
    variable: str = "",
    percentage: bool = False,
    barmode: str = "stack",
) -> None:
    y_title = "Frequency (%)" if percentage else "Frequency"
    fig.update_layout(
        title_text=plot_title,
        xaxis_title=variable,
        yaxis_title=y_title,
        bargap=0,
        bargroupgap=0.05,
        barmode=barmode,
        template="plotly_white",
    )
