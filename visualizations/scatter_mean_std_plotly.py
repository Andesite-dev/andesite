from __future__ import annotations

import plotly.graph_objects as go

from .base_plotly import AndesPlotlyTheme


def build_scatter_mean_std_figure(
    results,
    *,
    x_title: str = "Mean",
    y_title: str = "Standard Deviation",
    plot_title: str = "",
    category_name: str = "",
    theme: type[AndesPlotlyTheme] = AndesPlotlyTheme,
) -> go.Figure | None:
    if results is None:
        return None

    fig = go.Figure()

    for i, ((mean_x, mean_y), cat) in enumerate(results):
        cat_str = str(cat)
        hex_color = theme.palette[i % len(theme.palette)]
        fig.add_trace(
            go.Scatter(
                x=[mean_x],
                y=[mean_y],
                mode="markers",
                name=cat_str,
                marker={
                    "size": 40,
                    "color": theme.palette_rgba(hex_color, theme.scatter_marker_opacity),
                    "line": {
                        "color": hex_color,
                        "width": theme.scatter_marker_border_width,
                    },
                },
                hovertemplate=(
                    f"<b>{category_name}</b>: {cat_str}<br>"
                    f"<b>Mean</b>: %{{x:.3f}}<br>"
                    f"<b>Standard Deviation</b>: %{{y:.3f}}<extra></extra>"
                ),
            )
        )

    return fig


def apply_scatter_mean_std_layout(
    fig,
    *,
    plot_title: str = "",
    x_title: str = "Mean",
    y_title: str = "Standard Deviation",
) -> None:
    fig.update_layout(
        title_text=plot_title,
        xaxis_title=x_title or "Mean",
        yaxis_title=y_title or "Standard Deviation",
        hovermode="closest",
        template="plotly_white",
    )
