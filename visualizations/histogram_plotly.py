from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .base_plotly import AndesPlotlyTheme


def build_histogram_figure(
    hist_data,
    tables,
    flag,
    kde_result,
    *,
    plot_title: str = "",
    percentage: bool = False,
    theme: type[AndesPlotlyTheme] = AndesPlotlyTheme,
) -> go.Figure | None:
    _blue = theme.primary_color
    _teal = theme.secondary_color

    hover_template = (
        "Ranges: %{x:.3f} - %{meta:.3f} <br>Percentage: %{y:.2f} %<extra></extra>"
        if percentage
        else "Ranges: %{x:.3f} - %{meta:.3f} <br>Frecuency: %{y}<extra></extra>"
    )

    if not hist_data:
        return None

    main_bins = list(hist_data)
    tail_bins = []

    if flag:
        lower_limit = flag.get("lower_limit")
        upper_limit = flag.get("upper_limit")
        stats = tables[0]

        has_left_tail = lower_limit is not None and stats.loc[0, "min"] < lower_limit
        has_right_tail = upper_limit is not None and stats.loc[0, "max"] > upper_limit

        if has_left_tail:
            tail_bins.append(main_bins[0])
            main_bins = main_bins[1:]

        if has_right_tail:
            tail_bins.append(main_bins[-1])
            main_bins = main_bins[:-1]

    fig = go.Figure()

    if main_bins:
        x_main = np.array([float((b[1] + b[2]) / 2) for b in main_bins])
        y_main = np.array([float(b[0]) for b in main_bins])
        w_main = np.array([float(b[2] - b[1]) for b in main_bins])
        fig.add_trace(
            go.Bar(
                x=x_main,
                y=y_main,
                width=w_main,
                meta=x_main + w_main,
                name=plot_title or "Histogram",
                marker={
                    "color": theme.palette_rgba(_blue, theme.bar_color_base_opacity),
                    "line": {
                        "color": theme.palette_rgba(_blue, theme.bar_color_border_opacity),
                        "width": theme.bar_color_border_width,
                    },
                },
                showlegend=False,
                hovertemplate=hover_template,
            )
        )

    if tail_bins:
        x_tail = np.array([float((b[1] + b[2]) / 2) for b in tail_bins])
        y_tail = np.array([float(b[0]) for b in tail_bins])
        w_tail = np.array([float(b[2] - b[1]) for b in tail_bins])
        fig.add_trace(
            go.Bar(
                x=x_tail,
                y=y_tail,
                width=w_tail,
                meta=x_tail + w_tail,
                name="Narrow bars (tails)",
                marker={
                    "color": theme.palette_rgba(_teal, theme.bar_narrow_color_base_opacity),
                    "line": {
                        "color": theme.palette_rgba(_teal, theme.bar_narrow_color_border_opacity),
                        "width": theme.bar_narrow_color_border_width,
                    },
                },
                showlegend=False,
                hovertemplate=hover_template,
            )
        )

    if kde_result is not None:
        x_kde, y_kde_raw = kde_result
        all_bins = main_bins + tail_bins
        hist_integral = sum(b[0] * (b[2] - b[1]) for b in all_bins)
        y_kde = y_kde_raw * hist_integral
        fig.add_trace(
            go.Scatter(
                x=x_kde,
                y=y_kde,
                mode="lines",
                name="KDE",
                line={
                    "color": theme.palette_rgba(_blue, 1.0),
                    "width": theme.bar_kde_line_width,
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )

    return fig


def apply_histogram_layout(
    fig,
    *,
    plot_title: str = "",
    variable: str = "",
    cumulative: bool = False,
    percentage: bool = False,
) -> None:
    if cumulative:
        y_title = "Cumulative frequency (%)" if percentage else "Cumulative frequency"
    elif percentage:
        y_title = "Frequency (%)"
    else:
        y_title = "Frequency"

    fig.update_layout(
        title_text=plot_title,
        xaxis_title=variable,
        yaxis_title=y_title,
        bargap=0,
        bargroupgap=0,
        template="plotly_white",
        uirevision=f"hist:{variable}",
    )

    if percentage:
        fig.update_layout({"yaxis": {"ticksuffix": "%"}})

    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showgrid=False)

    has_negative_x = any(
        min(trace.x) < 0
        for trace in fig.data
        if hasattr(trace, "x") and trace.x is not None and len(trace.x) > 0
    )
    if has_negative_x:
        fig.update_layout(yaxis=dict(anchor="free", position=0))
