from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .base_plotly import AndesPlotlyTheme


def build_scatter_figure(
    results,
    *,
    x_title: str = "X",
    y_title: str = "Y",
    plot_title: str = "",
    category_name: str = "",
    show_identity_line: bool = False,
    show_regression_line: bool = False,
    show_zero_intercept_regression_line: bool = False,
    normalized_axis: bool = False,
    eda_object=None,
    theme: type[AndesPlotlyTheme] = AndesPlotlyTheme,
) -> go.Figure | None:
    if results is None:
        return None

    _limits, data = results

    if isinstance(data, dict) and data.get("__category_mode__"):
        per_cat = data["per_cat"]
        fig = go.Figure()
        total_points = sum(len(cx) for cx, _ in per_cat.values())
        scatter_cls = go.Scattergl if total_points > 100_000 else go.Scatter
        for i, (cat_str, (cx, cy)) in enumerate(per_cat.items()):
            hex_color = theme.palette[i % len(theme.palette)]
            fig.add_trace(
                scatter_cls(
                    x=cx,
                    y=cy,
                    mode="markers",
                    name=cat_str,
                    marker={
                        "size": theme.scatter_marker_size,
                        "color": theme.palette_rgba(hex_color, theme.scatter_marker_opacity),
                        "line": {
                            "color": hex_color,
                            "width": theme.scatter_marker_border_width,
                        },
                    },
                    hovertemplate=(
                        f"<b>{category_name}</b>: {cat_str}<br>"
                        f"<b>{x_title}</b>: %{{x:.3f}}<br>"
                        f"<b>{y_title}</b>: %{{y:.3f}}<extra></extra>"
                    ),
                )
            )
        return fig

    points, _correlations = data

    if points is None or len(points) == 0:
        return None

    x_values = points[:, 0]
    y_values = points[:, 1]

    fig = go.Figure()
    scatter_cls = go.Scattergl if len(x_values) > 10_000 else go.Scatter

    fig.add_trace(
        scatter_cls(
            x=x_values,
            y=y_values,
            mode="markers",
            name=plot_title,
            marker={
                "size": theme.scatter_marker_size,
                "color": theme.palette_rgba(theme.palette[0], theme.scatter_marker_opacity),
                "line": {
                    "color": theme.palette[0],
                    "width": theme.scatter_marker_border_width,
                },
            },
            hovertemplate=(
                f"<b>{x_title}</b>: %{{x:.3f}}<br>"
                f"<b>{y_title}</b>: %{{y:.3f}}<extra></extra>"
            ),
            showlegend=False,
        )
    )

    if show_identity_line:
        max_cut = np.max(points)
        min_cut = np.min(points)
        if max_cut - min_cut <= 200:
            fig.add_trace(
                go.Scatter(
                    x=[0, max_cut],
                    y=[0, max_cut],
                    mode="lines",
                    name="Identity line",
                    line={"color": "red", "width": 2, "dash": "dash"},
                    hoverinfo="skip",
                )
            )

    if show_regression_line and eda_object is not None:
        slope, intercept = eda_object.calculate_regression_line(x_values, y_values)
        x_line = np.array([np.min(x_values), np.max(x_values)])
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=slope * x_line + intercept,
                mode="lines",
                name=f"Regression (y={slope:.3f}x+{intercept:.3f})",
                line={"color": "green", "width": 2},
                hoverinfo="skip",
            )
        )

    if show_zero_intercept_regression_line and eda_object is not None:
        slope, _ = eda_object.calculate_regression_line(x_values, y_values)
        x_line = np.array([0, np.max(x_values)])
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=slope * x_line,
                mode="lines",
                name=f"Zero intercept (y={slope:.3f}x)",
                line={"color": "orange", "width": 2},
                hoverinfo="skip",
            )
        )

    return fig


def build_same_scale_scatter_figure(
    results,
    *,
    x_title: str = "X",
    y_title: str = "Y",
    plot_title: str = "",
    show_identity_line: bool = False,
    show_regression_line: bool = False,
    show_zero_intercept_regression_line: bool = False,
    eda_object=None,
    theme: type[AndesPlotlyTheme] = AndesPlotlyTheme,
) -> go.Figure | None:
    if results is None:
        return None

    _limits, (points, _correlations) = results

    if points is None or len(points) == 0:
        return None

    x_values = points[:, 0]
    y_values = points[:, 1]

    fig = go.Figure()
    scatter_cls = go.Scattergl if len(x_values) > 100_000 else go.Scatter

    fig.add_trace(
        scatter_cls(
            x=x_values,
            y=y_values,
            mode="markers",
            name=plot_title,
            marker={
                "size": 8,
                "color": theme.palette_rgba(theme.palette[0], 0.4),
                "line": {"color": theme.palette[0], "width": 1.5},
            },
            hovertemplate="<b>X</b>: %{x:.3f}<br><b>Y</b>: %{y:.3f}<extra></extra>",
        )
    )

    combined_min = float(min(np.min(x_values), np.min(y_values)))
    combined_max = float(max(np.max(x_values), np.max(y_values)))

    if show_identity_line:
        fig.add_trace(
            go.Scatter(
                x=[combined_min, combined_max],
                y=[combined_min, combined_max],
                mode="lines",
                name="Identity line",
                line={"color": "red", "width": 2, "dash": "dash"},
                hoverinfo="skip",
            )
        )

    if show_regression_line and eda_object is not None:
        slope, intercept = eda_object.calculate_regression_line(x_values, y_values)
        x_line = np.array([0, np.max(x_values)])
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=slope * x_line + intercept,
                mode="lines",
                name=f"Regression (y={slope:.3f}x+{intercept:.3f})",
                line={"color": "green", "width": 2},
                hoverinfo="skip",
            )
        )

    if show_zero_intercept_regression_line and eda_object is not None:
        slope, _ = eda_object.calculate_regression_line(x_values, y_values)
        x_line = np.array([0, np.max(x_values)])
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=slope * x_line,
                mode="lines",
                name=f"Zero intercept (y={slope:.3f}x)",
                line={"color": "orange", "width": 2},
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        xaxis={"range": [combined_min, combined_max]},
        yaxis={"range": [combined_min, combined_max]},
    )

    return fig


def apply_scatter_layout(
    fig,
    *,
    plot_title: str = "",
    x_title: str = "X",
    y_title: str = "Y",
    normalized_axis: bool = False,
) -> None:
    layout: dict = {
        "title_text": plot_title,
        "xaxis_title": x_title or "X",
        "yaxis_title": y_title or "Y",
        "hovermode": "closest",
        "template": "plotly_white",
    }
    if normalized_axis:
        layout["yaxis"] = {"scaleanchor": "x", "scaleratio": 1}
    fig.update_layout(**layout)


def apply_same_scale_scatter_layout(
    fig,
    *,
    plot_title: str = "",
    x_title: str = "X",
    y_title: str = "Y",
) -> None:
    fig.update_layout(
        title_text=plot_title,
        xaxis_title=x_title or "X",
        yaxis_title=y_title or "Y",
        hovermode="closest",
        template="plotly_white",
    )
