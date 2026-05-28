from __future__ import annotations

import plotly.graph_objects as go

from .base_plotly import AndesPlotlyTheme

_REF_LINE_COLOR = "#888888"


def build_pp_figure(
    results,
    *,
    plot_title: str = "",
    theme: type[AndesPlotlyTheme] = AndesPlotlyTheme,
) -> go.Figure | None:
    """Build a Probability-Probability plot.

    results: either
      - list of (x, y) float pairs (PlotPp — single series), or
      - list of (pairs_list, category_str) tuples (PlotCategoryPp — multi series).
    Both cases use axes fixed to [0, 1] with a 1:1 reference line.
    """
    if not results:
        return None

    fig = go.Figure()

    # Detect category mode: first element is (list_of_pairs, str)
    if isinstance(results[0], (list, tuple)) and len(results[0]) == 2 and isinstance(results[0][1], str):
        for i, (pairs, category) in enumerate(results):
            xs = [float(p[0]) for p in pairs]
            ys = [float(p[1]) for p in pairs]
            color = theme.palette[i % len(theme.palette)]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers",
                    name=str(category),
                    marker={
                        "color": theme.palette_rgba(color, theme.scatter_marker_opacity),
                        "size": theme.scatter_marker_size,
                        "line": {"color": color, "width": theme.scatter_marker_border_width},
                    },
                )
            )
    else:
        # Direct list of (x, y) pairs — single series
        xs = [float(p[0]) for p in results]
        ys = [float(p[1]) for p in results]
        color = theme.palette[0]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                name=plot_title or "PP",
                marker={
                    "color": theme.palette_rgba(color, theme.scatter_marker_opacity),
                    "size": theme.scatter_marker_size,
                    "line": {"color": color, "width": theme.scatter_marker_border_width},
                },
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="1:1 line",
            line={"color": _REF_LINE_COLOR, "width": 2, "dash": "dash"},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    return fig


def apply_pp_layout(
    fig,
    *,
    plot_title: str = "",
    x_title: str = "Theoretical",
    y_title: str = "Empirical",
) -> None:
    AndesPlotlyTheme.apply_common_layout(fig)
    fig.update_layout(
        title_text=plot_title,
        hovermode="closest",
        template="plotly_white",
        yaxis={"scaleanchor": "x", "scaleratio": 1},
    )
    fig.update_xaxes(title_text=x_title, range=[0, 1])
    fig.update_yaxes(title_text=y_title, range=[0, 1])
