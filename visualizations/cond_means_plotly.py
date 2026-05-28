from __future__ import annotations

import plotly.graph_objects as go

from .base_plotly import AndesPlotlyTheme


def build_cond_means_figure(
    results,
    *,
    theme: type[AndesPlotlyTheme] = AndesPlotlyTheme,
) -> go.Figure | None:
    """Accept raw output from CondMeansCalculation.

    Without category: results is a tuple/list where results[1] is [(x, y), ...].
    With category: results is [(pairs_obj, category_str), ...] where pairs_obj[1] is [(x, y), ...].
    """
    if results is None:
        return None

    fig = go.Figure()

    if isinstance(results[0], tuple):
        for i, (pairs_obj, category) in enumerate(results):
            pairs = pairs_obj[1]
            xs = [float(p[0]) for p in pairs]
            ys = [float(p[1]) for p in pairs]
            color = theme.palette[i % len(theme.palette)]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines+markers",
                    name=str(category),
                    marker={"color": color, "size": 8},
                    line={"color": color, "width": 2},
                )
            )
    else:
        pairs = results[1]
        xs = [float(p[0]) for p in pairs]
        ys = [float(p[1]) for p in pairs]
        color = theme.palette[0]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                marker={"color": color, "size": 8},
                line={"color": color, "width": 2},
                showlegend=False,
            )
        )

    return fig


def apply_cond_means_layout(
    fig,
    *,
    plot_title: str = "",
    x_title: str = "",
    y_title: str = "",
) -> None:
    AndesPlotlyTheme.apply_common_layout(fig)
    fig.update_layout(
        title_text=plot_title,
        hovermode="closest",
        template="plotly_white",
    )
    fig.update_xaxes(title_text=x_title)
    fig.update_yaxes(title_text=y_title)
