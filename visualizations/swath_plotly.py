from __future__ import annotations

import plotly.graph_objects as go

from .base_plotly import AndesPlotlyTheme


def build_swath_figure(
    results,
    *,
    theme: type[AndesPlotlyTheme] = AndesPlotlyTheme,
) -> go.Figure | None:
    """Return the swath plot figure built by export_moving_means.

    results: a go.Figure already assembled by andesite.clasification.validation.export_moving_means.
    """
    if results is None:
        return None
    return results


def apply_swath_layout(
    fig,
    *,
    plot_title: str = "",
) -> None:
    AndesPlotlyTheme.apply_common_layout(fig)
    fig.update_layout(
        title_text=plot_title,
        width=None,
        height=None,
        template="plotly_white",
    )
    fig.update_layout(yaxis2={"showgrid": False})
