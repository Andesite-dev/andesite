from __future__ import annotations

import plotly.graph_objects as go

from .base_plotly import AndesPlotlyTheme


def build_correlation_figure(
    corr_matrix,
    *,
    theme: type[AndesPlotlyTheme] = AndesPlotlyTheme,
) -> go.Figure | None:
    if corr_matrix is None or corr_matrix.empty:
        return None

    z = corr_matrix.to_numpy()[::-1]
    heatmap = go.Heatmap(
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index[::-1].tolist(),
        z=z,
        colorscale="Deep",
        zmid=0,
        zmin=-1,
        zmax=1,
        hovertemplate=(
            "<b>Variables:</b><br>"
            "%{x}<br>%{y}<br>"
            "<b>Correlación:</b><br>%{z:.2f}<extra></extra>"
        ),
        hoverlabel={"font_size": 14, "font_family": "Helvetica"},
        colorbar={"tickfont": {"color": theme.colorbar_tick_color}},
        text=z,
        texttemplate="%{text:.2f}",
    )

    return go.Figure(data=[heatmap])


def apply_correlation_layout(fig) -> None:
    fig.update_layout(title_text="Correlation Matrix")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False, scaleanchor="x", scaleratio=1)
