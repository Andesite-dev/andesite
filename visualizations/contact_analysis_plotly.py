from __future__ import annotations

import plotly.graph_objects as go

from .base_plotly import AndesPlotlyTheme


def build_contact_analysis_figure(
    df,
    *,
    theme: type[AndesPlotlyTheme] = AndesPlotlyTheme,
) -> go.Figure | None:
    """Accept the DataFrame returned by PlotContactAnalysis.eda_calculate (steps > 0 filtered).

    Expects columns: 'steps', 'head_<label>', 'tail_<label>'.
    Head values are plotted at +steps/2, tail values at -steps/2.
    """
    if df is None or df.empty:
        return None

    head_cols = [c for c in df.columns if c.startswith("head_")]
    tail_cols = [c for c in df.columns if c.startswith("tail_")]

    if not head_cols or not tail_cols:
        return None

    head_col = head_cols[0]
    tail_col = tail_cols[0]

    try:
        head_label = head_col.split("_")[1]
        tail_label = tail_col.split("_")[1]
        target_var = tail_col.split("_")[2] if len(tail_col.split("_")) > 2 else "Variable"
    except IndexError:
        head_label, tail_label, target_var = "Head", "Tail", "Variable"

    steps = df["steps"].values
    x_head = steps / 2.0
    x_tail = -steps / 2.0
    y_head = df[head_col].values
    y_tail = df[tail_col].values

    tail_pts = sorted(zip(x_tail, y_tail))
    head_pts = sorted(zip(x_head, y_head))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[p[0] for p in tail_pts],
            y=[p[1] for p in tail_pts],
            mode="lines+markers",
            name=tail_label,
            marker={"color": theme.palette[0], "size": 8},
            line={"color": theme.palette[0], "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[p[0] for p in head_pts],
            y=[p[1] for p in head_pts],
            mode="lines+markers",
            name=head_label,
            marker={"color": theme.palette[1], "size": 8},
            line={"color": theme.palette[1], "width": 2},
        )
    )
    return fig


def apply_contact_analysis_layout(
    fig,
    *,
    plot_title: str = "",
    x_title: str = "Distance to UG contact",
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
