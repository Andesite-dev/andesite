"""
plotly_plots.py — Nevado Plotly theming utility.

Apply consistent dark / light styling to any go.Figure before returning it
from a plot function.  Works with 2-D figures, figures with multiple subplot
axes (make_subplots / secondary Y), and 3-D scene figures.

Quick usage
-----------
    from plotly_plots import apply_nevado_theme, PLOTLY_CONFIG, PLOTLY_3D_CONFIG

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))
    fig = apply_nevado_theme(fig, is_dark=True, xaxis_title="Time", yaxis_title="Value")

    # In a Reflex page:
    rx.plotly(data=MyState.my_fig, config=PLOTLY_CONFIG, width="100%", height="400px")
"""

from __future__ import annotations

from dataclasses import dataclass

import plotly.graph_objs as go

@dataclass(frozen=True)
class NevadoColors:
    """All color tokens consumed by Plotly layout / trace helpers."""

    # Backgrounds — always transparent so the app CSS shines through
    bg: str
    bg_light: str

    # Typography
    text: str
    text_muted: str

    # Brand palette (maps to --primary / --secondary / success in styles.css)
    primary: str
    secondary: str
    tertiary: str

    # Axes / grid
    axis: str        # tick labels, axis line
    grid: str        # major gridlines
    zeroline: str    # zero-value line
    border: str      # axis border / line

    # Trace helpers for ensemble / simulation plots
    sim_trace: str   # semi-transparent for background simulation lines
    mean_trace: str  # opaque for mean / highlight lines

    # Colorbars
    colorbar_tick: str

    # 3-D wireframe edges
    edge_3d: str


_DARK = NevadoColors(
    # -- backgrounds
    bg="rgba(0,0,0,0)",
    bg_light="rgba(0,0,0,0)",
    # -- typography  (--text / --text-muted in styles.css dark mode)
    text="#f0f0f2",
    text_muted="#aaaaaa",
    # -- brand  (#6FC3DF = oklch(0.76 0.1 220), same hex used in plots.py)
    primary="#6FC3DF",
    secondary="#CF8A5F",
    tertiary="#7DB599",
    # -- axes / grid  (hardcoded in plot_3d_grid / plot_tonnage_grade)
    axis="#aaaaaa",
    grid="#333333",
    zeroline="#444444",
    border="#606060",
    # -- ensemble traces  (from plot_tonnage_grade is_dark branch)
    sim_trace="rgba(111, 195, 223, 0.18)",
    mean_trace="rgba(240, 240, 240, 0.92)",
    # -- colorbars  (from create_simulation_heatmap)
    colorbar_tick="#f8f8ff",
    # -- 3-D wireframe  (_EDGE_COLOR in plots.py)
    edge_3d="#000000",
)

_LIGHT = NevadoColors(
    # -- backgrounds
    bg="rgba(0,0,0,0)",
    bg_light="rgba(0,0,0,0)",
    # -- typography  (--text / --text-muted in styles.css light mode)
    text="#181818",
    text_muted="#555555",
    # -- brand  (--primary light = hsl(203 33.3% 35.3%) ≈ #3d6178)
    primary="#3d6178",
    secondary="#7a4520",
    tertiary="#336b4a",
    # -- axes / grid  (from plot_tonnage_grade light branch)
    axis="#444444",
    grid="#cccccc",
    zeroline="#999999",
    border="#909090",
    # -- ensemble traces  (from plot_tonnage_grade light branch)
    sim_trace="rgba(111, 195, 223, 0.35)",
    mean_trace="rgba(0, 0, 0, 0.87)",
    # -- colorbars
    colorbar_tick="#333333",
    # -- 3-D wireframe
    edge_3d="#ffffff",
)


def get_nevado_colors(is_dark: bool = True) -> NevadoColors:
    """Return the color token set for the requested theme."""
    return _DARK if is_dark else _LIGHT

# Ordered to lead with the three brand colors already used in the existing
# variogram/geology plots, then extend with complementary hues.
NEVADO_QUALITATIVE_DARK: list[str] = [
    "#6FC3DF",  # primary — steel blue
    "#CF8A5F",  # secondary — burnt sienna
    "#7DB599",  # tertiary — sage green
    "#9B7DC4",  # lavender
    "#D4A85A",  # amber
    "#D47A7A",  # dusty rose
    "#5FA8D4",  # sky blue
    "#A8C47A",  # lime
    "#C47AA8",  # mauve
    "#7AA8C4",  # slate
]

NEVADO_QUALITATIVE_LIGHT: list[str] = [
    "#3d6178",  # primary — dark steel
    "#7a4520",  # secondary — dark sienna
    "#336b4a",  # tertiary — dark sage
    "#5a3e8a",  # dark lavender
    "#8a6420",  # dark amber
    "#8a3c3c",  # dark rose
    "#2a6a8a",  # dark sky
    "#5a7a2a",  # dark lime
    "#7a3a6a",  # dark mauve
    "#2a5a7a",  # dark slate
]

#: Standard config for 2-D / heatmap figures — no toolbar, no Plotly logo.
PLOTLY_CONFIG: dict = {
    "displayModeBar": False,
    "displaylogo": False,
    "scrollZoom": False,
}

#: Config for 3-D scene figures — scroll-zoom enabled for orbit navigation.
PLOTLY_3D_CONFIG: dict = {
    "displayModeBar": False,
    "displaylogo": False,
    "scrollZoom": True,
}


_FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"

_DEFAULT_MARGIN = {"l": 40, "r": 20, "t": 30, "b": 40}

_MESH3D_LIGHTING = {
    "ambient": 0.5,
    "diffuse": 1.0,
    "fresnel": 4.0,
    "specular": 0.5,
    "roughness": 0.5,
}

def apply_nevado_theme(
    fig: go.Figure,
    is_dark: bool = True,
    *,
    margin: dict | None = None,
    title: str | None = None,
    height: int | None = None,
    width: int | None = None,
    show_legend: bool = True,
    legend_title: str | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    colorbar: bool = True,
) -> go.Figure:
    """Apply the Nevado dark / light theme to *any* ``go.Figure`` in-place.

    This function stamps consistent styling on top of whatever data traces
    already exist.  It never touches trace data or axis *ranges* — only
    cosmetic properties (colors, fonts, backgrounds, margins).

    Parameters
    ----------
    fig:
        Any ``go.Figure`` — 2-D, 3-D, subplots with secondary Y, heatmap, etc.
    is_dark:
        ``True`` (default) for the dark theme; ``False`` for light.
    margin:
        Override the default ``{"l": 40, "r": 20, "t": 30, "b": 40}``.
    title:
        Optional figure title text.
    height, width:
        Optional figure dimensions in pixels.
    show_legend:
        Whether to show the legend.  Defaults to ``True``.
    legend_title:
        Optional legend title text.
    xaxis_title, yaxis_title:
        If supplied, sets the title on the *first* x / y axis only.
        Leave ``None`` to preserve whatever title was already set.
    colorbar:
        Apply consistent colorbar font styling to all coloraxis / traces.
        Defaults to ``True``.

    Returns
    -------
    go.Figure
        The same figure (mutated in-place), returned for chaining.

    Examples
    --------
    Basic 2-D plot::

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6], name="series"))
        apply_nevado_theme(fig, xaxis_title="X", yaxis_title="Y")

    Dual-Y subplot (make_subplots)::

        from plotly.subplots import make_subplots
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(...), secondary_y=False)
        fig.add_trace(go.Scatter(...), secondary_y=True)
        apply_nevado_theme(fig)
        # Both yaxis and yaxis2 receive identical axis styling.

    3-D scene figure::

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=..., y=..., z=..., mode="markers"))
        apply_nevado_theme(fig, is_dark=False)

    Histogram / Violin / Box — no special handling needed::

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=data, name="dist"))
        apply_nevado_theme(fig, xaxis_title="Value", yaxis_title="Count")
    """
    c = get_nevado_colors(is_dark)
    m = margin if margin is not None else _DEFAULT_MARGIN.copy()


    layout_kwargs: dict = {
        "paper_bgcolor": c.bg,
        "plot_bgcolor": c.bg,
        "margin": m,
        "font": {"family": _FONT_FAMILY, "color": c.text},
        "showlegend": show_legend,
        "legend": {
            "font": {"color": c.text_muted, "family": _FONT_FAMILY},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
        },
    }

    if title is not None:
        layout_kwargs["title"] = {
            "text": title,
            "font": {"color": c.text, "size": 14, "family": _FONT_FAMILY},
        }

    if height is not None:
        layout_kwargs["height"] = height
    if width is not None:
        layout_kwargs["width"] = width

    if legend_title is not None:
        layout_kwargs["legend"]["title"] = {
            "text": legend_title,
            "font": {"color": c.text_muted},
        }

    fig.update_layout(**layout_kwargs)

    _has_3d = _figure_has_3d(fig)

    if not _has_3d:
        x_kwargs: dict = {
            "color": c.axis,
            "gridcolor": c.grid,
            "zerolinecolor": c.zeroline,
            "linecolor": c.border,
            "tickcolor": c.axis,
        }
        y_kwargs: dict = dict(x_kwargs)

        if xaxis_title is not None:
            x_kwargs["title"] = {"text": xaxis_title, "font": {"color": c.text_muted}}
        if yaxis_title is not None:
            y_kwargs["title"] = {"text": yaxis_title, "font": {"color": c.text_muted}}

        fig.update_xaxes(**x_kwargs)
        fig.update_yaxes(**y_kwargs)

    if _has_3d:
        _axis_3d: dict = {
            "color": c.axis,
            "gridcolor": c.grid,
            "backgroundcolor": "rgba(0,0,0,0)",
            "showgrid": True,
            "zeroline": False,
        }
        scene_kwargs: dict = {
            "xaxis": dict(_axis_3d),
            "yaxis": dict(_axis_3d),
            "zaxis": dict(_axis_3d),
            "bgcolor": "rgba(0,0,0,0)",
        }
        if xaxis_title is not None:
            scene_kwargs["xaxis"]["title"] = xaxis_title
        if yaxis_title is not None:
            scene_kwargs["yaxis"]["title"] = yaxis_title

        fig.update_scenes(**scene_kwargs)

    if colorbar:
        _colorbar_font = {
            "tickfont": {
                "size": 13,
                "color": c.colorbar_tick,
                "family": _FONT_FAMILY,
            },
        }
        for trace in fig.data:
            if hasattr(trace, "colorbar") and trace.colorbar is not None:
                trace.update(colorbar=_colorbar_font)
            # Heatmap exposes colorbar differently — both paths work via the
            # same attribute name so the check above covers it.

    return fig


def nevado_axis(
    is_dark: bool = True,
    title: str | None = None,
    show_grid: bool = True,
) -> dict:
    """Return a dict suitable for ``fig.update_layout(xaxis=nevado_axis(...))``.

    Useful when you need fine-grained per-axis control rather than the
    blanket ``apply_nevado_theme`` approach.
    """
    c = get_nevado_colors(is_dark)
    d: dict = {
        "color": c.axis,
        "gridcolor": c.grid if show_grid else "rgba(0,0,0,0)",
        "showgrid": show_grid,
        "zerolinecolor": c.zeroline,
        "linecolor": c.border,
        "tickcolor": c.axis,
    }
    if title is not None:
        d["title"] = {"text": title, "font": {"color": c.text_muted}}
    return d


def nevado_colorbar(is_dark: bool = True, title: str | None = None) -> dict:
    """Return a ``colorbar`` dict for Heatmap / Scatter / Mesh3d traces.

    Usage::

        fig.add_trace(go.Heatmap(
            x=..., y=..., z=...,
            colorbar=nevado_colorbar(is_dark=True, title="Cu %"),
        ))
    """
    c = get_nevado_colors(is_dark)
    d: dict = {
        "tickfont": {"size": 13, "color": c.colorbar_tick, "family": _FONT_FAMILY},
        "len": 1.0,
    }
    if title is not None:
        d["title"] = {"text": title, "font": {"color": c.colorbar_tick}}
    return d


def nevado_mesh_lighting() -> dict:
    """Return the standard Mesh3d ``lighting`` dict used across all 3-D plots."""
    return dict(_MESH3D_LIGHTING)


def _figure_has_3d(fig: go.Figure) -> bool:
    """Return True if the figure contains any 3-D trace types."""
    _3d_types = (go.Scatter3d, go.Mesh3d, go.Cone, go.Streamtube, go.Volume, go.Isosurface)
    return any(isinstance(t, _3d_types) for t in fig.data)


if __name__ == "__main__":
    from plotly.subplots import make_subplots

    # 2-D scatter
    fig2d = go.Figure()
    fig2d.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6], name="series"))
    apply_nevado_theme(fig2d, xaxis_title="X", yaxis_title="Y")
    assert fig2d.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert fig2d.layout.xaxis.color == "#aaaaaa"
    print("[OK] 2-D scatter — dark theme")

    # Light theme
    fig2d_light = go.Figure()
    fig2d_light.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))
    apply_nevado_theme(fig2d_light, is_dark=False)
    assert fig2d_light.layout.xaxis.color == "#444444"
    print("[OK] 2-D scatter — light theme")

    # Dual-Y subplot
    fig_sub = make_subplots(specs=[[{"secondary_y": True}]])
    fig_sub.add_trace(go.Scatter(x=[1, 2], y=[3, 4]), secondary_y=False)
    fig_sub.add_trace(go.Scatter(x=[1, 2], y=[0.1, 0.2]), secondary_y=True)
    apply_nevado_theme(fig_sub)
    assert fig_sub.layout.yaxis2.color == "#aaaaaa"
    print("[OK] Dual-Y subplot")

    # 3-D scene
    fig3d = go.Figure()
    fig3d.add_trace(go.Scatter3d(x=[0, 1], y=[0, 1], z=[0, 1], mode="markers"))
    apply_nevado_theme(fig3d, is_dark=True)
    assert fig3d.layout.scene.bgcolor == "rgba(0,0,0,0)"
    print("[OK] 3-D scene")

    # Heatmap colorbar
    fig_hm = go.Figure()
    fig_hm.add_trace(go.Heatmap(x=[1, 2], y=[1, 2], z=[[1, 2], [3, 4]],
                                colorbar=nevado_colorbar(is_dark=True)))
    apply_nevado_theme(fig_hm)
    print("[OK] Heatmap with colorbar")

    print("\nAll checks passed.")
