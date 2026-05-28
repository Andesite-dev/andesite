from __future__ import annotations

andes_scatter_palette = {
    "andes_deep_blue": {
        "hex": "#3F648E",
        "oklch": "oklch(0.494 0.080 252.5)",
        "hsl": "hsl(212 39% 40%)",
    },
    "andes_aqua_teal": {
        "hex": "#5AB0AB",
        "oklch": "oklch(0.702 0.084 190.4)",
        "hsl": "hsl(177 35% 52%)",
    },
    "andes_moss_green": {
        "hex": "#80A16B",
        "oklch": "oklch(0.670 0.085 133.8)",
        "hsl": "hsl(97 22% 53%)",
    },
    "andes_slate_purple": {
        "hex": "#736CA4",
        "oklch": "oklch(0.561 0.086 289.0)",
        "hsl": "hsl(248 24% 53%)",
    },
    "andes_graphite": {
        "hex": "#4A4A4A",
        "oklch": "oklch(0.409 0.000 89.9)",
        "hsl": "hsl(0 0% 29%)",
    },
    "andes_pale_mint": {
        "hex": "#9DCAC5",
        "oklch": "oklch(0.806 0.048 187.9)",
        "hsl": "hsl(173 30% 70%)",
    },
    "andes_steel_blue": {
        "hex": "#6A96C5",
        "oklch": "oklch(0.660 0.085 250.8)",
        "hsl": "hsl(211 44% 59%)",
    },
}

PALETTE_HEX = [v["hex"] for v in andes_scatter_palette.values()]


class AndesPlotlyTheme:
    """Pure-Python Plotly styling constants and layout helpers. No PyQt5 dependency."""

    # Palette exposed for theme-aware build_ functions
    palette: list[str] = PALETTE_HEX
    primary_color: str = andes_scatter_palette["andes_deep_blue"]["hex"]    # "#3F648E"
    secondary_color: str = andes_scatter_palette["andes_aqua_teal"]["hex"]  # "#5AB0AB"
    colorbar_tick_color: str = "black"

    font_family = "sans-serif"
    font_color = "black"

    title_font_size = 28
    axis_title_font_size = 24
    axis_tick_font_size = 15
    legend_font_size = 20
    legend_title_font_size = 20

    grid_color = "#A0A0A0"
    grid_width = 1.3
    grid_dash = "dash"
    zero_line_color = "black"
    axis_line_color = "black"
    axis_line_width = 2.0
    tick_color = "black"
    tick_width = 1.0
    tick_len = 4
    tick_direction = "outside"
    tick_xlabel_position = "outside"
    tick_ylabel_position = "outside"

    plot_bgcolor = "#ffffff"
    paper_bgcolor = "#ffffff"

    bar_color_base_opacity = 0.4
    bar_color_border_opacity = 0.9
    bar_color_border_width = 2.0
    bar_kde_line_width = 4
    bar_narrow_color_base_opacity = 0.4
    bar_narrow_color_border_opacity = 0.9
    bar_narrow_color_border_width = 2.0

    violin_color_base_opacity = 0.4
    violin_color_border_opacity = 0.9
    violin_color_border_width = 2.0
    violin_box_color_border_width = 2.0
    violin_box_color_border_opacity = 1.0
    violin_outlier_color_opacity = 0.4
    violin_outlier_border_opacity = 0.4
    violin_outlier_border_width = 1.0

    scatter_marker_size = 10
    scatter_marker_opacity = 0.4
    scatter_marker_border_opacity = 1.0
    scatter_marker_border_width = 1.5

    @staticmethod
    def palette_rgba(hex_color: str, alpha: float) -> str:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    @staticmethod
    def fig_config() -> dict:
        return {
            "displayModeBar": True,
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": False,
            "doubleClick": "reset",
            "toImageButtonOptions": {
                "format": "png",
                "filename": "plot",
                "height": None,
                "width": None,
                "scale": 2,
            },
        }

    @classmethod
    def apply_common_layout(cls, fig, *, show_legend: bool = True, uirevision: str = "plot") -> None:
        fig.update_layout(
            autosize=True,
            showlegend=show_legend,
            margin={"l": 40, "r": 20, "t": 50, "b": 80},
            font={
                "family": cls.font_family,
                "color": cls.font_color,
                "size": cls.axis_tick_font_size,
            },
            title={
                "x": 0.5,
                "xanchor": "center",
                "font": {
                    "family": cls.font_family,
                    "size": cls.title_font_size,
                    "color": cls.font_color,
                },
            },
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": -0.25,
                "xanchor": "center",
                "x": 0.5,
                "font": {
                    "family": "sans-serif",
                    "size": cls.legend_font_size,
                    "color": "black",
                },
                "title": {
                    "text": "",
                    "font": {
                        "family": "sans-serif",
                        "size": cls.legend_title_font_size,
                        "color": "black",
                    },
                },
            },
            uirevision=uirevision,
            template="plotly_white",
        )

        try:
            fig.update_xaxes(
                automargin=True,
                showgrid=False,
                gridcolor=cls.grid_color,
                gridwidth=cls.grid_width,
                griddash=cls.grid_dash,
                zeroline=False,
                showline=True,
                linecolor=cls.axis_line_color,
                linewidth=cls.axis_line_width,
                ticks=cls.tick_direction,
                tickcolor=cls.tick_color,
                ticklabelposition=cls.tick_xlabel_position,
                tickwidth=cls.tick_width,
                ticklen=cls.tick_len,
                title_font={
                    "family": cls.font_family,
                    "size": cls.axis_title_font_size,
                    "color": cls.font_color,
                },
                tickfont={
                    "family": cls.font_family,
                    "size": cls.axis_tick_font_size,
                    "color": cls.font_color,
                },
            )
            fig.update_yaxes(
                automargin=True,
                showgrid=True,
                gridcolor=cls.grid_color,
                gridwidth=cls.grid_width,
                griddash=cls.grid_dash,
                zeroline=False,
                zerolinecolor=cls.zero_line_color,
                zerolinewidth=1.0,
                showline=True,
                linecolor=cls.axis_line_color,
                ticklabelposition=cls.tick_ylabel_position,
                linewidth=cls.axis_line_width,
                ticks=cls.tick_direction,
                tickcolor=cls.tick_color,
                tickwidth=cls.tick_width,
                ticklen=cls.tick_len,
                title_font={
                    "family": cls.font_family,
                    "size": cls.axis_title_font_size,
                    "color": cls.font_color,
                },
                tickfont={
                    "family": cls.font_family,
                    "size": cls.axis_tick_font_size,
                    "color": cls.font_color,
                },
            )
        except Exception:  # nosec B110
            pass


# ---------------------------------------------------------------------------
# Nevado theme — import Nevado palettes at module level (no circular dep)
# ---------------------------------------------------------------------------

from .plotly_plots_nevado import (  # noqa: E402
    NEVADO_QUALITATIVE_DARK,
    NEVADO_QUALITATIVE_LIGHT,
)


class NevadoPlotlyThemeDark(AndesPlotlyTheme):
    """Nevado dark theme — transparent backgrounds, steel-blue primary palette."""

    palette: list[str] = NEVADO_QUALITATIVE_DARK
    primary_color: str = "#6FC3DF"
    secondary_color: str = "#CF8A5F"
    colorbar_tick_color: str = "#f0f0f2"

    plot_bgcolor: str = "rgba(0,0,0,0)"
    paper_bgcolor: str = "rgba(0,0,0,0)"
    font_color: str = "#f0f0f2"
    grid_color: str = "#333333"
    axis_line_color: str = "#606060"
    zero_line_color: str = "#444444"
    tick_color: str = "#aaaaaa"

    @classmethod
    def apply_common_layout(cls, fig, *, show_legend: bool = True, uirevision: str | None = None, **kwargs) -> None:
        from .plotly_plots_nevado import apply_nevado_theme
        apply_nevado_theme(fig, is_dark=True, show_legend=show_legend, **kwargs)


class NevadoPlotlyThemeLight(AndesPlotlyTheme):
    """Nevado light theme — transparent backgrounds, dark steel-blue primary palette."""

    palette: list[str] = NEVADO_QUALITATIVE_LIGHT
    primary_color: str = "#3d6178"
    secondary_color: str = "#7a4520"
    colorbar_tick_color: str = "#181818"

    plot_bgcolor: str = "rgba(0,0,0,0)"
    paper_bgcolor: str = "rgba(0,0,0,0)"
    font_color: str = "#181818"
    grid_color: str = "#cccccc"
    axis_line_color: str = "#909090"
    zero_line_color: str = "#999999"
    tick_color: str = "#444444"

    @classmethod
    def apply_common_layout(cls, fig, *, show_legend: bool = True, uirevision: str | None = None, **kwargs) -> None:
        from .plotly_plots_nevado import apply_nevado_theme
        apply_nevado_theme(fig, is_dark=False, show_legend=show_legend, **kwargs)


class NevadoPlotlyTheme:
    """Namespace — use ``NevadoPlotlyTheme.dark`` or ``NevadoPlotlyTheme.light``."""

    dark = NevadoPlotlyThemeDark
    light = NevadoPlotlyThemeLight
