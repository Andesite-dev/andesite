from .cross_validation import crossval_plot, regression_report
from .histogram import histograma
from .inspector import (
    DataframeInspector,
    plot_cross_validation,
    plot_feature_correlation,
    plot_feature_importances,
    plot_timeseries_grouped,
)
from .slicer_view_model import pixelplt

__all__ = [
    "DataframeInspector",
    "crossval_plot",
    "histograma",
    "pixelplt",
    "plot_cross_validation",
    "plot_feature_correlation",
    "plot_feature_importances",
    "plot_timeseries_grouped",
    "regression_report",
]
