from typing import List, Literal, Optional, Union, TypeAlias # type: ignore
from sklearn.pipeline import Pipeline
from ..utils import (
    MethodNotImplementedException,
    NotNumericColumnException,
    NotCategoricalColumnException,
    NotDateTimeColumnException,
    ColumnNotFoundedException,
    SameColumnNameException,
    EmptyArrayException
)
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import polars as pl
import pandas as pd
import numpy as np

MARKER_COLOR = "#D7282F"

DataFrameType: TypeAlias = Union[pd.DataFrame, pl.DataFrame]

class DataframeInspector():
    ENGINES = ["polars", "pandas"]
    DATETIME_DTYPES_POLARS = [pl.Date, pl.Datetime]
    STRING_DTYPES_POLARS = [pl.Utf8, pl.String, pl.Categorical, pl.Enum]
    NUMERIC_DTYPES_POLARS = [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64]
    CATEGORICAL_ID_THRESHOLD = 40
    NCOLS_GRID = 3
    SUBPLOTS_VERTICAL_SPACING = 0.25
    SUBPLOTS_HORIZONTAL_SPACING = 0.06

    def __init__(self, dataframe: DataFrameType, engine: str = "polars"): # type: ignore
        if isinstance(dataframe, pd.DataFrame):
            self.dataframe = pl.DataFrame(dataframe)
        else:
            self.dataframe = dataframe
        self.engine = engine
        self.__check_possible_engines()
        self.__check_engine_matching()
        self.__classify_column_types()

    def __check_possible_engines(self):
        assert self.engine in self.ENGINES, f"Please specify a valid engine value\n{self.ENGINES}"

    def __check_engine_matching(self):
        if self.engine == "polars" and not isinstance(self.dataframe, pl.DataFrame):
            raise ValueError("The dataframe should be of type polars.DataFrame")
        elif self.engine == "pandas" and not isinstance(self.dataframe, pd.DataFrame):
            raise ValueError("The dataframe should be of type pandas.DataFrame")

    def __classify_column_types(self):
        self.categorical_columns: List[str] = []
        self.numerical_columns: List[str] = []
        self.id_columns: List[str] = []
        self.datetime_columns: List[str] = []
        for c in self.dataframe.columns:
            if isinstance(self.dataframe, pl.DataFrame):
                unique_samples = self.dataframe[c].unique().count()
                dtype = self.dataframe[c].dtype
                if dtype in self.DATETIME_DTYPES_POLARS:
                    self.datetime_columns.append(c)
                elif (unique_samples < self.CATEGORICAL_ID_THRESHOLD) and dtype not in [pl.Float32, pl.Float64]:
                    self.categorical_columns.append(c)
                else:
                    if dtype in self.NUMERIC_DTYPES_POLARS:
                        self.numerical_columns.append(c)
                    else:
                        self.id_columns.append(c)

    def _get_categorical_columns(self) -> List[str]:
        return self.categorical_columns

    def _get_numerical_columns(self) -> List[str]:
        return self.numerical_columns

    def _get_id_columns(self) -> List[str]:
        return self.id_columns

    def _get_datetime_columns(self) -> List[str]:
        return self.datetime_columns

    def scatter(self) -> go.Figure:
        raise MethodNotImplementedException()

    def box_plot(
        self,
        col: str,
        title: str = "",
        figsize: tuple = (400, 350)
    ) -> go.Figure:
        if self.dataframe[col].dtype not in self.NUMERIC_DTYPES_POLARS:
            raise NotNumericColumnException(f"`{col}` is not a numeric column")
        if col not in self.dataframe.columns:
            raise ColumnNotFoundedException(f"`{col}` not founded in dataframe")
        fig = go.Figure()
        fig.add_trace(
            go.Box(
                y = self.dataframe[col],
                marker_color = MARKER_COLOR,
                name = col,
                hovertemplate = f"{col}" + ": %{y:.2f}<br><extra></extra>",
                showlegend = False
            )
        )
        fig.update_layout(
            margin = dict(l=10, r=20, t=40, b=10),
            height = figsize[1],
            width = figsize[0],
            title = title,
        )
        return fig

    def grid_boxplot(
        self,
        columns: List[str] = [],
        column_selector: str = "pick",
        all_positives: bool = False
    ) -> go.Figure:
        if (len(columns) == 0) and (column_selector != "all"):
            raise EmptyArrayException("<columns> parameter got empty. Please provide a list of columns or choose `all` on <column_selector> parameter")
        for c in columns:
            if self.dataframe[c].dtype not in self.NUMERIC_DTYPES_POLARS:
                raise NotNumericColumnException(f"`{c}` is not a numeric column")
            if c not in self.dataframe.columns:
                raise ColumnNotFoundedException(f"`{c}` not founded in dataframe")

        if column_selector == "all":
            columns = [col for col in self.dataframe.columns if self.dataframe[col].dtype in self.NUMERIC_DTYPES_POLARS]
        # Create a grid of nrows, 3 columns
        ncols = 3
        nrows = int(np.ceil(len(columns) / ncols))
        # pad_size is the number of empty spaces needed to fill the grid
        pad_size = (3 - len(columns) % 3) % 3
        # Padded array of columns to fill the grid with empty spaces at the end
        padded_arr = np.pad(columns, (0, pad_size), mode = 'constant', constant_values = "").reshape(-1, 3)
        fig = make_subplots(
            rows = nrows,
            cols = ncols,
            subplot_titles = columns,
            vertical_spacing = self.SUBPLOTS_VERTICAL_SPACING / nrows,
            horizontal_spacing = self.SUBPLOTS_HORIZONTAL_SPACING
        )

        # Add box plots to the grid subplots based on the padded_arr array
        for row in range(nrows):
            for col in range(ncols):
                variable_name = padded_arr[row, col]
                if variable_name == "":
                    continue
                if all_positives:
                    data = self.dataframe.select(variable_name).filter(pl.col(variable_name) >= 0)[variable_name]
                else:
                    data = self.dataframe[variable_name]
                fig.add_trace(
                    go.Box(
                    y = data,
                    marker_color = MARKER_COLOR,
                    name = variable_name,
                    hovertemplate = f"{variable_name}" + ": %{y:.2f}<br><extra></extra>",
                    showlegend = False
                ), row = row + 1, col = col + 1)
        fig.update_layout(
            width = 350*ncols,
            height = 300*nrows,
            margin = dict(t=30,l=20,r=20,b=20)
        )
        return fig

    # TODO: split-by Histogram (grid of histograms based on a column categoric)
    def histogram(
        self,
        var: str,
        nbins: int,
        start: Optional[Union[int, float]] = None,
        end: Optional[Union[int, float]] = None,
        title: str = "",
        color: str = MARKER_COLOR,
        split_by: Optional[str] = None,
        all_positives: bool = False,
        xaxis_title: Optional[str] = None,
        yaxis_title: Optional[str] = None,
        figsize: tuple = (400, 350)
    ) -> go.Figure:
        if self.dataframe[var].dtype not in self.NUMERIC_DTYPES_POLARS:
            raise NotNumericColumnException(f"`{var}` is not a numeric column")
        if var not in self.dataframe.columns:
            raise ColumnNotFoundedException(f"`{var}` not founded in dataframe")

        if split_by:
            if split_by not in self.dataframe.columns:
                raise ColumnNotFoundedException(f"`{split_by}` not founded in dataframe")
            unique_cats = self.dataframe.drop_nulls()[split_by].unique(maintain_order = True).to_list()
            ncols = 3
            nrows = int(np.ceil(len(unique_cats) / ncols))
            # pad_size is the number of empty spaces needed to fill the grid
            pad_size = (3 - len(unique_cats) % 3) % 3
            # Padded array of columns to fill the grid with empty spaces at the end
            padded_arr = np.pad(unique_cats, (0, pad_size), mode = 'constant', constant_values = "").reshape(-1, 3)
            fig = make_subplots(
                rows = nrows,
                cols = ncols,
                subplot_titles = unique_cats,
                vertical_spacing = self.SUBPLOTS_VERTICAL_SPACING / nrows,
                horizontal_spacing = self.SUBPLOTS_HORIZONTAL_SPACING
            )
            for row in range(nrows):
                for col in range(ncols):
                    variable_name = padded_arr[row, col]
                    if variable_name == "":
                        continue
                    subplot_data = self.dataframe.filter(pl.col(split_by) == variable_name)
                    if all_positives:
                        start_bins = subplot_data.select(var).filter(pl.col(var) >= 0)[var].min() # type: ignore
                        end_bins = subplot_data.select(var).filter(pl.col(var) >= 0)[var].max() # type: ignore
                    else:
                        start_bins = subplot_data[var].min() # type: ignore
                        end_bins = subplot_data[var].max() # type: ignore
                    fig.add_trace(
                        go.Histogram(
                            x = subplot_data[var],
                            xbins = dict(
                                start = start_bins,
                                end = end_bins,
                                size = (end_bins - start_bins)/nbins # type: ignore
                            ),
                            marker_color = color,
                            showlegend = False,
                            opacity = 0.8,
                            meta = str(variable_name),
                            marker_line = dict(width = 1, color = 'black'),
                            histnorm = 'percent',
                            hovertemplate = "Groupo: %{meta}<br>" + f"{var}" + ": %{x}<br>Porcentaje: %{y}<br><extra></extra>"
                        ), row = row + 1, col = col + 1)
            fig.update_layout(
                width = 350 * ncols,
                height = 300 * nrows,
                title = {
                    "text": f"<b>Histogram {var} split by {split_by}</b>",
                },
                margin = dict(t=60,l=20,r=20,b=20)
            )
            for i in range(1, nrows * ncols + 1):
                fig.update_layout({f'yaxis{i}': {'ticksuffix': '%'}})

            for row in range(1, nrows + 1):
                axis_id = f'yaxis{(row - 1) * ncols + 1}'
                fig.update_layout({axis_id: {'title': 'Porcentaje (%)'}})
        else:
            if (start is not None) and (start is not None):
                fig = go.Figure()
                fig.add_trace(
                    go.Histogram(
                        x = self.dataframe[var],
                        xbins = dict(
                            start = start,
                            end = end,
                            size = (end - start)/nbins # type: ignore
                        ),
                        marker_color = MARKER_COLOR,
                        opacity = 0.8,
                        marker_line = dict(width = 1, color = 'black'),
                        histnorm = 'percent',
                        hovertemplate= f"{var}" + ": %{x}<br>Porcentaje: %{y}<br><extra></extra>"
                    )
                )
                fig.update_layout(
                    title = {
                        "text": f"<b>{title}</b>",
                    },
                    xaxis = {
                        'title': {
                            'text': "Rangos" if not xaxis_title else xaxis_title,
                        },
                    },
                    yaxis = {
                        'title': {
                            'text': 'Porcentaje (%)' if not yaxis_title else yaxis_title,
                        },
                        'ticksuffix': '% ',
                    },
                    barmode = 'relative',
                    # hovermode = "x",
                    height = figsize[1],
                    width = figsize[0],
                    margin = dict(t=60,l=20,r=20,b=20)
                )
            else:
                raise Exception("Provide a valid value for <start> an <end> parameters")
        return fig

    def grid_histogram(
        self,
        nbins: int = 15,
        columns: List[str] = [],
        all_positives: bool = False,
        column_selector: str = "pick"
    ) -> go.Figure:
        if (len(columns) == 0) and (column_selector != "all"):
            raise EmptyArrayException("<columns> parameter got empty. Please provide a list of columns or choose `all` on <column_selector> parameter")
        for c in columns:
            if self.dataframe[c].dtype not in self.NUMERIC_DTYPES_POLARS:
                raise NotNumericColumnException(f"`{c}` is not a numeric column")
            if c not in self.dataframe.columns:
                raise ColumnNotFoundedException(f"`{c}` not founded in dataframe")

        if column_selector == "all":
            columns = [col for col in self.dataframe.columns if self.dataframe[col].dtype in self.NUMERIC_DTYPES_POLARS]
        # Create a grid of nrows, 3 columns
        ncols = 3
        nrows = int(np.ceil(len(columns) / ncols))
        # pad_size is the number of empty spaces needed to fill the grid
        pad_size = (3 - len(columns) % 3) % 3
        # Padded array of columns to fill the grid with empty spaces at the end
        padded_arr = np.pad(columns, (0, pad_size), mode = 'constant', constant_values = "").reshape(-1, 3)
        fig = make_subplots(
            rows = nrows,
            cols = ncols,
            subplot_titles = columns,
            vertical_spacing = self.SUBPLOTS_VERTICAL_SPACING / nrows,
            horizontal_spacing = self.SUBPLOTS_HORIZONTAL_SPACING
        )

        # Add box plots to the grid subplots based on the padded_arr array
        for row in range(nrows):
            for col in range(ncols):
                variable_name = padded_arr[row, col]
                if variable_name == "":
                    continue
                if all_positives:
                    start = self.dataframe.select(variable_name).filter(pl.col(variable_name) >= 0)[variable_name].min()
                    end = self.dataframe.select(variable_name).filter(pl.col(variable_name) >= 0)[variable_name].max()
                else:
                    start = self.dataframe[variable_name].min()
                    end = self.dataframe[variable_name].max()
                fig.add_trace(
                    go.Histogram(
                        x = self.dataframe[variable_name],
                        xbins = dict(
                            start = start,
                            end = end,
                            size = (end - start)/nbins
                        ),
                        marker_color = MARKER_COLOR,
                        showlegend = False,
                        opacity = 0.8,
                        marker_line = dict(width = 1, color = 'black'),
                        histnorm = 'percent',
                        hovertemplate= f"{variable_name}" + ": %{x}<br>Porcentaje: %{y}<br><extra></extra>"
                    ), row = row + 1, col = col + 1)
        fig.update_layout(
            width = 350 * ncols,
            height = 300 * nrows,
            margin = dict(t=30,l=20,r=20,b=20)
        )
        for i in range(1, nrows * ncols + 1):
            fig.update_layout({f'yaxis{i}': {'ticksuffix': '%'}})
        for row in range(1, nrows + 1):
            axis_id = f'yaxis{(row - 1) * ncols + 1}'
            fig.update_layout({axis_id: {'title': 'Porcentaje (%)', 'ticksuffix': '%'}})
        return fig

    #TODO: Agregar split_by como en histograma
    def frequency_bar(
            self,
            col: str,
            as_percentage: bool = False,
            hue: Optional[str] = None,
            title: str = "",
            figsize: tuple = (400, 350)
        ) -> go.Figure:
            if self.dataframe[col].dtype not in self.STRING_DTYPES_POLARS:
                raise NotCategoricalColumnException("Please provide a categorical column on parameter <col>")
            if hue and (self.dataframe[hue].dtype not in self.STRING_DTYPES_POLARS):
                raise NotCategoricalColumnException("Please provide a categorical column on parameter <hue>")
            if col not in self.dataframe.columns:
                raise ColumnNotFoundedException(f"`{col}` not founded on dataframe")
            if hue:
                if hue not in self.dataframe.columns:
                    raise ColumnNotFoundedException(f"`{hue}` not founded on dataframe")
            if hue == col:
                raise SameColumnNameException("<hue> value cannot be the same as <col>")

            col_value_counts = self.dataframe[col].drop_nulls().value_counts().sort("count", descending = True)
            x_axis_order = col_value_counts[col].to_list()
            fig = go.Figure()
            if hue:
                sub_df = self.dataframe[[col, hue]].drop_nulls().to_pandas()
                for c in sub_df.columns:
                    sub_df[c] = sub_df[c].astype("category")
                for n, g in sub_df.groupby(hue, observed = True):
                    fig.add_bar(
                        x = g[col].value_counts().reindex(x_axis_order).index.to_list(),
                        y = g[col].value_counts().reindex(x_axis_order).to_numpy(),
                        legendgrouptitle_text = str(hue),
                        name = str(n),
                        meta = str(n),
                        marker_line=dict(width = 1, color = "#333"),
                        hovertemplate = "%{meta}: %{y}<extra></extra>"
                    )
            else:
                barchart_dataframe = (
                    self.dataframe.select(pl.col(col).cast(pl.String).alias(col))
                    .group_by(col)
                    .agg([pl.col(col).count().alias("count")])
                    .with_columns(
                        (pl.col("count") / pl.col("count").sum() * 100).alias("relative_count"),
                    )
                    .sort("count", descending = True)
                )
                y_value = "relative_count" if as_percentage else "count"
                perc_txt = "%{y:.3f}%" if as_percentage else "%{y}"
                hover_template = f"{col}:" + " %{x}<br>" + f"{y_value}: " + perc_txt + "<extra></extra>"
                fig.add_bar(
                        x = barchart_dataframe[col],
                        y = barchart_dataframe[y_value],
                        marker = dict(color = MARKER_COLOR),
                        hovertemplate = hover_template
                    )
            fig.update_xaxes(
                categoryorder = "array",
                categoryarray = x_axis_order
            )
            fig.update_layout(
                height = figsize[1],
                width = figsize[0],
                barmode = "stack",
                title = {
                    "text": f"<b>{title}</b>",
                },
                xaxis = {
                    'title': {
                        'text': f"{col}",
                    },
                    'tickangle': -45
                },
                yaxis = {
                    'title': {
                        'text': 'Porcentaje (%)' if (as_percentage and not hue) else 'Cantidad',
                    },
                    'ticksuffix': '% ' if (as_percentage and not hue) else '',
                },
                margin = dict(t=40,l=20,r=20,b=20)
            )
            return fig

    def grid_frequency_bar(
        self,
        columns: List[str] = [],
        as_percentage: bool = False,
        column_selector: str = "pick",
        include_nulls: bool = False
    ):
        if column_selector == "all":
            columns = [col for col in self.dataframe.columns if self.dataframe[col].dtype in self.STRING_DTYPES_POLARS]
        # Create a grid of nrows, 3 columns
        ncols = self.NCOLS_GRID
        nrows = int(np.ceil(len(columns) / ncols))
        # pad_size is the number of empty spaces needed to fill the grid
        pad_size = (self.NCOLS_GRID - len(columns) % self.NCOLS_GRID) % self.NCOLS_GRID
        # Padded array of columns to fill the grid with empty spaces at the end
        padded_arr = np.pad(columns, (0, pad_size), mode = 'constant', constant_values = "").reshape(-1, self.NCOLS_GRID)
        fig = make_subplots(
            rows = nrows,
            cols = ncols,
            subplot_titles = columns,
            vertical_spacing = self.SUBPLOTS_VERTICAL_SPACING / nrows,
            horizontal_spacing = self.SUBPLOTS_HORIZONTAL_SPACING
        )
        for row in range(nrows):
            for col in range(ncols):
                variable_name = padded_arr[row, col]
                if variable_name == "":
                    continue
                barchart_dataframe = (
                    self.dataframe.select(pl.col(variable_name).cast(pl.String).alias(variable_name))
                    .group_by(variable_name)
                    .agg([pl.col(variable_name).count().cast(pl.Int32).alias("count")])
                    .with_columns(
                        (pl.col("count") / pl.col("count").sum() * 100).cast(pl.Float32).alias("relative_count"),
                    )
                    .filter(pl.col(variable_name).is_not_null())
                    .sort("count", descending = True)
                )
                if include_nulls:
                    null_df = pl.DataFrame({
                        variable_name: "NULL",
                        "count": self.dataframe.select(variable_name).null_count().item(),
                        "relative_count": 0.0,
                    }, schema_overrides={variable_name: pl.String, "count": pl.Int32, "relative_count": pl.Float32})
                    barchart_dataframe = (
                        pl.concat([barchart_dataframe, null_df])
                        .with_columns(
                            (pl.col("count") / pl.col("count").sum() * 100).cast(pl.Float32).alias("relative_count"),
                        )
                        .sort("count", descending = True)
                    )
                y_value = "relative_count" if as_percentage else "count"
                perc_txt = "%{y:.3f}%" if as_percentage else "%{y}"
                hover_template = f"{variable_name}:" + " %{x}<br>" + f"{y_value}: " + perc_txt + "<extra></extra>"
                fig.add_trace(
                    go.Bar(
                        x = barchart_dataframe[variable_name],
                        y = barchart_dataframe[y_value],
                        marker = dict(color = MARKER_COLOR),
                        hovertemplate = hover_template,
                        showlegend = False
                    ), row = row + 1, col = col + 1)
        fig.update_layout(
            width = 350 * ncols,
            height = 300 * nrows,
            barmode = "stack",
            margin = dict(t=40,l=20,r=20,b=20)
        )
        # Apply the tickangle to all x-axes
        for i in range(1, nrows * ncols + 1):
            fig.update_layout({f'xaxis{i}': {'tickangle': -45}})
            if as_percentage:
                fig.update_layout({f'yaxis{i}': {'ticksuffix': '%'}})

        for row in range(1, nrows + 1):
            axis_id = f'yaxis{(row - 1) * ncols + 1}'
            if as_percentage:
                fig.update_layout({axis_id: {'title': 'Porcentaje (%)', 'ticksuffix': '%'}})
            else:
                fig.update_layout({axis_id: {'title': 'Cantidad'}})
        return fig

    # TODO: on bar() method add Hue column like in frequency_bar() method on DataframeInspector() class
    def bar(
        self,
        xcol: str,
        ycol: str,
        xaxis_title: str = "",
        yaxis_title: str = "",
        title: str = "",
        include_nulls: bool = False,
        color: str = MARKER_COLOR,
        orientation: str = "v",
        as_percentage: bool = False,
        x_order: Optional[List[str]] = None,
        figsize: tuple = (400, 350)
    ) -> go.Figure:
        data = (
            self.dataframe
            .filter(pl.col(xcol).is_not_null())
            .with_columns([
                pl.col(ycol).sum().alias(f"{ycol}_total")
            ])
            .group_by(xcol)
            .agg([
                pl.col(ycol).sum().alias(ycol),
                (pl.col(ycol).sum()*100/pl.col(f"{ycol}_total").first()).alias("relative_sum")
            ])
            .sort(ycol, descending = True)
        )
        y_value = "relative_sum" if as_percentage else ycol
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x = data[xcol] if orientation == "v" else data[y_value],
                y = data[y_value] if orientation == "v" else data[xcol],
                showlegend = False,
                marker = dict(color = color),
                orientation = orientation,
                hovertemplate = f"{xcol}: " + "%{x}<br>" + f"{ycol}: " + "%{y:.2f}" + "<extra></extra>"
            )
        )
        fig.update_layout(
            height = figsize[1],
            width = figsize[0],
            barmode = "stack",
            title = {
                "text": f"<b>{title}</b>",
            },
            xaxis = {
                'title': {
                    'text': f"{xcol}" if xaxis_title == "" else xaxis_title,
                },
                'tickangle': -45
            },
            yaxis = {
                'title': {
                    'text': f"{ycol}" if yaxis_title == "" else yaxis_title,
                },
                'ticksuffix': '% ' if as_percentage else '',
            },
            margin = dict(t=40,l=20,r=20,b=20)
        )
        return fig

    def timeseries(
            self,
            x: str,
            y: str,
            group: Literal["year", "month", "day"] = "month",
            color: str = MARKER_COLOR,
            title: str = '',
            xtickformat: str = "%d %b %Y",
            xdticks: str = "M2",
            show_slider: bool = True
        ) -> go.Figure:

        if self.dataframe[x].dtype not in self.DATETIME_DTYPES_POLARS:
            raise NotDateTimeColumnException("Please provide a datetime column on X axis")
        if self.dataframe[y].dtype not in self.NUMERIC_DTYPES_POLARS:
            raise NotNumericColumnException("Please provide a numeric column on Y axis")
        if x not in self.dataframe.columns:
            raise ColumnNotFoundedException(f"`{x}` not founded in dataframe")
        if y not in self.dataframe.columns:
            raise ColumnNotFoundedException(f"`{y}` not founded in dataframe")

        fmt_date_grouping = {
            "month": {
                "grouping_by": [pl.col(x).dt.year().alias("year"), pl.col(x).dt.month().alias("month")],
                "datetime_year": "year",
                "datetime_month": "month",
                "datetime_day": 1,
                "width": 24*30
            },
            "year": {
                "grouping_by": [pl.col(x).dt.year().alias("year")],
                "datetime_year": "year",
                "datetime_month": 1,
                "datetime_day": 1,
                "width": 24*30*12
            },
            "day": {
                "grouping_by": [pl.col(x).dt.year().alias("year"), pl.col(x).dt.month().alias("month"), pl.col(x).dt.day().alias("day")],
                "datetime_year": "year",
                "datetime_month": "month",
                "datetime_day": "day",
                "width": 24
            },
        }

        group_df = (
            self.dataframe
            .with_columns([
                pl.col(x).dt.year().alias("year"),
                pl.col(x).dt.month().alias("month"),
                pl.col(x).dt.day().alias("day")
            ])
            .group_by(fmt_date_grouping[group].get("grouping_by"))
            .agg([
                pl.col(y).sum().alias(y)
            ])
            .with_columns([
                pl.datetime(
                    year = fmt_date_grouping[group].get("datetime_year"),
                    month = fmt_date_grouping[group].get("datetime_month"),
                    day = fmt_date_grouping[group].get("datetime_day")
                )
                .dt.date()
                .alias("date")
            ])
            .sort("date")
        )

        bar_with = 1000*3600 * int(fmt_date_grouping[group].get("width")) # type: ignore

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x = group_df["date"],
                y = group_df[y],
                width = bar_with,
                marker = dict(color = color),
                name = str(y),
                showlegend = False,
                meta = str(y),
                hovertemplate = "Fecha: %{x}<br>" + f"{y}: " + "%{y:.2f}<extra></extra>"
            )
        )
        fig.update_xaxes(
            rangeslider = dict(
                visible = show_slider,
                thickness = 0.05,
            ),
            tickformat = xtickformat,
            dtick = xdticks,
            tickangle = -45
        )
        fig.update_layout(
            margin = dict(t=40,l=20,r=30,b=30),
            title = title,
            xaxis_title = x,
            yaxis_title = y,
        )
        return fig

    def inspect_column_values(self) -> pl.DataFrame:
        results = []
        row_count: int = self.dataframe.shape[0]

        for c in self.dataframe.columns:
            if isinstance(self.dataframe, pl.DataFrame):
                dtype = self.dataframe[c].dtype
                nunique = self.dataframe[c].unique().count() if dtype in self.STRING_DTYPES_POLARS else 0
                null_count = self.dataframe[c].null_count()
                empty_count = self.dataframe.filter(pl.col(c) == "")[c].count() if dtype in self.STRING_DTYPES_POLARS else 0
                none_count = self.dataframe.filter(pl.col(c).is_in(["None", "NONE"]))[c].count() if dtype in self.STRING_DTYPES_POLARS else 0
                nan_count = self.dataframe.filter(pl.col(c).is_nan())[c].count() if dtype in self.NUMERIC_DTYPES_POLARS else 0
                neg_count = self.dataframe.filter(pl.col(c) < 0)[c].count() if dtype in self.NUMERIC_DTYPES_POLARS else 0
                special_char_count = self.dataframe.filter(pl.col(c).str.contains(r"[^\w]|[\s]"))[c].count() if dtype in self.STRING_DTYPES_POLARS else 0

                results.append({
                    "column": c,
                    "dtype": str(dtype),
                    "nuniques": nunique,
                    "null_count": null_count,
                    "null_pct": np.round((null_count * 100 / row_count), 3),
                    "neg_count": neg_count,
                    "neg_pct": np.round((neg_count * 100 / row_count), 3),
                    "empty_count": empty_count,
                    "empty_pct": np.round((empty_count * 100 / row_count), 3),
                    "none_count": none_count,
                    "none_pct": np.round((none_count * 100 / row_count), 3),
                    "nan_count": nan_count,
                    "nan_pct": np.round((nan_count * 100 / row_count), 3),
                    "special_char_count": special_char_count,
                })

        return pl.DataFrame(results)



def plot_cross_validation(y_true, y_pred, color: str = MARKER_COLOR, title = "", figsize: tuple = (400, 400)):
    max_value = max(np.max(y_true), np.max(y_pred))
    min_value = min(np.min(y_true), np.min(y_pred))

    fig = go.Figure()
    fig.add_scatter(
        x = y_pred,
        y = y_true,
        mode = 'markers',
        marker = {
            "size": 7,
            'color': color,
            'opacity': 0.3
        },
        showlegend = False,
        hovertemplate = f"Predicted" + ": %{x}<br>" +
        f"True" + ": %{y}<br>" +
        "<extra></extra>"
    )
    fig.add_scatter(
        x = (min_value, max_value),
        y = (min_value, max_value),
        mode = 'lines',
        opacity = 0.3,
        line = {
            'color': 'Red',
            'width': 3,
        },
        showlegend = False,
        hovertemplate = "<extra></extra>"
    )
    fig.update_layout(
        height = figsize[1],
        width = figsize[0],
        margin = dict(t=40, l=15, r=40, b=20),
        title = {
            "text": f"{title}",
            "font": {
                "family": "Helvetica",
                "size": 22,
            },
            'y':0.98
        },
        xaxis = {
            'title' : {
                'text': "Predicted values",
                'font': {
                    "family": "Helvetica",
                    'size': 18,
                    'color': 'black'
                    }
                },
            'tickfont': dict(color = "black"),
            'showgrid' : True,
            'range': [min_value, max_value]
        },
        yaxis = {
            'title' : {
                'text': "Real values",
                'font': {
                    "family": "Helvetica",
                    'size': 18,
                    'color': 'black'
                    }
            },
            'tickfont': dict(color = "black"),
            'showgrid' : True,
            'range': [min_value, max_value]
        },
    )
    return fig


def plot_timeseries_grouped(dataframe, groupvariable, subgroupvariables, x, y, title = '', xtickformat = "%b %Y", xdticks = "M2", show_slider = True):
    fig = go.Figure()
    for machine in subgroupvariables:
        machine_data = dataframe[dataframe[groupvariable] == machine].sort_values(by = x, ascending = True)
        fig.add_trace(
            go.Bar(
                x = machine_data[x],
                y = machine_data[y].to_numpy(),
                width = 1000*3600*24, # Recordar que width esta en milisegundos para Timeseries
                name = machine,
                showlegend = True,
                meta = str(machine),
                hovertemplate = "Maquina: %{meta}<br>Fecha: %{x}<br>Metros: %{y:.2f}<extra></extra>"
            )
        )
    fig.update_xaxes(
        rangeslider = dict(
            visible = show_slider,
            thickness = 0.05,
        ),
        tickformat = xtickformat,
        dtick = xdticks,
        tickangle = -45
    )
    fig.update_layout(
        margin = dict(t=40,l=20,r=20,b=30),
        title = title,
        xaxis_title = x,
        yaxis_title = y,
    )
    return fig

def plot_feature_importances(model: Pipeline, column_names: Optional[List[str]] = None, figsize: tuple = (900, 500)):
    if isinstance(model, Pipeline):
        importances_df = pd.DataFrame({
            'feature': model.named_steps["preprocessor"].get_feature_names_out().tolist(),
            'importance': model.named_steps["regressor"].feature_importances_.tolist()
        })
    else:
        importances_df = pd.DataFrame({
            'feature': model.feature_names_in_.tolist(),
            'importance': model.feature_importances_.tolist()
        })

    importances_df = importances_df.sort_values(by = 'importance', ascending = True)

    fig = go.Figure(data = [
        go.Bar(
            x = importances_df['importance'],
            y = importances_df['feature'],
            orientation = 'h',
            marker = dict(color = 'skyblue'),
            hovertemplate = "Feature: %{y}<br>Importance: %{x:.4f}<br><extra></extra>"
        )
    ])
    fig.update_layout(
        height = figsize[1],
        width = figsize[0],
        title_text = "Importancia de variables del modelo",
        margin = dict(t=30,l=20,r=20,b=20)
    )
    return fig

def plot_feature_correlation(data, threshold = 0.3, explicit_corr = False, figsize = (900, 500)):
    corr_data = data.corr(method= "pearson").round(2) # ['pearson', 'kendall', 'spearman']
    cols_to_keep = corr_data.where(corr_data.abs() >= threshold).dropna(axis = 1, thresh = 2).columns.tolist()
    corr_matrix = corr_data.loc[cols_to_keep, cols_to_keep]

    fig = go.Figure(
        go.Heatmap(
            x = corr_matrix.index,
            y = corr_matrix.index[::-1],
            z = corr_matrix.to_numpy()[::-1],
            colorscale = "Blues",
            hovertemplate = (
                "<b>Variables:</b><br>" +
                "%{x} <br>%{y} <br><b>Correlacion:</b><br>%{z:.2f}<extra></extra>"
            ),
            hoverlabel = dict(
                font_size = 14,
                font_family = "Helvetica"
            ),
            colorbar = {
                "tickfont": {
                    'color': 'black'
                },
            },
        )
    )
    if explicit_corr:
      fig.update_traces(
          text = corr_matrix.to_numpy()[::-1],
          texttemplate = "%{text:.2f}",
      )
    fig.update_layout(
        height = figsize[1],
        width = figsize[0],
        title = {
            "text": f"<b>Matriz de correlacion de variables</b>",
            "font": {
                "family": "Helvetica",
                "size": 23,
            },
            'y': 0.99
        },
        xaxis = {
            'showgrid': False
        },
        yaxis = {
            'showgrid': False
        },
        margin = dict(t=30,l=20,r=20,b=20)
    )
    return fig
