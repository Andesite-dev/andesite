import pandas as pd
import numpy as np
import time
import os
import re
import plotly.graph_objs as go
import dask.dataframe as dd

def histograma(df: pd.DataFrame, var: str, start: float, end: float, size: float, title: str) -> go.Figure:
    """Generates a histogram plot for a specified variable in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data.
    var : str
        The variable for which the histogram is to be plotted.
    start : float
        The starting value for the bins.
    end : float
        The ending value for the bins.
    size : float
        The size of each bin.
    title : str
        The title of the histogram plot.

    Returns
    -------
    go.Figure
        The Plotly Figure object representing the histogram.
    """
    # TODO examples
    # Create an empty Figure object
    fig = go.Figure()
    # Add a histogram trace to the Figure
    fig.add_trace(go.Histogram(
        x=df[var],  # Data for the histogram
        xbins=dict(
            start=start,
            end=end,
            size=size
        ),
        marker_color='#4a9a75',  # Color of the bars
        opacity=0.8,
        marker_line=dict(width=1, color='black'),
        histnorm='percent',  # Normalize to percentage
        hovertemplate="Largo: %{x}<br>Porcentaje: %{y}<br><extra></extra>"
    )
    )
    # Update layout settings for the Figure
    fig.update_layout(
        title={
            # Title of the plot
            "text": f"<b>{title}</b>",
            # Modify font settings for the title
            "font": {
                "family": "Helvetica",
                "size": 24,
            },
            'y': 0.98
        },
        xaxis={
            'title': {
                # Title of the X-axis
                'text': "Ranges",
                # Modify font settings for the X-axis title
                'font': {
                    "family": "Helvetica",
                    'size': 16,
                }
            },
        },
        yaxis={
            'title': {
                # Title of the Y-axis
                'text': 'Percentage (%)',
                # Modify font settings for the Y-axis title
                'font': {
                    "family": "Helvetica",
                    'size': 16,
                }
            },
            # Add a percentage suffix to Y-axis ticks
            'ticksuffix': '% ',
        },
        # Set bar mode to 'relative' for stacked bars
        barmode='relative',
        # Set the height and width of the plot
        height=600,
        width=1000,
        # Set margin values
        margin=dict(
            t=50,
            l=20,
            r=20,
            b=20)
    )
    # Show the created Figure
    return fig