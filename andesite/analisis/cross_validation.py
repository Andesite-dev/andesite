import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix

def regression_report(true_values, estimated_values):
    mse = mean_squared_error(true_values, estimated_values)
    rmse = mean_squared_error(true_values, estimated_values, squared=False)
    mae = mean_absolute_error(true_values, estimated_values)
    r2 = r2_score(true_values, estimated_values)
    corrcoef = np.corrcoef(estimated_values, true_values)[0, 1]

    n = len(true_values)
    p = 1
    r2_adjusted = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)
    metrics_names = ['MSE', 'RMSE', 'MAE', 'R2', 'R2-adj', 'corr']
    metrics_results = [mse, rmse, mae, r2, r2_adjusted, corrcoef]
    report = {key:np.round(value, 2) for (key,value) in zip(metrics_names, metrics_results)}
    return report

def crossval_plot(report_results: dict, estimate_values, true_values, export: bool = False):
    mse, rmse = report_results.get('MSE'), report_results.get('RMSE')
    mae, r2 = report_results.get('MAE'), report_results.get('R2')
    r2_adj, corr = report_results.get('R2-adj'), report_results.get('corr')
    rango_x = np.arange(0, np.ceil(np.max(true_values)).astype(int))
    fig = go.Figure()
    fig.add_annotation(
        x = 90,
        y = 8,
        text = f"<b>MSE = {mse}<br>RMSE = {rmse}<br>MAE = {mae}<br>R2 = {r2}<br>R2-adj = {r2_adj}<br>Corr = {corr}</b>",
        showarrow = False,
        yshift = 12,
        font = dict(
            size = 12,
            color = "black"
        )
    )
    fig.add_scatter(
        x = estimate_values,
        y = true_values,
        mode = 'markers',
        marker = {
            "size": 7,
            'color': '#48494B',
            'opacity': 0.3
        },
        showlegend = False,
        hovertemplate= f"Predicted" + ": %{x}<br>" +
        f"True" + ": %{y}<br>" +
        "<extra></extra>"
    )
    fig.add_scatter(
        x = rango_x,
        y = rango_x,
        mode = 'lines',
        line = {
            'color': 'Red',
            'width': 3,
        },
        showlegend = False,
        hovertemplate = "<extra></extra>"
    )
    fig.update_layout(
        height = 600,
        width = 600,
        margin = dict(t=40, l=15, r=40, b=20),
        title = {
            "text": "Cross Validation",
            "font": {
                "family": "Helvetica",
                "size": 24,
            },
            'y':0.98
        },
        xaxis = {
            'title' : {
                'text': "Predicted",
                'font': {
                    "family": "Helvetica",
                    'size': 18,
                    'color': 'black'
                    }
                },
            'tickfont': dict(color = "black"),
            'showgrid' : True,
            'tick0': 0,
            'dtick': 10,
            'range': [0, np.max(true_values)]
        },
        yaxis = {
            'title' : {
                'text': "True",
                'font': {
                    "family": "Helvetica",
                    'size': 18,
                    'color': 'black'
                    }
            },
            'tickfont': dict(color = "black"),
            'showgrid' : True,
            'tick0': 0,
            'dtick': 10,
            'range': [0, np.max(true_values)]
        },
    )
    if export:
        fig.write_html('validation.html')
    return fig