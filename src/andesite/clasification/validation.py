import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def export_moving_means_cat(
    comp_df,
    model_df,
    model_plane_col,
    model_var_col,
    model_cat_col,
    comp_plane_col,
    comp_var_col,
    n_intervals,
    title,
    export = False
):
    model_cats_means_arr = []
    model_cats_counts_vals = []
    for _, data in model_df.groupby(model_cat_col):
        hist_coord_model, model_bins = np.histogram(data[model_plane_col], n_intervals)
        cum_hist_model = np.zeros(n_intervals + 1, dtype=np.int32)
        cum_hist_model[1:] = np.cumsum(hist_coord_model)
        model_vals = data[model_var_col].to_numpy()
        model_means_arr = np.zeros(n_intervals)
        model_counts_vals = np.zeros(n_intervals)
        for k in range(n_intervals):
            model_data = model_vals[cum_hist_model[k]:cum_hist_model[k + 1]]
            mask = (model_data != -999.0)
            model_means_arr[k] = np.mean(model_data[mask])
            model_counts_vals[k] = len(model_data)
        model_cats_means_arr.append(model_means_arr)
        model_cats_counts_vals.append(model_counts_vals)
    _, model_bins = np.histogram(model_df[model_plane_col], n_intervals)
    hist_coord_comp, _ = np.histogram(comp_df[comp_plane_col], bins=model_bins)
    axis_coord = np.round([(model_bins[i] + model_bins[i + 1]) / 2 for i in range(n_intervals)], 2)
    cum_hist_comp = np.zeros(n_intervals + 1, dtype=np.int32)
    cum_hist_comp[1:] = np.cumsum(hist_coord_comp)
    comp_vals = comp_df[comp_var_col].to_numpy()
    comp_means_arr = np.zeros(n_intervals)
    comp_counts_vals = np.zeros(n_intervals)
    for k in range(n_intervals):
        comp_data = comp_vals[cum_hist_comp[k]:cum_hist_comp[k + 1]]
        comp_means_arr[k] = np.mean(comp_data)
        comp_counts_vals[k] = len(comp_data)
    swath_df = pd.DataFrame({
        'intervals': axis_coord,
        'composites': comp_means_arr,
        'composites_counts': comp_counts_vals,
        'model_1': model_cats_means_arr[0],
        'model_counts_1': model_cats_counts_vals[0],
        'model_2': model_cats_means_arr[1],
        'model_counts_2': model_cats_counts_vals[1],
        'model_3': model_cats_means_arr[2],
        'model_counts_3': model_cats_counts_vals[2]
    })
    swath_df = swath_df.fillna(method='ffill')
    cat_colors = ['green', 'yellow', 'red']
    cat_text = ['Medido', 'Indicado', 'Inferido']
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig1 = go.Scatter(
            x=swath_df['intervals'],
            y=swath_df['composites'],
            name='Compositos',
            mode='markers+lines',
            line = {
                'color': '#d0786e',
            },
            text=swath_df['composites_counts'],
            hovertemplate= "Vista: %{x}<br>" +
            "RQD: %{y}<br>" +
            "Cantidad datos: %{text:,.0f}" +
            "<extra></extra>"
    )
    for i in range(3):
        fig.add_trace(go.Scatter(
                x=swath_df['intervals'],
                y=swath_df[f'model_{i+1}'],
                name=f'{cat_text[i]}',
                mode='markers+lines',
                line = {
                    'color': cat_colors[i],
                },
                hovertemplate= "Vista: %{x}<br>" +
                "RQD: %{y}<br>" +
                "<extra></extra>"
        ), secondary_y = False)
    fig3 = go.Bar(
        x = swath_df['intervals'],
        y = swath_df['composites_counts'],
        name = 'N° compositos',
        marker_color = '#d0786e',
        showlegend = True,
        opacity = 0.35,
        hoverinfo = 'none'
    )
    fig.add_trace(fig1, secondary_y = False)
    fig.add_trace(fig3, secondary_y = True)
    fig.update_layout(
        width=1000,
        height=550,
        margin=dict(l=30, r=20, t=40, b=30),
        title={
            'text': f'Medias condicionales {title}'
        },
        xaxis={
            'title': f'Coordenadas eje {model_plane_col}',
            'range': [swath_df['intervals'].min(), swath_df['intervals'].max()]
        },
        yaxis={
            'title': 'RQD (%)',
            'range': [0, 100],
            'tick0': 0,
            'dtick': 10,
            'ticksuffix': '%'
        },
        yaxis2={
            'showgrid': False,
            'title': 'Cantidad de datos de composito'
        }
    )
    if export:
        fig.write_html(f"conditional-means-cat-{model_plane_col}.html")
    return fig

def export_moving_means(comp_df, model_df, model_plane_col, model_var_col, comp_plane_col, comp_var_col, n_intervals, title, export=False):
    hist_coord_model, model_bins = np.histogram(model_df[model_plane_col], n_intervals)
    hist_coord_comp, _ = np.histogram(comp_df[comp_plane_col], bins=model_bins)
    axis_coord = np.round([(model_bins[i] + model_bins[i + 1]) / 2 for i in range(n_intervals)], 2)
    cum_hist_model = np.zeros(n_intervals + 1, dtype=np.int32)
    cum_hist_comp = np.zeros(n_intervals + 1, dtype=np.int32)
    cum_hist_model[1:] = np.cumsum(hist_coord_model)
    cum_hist_comp[1:] = np.cumsum(hist_coord_comp)
    comp_vals = comp_df[comp_var_col].to_numpy()
    model_vals = model_df[model_var_col].to_numpy()
    comp_means_arr = np.zeros(n_intervals)
    model_means_arr = np.zeros(n_intervals)
    comp_counts_vals = np.zeros(n_intervals)
    model_counts_vals = np.zeros(n_intervals)
    for k in range(n_intervals):
        model_data = model_vals[cum_hist_model[k]:cum_hist_model[k + 1]]
        comp_data = comp_vals[cum_hist_comp[k]:cum_hist_comp[k + 1]]
        comp_means_arr[k] = np.mean(comp_data)
        comp_counts_vals[k] = len(comp_data)
        mask = (model_data != -999.0)
        model_means_arr[k] = np.mean(model_data[mask])
        model_counts_vals[k] = len(model_data)
    swath_df = pd.DataFrame({
        'intervals': axis_coord,
        'composites': comp_means_arr,
        'composites_counts': comp_counts_vals,
        'model': model_means_arr,
        'model_counts': model_counts_vals
    })
    swath_df = swath_df.fillna(method='ffill')
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig1 = go.Scatter(
            x=swath_df['intervals'], 
            y=swath_df['composites'], 
            name='Compositos',
            mode='markers+lines',
            line = {
                'color': '#d0786e',
            },
            text=swath_df['composites_counts'],
            hovertemplate= "Vista: %{x}<br>" +
            "RQD: %{y}<br>" +
            "Cantidad datos: %{text:,.0f}" +
            "<extra></extra>"
    )
    fig2 = go.Scatter(
            x=swath_df['intervals'], 
            y=swath_df['model'],
            name='Modelo',
            mode='markers+lines',
            line = {
                'color': '#42bbac',
            },
            text=swath_df['model_counts'],
            hovertemplate= "Vista: %{x}<br>" +
            "RQD: %{y}<br>" +
            "Cantidad datos: %{text:,.0f}" +
            "<extra></extra>"
    )
    fig3 = go.Bar(
        x = swath_df['intervals'], 
        y = swath_df['composites_counts'],
        name = 'N° compositos',
        marker_color = '#d0786e',
        showlegend = True,
        opacity = 0.5,
        hoverinfo = 'none'
    )
    fig.add_trace(fig1, secondary_y = False)
    fig.add_trace(fig2, secondary_y = False)
    fig.add_trace(fig3, secondary_y = True)
    fig.update_layout(
        width=1000,
        height=550,
        margin=dict(l=30, r=20, t=50, b=30),
        title={
            'text': f'Medias condicionales {title}'
        },
        xaxis={
            'title': f'Coordenadas eje {model_plane_col}',
            'range': [axis_coord[0], axis_coord[-1]]
        },
        yaxis={
            'title': 'RQD (%)',
            'range': [0, 100],
            'tick0': 0,
            'dtick': 10,
            'ticksuffix': '%'
        },
        yaxis2={
            'showgrid': False,
            'title': 'Cantidad de datos de composito'
        }
    )
    if export:
        fig.write_html(f"conditional-means-{model_plane_col}.html")
    return fig