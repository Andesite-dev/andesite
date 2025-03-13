import plotly.graph_objects as go
from andesite.datafiles.grid import GridDatafile
from andesite.utils.manipulations import find_pattern_on_list
from andesite.utils._globals import *

def pixelplt(cmp_df, model_df: GridDatafile, target_model_col, cutoff, slice='XY', export=False):
    model_x_col = find_pattern_on_list(model_df.columns, POSSIBLE_X_COLUMNS)
    model_y_col = find_pattern_on_list(model_df.columns, POSSIBLE_Y_COLUMNS)
    model_z_col = find_pattern_on_list(model_df.columns, POSSIBLE_Z_COLUMNS)
    cmp_x_col = find_pattern_on_list(cmp_df.columns, POSSIBLE_X_COLUMNS)
    cmp_y_col = find_pattern_on_list(cmp_df.columns, POSSIBLE_Y_COLUMNS)
    cmp_z_col = find_pattern_on_list(cmp_df.columns, POSSIBLE_Z_COLUMNS)
    model_coordinates = model_df.get_metadata().get('coordinates')
    model_df = model_df.load()

    if slice == 'XY':
        if model_z_col == '' or model_x_col == '' or model_y_col == '':
            model_x_col, model_y_col, model_z_col = model_coordinates
        mini_block_data = model_df[(model_df[model_z_col] == cutoff)]
        mini_cmp_data = cmp_df.query(f'({cmp_z_col} < {cutoff + 0.5}) and ({cmp_z_col} > {cutoff - 0.5})')
        xblock_min, xblock_max = mini_block_data[model_x_col].min(), mini_block_data[model_x_col].max()
        yblock_min, yblock_max = mini_block_data[model_y_col].min(), mini_block_data[model_y_col].max()
        var_max = mini_block_data[target_model_col].max()
    fig = go.Figure()

    # Create a heatmap for mini_block_data
    fig.add_heatmap(
        x=mini_block_data[model_x_col],
        y=mini_block_data[model_y_col],
        z=mini_block_data[target_model_col],
        colorscale='Jet',  # You can change the colorscale
        colorbar=dict(
            title=f'{target_model_col}'
        ),
        zmin = 0,
        zmax = var_max,
        hovertemplate= "Este: %{x:,.0f}<br>" +
                        "Norte: %{y:,.0f}<br>" +
                        "Varianza: %{z:,.3f}" +
                        "<extra></extra>"
    )

    # Add the points from mini_cmp_data
    fig.add_scatter(
        x=mini_cmp_data[cmp_x_col],
        y=mini_cmp_data[cmp_y_col],
        mode='markers',
        marker=dict(
            size=3.7,
            color='red',
        ),
        showlegend=False,
        hovertemplate= "Este: %{x:,.0f}<br>" +
                        "Norte: %{y:,.0f}<br>" +
                        "<extra></extra>"
    )

    fig.update_layout(
        title={
            # Title of the plot
            "text": f"<b>View {cutoff} Z</b>",
            # Modify font settings for the title
            "font": {
                "family": "Helvetica",
                "size": 24,
            },
            'y': 0.98
        },
        margin = dict(l=20, r=20, t=40, b=20),
        width = 800,
        height = 600,
        xaxis = {
            'title': {
                'text': 'X',
                'font': {
                    "family": "Helvetica",
                    'size': 16,
                }
            },
            'range': [xblock_min, xblock_max]
        },
        yaxis = {
            'title': {
                'text': 'Y',
                'font': {
                    "family": "Helvetica",
                    'size': 16,
                }
            },
            'range': [yblock_min, yblock_max]
        },
    )
    # TODO: dentro de export, puedo ver si la variable es str, en ese caso reemplazar el nombre
    #       del archivo en write_html()
    if export:
        fig.write_html('heatmap.html')

    return fig


# pixelplt(
#     cmp_df = composites,
#     model_df = blockmodel,
#     target_model_col = 'EstimationVariance',
#     cutoff = 705,
#     slice = 'XY',
#     export = True
# )