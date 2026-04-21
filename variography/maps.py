

import os
import re
import shutil
import subprocess
import tempfile
import numpy as np
import pandas as pd
from andesite.utils.files import grab_index_coordinates, grab_index_target, read_file_from_gslib, transform_datafile_to_gslib
from andesite.utils.manipulations import globalize_backslashes
import plotly.graph_objects as go
import plotly.express as px
from icecream import ic

class VarMap:

    def __init__(self, input_drillholes, coordinates, input_grades, plane, n_directions, lag_count, lag_size, legend_type, out_file=None):
        self.input_drillholes_path = input_drillholes
        self.coordinates = coordinates
        self.input_grades = input_grades
        self.plane = plane
        self.n_directions = n_directions
        self.lag_count = lag_count
        self.lag_size = lag_size
        self.out_file = out_file
        self.legend_type = legend_type
        self.bandh = 999999
        self.bandv = 999999

    def set_workdir(self, workdir):
        self.workdir = workdir

    def hint(self):
        self.real_path_filename = transform_datafile_to_gslib(self.input_drillholes_path)
        df = read_file_from_gslib(self.real_path_filename).compute()
        statistics_df = df[self.coordinates].describe()
        differences = statistics_df.loc['max'] - statistics_df.loc['min']
        # Add the differences as a new row or column to the statistics DataFrame
        statistics_df.loc['ranges'] = differences
        print('The coordinates dimentions are:')
        print(np.round(statistics_df, 3))


    def gamv_formatted(self, gamv_file):
        with open(gamv_file, 'r') as f:
            lines = f.readlines()
            variable = re.search(r'tail:([^\s]+)', lines[0].strip()).group(1)
        with open(gamv_file, 'w') as file:
            lines[0] = f'GAMV\n6\nindex\nsteps\ngamma\npairs\nhead_{variable}\ntail_{variable}\n'
            file.writelines(lines)

    def create_gamv_params_temp(self, stand_sills=False):
        self.real_path_filename = transform_datafile_to_gslib(self.input_drillholes_path)
        ix, iy, iz = grab_index_coordinates(self.real_path_filename, self.coordinates)
        igrade = grab_index_target(self.real_path_filename, self.input_grades)
        assert self.plane.upper() in ['XY', 'XZ', 'YZ'], "Choose one of these planes XY, YZ, XZ"

        real_direction_step = 180/self.n_directions
        dirtol = 90/self.n_directions
        self._paths = []

        if self.plane.upper() == 'XY':
            directions = [d for d in np.arange(0, 180, real_direction_step)]
        else:
            directions = [90 - d for d in np.arange(0, 180, real_direction_step)]
        for dir in directions:
            temp_params_path = tempfile.NamedTemporaryFile(prefix="params_gamv"+'_', suffix=".par", delete=False)
            temp_out_filename_path = os.path.join(tempfile.gettempdir(), f'results_{dir}.out')
            fmt_params_path = globalize_backslashes(temp_params_path.name)
            fmt_out_path = globalize_backslashes(temp_out_filename_path)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            with open(os.path.join(current_dir, '../utils/bin/gamv-generic.par'), 'r') as file:
                lines = file.readlines()
            with open(fmt_params_path, 'w') as f:
                lines[1] = f'{globalize_backslashes(self.real_path_filename)}                      -file with data\n'
                lines[2] = f'{ix}   {iy}   {iz}                         -   columns for X, Y, Z coordinates\n'
                lines[3] = f'1   {igrade}                             -   number of variables,col numbers\n'
                lines[5] = f'{fmt_out_path}                   - file for variogram output\n'
                lines[6] = f'{self.lag_count}                                - number of lags\n'
                lines[7] = f'{self.lag_size}                               - lag separation distance\n'
                lines[8] = f'{self.lag_size / 2.0}                              -lag tolerance\n'
                if self.plane.upper() == 'XY':
                    lines[10] = f'{dir} {dirtol} {self.bandh} 0.0 999999 {self.bandv}   -azm,atol,bandh,dip,dtol,bandv\n'
                elif self.plane.upper() == 'YZ':
                    lines[10] = f'0.0 999999 {self.bandh} {dir} {dirtol} {self.bandv}   -azm,atol,bandh,dip,dtol,bandv\n'
                else:
                    lines[10] = f'90.0 999999 {self.bandh} {dir} {dirtol} {self.bandv}   -azm,atol,bandh,dip,dtol,bandv\n'
                lines[11] = f'{int(stand_sills)}                                 -standardize sills? (0=no, 1=yes)\n'
                f.writelines(lines)
            self._paths.append([fmt_params_path, fmt_out_path])

    def get_coordinates(self, azimuth, step_size):
        # Convert azimuth from degrees to radians
        azimuth_rad = np.radians(azimuth)

        # Create a vectorized version of the sine and cosine functions
        sin = np.vectorize(np.sin)
        cos = np.vectorize(np.cos)

        # Initialize the x, y, and z components
        x = np.zeros_like(azimuth)
        y = np.zeros_like(azimuth)
        z = np.zeros_like(azimuth)

        # Set the x, y, and z components based on the plane
        if self.plane.upper() == 'XY':
            x = step_size * cos(azimuth_rad)
            y = step_size * sin(azimuth_rad)
        elif self.plane.upper() == 'XZ':
            x = step_size * cos(azimuth_rad)
            z = step_size * sin(azimuth_rad)
        elif self.plane.upper() == 'YZ':
            y = step_size * cos(azimuth_rad)
            z = step_size * sin(azimuth_rad)
        else:
            raise ValueError(f"Invalid plane '{self.plane}', must be XY, XZ, or YZ")

        # Return the coordinates as a numpy array
        return np.round(np.stack((x, y, z), axis=-1), 4)

    def run_gamv(self, parameters):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # logger.debug(f'running >>>{os.path.join(current_dir, "bin/gamv_OpenMP.exe")} {parameters}')
        si = subprocess.STARTUPINFO()
        CREATE_NO_WINDOW = 0x08000000
        output = subprocess.check_output([globalize_backslashes(os.path.join(current_dir, '../utils/bin/gamv_OpenMP.exe')), f'{parameters}'], creationflags=CREATE_NO_WINDOW)
        output_str = output.decode("utf-8")
        if "GAMV Version: 3.100 Finished" in output_str:
            return
        else:
            raise Exception(f'Something wrong happend after run\n>>> bin/gamv_openMP.exe {parameters}')

    def complete_varmap_radius(self):
        if self.plane.upper() == 'XY':
            angles = []
            directions = [np.round(d, 2) for d in np.arange(0, 180, 180/self.n_directions)]
            for d, g in zip(directions, self._paths):
                mirror_file = os.path.join(tempfile.gettempdir(), f'results_{d + 180}.out')
                angles.append(d)
                shutil.copyfile(g[1], mirror_file)
                angles.append(d + 180)
        else:
            angles = []
            directions = [90 - d for d in np.arange(0, 180, 180/self.n_directions)]
            for d, g in zip(directions, self._paths):
                if d > 0:
                    new_result_path = os.path.join(tempfile.gettempdir(), f'results_{360 - d}.out')
                    os.rename(g[1], new_result_path)
                    angles.append(360 - d)
                    shutil.copyfile(new_result_path, os.path.join(tempfile.gettempdir(), f'results_{((360 - d) + 180)%360}.out'))
                    angles.append(((360 - d) + 180)%360)
                elif d == 0:
                    angles.append(d)
                    shutil.copyfile(g[1], os.path.join(tempfile.gettempdir(), f'results_{d + 180}.out'))
                    angles.append(d + 180)
                else:
                    new_result_path = os.path.join(tempfile.gettempdir(), f'results_{-d}.out')
                    os.rename(g[1], new_result_path)
                    angles.append(-d)
                    shutil.copyfile(new_result_path, os.path.join(tempfile.gettempdir(), f'results_{-d + 180}.out'))
                    angles.append(-d + 180)
        return angles

    def transform_varmap_to_dataframe(self):
        for out in self._paths:
            self.gamv_formatted(out[1])

        angles = self.complete_varmap_radius()

        self.tickvalues = np.arange(0, (self.lag_count + 1)*self.lag_size, self.lag_size, dtype=np.float32)


        varmap_df = pd.DataFrame()
        for angle in angles:
            resuts_path = os.path.join(tempfile.gettempdir(), f'results_{angle}.out')
            temp_gamv_df = read_file_from_gslib(globalize_backslashes(resuts_path)).compute()
            temp_df = pd.DataFrame()
            temp_df['variogram'] = temp_gamv_df['gamma'].to_numpy()[2:]
            temp_df['direction'] = angle
            temp_df['steps'] = self.tickvalues[1:]
            temp_df['frequency'] = [self.lag_size]*(self.lag_count)
            temp_df['pairs'] = temp_gamv_df['pairs'].to_numpy()[2:]
            varmap_df = pd.concat([varmap_df, temp_df], axis=0)
        varmap_df.fillna(0, inplace=True)

        return varmap_df

    def calculate(self, stand_sills = False):
        self.create_gamv_params_temp(stand_sills = stand_sills)
        for par in self._paths:
            self.run_gamv(par[0])

        self.varmap_df = self.transform_varmap_to_dataframe()
        self.clear()
        return self.varmap_df

    def varmap_points(self, varmap_df):
        points = self.get_coordinates(varmap_df['direction'], varmap_df['steps'])
        values = varmap_df.loc[:, ['variogram', 'pairs']].to_numpy()

        return points, values

    def clear(self):
        for file in os.listdir(tempfile.gettempdir()):
            if file.startswith('param') or file.startswith('result'):
                os.remove(os.path.join(tempfile.gettempdir(), file))
        os.remove(self.real_path_filename)

    def plot(self, varmap_df, title = None, figsize=(700, 650), export=False):
        """
        Perform a variographic map plot. To user this plot, several colobar can be used, some of
        them are:
        - px.colors.sequential.Jet
        - px.colors.sequential.Rainbow
        - px.colors.diverging.Portland

        Parameters
        ----------
        varmap_df : pandas.DataFrame
            Dataframe containing the variogram data
        legend_type : str, optional
            Type of colorbar to use, can be 'discrete' or 'continuous', by default 'discrete'
        title : str, optional
            Title of the plot, by default None
        export : bool, optional
            Whether to export the plot, by default False
        """

        tickvals = np.sort(varmap_df['direction'].unique())
        if self.plane.upper() == 'XY':
            ticktext = tickvals
        else:
            ticktext = [f'-{t}' if t > 0 and t <= 90 else 0.0
                        if t == 0 else '' if t > 90 and t < 270
                        else f'{360 - t}' for t in tickvals]
        print(ticktext)

        if self.plane.upper() == 'XY':
            rotation = 90
        else:
            rotation = 0

        fig = go.Figure()

        varmap_frequency = varmap_df['frequency']
        varmap_directions = varmap_df['direction']
        varmap_values = varmap_df['variogram']

        barpolar_width = tickvals[1] - tickvals[0]

        if self.legend_type == "discrete":
            cmin = 0
            cmax = 1.2
            norm_07 = (0.7 - cmin) / (cmax - cmin)
            norm_1 = (1.0 - cmin) / (cmax - cmin)
            colorscale = [
                [0.0, '#ADDEA5'],
                [norm_07, '#ADDEA5'],
                [norm_07, '#FFD684'],
                [norm_1, '#FFD684'],
                [norm_1, '#C6294A'],
                [1.0, '#C6294A']
            ]
            center1 = 0.35
            center2 = 0.85
            center3 = (1 + cmax) / 2
            tickvals_colorbar = [center1, center2, center3]
            ticktext_colorbar = ['0 - 0.7', '0.7 - 1', '> 1']

            barpolar_marker = {
                'color': varmap_values,
                'colorbar': {
                    'title': {
                        'text': self.input_grades,
                        'font': {
                            'size': 16
                        }
                    },
                    'thickness': 28,
                    'tickvals': tickvals_colorbar,
                    'ticktext': ticktext_colorbar,
                    'tickmode': 'array'
                },
                'cmin': cmin,
                'cmax': cmax,
            }
        else:
            cmin = varmap_values.quantile(0.01) #varmap_df[varmap_df['steps'] == lag_size]['variogram'].min(),
            cmax = varmap_values.quantile(0.99) #varmap_df['variogram'].max()

            barpolar_marker = {
                'color': varmap_values,
                'colorbar': {
                    'title': {
                    'text': self.input_grades,
                    'font': {
                        'size': 16
                    }
                },
                'thickness': 28,
                },
                'cmin': cmin,
                'cmax': cmax
            }

        fig.add_trace(go.Barpolar(
            r = varmap_frequency,
            theta = varmap_directions,
            width = barpolar_width,
            text = [f'Paso: {s}<br>Pares: {p}' for s, p in list(zip(varmap_df['steps'], varmap_df['pairs']))],
            marker_colorscale = px.colors.sequential.Jet if self.legend_type == "continuous" else colorscale,
            marker = barpolar_marker,
            hovertemplate="Direction: %{theta}<br>Variogram: %{marker.color:.2f}<br>%{text}<extra></extra>",
            showlegend=False
        ))
        fig.update_layout(
            width = figsize[0],
            height = figsize[1],
            title = f'Variographic Map {self.input_grades} plane {self.plane.upper()}',
            font_size = 14,
            legend_font_size = 14,
            polar = {
                'angularaxis': {
                    'rotation': rotation,
                    'direction': 'clockwise',
                    'gridcolor': '#858585',
                    'showgrid': True,
                    'tickvals': tickvals,
                    'ticktext': ticktext
                },
                'radialaxis': {
                    'tickvals' : self.tickvalues,
                    'ticktext': [f'{d:.1f}°' for d in self.tickvalues][:-1],
                    'gridcolor': '#858585',
                    'tickfont': {
                        'size': 14,
                        'family': 'Arial Black',
                        'color': 'black'
                    },
                    'angle': 25,
                    'tickangle' : 45,
                    'range': [0, self.lag_size*(self.lag_count)],
                    'showgrid': False,
                    'visible': False
                }
            },
            # template = 'plotly_white',
        )
        if export:
            fig.write_html(f"varmap-{self.input_grades}-{self.plane.upper()}.html")

        return fig
