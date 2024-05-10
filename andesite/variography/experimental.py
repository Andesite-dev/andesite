import os
import re
import subprocess
import tempfile
import pandas as pd
from andesite.utils.files import grab_index_coordinates, grab_index_target, read_file_from_gslib, transform_datafile_to_gslib
from andesite.utils.manipulations import globalize_backslashes
import plotly.graph_objects as go
from icecream import ic

class VariogramDatafile:

    def __init__(self, variogram: pd.DataFrame, parameters: dict):
        self.variogram = variogram
        self.parameters = parameters

    def load(self):
        return self.variogram.copy()

    def get_metadata(self):
        return self.parameters

class Variogram:

    def __init__(self, input_drillholes, coordinates, input_grades, params):
        self.input_drillholes_path = input_drillholes
        self.coordinates = coordinates
        self.input_grades = input_grades
        self.azim = params['azimuth']
        self.atol = params['azimuth_tolerance']
        self.dip = params['dip']
        self.diptol = params['dip_tolerance']
        self.lag_count = params['lag_count']
        self.lag_size = params['lag_size']
        self.lag_tol = params['lag_tolerance']
        self.bandh = params['horizontal_bandwidth']
        self.bandv = params['vertical_bandwidth']
        self.metadata = {
            'path': self.input_drillholes_path,
            'grades': self.input_grades,
            'coordinates': self.coordinates,
            'parameters': params
        }

    def get_metadata(self):
        return self.metadata

    def normalize_dip(self, value):
        result = (value - 90) % 180
        if result > 90:
            result -= 180
        return result

    def clear(self):
        for file in [self.fmt_out_path, self.fmt_params_path]:
            os.remove(file)

    def create_params_temp(self, elipsoid=False, stand_sills=False):
        real_path_filename = transform_datafile_to_gslib(self.input_drillholes_path)
        ix, iy, iz = grab_index_coordinates(real_path_filename, self.coordinates)
        igrade = grab_index_target(real_path_filename, self.input_grades)

        temp_params_path = tempfile.NamedTemporaryFile(prefix="params_gamv"+'_', suffix=".par", delete=False)
        temp_out_filename_path = tempfile.NamedTemporaryFile(prefix="gamv"+'_', suffix=".out", delete=False)
        self.fmt_params_path = globalize_backslashes(temp_params_path.name)
        self.fmt_out_path = globalize_backslashes(temp_out_filename_path.name)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(current_dir, '../utils/bin/gamv-generic.par'), 'r') as file:
            lines = file.readlines()
        with open(self.fmt_params_path, 'w') as f:
            lines[1] = f'{globalize_backslashes(real_path_filename)}                      -file with data\n'
            lines[2] = f'{ix}   {iy}   {iz}                         -   columns for X, Y, Z coordinates\n'
            lines[3] = f'1   {igrade}                             -   number of variables,col numbers\n'
            lines[5] = f'{self.fmt_out_path}                   - file for variogram output\n'
            lines[6] = f'{int(self.lag_count)}                                - number of lags\n'
            lines[7] = f'{self.lag_size}                               - lag separation distance\n'
            lines[8] = f'{self.lag_tol}                              -lag tolerance\n'
            lines[10] = f'{self.azim} {self.atol} {self.bandh} {self.dip} {self.diptol} {self.bandh}   -azm,atol,bandh,dip,dtol,bandv\n'
            if not elipsoid:
                lines[11] = f'{int(stand_sills)}                                 -standardize sills? (0=no, 1=yes)\n'
                self.metadata.update({
                    'directions': [
                        [self.azim, self.dip]
                    ]
                })
            else:
                lines[9] = '3                                 -number of directions\n'
                lines[11] = f'{self.azim + 90} {self.atol} {self.bandh} {self.dip} {self.diptol} {self.bandh}   -azm,atol,bandh,dip,dtol,bandv\n'
                lines[12] = f'{self.azim} {self.atol} {self.bandh} {self.normalize_dip(self.dip)} {self.diptol} {self.bandh}   -azm,atol,bandh,dip,dtol,bandv\n'
                lines[13] = f'{int(stand_sills)}                                  -standardize sills? (0=no, 1=yes)\n'
                lines[14] = '1                                 -number of variograms\n'
                lines[15] = '1   1   1                         -tail var., head var., variogram type\n'
                self.metadata.update({
                    'directions': [
                        [self.azim, self.dip],
                        [self.azim + 90, self.dip],
                        [self.azim, self.normalize_dip(self.dip)]
                    ]
                })
            f.writelines(lines)
        # print(f'{self.fmt_params_path} created sucessfully!')

    def gamv_formatted_elipsoid(self, gamv_file):
        with open(gamv_file, 'r') as f:
            file_content = f.read()

        # Split the file content into individual sets
        sets = file_content.strip().split('Semivariogram')
        for index, set_content in enumerate(sets[1:], start=1):
            # Extract variable name for each set
            variable = re.search(r'tail:([^\s]+)', set_content).group(1)
            # Remove the first line (unwanted line) from each set
            set_content_lines = set_content.strip().split('\n')[1:]
            formatted_content = '\n'.join(set_content_lines)

            # Format the set content
            formatted_content = f'GAMV\n6\nindex\nsteps\ngamma\npairs\nhead_{variable}\ntail_{variable}\n{formatted_content}'
            # Write the formatted content to a new file or overwrite the original file
            with open(f'{os.path.splitext(gamv_file)[0]}_{index}.out', 'w') as file:
                file.write(formatted_content)

    def gamv_formatted(self, gamv_file):
        with open(gamv_file, 'r') as f:
            lines = f.readlines()
            variable = re.search(r'tail:([^\s]+)', lines[0].strip()).group(1)
        with open(gamv_file, 'w') as file:
            lines[0] = f'GAMV\n6\nindex\nsteps\ngamma\npairs\nhead_{variable}\ntail_{variable}\n'
            file.writelines(lines)

    def run_gamv(self, parameters):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        si = subprocess.STARTUPINFO()
        CREATE_NO_WINDOW = 0x08000000
        output = subprocess.check_output([globalize_backslashes(os.path.join(current_dir, '../utils/bin/gamv_OpenMP.exe')), f'{parameters}'], creationflags=CREATE_NO_WINDOW)
        output_str = output.decode("utf-8")
        if "GAMV Version: 3.000 Finished" in output_str:
            return
        else:
            raise Exception(f'Something wrong happend after run\n>>> bin/gamv_openMP.exe {parameters}\n{output_str}')

    def single_semivariogram(self, stand_sills: bool = False) -> VariogramDatafile:
        self.metadata.update({
            'norm': stand_sills
        })
        self.create_params_temp(elipsoid=False, stand_sills=stand_sills)

        self.run_gamv(self.fmt_params_path)
        print(f'{self.fmt_out_path} file created!')
        self.gamv_formatted(self.fmt_out_path)

        df = read_file_from_gslib(self.fmt_out_path).compute()
        df.rename(columns={'steps': 'steps_dir1', 'gamma': 'gamma_dir1', 'pairs': 'pairs_dir1'}, inplace=True)
        self.metadata.update({
            'vartype': 'single'
        })
        self.clear()
        return VariogramDatafile(variogram=df.iloc[1:], parameters=self.metadata)

    def elipsoid_variogram(self, stand_sills: bool = False) -> VariogramDatafile:
        self.metadata.update({
            'norm': stand_sills
        })
        self.create_params_temp(elipsoid=True, stand_sills=stand_sills)
        self.run_gamv(self.fmt_params_path)

        self.gamv_formatted_elipsoid(self.fmt_out_path)
        print(f'{self.fmt_out_path} file created!')
        df_1 = read_file_from_gslib(f'{os.path.splitext(self.fmt_out_path)[0]}_1.out').compute()
        df_2 = read_file_from_gslib(f'{os.path.splitext(self.fmt_out_path)[0]}_2.out').compute()
        df_3 = read_file_from_gslib(f'{os.path.splitext(self.fmt_out_path)[0]}_3.out').compute()
        df_1.rename(columns={'steps': 'steps_dir1', 'gamma': 'gamma_dir1', 'pairs': 'pairs_dir1'}, inplace=True)
        df_2.rename(columns={'steps': 'steps_dir2', 'gamma': 'gamma_dir2', 'pairs': 'pairs_dir2'}, inplace=True)
        df_3.rename(columns={'steps': 'steps_dir3', 'gamma': 'gamma_dir3', 'pairs': 'pairs_dir3'}, inplace=True)

        self.metadata.update({
            'vartype': 'elipsoid'
        })
        # self.clear()
        return VariogramDatafile(variogram=pd.concat([df_1, df_2, df_3], axis=1).iloc[1:], parameters=self.metadata)


    def plot(self, variogram_dataframe, show_pairs=False, export=False):
        if (variogram_dataframe.shape[1]) > 7:
            dirs = [f'{self.azim}/{self.dip}', f'{self.azim + 90}/{self.dip}', f'{self.azim}/{self.normalize_dip(self.dip)}']
            title = f'Ortogonal Variograms for {dirs[0]}'
            gamma_max = variogram_dataframe[['gamma_dir1', 'gamma_dir2', 'gamma_dir3']].max().max()
        else:
            dirs = [f'{self.azim}/{self.dip}']
            title = f'Experimental Variogram {dirs[0]}'
            gamma_max = variogram_dataframe['gamma_dir1'].max()
        colors = ['black', 'blue', 'green']
        fig = go.Figure()
        for i, n in enumerate(dirs):
            fig.add_trace(
                go.Scatter(
                    # Para casos donde los lags de algunas direcciones den 0 (porque se acabaron los datos)
                    # Esto de puede dejar como el lag principal, ya que los valores varian muy poco
                    x = variogram_dataframe[f'steps_dir{i+1}'],
                    y = variogram_dataframe[f'gamma_dir{i+1}'],
                    mode = 'markers+lines',
                    marker = {
                        'color': 'red',
                        # Editar esta parte en caso de tener pocos o muchos pares
                        'size': variogram_dataframe[f'pairs_dir{i+1}']*0.00088 if show_pairs else variogram_dataframe[f'pairs_dir{i+1}']*0
                    },
                    line = {
                        'color': colors[i],
                        'width': 2
                    },
                    name = n,
                    opacity = 0.85,
                    text = variogram_dataframe[f'pairs_dir{i+1}'],
                    textposition = "top center",
                    hovertemplate= "Steps: %{x}<br>" +
                    "Variogram: %{y}<br>" +
                    "Pairs: %{text}" +
                    "<extra></extra>",
                )
            )
        # fig.add_trace(
        #     go.Scatter(
        #         x = [0., lag1[-1]],
        #         y = [sill, sill],
        #         mode = 'lines',
        #         line = {
        #             'color': 'black',
        #             'width': 2,
        #             'dash': 'dash'
        #         },
        #         showlegend = False,
        #         opacity = 0.7
        #     )
        # )
        fig.update_layout(
            width = 800,
            margin = dict(l=20, b=20, t=40, r=20),
            title = {
                'text': title
            },
            xaxis = {
                'title': 'Step',
                'range': [0, variogram_dataframe['steps_dir1'].max()]
            },
            yaxis = {
                'title': 'Variograma',
                'range': [0, gamma_max],
                # 'tick0': 0,
                # 'dtick': 0.1
            }
        )
        if export:
            fig.write_html(f'Variogram-{self.azim}-{self.dip}.html')
        return fig