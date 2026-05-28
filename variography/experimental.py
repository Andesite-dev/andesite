import os
import subprocess
import tempfile
import pandas as pd
from andesite.utils.files import grab_index_coordinates, grab_index_target, read_file_from_gslib, transform_datafile_to_gslib
from andesite.utils.manipulations import globalize_backslashes
import plotly.graph_objects as go
from icecream import ic
from andesite.visualizations.base_plotly import PALETTE_HEX
from andesite.visualizations.plotly_plots_nevado import NEVADO_QUALITATIVE_DARK, NEVADO_QUALITATIVE_LIGHT

_PALETTE_MAP = {
    'andes': PALETTE_HEX,
    'nevado_dark': NEVADO_QUALITATIVE_DARK,
    'nevado_light': NEVADO_QUALITATIVE_LIGHT,
}
_DEFAULT_COLORS = ['black', 'blue', 'green']

def _resolve_palette(palette):
    if palette is None:
        return _DEFAULT_COLORS
    colors = _PALETTE_MAP.get(palette)
    if colors is None:
        raise ValueError(f"Unknown palette '{palette}'. Choose from: {list(_PALETTE_MAP)}")
    return list(colors[:3])

class VariogramDatafile:

    def __init__(self, variogram: pd.DataFrame, parameters: dict):
        self.variogram = variogram
        self.parameters = parameters

    def load(self):
        return self.variogram.copy()

    def get_metadata(self):
        return self.parameters

    def add_direction(self, other: 'VariogramDatafile') -> 'VariogramDatafile':
        n_existing = len(self.variogram.columns) // 3
        n_adding   = len(other.variogram.columns) // 3
        rename_map = {}
        for i in range(1, n_adding + 1):
            j = n_existing + i
            rename_map[f'steps_dir{i}'] = f'steps_dir{j}'
            rename_map[f'gamma_dir{i}'] = f'gamma_dir{j}'
            rename_map[f'pairs_dir{i}'] = f'pairs_dir{j}'
        new_df = pd.concat(
            [self.variogram, other.variogram.rename(columns=rename_map)], axis=1
        )
        new_params = {
            **self.parameters,
            'directions': (self.parameters.get('directions', []) +
                           other.parameters.get('directions', [])),
            'vartype': 'multi',
        }
        return VariogramDatafile(variogram=new_df, parameters=new_params)

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
        self.weight_column = params.get('weight_column', None)
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
        iweight = grab_index_target(real_path_filename, self.weight_column) if self.weight_column else 0

        temp_params_path = tempfile.NamedTemporaryFile(prefix="params_gamv"+'_', suffix=".par", delete=False)
        self.fmt_params_path = globalize_backslashes(temp_params_path.name)
        temp_params_path.close()
        temp_out_filename_path = tempfile.NamedTemporaryFile(prefix="gamv"+'_', suffix=".out", delete=False)
        self.fmt_out_path = globalize_backslashes(temp_out_filename_path.name)
        temp_out_filename_path.close()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(current_dir, '../utils/bin/gamv-generic.par'), 'r') as file:
            lines = file.readlines()
        with open(self.fmt_params_path, 'w') as f:
            lines[1] = f'{globalize_backslashes(real_path_filename)}                      -file with data\n'
            lines[2] = f'0   {ix}   {iy}   {iz}   {iweight}   0   0         -columns: BHID,X,Y,Z,WT,FROM,TO\n'
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
        print(f'{self.fmt_params_path} created sucessfully!')

    def gamv_formatted_elipsoid(self, gamv_file):
        with open(gamv_file, 'r') as f:
            content = f.read()
        parts = content.strip().split('Semivariogram')
        for index, section in enumerate(parts[1:], start=1):
            out_path = f'{os.path.splitext(gamv_file)[0]}_{index}.out'
            with open(out_path, 'w') as f:
                f.write('Semivariogram' + section)

    def run_gamv(self, parameters):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        si = subprocess.STARTUPINFO()
        CREATE_NO_WINDOW = 0x08000000
        output = subprocess.check_output([globalize_backslashes(os.path.join(current_dir, '../utils/bin/gamv_OpenMP.exe')), f'{parameters}'], creationflags=CREATE_NO_WINDOW)
        output_str = output.decode("utf-8")
        if "GAMV elapsed time:" in output_str:
            return
        else:
            raise Exception(f'Something wrong happend after run\n>>> bin/gamv_OpenMP.exe {parameters}\n{output_str}')

    def single_semivariogram(self, stand_sills: bool = False) -> VariogramDatafile:
        self.metadata.update({
            'norm': stand_sills
        })
        self.create_params_temp(elipsoid=False, stand_sills=stand_sills)

        self.run_gamv(self.fmt_params_path)
        actual_out_path = f'{os.path.splitext(self.fmt_out_path)[0]}_{self.input_grades}_semivariogram.out'
        if not os.path.exists(actual_out_path) or os.path.getsize(actual_out_path) == 0:
            raise RuntimeError(f"gamv produced no output at {actual_out_path}")
        df = read_file_from_gslib(actual_out_path).compute()
        df.rename(columns={'steps': 'steps_dir1', 'gamma': 'gamma_dir1', 'pairs': 'pairs_dir1'}, inplace=True)
        self.metadata.update({
            'vartype': 'single'
        })
        return VariogramDatafile(variogram=df, parameters=self.metadata)

    def elipsoid_variogram(self, stand_sills: bool = False) -> VariogramDatafile:
        self.metadata.update({
            'norm': stand_sills
        })
        self.create_params_temp(elipsoid=True, stand_sills=stand_sills)
        self.run_gamv(self.fmt_params_path)
        actual_out_path = f'{os.path.splitext(self.fmt_out_path)[0]}_{self.input_grades}_semivariogram.out'
        if not os.path.exists(actual_out_path) or os.path.getsize(actual_out_path) == 0:
            raise RuntimeError(f"gamv produced no output at {actual_out_path}")

        self.gamv_formatted_elipsoid(actual_out_path)
        print(f'{actual_out_path} file created!')
        base = os.path.splitext(actual_out_path)[0]
        df_1 = read_file_from_gslib(f'{base}_1.out').compute()
        df_2 = read_file_from_gslib(f'{base}_2.out').compute()
        df_3 = read_file_from_gslib(f'{base}_3.out').compute()
        df_1.rename(columns={'steps': 'steps_dir1', 'gamma': 'gamma_dir1', 'pairs': 'pairs_dir1'}, inplace=True)
        df_2.rename(columns={'steps': 'steps_dir2', 'gamma': 'gamma_dir2', 'pairs': 'pairs_dir2'}, inplace=True)
        df_3.rename(columns={'steps': 'steps_dir3', 'gamma': 'gamma_dir3', 'pairs': 'pairs_dir3'}, inplace=True)

        self.metadata.update({
            'vartype': 'elipsoid'
        })
        return VariogramDatafile(variogram=pd.concat([df_1, df_2, df_3], axis=1), parameters=self.metadata)


    def plot(self, variogram_datafile, show_pairs=False, export=False, palette=None):
        if isinstance(variogram_datafile, VariogramDatafile):
            df = variogram_datafile.load()
            directions = variogram_datafile.get_metadata().get('directions', [])
            dirs = [f'{d[0]}/{d[1]}' for d in directions]
        else:
            df = variogram_datafile
            n_dirs = df.shape[1] // 3
            dirs = [f'{self.azim}/{self.dip}'] if n_dirs == 1 else [f'Dir {i+1}' for i in range(n_dirs)]

        n_dirs = len(dirs)
        gamma_max = df[[f'gamma_dir{i+1}' for i in range(n_dirs)]].max().max()
        title = f'Experimental Variogram {"  |  ".join(dirs)}'
        colors = _resolve_palette(palette)
        if n_dirs > len(colors):
            colors = (colors * ((n_dirs // len(colors)) + 1))[:n_dirs]

        MIN_MARKER = 10
        MAX_MARKER = 40
        fig = go.Figure()
        for i, n in enumerate(dirs):
            mask = df[f'steps_dir{i+1}'] != 0.0
            x_vals = df.loc[mask, f'steps_dir{i+1}']
            y_vals = df.loc[mask, f'gamma_dir{i+1}']
            pairs_col = df.loc[mask, f'pairs_dir{i+1}']
            if show_pairs:
                col_min, col_max = pairs_col.min(), pairs_col.max()
                if col_max > col_min:
                    marker_size = MIN_MARKER + (pairs_col - col_min) / (col_max - col_min) * (MAX_MARKER - MIN_MARKER)
                else:
                    marker_size = pd.Series([MIN_MARKER] * len(pairs_col), index=pairs_col.index)
            else:
                marker_size = 12
            fig.add_trace(
                go.Scatter(
                    x = x_vals,
                    y = y_vals,
                    mode = 'markers+lines',
                    marker = {
                        'color': colors[i],
                        'size': marker_size
                    },
                    line = {
                        'color': colors[i],
                        'width': 3,
                        'dash' : "dash"
                    },
                    name = n,
                    opacity = 0.9,
                    text = pairs_col,
                    textposition = "top center",
                    hovertemplate= "Steps: %{x}<br>" +
                    "Variogram: %{y}<br>" +
                    "Pairs: %{text}" +
                    "<extra></extra>",
                )
            )
        fig.update_layout(
            # width = 800,
            margin = dict(l=20, b=20, t=40, r=20),
            title = {'text': title},
            xaxis = {
                'title': 'Step',
                'range': [0, df['steps_dir1'].max()]
            },
            yaxis = {
                'title': 'Variograma',
                'range': [0, gamma_max],
            }
        )
        if export:
            fig.write_html(f'Variogram-{self.azim}-{self.dip}.html')
        return fig


if __name__ == '__main__':
    DATAFILE = r'..\tests\data\sondajes_cerro_blanco_final.dat'
    COORDINATES = ['Este', 'Norte', 'Cota']
    GRADE = 'AuGrade'

    base_params = {
        "azimuth_tolerance": 22.5,
        "horizontal_bandwidth": 99999.0,
        "dip": 81.0,
        "dip_tolerance": 22.5,
        "vertical_bandwidth": 99999.0,
        "lag_tolerance": 6.0,
        "lag_count": 10.0,
        "lag_size": 12.0,
        "value": GRADE,
    }

    # Direction 1 — azimuth 0
    v1 = Variogram(DATAFILE, COORDINATES, GRADE, {**base_params, "azimuth": 0.0})
    r1 = v1.single_semivariogram()

    # Direction 2 — azimuth 90
    v2 = Variogram(DATAFILE, COORDINATES, GRADE, {**base_params, "azimuth": 90.0})
    r2 = v2.single_semivariogram()

    # Plot both directions on the same figure
    combined = r1.add_direction(r2)
    fig_combined = v1.plot(combined, show_pairs=False, palette='andes', export=False)
    fig_combined.show()