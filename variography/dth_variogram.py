import os
import subprocess
import tempfile

from andesite.utils.files import (
    dataframe_to_gslib,
    grab_index_coordinates,
    grab_index_target,
    read_file_from_gslib,
)
from andesite.utils.manipulations import globalize_backslashes
from andesite.variography.experimental import VariogramDatafile


class DTHVariographyTask:

    def __init__(self, input_drillholes, coordinates, input_grades, bhid_col, from_col, to_col, params):
        self.input_drillholes_path = input_drillholes
        self.coordinates = coordinates
        self.input_grades = input_grades
        self.bhid_col = bhid_col
        self.from_col = from_col
        self.to_col = to_col
        self.azim = params.get('azimuth', 0.0)
        self.atol = params.get('azimuth_tolerance', 90.0)
        self.dip = params.get('dip', 90.0)
        self.diptol = params.get('dip_tolerance', 90.0)
        self.lag_count = params['lag_count']
        self.lag_size = params['lag_size']
        self.lag_tol = params.get('lag_tolerance', params['lag_size'] / 2.0)
        self.bandh = params.get('horizontal_bandwidth', 99999.0)
        self.bandv = params.get('vertical_bandwidth', 99999.0)
        self.metadata = {
            'path': self.input_drillholes_path,
            'grades': self.input_grades,
            'coordinates': self.coordinates,
            'bhid_col': bhid_col,
            'from_col': from_col,
            'to_col': to_col,
            'parameters': params,
        }

    def get_metadata(self):
        return self.metadata

    def clear(self):
        for path in [self.fmt_out_path, self.fmt_params_path]:
            try:
                os.remove(path)
            except OSError:
                pass

    def create_params_temp(self):
        print(f"[DTHVariographyTask] input={self.input_drillholes_path}")
        df = read_file_from_gslib(self.input_drillholes_path).compute()
        print(f"[DTHVariographyTask] columns={list(df.columns)}")
        print(f"[DTHVariographyTask] bhid_col={self.bhid_col!r}, from={self.from_col!r}, to={self.to_col!r}, grade={self.input_grades!r}")

        df = df.sort_values(by=[self.bhid_col, self.from_col, self.to_col])
        base_filename = os.path.splitext(os.path.basename(self.input_drillholes_path))[0]
        real_path_filename = globalize_backslashes(
            os.path.join(tempfile.gettempdir(), f'{base_filename}.dat')
        )
        dataframe_to_gslib(df, real_path_filename)

        ibhid = grab_index_target(real_path_filename, self.bhid_col)
        ix, iy, iz = grab_index_coordinates(real_path_filename, self.coordinates)
        ifrom = grab_index_target(real_path_filename, self.from_col)
        ito = grab_index_target(real_path_filename, self.to_col)
        igrade = grab_index_target(real_path_filename, self.input_grades)
        print(f"[DTHVariographyTask] indices: ibhid={ibhid}, ix={ix}, iy={iy}, iz={iz}, ifrom={ifrom}, ito={ito}, igrade={igrade}")

        temp_params_path = tempfile.NamedTemporaryFile(prefix='params_gamv_', suffix='.par', delete=False)
        self.fmt_params_path = globalize_backslashes(temp_params_path.name)
        temp_params_path.close()
        temp_out_path = tempfile.NamedTemporaryFile(prefix='gamv_', suffix='.out', delete=False)
        self.fmt_out_path = globalize_backslashes(temp_out_path.name)
        temp_out_path.close()

        current_dir = os.path.dirname(os.path.abspath(__file__))
        par_template = os.path.join(current_dir, '../utils/bin/gamv-generic.par')
        with open(par_template, 'r') as f:
            lines = f.readlines()

        with open(self.fmt_params_path, 'w') as f:
            lines[1] = f'{globalize_backslashes(real_path_filename)}            -file with data\n'
            lines[2] = f'{ibhid}   {ix}   {iy}   {iz}   0   {ifrom}   {ito}         -columns: BHID,X,Y,Z,WT,FROM,TO\n'
            lines[3] = f'1   {igrade}                         -   number of variables,col numbers\n'
            lines[5] = f'{self.fmt_out_path}                         -file for variogram output\n'
            lines[6] = f'{int(self.lag_count)}                                -number of lags\n'
            lines[7] = f'{self.lag_size}                              -lag separation distance\n'
            lines[8] = f'{self.lag_tol}                               -lag tolerance\n'
            lines[10] = f'{self.azim} {self.atol} {self.bandh} {self.dip} {self.diptol} {self.bandv}   -azm,atol,bandh,dip,dtol,bandv\n'
            lines[11] = '0                                 -standardize sills? (0=no, 1=yes)\n'
            lines[12] = '1                                 -number of variograms\n'
            lines[13] = '1   1   11                         -tail var., head var., variogram type\n'
            f.writelines(lines)

        self.metadata['real_path'] = real_path_filename

    def run_gamv(self, parameters):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        CREATE_NO_WINDOW = 0x08000000
        exe = globalize_backslashes(os.path.join(current_dir, '../utils/bin/gamv_OpenMP.exe'))
        print(f'running >>> {os.path.join(current_dir, exe)}  {parameters}')
        output = subprocess.check_output([exe, parameters], creationflags=CREATE_NO_WINDOW)
        output_str = output.decode('utf-8')
        if 'GAMV elapsed time:' in output_str:
            return
        raise Exception(f'gamv_OpenMP.exe failed\n>>> {exe} {parameters}\n{output_str}')

    def gamv_formatted_dth(self, gamv_file):
        """Split concatenated Down-the-Hole Variogram sections into separate Geo-EAS files."""
        with open(gamv_file, 'r') as f:
            content = f.read()
        parts = content.strip().split('Down-the-Hole Variogram')
        out_paths = []
        for index, section in enumerate(parts[1:], start=1):
            out_path = f'{os.path.splitext(gamv_file)[0]}_{index}.out'
            with open(out_path, 'w') as f:
                f.write('Down-the-Hole Variogram' + section)
            out_paths.append(out_path)
        return out_paths

    def single_semivariogram(self) -> VariogramDatafile:
        self.create_params_temp()
        self.run_gamv(self.fmt_params_path)

        actual_out_path = f'{os.path.splitext(self.fmt_out_path)[0]}_{self.input_grades}_dth_semivariogram.out'
        if not os.path.exists(actual_out_path) or os.path.getsize(actual_out_path) == 0:
            raise RuntimeError(f"gamv produced no output at {actual_out_path}")

        out_paths = self.gamv_formatted_dth(actual_out_path)
        if not out_paths:
            raise Exception('DTH variogram produced no output sections')

        df = read_file_from_gslib(out_paths[0]).compute()
        df.rename(columns={'steps': 'steps_dir1', 'gamma': 'gamma_dir1', 'pairs': 'pairs_dir1'}, inplace=True)
        self.metadata['vartype'] = 'dth'
        return VariogramDatafile(variogram=df, parameters=self.metadata)
