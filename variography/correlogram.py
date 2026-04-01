import os
import re
import subprocess
import tempfile
from andesite.utils.files import grab_index_coordinates, load_datafile, read_file_from_gslib, dataframe_to_gslib
from andesite.utils.manipulations import globalize_backslashes
from itertools import combinations
class CorrelogramTask:

    def __init__(self, input_drillholes, coordinates, input_grades, input_ug, params):
        self.input_drillholes_path = input_drillholes
        self.coordinates = coordinates
        self.input_grades = input_grades
        self.input_ug = input_ug
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
            'ug': self.input_ug,
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

    def create_params_temp(self, n_ugs, idx_ugs, input_drillholes_path, ug_pairs, elipsoid=False, stand_sills=False):
        ix, iy, iz = grab_index_coordinates(input_drillholes_path, self.coordinates)

        temp_params_path = tempfile.NamedTemporaryFile(prefix="params_gamv"+'_', suffix=".par", delete=False)
        temp_out_filename_path = tempfile.NamedTemporaryFile(prefix="gamv"+'_', suffix=".out", delete=False)
        self.fmt_params_path = globalize_backslashes(temp_params_path.name)
        self.fmt_out_path = globalize_backslashes(temp_out_filename_path.name)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(current_dir, '../utils/bin/gamv-generic.par'), 'r', encoding='utf-8') as file:
            lines = file.readlines()
        with open(self.fmt_params_path, 'w', encoding='utf-8') as f:
            lines[1] = f'{globalize_backslashes(input_drillholes_path)}                      -file with data\n'
            lines[2] = f'{ix}   {iy}   {iz}                         -   columns for X, Y, Z coordinates\n'
            lines[3] = f'{n_ugs}   {idx_ugs}                             -   number of variables,col numbers\n'
            lines[5] = f'{self.fmt_out_path}                   - file for variogram output\n'
            lines[6] = f'{int(self.lag_count)}                                - number of lags\n'
            lines[7] = f'{self.lag_size}                               - lag separation distance\n'
            lines[8] = f'{self.lag_tol}                              -lag tolerance\n'
            lines[10] = f'{self.azim} {self.atol} {self.bandh} {self.dip} {self.diptol} {self.bandv}   -azm,atol,bandh,dip,dtol,bandv\n'
            lines[11] = f'{int(stand_sills)}                                 -standardize sills? (0=no, 1=yes)\n'
            lines[12] = f'{len(ug_pairs)}                                 -number of variograms\n'
            lines = lines[:13]
            for ug_pair in ug_pairs:
                lines.append(f'{ug_pair[0]}   {ug_pair[1]}   4                         -tail var., head var., variogram type\n')
            f.writelines(lines)

    def gamv_correlogram_formatted(self, gamv_file):
        with open(gamv_file, 'r', encoding='utf-8', errors='replace') as f:
            file_content = f.read()

        formatted_correlograms_paths = []

        # Split the file content into individual sets
        sets = file_content.strip().split('Correlogram')
        for index, set_content in enumerate(sets[1:], start=1):
            # Extract variable name for each set
            tail = re.search(r'tail:([^\s]+)', set_content).group(1)
            head = re.search(r'head:([^\s]+)', set_content).group(1)
            # Remove the first line (unwanted line) from each set
            set_content_lines = set_content.strip().split('\n')[1:]
            formatted_content = '\n'.join(set_content_lines)

            # Format the set content
            formatted_content = f'Correlogram\n8\nindex\nsteps\ngamma\npairs\nhead_{head}\ntail_{tail}\nvhead_{head}\nvtail_{tail}\n{formatted_content}'
            # Write the formatted content to a new file or overwrite the original file
            formatted_output_path = f'{os.path.splitext(gamv_file)[0]}_{index}.out'
            print(f"Writting on {formatted_output_path} ...")
            with open(formatted_output_path, 'w', encoding='utf-8') as file:
                file.write(formatted_content)

            formatted_correlograms_paths.append(formatted_output_path)

        return formatted_correlograms_paths

    def run_gamv(self, parameters):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        si = subprocess.STARTUPINFO()
        CREATE_NO_WINDOW = 0x08000000
        output = subprocess.check_output([globalize_backslashes(os.path.join(current_dir, '../utils/bin/gamv_OpenMP.exe')), f'{parameters}'], creationflags=CREATE_NO_WINDOW)
        output_str = output.decode("utf-8")
        if "GAMV Version: 3.100 Finished" in output_str:
            return
        else:
            raise Exception(f'Something wrong happend after run\n>>> bin/gamv_openMP.exe {parameters}\n{output_str}')

    def calculate(self):
        datafile = load_datafile(self.input_drillholes_path)
        unique_ugs = datafile[self.input_ug].unique().tolist()
        cross_ugs = list(combinations(range(1, len(unique_ugs) + 1), 2))

        def target_ohe(row, ug):
            if int(row[self.input_ug]) == int(ug):
                return row[self.input_grades]
            return -999

        ug_cols_idx = []
        for ug_val in unique_ugs:
            col_name = f'UG{int(ug_val)}_{self.input_grades}'
            datafile[col_name] = datafile.apply(target_ohe, axis=1, args=(ug_val,))
            ug_cols_idx.append(str(datafile.columns.get_loc(col_name) + 1))

        not_obj_df = datafile.select_dtypes(exclude=['object'])
        datafile_ug_path = tempfile.NamedTemporaryFile(prefix="drillhole"+'_', suffix=".dat", delete=False)

        ug_cols_idx_params = "  ".join(ug_cols_idx)

        dataframe_to_gslib(not_obj_df, datafile_ug_path.name)

        self.create_params_temp(
            n_ugs = len(unique_ugs),
            idx_ugs = ug_cols_idx_params,
            input_drillholes_path = datafile_ug_path.name,
            ug_pairs = cross_ugs
        )
        self.run_gamv(self.fmt_params_path)

        correlograms_paths = self.gamv_correlogram_formatted(self.fmt_out_path)
        return correlograms_paths




