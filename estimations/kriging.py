import os
import tempfile
import subprocess
import numpy as np
from andesite.datafiles.grid import Grid, GridDatafile
from andesite.utils.manipulations import globalize_backslashes
from .estimation_exceptions import OutputNameNotProvidedException, SameOutputVariablesException
from andesite.utils.files import (
    dataframe_to_gslib,
    grab_index_coordinates,
    grab_index_target,
    load_datafile,
    read_file_from_gslib,
    transform_datafile_to_gslib,
)

KT3DSEQ_BIN_PATH = '../utils/bin/kt3d_Seq.exe'
KT3DPAR_BIN_PATH = '../utils/bin/kt3d_OpenMP.exe'
KT3DPARAMETERS_PATH = '../utils/bin/kt3d-generic.par'

class KrigingExecutor:
    def __init__(
            self, input_drillholes, coordinates, grades, variogram_structs, variogram_nugget,
            grid_mn, grid_n, grid_siz, output_datafile, parameters, out_vars,
            ug_enabled=False, ug_grid_datafile_path="", ug_col_name="", ug_target=0,
            dh_ug_col_name="", dhid_col_name="", jackknife_path=""):
        self.input_drillholes = input_drillholes
        self.coordinates = coordinates
        self.input_grades = grades
        self.variogram_structs = variogram_structs
        self.variogram_nugget = variogram_nugget
        self.grid_mn = grid_mn
        self.grid_n = grid_n
        self.grid_siz = grid_siz
        self.kriging_parameters = parameters
        self.output_datafile = output_datafile
        self.kriging_estimate = out_vars[0]
        self.kriging_variance = out_vars[1]
        self.n_structs = len(self.variogram_structs)
        self.ug_enabled = ug_enabled
        self.ug_grid_datafile_path = ug_grid_datafile_path
        self.ug_col_name = ug_col_name
        self.ug_target = ug_target
        self.ug_grid_gslib = ""
        self.dh_ug_col_name = dh_ug_col_name
        self.dhid_col_name = dhid_col_name
        self.jackknife_path = jackknife_path
        self.real_jackknife_path = ""

        self.grid = Grid(*self.grid_mn, *self.grid_siz, *self.grid_n)
        self.block_model = self.grid.create()

    def grab_varmodel_params(self, variogram_structures):
        """
        variogram model types
        1: spherical
        2: exponential
        3: Gaussian
        """
        angles = variogram_structures['angles']
        directions = variogram_structures['ranges']
        sill = variogram_structures['sill']
        model = variogram_structures['type']
        model_idx = 1 if model=='Spherical' else (2 if model=='Exponential' else 3)
        angles_fmt = f'{model_idx}    {sill}  {angles[0]}   {angles[1]}   {angles[2]}       -it,cc,ang1,ang2,ang3\n'
        directions_fmt = f'         {directions[0]}  {directions[1]}  {directions[2]}     -a_hmax, a_hmin, a_vert\n'
        return angles_fmt, directions_fmt

    def create_kt3d_params(self, mode: int = 0):
        # variables needed
        self.real_path_drillholes = transform_datafile_to_gslib(self.input_drillholes)
        ix, iy, iz = grab_index_coordinates(self.real_path_drillholes, self.coordinates)
        igrade = grab_index_target(self.real_path_drillholes, self.input_grades)

        # DHID cross-validation: encode string DHID column to integers and add to GSLIB
        dh_col_idx = 0
        if self.dhid_col_name:
            original_df = load_datafile(self.input_drillholes)
            dhid_vals = original_df[self.dhid_col_name]
            unique_ids = sorted(dhid_vals.unique())
            encoding = {v: i + 1 for i, v in enumerate(unique_ids)}
            encoded = dhid_vals.map(encoding).values
            gslib_df = read_file_from_gslib(self.real_path_drillholes).compute()
            gslib_df['DHID_ENCODED'] = encoded
            dataframe_to_gslib(gslib_df, self.real_path_drillholes)
            dh_col_idx = grab_index_target(self.real_path_drillholes, 'DHID_ENCODED')

        xmn, ymn, zmn = self.grid_mn
        nx, ny, nz = self.grid_n
        xsiz, ysiz, zsiz = self.grid_siz
        xdc, ydc, zdc = self.kriging_parameters['discretization']
        min_data, max_data, max_octants = self.kriging_parameters['search data']
        xradius, yradius, zradius = self.kriging_parameters['search radius']
        xangle, yangle, zangle = self.kriging_parameters['search angles']
        krig_type = 1 if self.kriging_parameters['type'] == 'Ordinary' else 0
        krig_mean = self.kriging_parameters['simple kriging'][0]

        # Creation of the parameters file for KT3D
        temp_params_path = tempfile.NamedTemporaryFile(prefix="params_kt3d"+'_', suffix=".par", delete=False)
        temp_out_filename_path = tempfile.NamedTemporaryFile(prefix="kt3d_", suffix=".out", delete=False)
        temp_dbg_filename_path = tempfile.NamedTemporaryFile(prefix="kt3d_", suffix=".dbg", delete=False)
        self.fmt_params_path = globalize_backslashes(temp_params_path.name)
        self.fmt_out_path = globalize_backslashes(temp_out_filename_path.name)
        self.fmt_dbg_path = globalize_backslashes(temp_dbg_filename_path.name)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(current_dir, KT3DPARAMETERS_PATH), 'r') as file:
            lines = file.readlines()

        lines[4] = f'{globalize_backslashes(self.real_path_drillholes)}                     -file with data\n'
        dh_ug_col_idx = 0
        if self.ug_enabled and self.dh_ug_col_name:
            dh_ug_col_idx = grab_index_target(self.real_path_drillholes, self.dh_ug_col_name)
        lines[5] = f'{dh_col_idx}  {ix}  {iy}  {iz}  {igrade}  0  {dh_ug_col_idx}   -   columns for DH,X,Y,Z,var,sec var,ugcol\n'
        lines[7] = f'{mode}                                -option: 0=grid, 1=cross, 2=jackknife\n'

        # Jackknife targets file and column indices (mode=2 only)
        if mode == 2 and self.jackknife_path:
            self.real_jackknife_path = transform_datafile_to_gslib(self.jackknife_path)
            jk_ix, jk_iy, jk_iz = grab_index_coordinates(self.real_jackknife_path, self.coordinates)
            jk_igrade = grab_index_target(self.real_jackknife_path, self.input_grades)
            lines[8] = f'{globalize_backslashes(self.real_jackknife_path)}     -file with jackknife data\n'
            lines[9] = f'{jk_ix}   {jk_iy}   {jk_iz}    {jk_igrade}    0     -   columns for X,Y,Z,vr and sec var\n'
        lines[11] = f'{self.fmt_dbg_path}                         -file for debugging output\n'
        lines[12] = f'{self.fmt_out_path}                         -file for kriged output\n'
        lines[13] = f'{int(nx)}   {xmn}    {xsiz}                -nx,xmn,xsiz\n'
        lines[14] = f'{int(ny)}   {ymn}    {ysiz}                -ny,ymn,ysiz\n'
        lines[15] = f'{int(nz)}   {zmn}    {zsiz}                  -nz,zmn,zsiz\n'
        lines[16] = f'{xdc}    {ydc}      {zdc}                    -x,y and z block discretization\n'
        lines[17] = f'{min_data}    {max_data}                          -min, max data for kriging\n'
        lines[18] = f'{max_octants}                               -max per octant (0-> not used)\n'
        lines[19] = f'{xradius}  {yradius}  {zradius}              -maximum search radii\n'
        lines[20] = f' {xangle}   {yangle}   {zangle}                 -angles for search ellipsoid\n'
        lines[21] = f'{krig_type}     {krig_mean}                      -0=SK,1=OK,2=non-st SK,3=exdrift\n'
        lines[26] = f'{self.n_structs}    {self.variogram_nugget}                        -nst, nugget effect\n'
        for i in range(self.n_structs):
            angles, directions = self.grab_varmodel_params(self.variogram_structs[i])
            lines[27 + i*2] = angles
            lines[28 + i*2] = directions

        cutoff = 27 + 2 * self.n_structs

        with open(self.fmt_params_path, 'w') as f:
            f.writelines(lines[:cutoff])
            if self.ug_enabled:
                self.ug_grid_gslib = transform_datafile_to_gslib(self.ug_grid_datafile_path)
                ug_col_idx = grab_index_target(self.ug_grid_gslib, self.ug_col_name)
                f.write(f'1                             -use ug kriging (0=no, 1=yes)\n')
                f.write(f'{globalize_backslashes(self.ug_grid_gslib)}                -external grid file\n')
                f.write(f'{ug_col_idx}                             -column for UG values in grid file\n')
                f.write(f'{self.ug_target}                             -target UG to krige\n')
            else:
                f.write('0                             -use ug kriging (0=no, 1=yes)\n')
                f.write('dummy.dat                     -external grid file\n')
                f.write('0                             -column for UG values in grid file\n')
                f.write('0                             -target UG to krige\n')

        print(f'File {self.fmt_params_path} created!')
        return self.fmt_params_path, self.fmt_out_path

    def run_kt3d(self, mode: int = 0):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        exe_file =  KT3DSEQ_BIN_PATH if mode == 2 else KT3DPAR_BIN_PATH
        print(f'running >>> {os.path.join(current_dir, exe_file)} {self.fmt_params_path}')
        CREATE_NO_WINDOW = 0x08000000
        output = subprocess.check_output([
            globalize_backslashes(os.path.join(current_dir, exe_file)), f'{self.fmt_params_path}'], creationflags=CREATE_NO_WINDOW, stderr=subprocess.STDOUT)
        output_str = output.decode("utf-8")
        if "KT3D Version: 3.500 Finished" in output_str:
            return True
        else:
            return False

    def check_kriging_variables(self):
        print("check_kriging_variables() starting ...")
        if self.kriging_estimate == self.kriging_variance:
            raise SameOutputVariablesException("Output variables has the same name")
        if self.kriging_estimate == '' or self.kriging_variance == '':
            raise OutputNameNotProvidedException("Please provide valid names for output variables")
        print("check_kriging_variables() completed!")

    def get_kt_results(self, output):
        kt3d_df = read_file_from_gslib(output).compute()
        return kt3d_df['Estimate'].to_numpy(), kt3d_df['EstimationVariance'].to_numpy()

    def clear(self):
        files = [self.fmt_params_path, self.fmt_out_path, self.fmt_dbg_path, self.real_path_drillholes]
        if self.ug_grid_gslib:
            files.append(self.ug_grid_gslib)
        if self.real_jackknife_path:
            files.append(self.real_jackknife_path)
        for file in files:
            try:
                os.remove(file)
            except:
                continue

    def _run_validation(self, mode: int):
        self.check_kriging_variables()
        params_path, output_path = self.create_kt3d_params(mode=mode)
        print(f"After creating the params file {params_path} the KT3D will start")
        self.kt_status = self.run_kt3d(mode=mode)
        if self.kt_status:
            df = read_file_from_gslib(output_path).compute()
            df = df.rename(columns={'ErrorEstimation': 'Error'})
            df['StandardizedError'] = (
                (df['True'] - df['Estimate']) / np.sqrt(df['EstimationVariance'])
            )
            df['AbsoluteError'] = np.abs(df['Error'])
            return df[['X', 'Y', 'Z', 'True', 'Estimate', 'EstimationVariance',
                        'Error', 'StandardizedError', 'AbsoluteError']]
        else:
            raise Exception(f'KT3D validation failed (mode={mode})\n>>> {params_path}')

    def cross_validation(self):
        return self._run_validation(mode=1)

    def jackknife(self):
        return self._run_validation(mode=2)

    def estimate(self, just_results=False):
        self.check_kriging_variables()
        params_path, output_path = self.create_kt3d_params()

        self.kt_status = self.run_kt3d()
        if self.kt_status:
            self.estimate_values, self.variance_values = self.get_kt_results(output_path)
        else:
            raise Exception(f'KT3D estimation failed\n>>> {params_path}')
        if just_results:
            return self.estimate_values, self.variance_values
        else:
            self.block_model[self.kriging_estimate] = self.estimate_values
            self.block_model[self.kriging_variance] = self.variance_values
            estimate_df = self.block_model.replace(-999.0, np.nan)
            return GridDatafile(estimate_df, self.grid.get_metadata())

    def save_kriging(self):
        assert self.kt_status == True, "Please run estimate() first"
        if self.output_datafile == '':
            raise OutputNameNotProvidedException("Please provide valid names for output variables")
        dataframe_to_gslib(self.block_model, self.output_datafile)