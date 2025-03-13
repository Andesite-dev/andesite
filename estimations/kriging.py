import os
import tempfile
import subprocess
import numpy as np
from andesite.datafiles.grid import Grid, GridDatafile
from andesite.utils.manipulations import globalize_backslashes
from .estimation_exceptions import OutputNameNotProvidedException, SameOutputVariablesException
from andesite.utils.files import dataframe_to_gslib, grab_index_coordinates, grab_index_target, read_file_from_gslib, transform_datafile_to_gslib

KT3DSEQ_BIN_PATH = '../utils/bin/kt3d_Seq.exe'
KT3DPAR_BIN_PATH = '../utils/bin/kt3d_OpenMP.exe'
KT3DPARAMETERS_PATH = '../utils/bin/kt3d-generic.par'

class KrigingExecutor:
    def __init__(self, input_drillholes, coordinates, grades, variogram_structs, variogram_nugget, grid_mn, grid_n, grid_siz, output_datafile, parameters, out_vars):
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

    def create_kt3d_params(self, cross_val: bool=False):
        # variables needed
        self.real_path_drillholes = transform_datafile_to_gslib(self.input_drillholes)
        ix, iy, iz = grab_index_coordinates(self.real_path_drillholes, self.coordinates)
        igrade = grab_index_target(self.real_path_drillholes, self.input_grades)
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
        with open(self.fmt_params_path, 'w') as f:
            lines[4] = f'{globalize_backslashes(self.real_path_drillholes)}                     -file with data\n'
            lines[5] = f'0  {ix}  {iy}  {iz}  {igrade}  0                 -   columns for DH,X,Y,Z,var,sec var\n'
            lines[7] = f'{int(cross_val)}                                -option: 0=grid, 1=cross, 2=jackknife\n'
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

            if self.n_structs == 1:
                lines[26] = f'2    {self.variogram_nugget}                        -nst, nugget effect\n'
                sill = np.float32(self.variogram_structs[0]['sill'])
                model = self.variogram_structs[0]['type']
                model_idx = 1 if model=='Spherical' else (2 if model=='Exponential' else 3)
                angles = self.variogram_structs[0]['angles']
                lines[27] = f'{model_idx}    {sill/2:.4f}  {angles[0]}   {angles[1]}   {angles[2]}       -it,cc,ang1,ang2,ang3\n'
                lines[28] = directions
                lines[29] = f'{model_idx}    {sill/2:.4f}  {angles[0]}   {angles[1]}   {angles[2]}       -it,cc,ang1,ang2,ang3\n'
                lines[30] = directions

            f.writelines(lines)

        print(f'File {self.fmt_params_path} created!')
        return self.fmt_params_path, self.fmt_out_path

    def run_kt3d(self, cross_val: bool = False):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        exe_file = KT3DSEQ_BIN_PATH if cross_val else KT3DPAR_BIN_PATH
        print(f'running >>>{os.path.join(current_dir, exe_file)} {self.fmt_params_path}')
        CREATE_NO_WINDOW = 0x08000000
        output = subprocess.check_output([globalize_backslashes(os.path.join(current_dir, exe_file)), f'{self.fmt_params_path}'], creationflags=CREATE_NO_WINDOW)
        output_str = output.decode("utf-8")
        if "KT3D Version: 3.000 Finished" in output_str:
            return True
        else:
            return False

    def format_kt3d_results(self):
        cores = os.cpu_count()

        target_out_file = self.fmt_out_path
        final_out_sim = 4000 + cores

        # Concatenate .out files
        for i in range(4001, final_out_sim + 1):
            file_sim_name = f"{self.fmt_out_path}{i}"
            with open(target_out_file, "ab") as target_file, open(file_sim_name, "rb") as source_file:
                target_file.write(source_file.read())
            os.remove(file_sim_name)
        print(f'File {target_out_file} joined correctly')

    def check_kriging_variables(self):
        if self.kriging_estimate == self.kriging_variance:
            raise SameOutputVariablesException("Output variables has the same name")
        if self.kriging_estimate == '' or self.kriging_variance == '':
            raise OutputNameNotProvidedException("Please provide valid names for output variables")

    def get_kt_results(self, output):
        kt3d_df = read_file_from_gslib(output).compute()
        # logger.debug(f'columns of file {os.path.basename(output)}: {kt3d_df.columns}')
        # logger.debug(f'shape of kriging datafile: {kt3d_df.shape}')
        return kt3d_df.iloc[:, 0].to_numpy(), kt3d_df.iloc[:, 1].to_numpy()

    def clear(self):
        for file in [self.fmt_params_path, self.fmt_out_path, self.fmt_dbg_path, self.real_path_drillholes]:
            try:
                os.remove(file)
            except:
                continue

    def cross_validation(self):
        self.check_kriging_variables()
        params_path, output_path = self.create_kt3d_params(cross_val=True)
        self.kt_status = self.run_kt3d(cross_val=True)
        if self.kt_status:
            raw_valcru_df = read_file_from_gslib(output_path).compute()
            valcru_df = raw_valcru_df.replace(-999.0, np.nan)
            return valcru_df
        else:
            raise Exception(f'Something wrong happend after run\n>>> bin/gamv_openMP.exe {params_path}')

    def estimate(self, just_results=False):
        self.check_kriging_variables()
        params_path, output_path = self.create_kt3d_params()

        self.kt_status = self.run_kt3d()
        if self.kt_status:
            self.format_kt3d_results()
            self.estimate_values, self.variance_values = self.get_kt_results(output_path)
        else:
            raise Exception(f'Something wrong happend after run\n>>> bin/gamv_openMP.exe {params_path}')
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