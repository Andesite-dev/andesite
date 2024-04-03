import os
import subprocess
import tempfile
from .estimation_exceptions import OutputNameNotProvidedException, SameOutputVariablesException
from andesite.utils.files import dataframe_to_gslib, grab_index_coordinates, grab_index_target, read_file_from_gslib, transform_datafile_to_gslib
from andesite.utils.manipulations import globalize_backslashes
from andesite.datafiles.grid import Grid

class Kriging:

    def __init__(self, input_drillholes, coordinates, grades, grid_metadata, variogram_model, output_datafile, parameters, estimate_name, variance_name):
        self.input_drillholes = input_drillholes
        self.coordinates = coordinates
        self.input_grades = grades
        self.grid_metadata = grid_metadata
        self.variogram_model = variogram_model
        self.output_datafile = output_datafile
        self.kriging_parameters = parameters
        self.kriging_estimate = estimate_name # Estas puden venir del run no especificamente de aquí
        self.kriging_variance = variance_name # Estas pueden venid del run no especificamente de inicio

    def grab_varmodel_params(self, variogram_structures):
        """
        variogram model types
        1: spherical
        2: exponential
        3: cubic
        4: Gaussian
        """
        angles = variogram_structures['angles']
        directions = variogram_structures['ranges']
        sill = variogram_structures['sill']
        model = variogram_structures['type']
        model_idx = 1 if model=='Spherical' else (2 if model=='Exponential' else 4)
        angles_fmt = f'{model_idx}    {sill}  {angles[0]}   {angles[1]}   {angles[2]}       -it,cc,ang1,ang2,ang3\n'
        directions_fmt = f'         {directions[0]}  {directions[1]}  {directions[2]}     -a_hmax, a_hmin, a_vert\n'
        return angles_fmt, directions_fmt

    def create_kt3d_params(self, cross_val: bool=False):
        # variables needed
        self.real_path_drillholes = transform_datafile_to_gslib(self.input_drillholes)
        ix, iy, iz = grab_index_coordinates(self.real_path_drillholes, self.coordinates)
        igrade = grab_index_target(self.real_path_drillholes, self.input_grades)
        xmn, ymn, zmn = self.grid_metadata['grid_mn']
        nx, ny, nz = self.grid_metadata['grid_n']
        xsiz, ysiz, zsiz = self.grid_metadata['grid_siz']
        xdc, ydc, zdc = self.kriging_parameters['discretization']
        min_data, max_data, max_octants = self.kriging_parameters['search data']
        xradius, yradius, zradius = self.kriging_parameters['search radius']
        xangle, yangle, zangle = self.kriging_parameters['search angles']
        krig_type = 1 if self.kriging_parameters['type'] == 'Ordinary' else 0
        krig_mean = self.kriging_parameters['simple kriging'][0]
        structures = self.variogram_model['structures']
        n_structures = len(structures)
        nugget = self.variogram_model['nugget']

        # Creation of the parameters file for KT3D
        temp_params_path = tempfile.NamedTemporaryFile(prefix="params_kt3d"+'_', suffix=".par", delete=False)
        temp_out_filename_path = tempfile.NamedTemporaryFile(prefix="kt3d_", suffix=".out", delete=False)
        temp_dbg_filename_path = tempfile.NamedTemporaryFile(prefix="kt3d_", suffix=".dbg", delete=False)
        self.fmt_params_path = globalize_backslashes(temp_params_path.name)
        self.fmt_out_path = globalize_backslashes(temp_out_filename_path.name)
        self.fmt_dbg_path = globalize_backslashes(temp_dbg_filename_path.name)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(current_dir, '../utils/bin/kt3d-generic.par'), 'r') as file:
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
            lines[26] = f'{n_structures}    {nugget}                        -nst, nugget effect\n'
            for i in range(n_structures):
                angles, directions = self.grab_varmodel_params(structures[i])
                lines[27 + i*2] = angles
                lines[28 + i*2] = directions
            f.writelines(lines)

        print(f'File {self.fmt_params_path} created!')
        return self.fmt_params_path, self.fmt_out_path

    def run_kt3d(self, cross_val: bool = False):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        exe_file = "kt3d_Seq.exe" if cross_val else "kt3d_OpenMP.exe"
        print(f'running >>>{os.path.join(current_dir, f"{exe_file}")} {self.fmt_params_path}')
        CREATE_NO_WINDOW = 0x08000000
        output = subprocess.check_output([globalize_backslashes(os.path.join(current_dir, f'../utils/bin/{exe_file}')), f'{self.fmt_params_path}'], creationflags=CREATE_NO_WINDOW)
        output_str = output.decode("utf-8")
        if "KT3D Version: 3.000 Finished" in output_str:
            return True
        else:
            return False

    def save_kriging_results(self, estimate_values, variance_values):
        self.output_datafile.add_variable(self.kriging_estimate, estimate_values)
        self.output_datafile.add_variable(self.kriging_variance, variance_values)

        self.output_datafile.save_data(self.output_datafile.get_metadata('path'))

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

    def get_kt_results(self, output):
        kt3d_df = read_file_from_gslib(output).compute()
        # logger.debug(f'columns of file {os.path.basename(output)}: {kt3d_df.columns}')
        # logger.debug(f'shape of kriging datafile: {kt3d_df.shape}')
        return kt3d_df.iloc[:, 0].to_numpy(), kt3d_df.iloc[:, 1].to_numpy()

    def clear(self):
        for file in [self.fmt_params_path, self.fmt_out_path, self.fmt_dbg_path, self.real_path_drillholes]:
            os.remove(file)

    def cross_validation(self):
        if self.kriging_estimate == self.kriging_variance:
            raise SameOutputVariablesException("Output variables has the same name")
        if self.kriging_estimate == '' or self.kriging_variance == '':
            raise OutputNameNotProvidedException("Please provide valid names for output variables")

        params_path, output_path = self.create_kt3d_params(cross_val=True)
        self.kt_status = self.run_kt3d(cross_val=True)
        if self.kt_status:
            return read_file_from_gslib(output_path).compute()
        else:
            raise Exception(f'Something wrong happend after run\n>>> bin/gamv_openMP.exe {params_path}')

    def estimate(self):
        if self.kriging_estimate == self.kriging_variance:
            raise SameOutputVariablesException("Output variables has the same name")
        if self.kriging_estimate == '' or self.kriging_variance == '':
            raise OutputNameNotProvidedException("Please provide valid names for output variables")

        params_path, output_path = self.create_kt3d_params()

        self.kt_status = self.run_kt3d()
        if self.kt_status:
            self.format_kt3d_results()
            self.estimate_values, self.variance_values = self.get_kt_results(output_path)
            self.output_datafile[self.kriging_estimate] = self.estimate_values
            self.output_datafile[self.kriging_variance] = self.variance_values
            return self.output_datafile
        else:
            raise Exception(f'Something wrong happend after run\n>>> bin/gamv_openMP.exe {params_path}')

    def save_kriging(self, output_name = ''):
        assert self.kt_status == True, "Please run estimate() first"
        if output_name == '':
            raise OutputNameNotProvidedException("Please provide valid names for output variables")
        dataframe_to_gslib(self.output_datafile, output_name)