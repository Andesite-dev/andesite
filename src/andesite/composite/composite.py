import os
import time
import pandas as pd
import numpy as np
from icecream import ic
from utils.manipulations import find_pattern_on_list, globalize_backslashes
from utils.files import read_file_from_gslib
from .categoric_compositing import one_drill_categoric, one_drill_categoric_multi
from .numeric_compositing import one_drill_numeric, one_drill_numeric_multi
from pandas.api.types import is_numeric_dtype
from utils._globals import (
    POSSIBLE_X_COLUMNS,
    POSSIBLE_Y_COLUMNS,
    POSSIBLE_Z_COLUMNS,
    POSSIBLE_TO_COLUMNS,
    POSSIBLE_AZIM_COLUMNS,
    POSSIBLE_DIP_COLUMNS,
    POSSIBLE_FROM_COLUMNS,
    POSSIBLE_HOLEID_COLUMNS,
    POSSIBLE_LENGTH_COLUMNS
)

class DatafileFactory:
    def __init__(self, path):
        self.path = path
        self.dataframe = self.read()
        self.grab_parameters()
        self.metadata = {
            'path': self.path,
            'holeid': self.holeid_col
        }

    def __repr__(self):
        return f"{self.__class__.__name__} instance from {globalize_backslashes(self.path)}"

    def read(self):
        if self.path.endswith('.csv'):
            return pd.read_csv(self.path)
        elif self.path.endswith('.hdf5'):
            return pd.read_hdf(self.path)
        else:
            return read_file_from_gslib(self.path).compute()

    def load(self):
        return self.dataframe

    def get_metadata(self):
        return self.metadata

    def grab_parameters(self):
        self.holeid_col = find_pattern_on_list(self.dataframe.columns, POSSIBLE_HOLEID_COLUMNS)

class Assay(DatafileFactory):
    def __init__(self, path):
        super().__init__(path)
        self.metadata.update({
            'from': self.from_col,
            'to': self.to_col,
            'length': self.length
        })

    def grab_parameters(self):
        super().grab_parameters()
        self.from_col = find_pattern_on_list(self.dataframe.columns, POSSIBLE_FROM_COLUMNS)
        self.to_col = find_pattern_on_list(self.dataframe.columns, POSSIBLE_TO_COLUMNS)
        self.length = find_pattern_on_list(self.dataframe.columns, POSSIBLE_LENGTH_COLUMNS)

class Collar(DatafileFactory):
    def __init__(self, path):
        super().__init__(path)
        self.metadata.update({
            'east': self.east,
            'north': self.north,
            'elevation': self.elevation,
            'length': self.length
        })

    def grab_parameters(self):
        super().grab_parameters()
        self.east = find_pattern_on_list(self.dataframe.columns, POSSIBLE_X_COLUMNS)
        self.north = find_pattern_on_list(self.dataframe.columns, POSSIBLE_Y_COLUMNS)
        self.elevation = find_pattern_on_list(self.dataframe.columns, POSSIBLE_Z_COLUMNS)
        self.length = find_pattern_on_list(self.dataframe.columns, POSSIBLE_LENGTH_COLUMNS)

class Survey(DatafileFactory):
    def __init__(self, path):
        super().__init__(path)
        self.metadata.update({
            'depth': self.depth,
            'azim': self.azim,
            'dip': self.dip
        })

    def grab_parameters(self):
        super().grab_parameters()
        self.depth = find_pattern_on_list(self.dataframe.columns, POSSIBLE_TO_COLUMNS)
        self.azim = find_pattern_on_list(self.dataframe.columns, POSSIBLE_AZIM_COLUMNS)
        self.dip = find_pattern_on_list(self.dataframe.columns, POSSIBLE_DIP_COLUMNS)

class DatafileComposite:
    """
    TODO: los missing drillholes estan mal calculados, muestran los que si estan no se porque
    TODO: __set_survey_tail no sabe que hacer cuando el primer dato no es 0, ej: si parte de 30.0 un sondaje
          en survey, despues no va a tener coordenas x, y, z entre el 0 y 30.
    TODO: Hay sondajes tipo Survey tienen este comportamiento, aveces mas de un sondaje no parte del 0, o incluso
          todos son así, haciendo más dificil el proceso. Existen tres maneras de solucionarlo:
          1. Crear otro method llamado __set_survey_head, pero tendria que por un lado el usuario decirme que la BD
             es así, y por otro lado debe ocurrir en todos los holes en Survey
          2. Solucionarlo dentro del mismo __set_survey_tail, haciendo todo esto automatizado, en caso de encontrarse
             con que más de un sondaje en su columna depth no parte de 0, entonces agregarle esa row en el df para
             poder continuar. Además se debe ordenar siempre la base de datos respecto a estas variables en caso
             de que algo ocurra mál, o que la BD venga con datos desordenados
          3. La BD de Survey realmente es así, por lo que no hay problema que no partan de 0 (cosa que veo dificil),
             aunque esto traeria como consecuencia que los valores de Collar esten corridos X metros en cada hole,
             como si realmente el "inicio" del sondaje en terminos de testigo, comenzara más abajo, cosa que tambien
             veo bastante extraña
    SOLUCION TEMPORAL ES REMOVER ESTAS COORDENADAS CON NAN INCLUSO SI TIENE VALORES COMPOSITADOS
    """

    def __init__(self, assay: Assay, collar: Collar, survey: Survey):
        assert isinstance(assay, Assay), "assay must be type <<Assay>>"
        assert isinstance(collar, Collar), "collar must be type <<Collar>>"
        assert isinstance(survey, Survey), "survey must be type <<Survey>>"
        
        self.assay = assay
        self.collar = collar
        self.survey = survey

        self.assay_df = self.assay.dataframe
        self.survey_df = self.survey.dataframe
        self.collar_df = self.collar.dataframe
        self.metadata = {}

        self.check_holeid_match()

    def check_holeid_match(self):
        # print('check_holeid_match() called')
        dhid_in_all = list(set(self.assay_df[self.assay.holeid_col].to_numpy()) & 
                           set(self.survey_df[self.survey.holeid_col].to_numpy()) & 
                           set(self.collar_df[self.collar.holeid_col].to_numpy()))
        self.final_collar = self.collar_df[self.collar_df[self.collar.holeid_col].isin(dhid_in_all)].reset_index(drop = True)
        self.final_assay = self.assay_df[self.assay_df[self.assay.holeid_col].isin(dhid_in_all)].reset_index(drop = True)
        self.final_survey = self.survey_df[self.survey_df[self.survey.holeid_col].isin(dhid_in_all)].reset_index(drop = True)


    def holeid_missmatch(self):
        # print('holeid_missmatch() called')
        dhid_not_in_all = list(set(self.assay_df[self.assay.holeid_col].to_numpy()) ^ 
                               set(self.survey_df[self.survey.holeid_col].to_numpy()) ^ 
                               set(self.collar_df[self.collar.holeid_col].to_numpy()))
        if len(dhid_not_in_all) > 0:
            np.savetxt('missing_drillholes.txt', np.array(dhid_not_in_all, dtype=np.object0), fmt="%s")
            return 'Los codigos de sondaje que no se encuentran en todas las bases de datos, los sondajes faltantes se guardan en <<missing_drillholes.txt>>'
        return 'No missmatch found on databases'
    
    def composite_functions_handler(self, df, holeid, mode, var_lst, composite_length):
        #print(f'composite_functions_handler() called for {holeid}')
        assert mode in ['numeric', 'categorical'], 'mode parameter must be <numeric> or <categorical>'
        if mode == 'numeric':
            if len(var_lst) > 1:
                return one_drill_numeric_multi(df, self.assay.holeid_col, holeid, self.assay.from_col, self.assay.to_col, var_lst, composite_length)
            else:
                return one_drill_numeric(df, self.assay.holeid_col, holeid, self.assay.from_col, self.assay.to_col, var_lst[0], composite_length)
        else:
            if len(var_lst) > 1:
                return one_drill_categoric_multi(df, self.assay.holeid_col, holeid, self.assay.from_col, self.assay.to_col, var_lst, composite_length)
            else:
                return one_drill_categoric(df, self.assay.holeid_col, holeid, self.assay.from_col, self.assay.to_col, var_lst[0], composite_length)

    def check_dtype(self, target):
        # print('check_dtype() called')
        if is_numeric_dtype(target):
            return 'numeric'
        return 'categorical'

    def set_target_values(self, vars):
        # print('set_target_values() called')
        vars_types = {}
        if isinstance(vars, str):
            vars_types[vars] =  self.check_dtype(self.assay_df[vars])
        else:
            for v in vars:
                vars_types[v] = self.check_dtype(self.assay_df[v])
        return vars_types
    
    def clasify_target_values(self, vars):
        # print('clasify_target_values() called')
        self.variables_dict = self.set_target_values(vars)
        self.num_vars, self.cat_vars = [], []
        for k, v in self.variables_dict.items():
            if v == 'numeric':
                self.num_vars.append(k)
            else:
                self.cat_vars.append(k)
        return self.num_vars, self.cat_vars


    def _compositing(self, vars, length):
        # Tell the user the missmatch of the datasets
        if isinstance(vars, str):
            vars = [vars]
        # self.holeid_missmatch()
        # Separate the numeric and the categorical variables in 2 lists
        numeric_vars, categorical_vars = self.clasify_target_values(vars)
        # Calculate the composite for both numeric and categorical variables
        comp_num_results, comp_cat_results = [], []
        if len(numeric_vars) > 0:
            comp_num_results = [self.composite_functions_handler(group, hole_id, 'numeric', numeric_vars, length) for hole_id, group in self.final_assay.groupby(self.assay.holeid_col)]
            df_numeric = pd.concat(comp_num_results)
        if len(categorical_vars) > 0:
            comp_cat_results = [self.composite_functions_handler(group, hole_id, 'categorical', categorical_vars, length) for hole_id, group in self.final_assay.groupby(self.assay.holeid_col)]
            df_categorical = pd.concat(comp_cat_results)
        # Join the results into a dataframe
        if (len(comp_num_results) > 0) and (len(comp_cat_results) > 0):
            assay_df_comp = df_numeric.merge(df_categorical, left_on=[self.assay.holeid_col, 'from', 'to'], right_on=[self.assay.holeid_col, 'from', 'to'], how='outer')
            return assay_df_comp
        elif (len(comp_num_results) > 0) and (len(comp_cat_results) == 0):
            return df_numeric
        else:
            return df_categorical
    
    def __set_survey_tail(self, row):
        # This simple function creates a "TO" column on the survey dataset specifically
        # it is only use in the context of composite the survey AZ and DIP
        hole_id = row[self.survey.holeid_col]
        try:
            next_hole_id = self.final_survey.loc[row.name + 1, self.survey.holeid_col]
        except:
            next_hole_id = ''

        if hole_id != next_hole_id:
            tail = self.final_collar.loc[self.final_collar[self.collar.holeid_col] == hole_id, self.collar.length].values[0]
            return tail
        else:
            next_depth = self.final_survey.loc[row.name + 1, self.survey.depth]
            return next_depth

    def new_coords(self, x, y, z, az, dip, length):
        # print('new_coords() called')
        # Convert azimuth and dip to radians
        az_rad = np.radians(az)
        dip_rad = np.radians(dip)
        
        # Calculate the changes in coordinates
        delta_z = length * np.sin(dip_rad)
        delta_y = length * np.cos(dip_rad) * np.cos(az_rad)
        delta_x = length * np.cos(dip_rad) * np.sin(az_rad)
        
        # Calculate the cumulative sum of changes
        new_z = np.round(z + np.cumsum(delta_z), 3)
        new_y = np.round(y + np.cumsum(delta_y), 3)
        new_x = np.round(x + np.cumsum(delta_x), 3)
        
        # Transpose and concatenate the arrays to represent coordinates as a 3D space
        # new_coords_3d = np.round(np.concatenate((new_x[:, np.newaxis], new_y[:, np.newaxis], new_z[:, np.newaxis]), axis=1), 3)
        
        return new_x, new_y, new_z

    def composite(self, vars, length, output_filename=None):
        # print('composite() called')
        composites_df = self._compositing(vars, length)
        # Composite the Survey Dataset
        self.final_survey[f'{self.survey.depth}tail'] = self.final_survey.apply(self.__set_survey_tail, axis=1)
        results_survey = [one_drill_numeric_multi(group, self.survey.holeid_col, hole_id, self.survey.depth, f'{self.survey.depth}tail', [self.survey.azim, self.survey.dip], length) for hole_id, group in self.final_survey.groupby(self.survey.holeid_col)]
        survey_df = pd.concat(results_survey)
        merge_df = composites_df.merge(
            survey_df, 
            left_on=[self.assay.holeid_col, 'from', 'to'], 
            right_on=[self.survey.holeid_col, 'from', 'to'], 
            how='outer'
        )
        merge_df = merge_df.sort_values(by=[self.assay.holeid_col, 'from', 'to'])
        i = 0
        for n, g in merge_df.groupby(self.assay.holeid_col):
            initial_coordinates = self.final_collar.loc[i, [self.collar.east, self.collar.north, self.collar.elevation]].to_numpy()
            length_arr = np.repeat(length, g.shape[0])
            length_arr[0] = length/2.0
            new_x, new_y, new_z = self.new_coords(*initial_coordinates, g[self.survey.azim], g[self.survey.dip], length_arr)
            merge_df.loc[merge_df[self.assay.holeid_col] == n, 'midx'] = new_x
            merge_df.loc[merge_df[self.assay.holeid_col] == n, 'midy'] = new_y
            merge_df.loc[merge_df[self.assay.holeid_col] == n, 'midz'] = new_z
            i += 1
        merge_df = merge_df.dropna(subset=['midx', 'midy', 'midz'], how='any')
        if isinstance(vars, str):
            vars = [vars]
        composites_final = merge_df[[self.assay.holeid_col, 'from', 'to', 'midx', 'midy', 'midz', *vars]]

        if output_filename is not None:
            composites_final.to_csv(f'{output_filename}.csv', index=False)
        return composites_final


# Compositacion Lab a 2 metros a 2 variables 1.35 segundos -> 11k compositos (na included)
# Compositacion de Parker a 1.5 metrso de 13 variables: 14.345 segundos -> 81k compositos (na included)
# Compositacion de RQD a 1.5 metros a 1 variable: 2 min 12.891 segundos -> 287k compositos (na included)