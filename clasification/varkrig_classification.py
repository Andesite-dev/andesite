from andesite.datafiles.grid import GridDatafile
import dask.array as da
import numpy as np

class KrigingVarianceClasification():

    def __init__(self, blockmodel: GridDatafile, estimate_col: str, variance_col: str):
        self.blockmodel = blockmodel
        self.estimate_col = estimate_col
        self.variance_col = variance_col


    def clasify(self, thresh_cat_1: np.float32, thresh_cat_2, just_results: bool = False) -> GridDatafile:
        self.blockmodel_dataframe = self.blockmodel.load()
        cat_array = da.where((
            self.blockmodel_dataframe[self.variance_col] <= thresh_cat_1) & (self.blockmodel_dataframe[self.estimate_col] >= 0),
            1,
            da.where((
                thresh_cat_1 < self.blockmodel_dataframe[self.variance_col]) & (self.blockmodel_dataframe[self.variance_col] <= thresh_cat_2) & (self.blockmodel_dataframe[self.estimate_col] >= 0),
                2,
                da.where(
                    (self.blockmodel_dataframe[self.variance_col] > thresh_cat_2) & (self.blockmodel_dataframe[self.estimate_col] >= 0),
                    3,
                4)
        ))
        if just_results:
            return cat_array
        self.blockmodel.add_variable('cat', cat_array)
        return self.blockmodel