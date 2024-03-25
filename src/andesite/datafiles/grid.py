import numpy as np
import pandas as pd
from utils.files import dataframe_to_gslib, read_file_from_gslib


class Grid:

    def __init__(self, ix, iy, iz, dx, dy, dz, nx, ny, nz):
        self.ix = ix
        self.iy = iy
        self.iz = iz
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.metadata = {
            'grid_mn': [self.ix, self.iy, self.iz],
            'grid_n': [self.nx, self.ny, self.nz],
            'grid_siz': [self.dx, self.dy, self.dz]
        }

    def get_metadata(self):
        return self.metadata

    def __set_coords_to_block(self):
        self.x_coords = np.tile(np.arange(self.ix, self.ix + self.dx*self.nx, self.dx), self.ny*self.nz)
        self.y_coords = np.tile(np.repeat(np.arange(self.iy, self.iy + self.dy*self.ny, self.dy), self.nx), self.nz)
        self.z_coords = np.repeat(np.arange(self.iz, self.iz + self.dz*self.nz, self.dz), self.nx*self.ny)
        return self.x_coords, self.y_coords, self.z_coords

    def create(self, export=False):
        self.__set_coords_to_block()
        block = pd.DataFrame({
            'x_block': self.x_coords,
            'y_block': self.y_coords,
            'z_block': self.z_coords
        })
        if export:
            dataframe_to_gslib(block, 'blockmodel.dat')
        return block
