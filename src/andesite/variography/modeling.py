import pandas as pd
import plotly.graph_objects as go
import numpy as np

from variography.experimental import VariogramDatafile

def rotation_matrix_azm_dip(azimuth, dip=0, rake=0):
    azimuth = np.deg2rad(azimuth)
    dip = -np.deg2rad(dip)
    rake = -np.deg2rad(rake)
    azimuth_matrix = np.array([[np.cos(azimuth), np.sin(azimuth), 0],
                               [-np.sin(azimuth), np.cos(azimuth), 0],
                               [0, 0, 1]])
    dip_matrix = np.array([[1, 0, 0],
                           [0, np.cos(dip), np.sin(dip)],
                           [0, -np.sin(dip), np.cos(dip)]])
    rake_matrix = np.array([[np.cos(rake), 0, -np.sin(rake)],
                            [0, 1, 0],
                            [np.sin(rake), 0, np.cos(rake)]])
    return np.dot(azimuth_matrix, np.dot(dip_matrix, rake_matrix))

def generate_lags(azimuth, dip, lag_distance, nlags, plot=False):
        lag_distance = np.float32(lag_distance)
        h = np.zeros((nlags+1, 3), dtype=np.float64)
        h[:, 1] = np.arange(nlags+1) * lag_distance
        if plot:
            h[0, :] = 1e-8
        rotmat = rotation_matrix_azm_dip(azimuth, dip)
        return np.matmul(h, rotmat.T)

class VariogramStructure(object):
    type_coding = {'Spherical': 1, 'Exponential': 2, 'Gaussian':3}

    def __init__(self, model_type, sill, angles, ranges):
        self.type = model_type
        self.type_code = self.type_coding[model_type]
        self.sill = sill
        self.ranges = [np.float32(rang) for rang in ranges]
        self.angles = tuple(angles)
        self.model = VariogramModelStandart(model_type, sill, ranges[0])

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            a = (self.type == other.type)
            b = (self.type_code == other.type_code)
            c = (self.sill == other.sill)
            d = (self.ranges == other.ranges)
            e = (self.angles == other.angles)
            return a and b and c and d and e

    def __call__(self, h, norm_values=False):
        if type(h) is not np.ndarray:
            raise Exception()
        if h.ndim > 2:
            raise Exception()

        a, b = -1, -1
        if h.ndim == 2:
            a, b = h.shape
            if b not in [2, 3]:
                raise Exception()
        if h.ndim == 1:
            lags = h
            return self.model(lags)

        if b == 2:
            new_h = np.zeros((len(h), 3))
            new_h[:, :2] = h
            h = new_h
            assert h.shape == (a, 3)

        rotmat = rotation_matrix_azm_dip(self.angles[0], self.angles[1], self.angles[2])
        lags = np.empty_like(h)
        rotated_h = np.matmul(h, rotmat)
        anis1 = (self.ranges[0]/self.ranges[1])
        anis2 = (self.ranges[0]/self.ranges[2])

        lags[:, 0] = rotated_h[:, 0]*anis1
        lags[:, 1] = rotated_h[:, 1]
        lags[:, 2] = rotated_h[:, 2]*anis2

        # we apply the norm 2 to h over the 1 axis
        lags_norm = np.linalg.norm(lags, 2, 1)

        assert (len(lags_norm) == h.shape[0])
        h_norm = np.linalg.norm(rotated_h, 2, 1)
        if norm_values:
            return h_norm, self.model.__call__(lags_norm)
        else:
            return self.model.__call__(lags_norm)

    def direction(self, azimuth, dip, lag_distance, nlags, norm_values=True, plot=False):
        return self.__call__(generate_lags(azimuth, dip, float(lag_distance), nlags, plot), norm_values)
    
class VariographyModel(object):
    def __init__(self, nugget=0, struct_dict=None):
        self.models = []
        self.nugget = float(nugget)

        # add nugget if present
        if nugget != 0:
            structure = NuggetModel(nugget)
            self.add_structure(structure)
        if struct_dict:
            for struct in struct_dict:
                # we obtain the class according to type
                self.add_structure(VariogramStructure(struct['type'], struct['sill'], struct['angles'], struct['ranges']))

    def __len__(self):
        if self.nugget == 0:
            return len(self.models)
        else:
            return len(self.models)-1

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__

    @property
    def structures(self):
        i = 0
        if self.nugget != 0:
            i = 1
        return self.models[i:]

    def add_structure(self, vario_structure):
        # vario_structure must be a VariogramStructure
        self.models.append(vario_structure)

    def delete_structure(self, index):
        del self.models[index]

    def __call__(self, h):
        if type(h) is not np.ndarray:
            raise Exception()

        # array of scalars h
        if h.ndim > 2:
            raise Exception()

        n = len(h)
        # array of lags of three dimensions

        a, b = -1, -1
        if h.ndim == 2:
            a, b = h.shape
            if b not in [2, 3]:
                raise Exception()

        num_models = len(self.models)
        if num_models == 0:
            raise AttributeError("Empty model")
        else:
            result = np.zeros(n)
            for k in range(num_models):
                result += self.models[k](h)
            return result

    def direction(self, azimuth, dip, lag_distance, nlags, norm_values=True, plot=False):
        num_models = len(self.models)
        if num_models == 0:
            raise AttributeError("Empty model")
        else:
            result = 0
            for k in range(num_models):
                r = self.models[k].direction(azimuth, dip, lag_distance, nlags, norm_values, plot)
                result += r[1]
            lags = self.models[0].direction(azimuth, dip, lag_distance, nlags, norm_values, plot)[0]
            return lags, result

class NuggetModel(object):
    def __init__(self, nugget):
        self.nugget = nugget

    def __call__(self, h):
        result = self.nugget*np.ones(len(h))
        result[h == 0] = 0
        return result

    def direction(self, azimuth, dip, lag_distance, nlags, norm_values=True, plot=False):
        h = np.linspace(0, nlags*lag_distance, nlags+1)
        if plot:
            h[0] = 1e-8
        if norm_values:
            return h, self.__call__(h)
        else:
            return self.__call__(h)

# Esta clase intermedia solo conecta el nombre de modelo con las clases
class VariogramModelStandart(object):
    def __init__(self, model_type, sill, a):
        self.type = model_type
        self.sill = sill
        self.range = a

        if model_type == 'Exponential':
            self.model = ExponentialModel(self.range)
        elif model_type == 'Gaussian':
            self.model = GaussianModel(self.range)
        elif model_type == 'Spherical':
            self.model = SphericalModel(self.range)

    def __call__(self, h):
        return self.sill*self.model(h)

# Modelos variograficos disponibles
class GaussianModel(object):
    def __init__(self, a):
        self.range = a
    def __call__(self, h):
        return 1 - np.exp(-np.power(3*h/self.range, 2))

class ExponentialModel(object):
    def __init__(self, a):
        self.range = a
    def __call__(self, h):
        return 1 - np.exp(-3.0 * h/self.range)

class SphericalModel(object):
    def __init__(self, a):
        self.range = a
    def __call__(self, h):
        result = 1.5 * h/self.range - 0.5 * np.power(h/self.range, 3)
        result[h >= self.range] = 1
        return result
    

class VariogramModeling:

    def __init__(self, experimental_variogram: VariogramDatafile, parameters: dict):
        """_summary_

        Parameters
        ----------
        experimental_variogram : pd.Dataframe
            values for experimental variogram in a pd.Dataframe format
        parameters : dict
            list of parameters like nugget effect and estructure parameters in this format
            {
                'nugget': 0.3,
                'structures': [
                    {
                        'type': 'Exponential',
                        'sill': 0.6,
                        'angles': (0, 0, 0),
                        'ranges': (100, 100, 100)
                    }
                ] 
            }
        """
        self.experimental_variogram = experimental_variogram
        self.parameters = parameters
        self.metadata = {
            'nugget': self.parameters.get('nugget'),
            'structures': self.parameters.get('structures')
        }

    def get_metadata(self):
        return self.metadata

    def modeling(self):
        nugget = self.parameters.get('nugget')
        vario_model = VariographyModel(nugget)
        self.total_sill = nugget
        for istr in range(len(self.parameters.get('structures'))):
            vario_model.add_structure(
                VariogramStructure(
                    model_type = self.parameters.get('structures')[istr]['type'], 
                    sill = self.parameters.get('structures')[istr]['sill'], 
                    angles = self.parameters.get('structures')[istr]['angles'], 
                    ranges = self.parameters.get('structures')[istr]['ranges']
            ))
            self.total_sill += self.parameters.get('structures')[istr]['sill']

        self.experimental_params = self.experimental_variogram.get_metadata()
        self.experimental_df = self.experimental_variogram.load()

        # Aqui se lee el archivo var experimental para saber los angulos
        self.max_lag = np.int32(self.experimental_df['steps_dir1'].max())
        self.experimental_dirs = self.experimental_params.get('directions')
        if self.experimental_params.get('vartype') == 'elipsoid':
            model_lags1, model_gamma1 = vario_model.direction(azimuth=self.experimental_dirs[0][0], dip=self.experimental_dirs[0][1], lag_distance=1, nlags=self.max_lag, norm_values=True, plot=True)
            model_lags2, model_gamma2 = vario_model.direction(azimuth=self.experimental_dirs[1][0], dip=self.experimental_dirs[1][1], lag_distance=1, nlags=self.max_lag, norm_values=True, plot=True)
            model_lags3, model_gamma3 = vario_model.direction(azimuth=self.experimental_dirs[2][0], dip=self.experimental_dirs[2][1], lag_distance=1, nlags=self.max_lag, norm_values=True, plot=True)
            self.model_df = pd.DataFrame({
                'model_lag1': model_lags1,
                'model_lag2': model_lags2,
                'model_lag3': model_lags3,
                'model_gamma1': model_gamma1,
                'model_gamma2': model_gamma2,
                'model_gamma3': model_gamma3
            })
            self.gamma_max = self.experimental_df[['gamma_dir1', 'gamma_dir2', 'gamma_dir3']].max().max()
        else:
            print('single')
            model_lags1, model_gamma1 = vario_model.direction(azimuth=self.experimental_dirs[0][0], dip=self.experimental_dirs[0][1], lag_distance=1, nlags=self.max_lag, norm_values=True, plot=True)
            self.model_df = pd.DataFrame({
                'model_lag1': model_lags1,
                'model_gamma1': model_gamma1
            })
            self.gamma_max = self.experimental_df['gamma_dir1'].max()

        return self.model_df

    def plot(self, export=False):
        colors = ['black', 'blue', 'green']
        fig = go.Figure()
        for i in range(0, int(self.model_df.shape[1]/2)):
            fig.add_trace(
                go.Scatter(
                    x = self.model_df[f'model_lag{i+1}'],
                    y = self.model_df[f'model_gamma{i+1}'],
                    name = f'Modelado {self.experimental_dirs[i][0]}/{self.experimental_dirs[i][1]}',
                    mode = 'lines',
                    line = {
                        'color': colors[i],
                        'width': 3
                    },
                    hovertemplate = "<b>Model</b><br>" +
                    "Step: %{x}<br>" +
                    "Variogram: %{y:.3f}<br>" +
                    "<extra></extra>",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x = self.experimental_df[f'steps_dir{i+1}'],
                    y = self.experimental_df[f'gamma_dir{i+1}'],
                    name = f'Variogram {self.experimental_dirs[i][0]}/{self.experimental_dirs[i][1]}',
                    mode = 'markers+lines',
                    line = {
                        'color': colors[i],
                        'width': 1,
                        'dash': 'dash'
                    },
                    marker = {
                        'color': colors[i],
                        'size': 8,
                        'symbol': 'star' 
                    },
                    opacity = 0.3,
                    hovertemplate = "<b>Experimental</b><br>" + 
                    "Step: %{x}<br>" +
                    "Variogram: %{y:.3f}<br>" +
                    "<extra></extra>",
                )
            )
        fig.update_layout(
            width = 800,
            margin = dict(l=20, b=15, t=40, r=20),
            title = {
                'text': f'Variogram Modeling direction {self.experimental_dirs[0][0]}/{self.experimental_dirs[0][1]}'
            },
            yaxis = {
                'title': 'Variogram',
                'range': [0, self.gamma_max]
            },
            xaxis = {
                'title': 'Step (m)',
                'range': [0, self.max_lag]
            },
            legend=dict(
                y=0.01,
                x=0.76
            )
        )
        if export:
            fig.write_html(f'modeling-{self.experimental_dirs[0, 0]}-{self.experimental_dirs[0, 1]}.html')
        return fig
    
