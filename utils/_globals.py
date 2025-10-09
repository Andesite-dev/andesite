import polars as pl

POSSIBLE_X_COLUMNS = ['X', 'east', 'este', 'midx', 'xm', 'centroid_x', 'xcenter']
POSSIBLE_Y_COLUMNS = ['y', 'north', 'norte', 'midy', 'ym', 'centroid_y', 'ycenter']
POSSIBLE_Z_COLUMNS = ['z', 'elev', 'cota', 'midz', 'zm', 'centroid_z', 'zcenter', 'RL']
POSSIBLE_HOLEID_COLUMNS = ['dhid', 'hole', 'codigo', 'bhid']
POSSIBLE_FROM_COLUMNS = ['from', 'desde', 'head']
POSSIBLE_TO_COLUMNS = ['to', 'hasta', 'tail', 'depth', 'at']
POSSIBLE_LENGTH_COLUMNS = ['length', 'largo', 'avance', 'depth']
POSSIBLE_AZIM_COLUMNS = ['az', 'rumbo', 'bear', 'brg']
POSSIBLE_DIP_COLUMNS = ['dip', 'incli', 'manteo']

DATETIME_DTYPES_POLARS = [pl.Date, pl.Datetime]
STRING_DTYPES_POLARS = [pl.Utf8, pl.String, pl.Categorical, pl.Enum]
NUMERIC_DTYPES_POLARS = [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64]
CATEGORICAL_ID_THRESHOLD = 40

