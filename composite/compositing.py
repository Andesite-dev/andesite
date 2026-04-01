from functools import reduce
import polars as pl
from typing import Optional, Union, Set, List, Any, Tuple, Literal
import numpy as np
from ..utils._globals import (
    NUMERIC_DTYPES_POLARS,
    STRING_DTYPES_POLARS,
    DATETIME_DTYPES_POLARS,
    CATEGORICAL_ID_THRESHOLD,
)
from ..utils._typing import (
    StringList,
    Number,
    Int,
    Float,
    Bool
)

def check_unique_sets(*args: Union[Set, List], verbose: bool = False) -> List[Any]:
    # Convert all inputs to sets
    sets = [set(arg) for arg in args]

    # Find common values by intersecting all sets
    common_values = set.intersection(*sets)

    # Convert sets to lists for the output
    if verbose:
        print(f"Number of common values: {len(common_values)}")
    return list(common_values)

class Assay(object):
    def __init__(
            self,
            dataframe: pl.DataFrame,
            dhid: str,
            from_col: str,
            to_col: str,
            target_variables: StringList,
            target_selector: Literal["all", "pick"] = "all",
            tag_name: str = "assay"
        ):
        self.dataframe = dataframe
        self.dhid = dhid
        self.from_col = from_col
        self.to_col = to_col
        self.target_variables = target_variables
        self.target_selector = target_selector
        self.tag_name = tag_name

        self.__target_selector()
        self.__autosort()
        self.__classify_column_types()
        self.__fix_casting()

    def __target_selector(self):
        if self.target_selector == "all":
            self.target_variables = list(set(self.dataframe.columns) - set([self.dhid, self.from_col, self.to_col]))

    def __fix_casting(self):
        self.dataframe = (
            self.dataframe
            .with_columns([
                pl.col(c).cast(pl.Int32).alias(c)
                for c in self.categorical_columns
                if self.dataframe[c].dtype in NUMERIC_DTYPES_POLARS
            ])
        )

    def __autosort(self):
        self.dataframe = (
            self.dataframe
            .select([self.dhid, self.from_col, self.to_col] + self.target_variables)
            .sort([self.dhid, self.from_col, self.to_col])
        )

    def __classify_column_types(self):
        self.categorical_columns: List[str] = []
        self.numerical_columns: List[str] = []
        self.id_columns: List[str] = []
        self.datetime_columns: List[str] = []
        for c in self.dataframe.columns:
            if isinstance(self.dataframe, pl.DataFrame):
                unique_samples = self.dataframe[c].n_unique()
                dtype = self.dataframe[c].dtype
                if dtype in DATETIME_DTYPES_POLARS:
                    self.datetime_columns.append(c)
                elif (unique_samples < CATEGORICAL_ID_THRESHOLD) and dtype not in [pl.Float32, pl.Float64]:
                    self.categorical_columns.append(c)
                else:
                    if dtype in NUMERIC_DTYPES_POLARS:
                        self.numerical_columns.append(c)
                    else:
                        self.id_columns.append(c)

    def _get_categorical_columns(self) -> List[str]:
        return self.categorical_columns

    def _get_numerical_columns(self) -> List[str]:
        return self.numerical_columns

    def _get_id_columns(self) -> List[str]:
        return self.id_columns

    def _get_datetime_columns(self) -> List[str]:
        return self.datetime_columns

    def target_categorical_columns(self) -> List[str]:
        return list(
            set(self.categorical_columns)
            - set([self.dhid, self.from_col, self.to_col])
        )

    def target_numerical_columns(self) -> List[str]:
        return list(
            set(self.numerical_columns)
            - set([self.dhid, self.from_col, self.to_col])
        )

    def filter_drillhole(self, by: str):
        return self.dataframe.filter(pl.col(self.dhid) == by)

    @staticmethod
    def __most_frequent_category(weights: np.ndarray, arr_vars: np.ndarray):
        # Step 1: Group by the first column
        first_column = arr_vars[:, 0]
        unique_groups, group_indices = np.unique(first_column, return_inverse=True)

        # Step 2: Sum proportions for each unique value in the first column
        grouped_sums = np.bincount(group_indices, weights=weights)

        # Step 3: Find the group with the maximum summed proportion
        max_group_idx = np.argmax(grouped_sums)

        # Step 4: Retrieve one representative row for the selected group
        # Since the grouping is based on the first column, any row matching the group will work
        result = arr_vars[first_column == unique_groups[max_group_idx]][0]
        return result

    #TODO: Change this method to accept any Assay() instances, not just the assay first object
    def categoric_multivar_regularization(self, hole_id: str, comp_length: Union[int, float]) -> pl.DataFrame:
        df = self.filter_drillhole(hole_id)
        self.assay_categorical_cols = self.target_categorical_columns()

        # Convert Polars columns to NumPy arrays for calculations
        drill_from = df[self.from_col].to_numpy()
        drill_to = df[self.to_col].to_numpy()

        # Calculate the number of composites
        n_comp = int(np.ceil(drill_to[-1] / comp_length))
        n_vars = len(self.assay_categorical_cols)

        # Create composite intervals
        comp_from = np.arange(0, n_comp * comp_length, comp_length, dtype=np.float32)
        comp_to = comp_from + comp_length

        if n_vars == 0:
            # print("Not categoric columns founded.")
            return pl.DataFrame({
                self.dhid: hole_id,
                "FROM": comp_from,
                "TO": comp_to,
            })

        comp_vars = np.full((n_comp, n_vars), np.nan, dtype="<U16")  # Result matrix for all variables
        drill_variables = np.stack([df[var].to_numpy() for var in self.assay_categorical_cols], axis=1)

        for icomp in range(n_comp):
            comp_start = comp_from[icomp]
            comp_end = comp_to[icomp]

            # Determine overlaps
            overlap = np.minimum(drill_to, comp_end) - np.maximum(drill_from, comp_start)
            mask = overlap < 0
            overlap[mask] = 0
            weights = overlap[~mask]
            values = drill_variables[~mask]

            result = (
                np.full((1, len(self.assay_categorical_cols)), np.nan)
                if np.size(weights) == 0
                else self.__most_frequent_category(weights, values)
            )
            comp_vars[icomp] = result

        # Create the final Polars DataFrame
        data = {
            self.dhid: hole_id,
            "FROM": comp_from,
            "TO": comp_to,
        }
        for i, var in enumerate(self.assay_categorical_cols):
            data[f"{var}"] = comp_vars[:, i]

        return pl.DataFrame(data)

    #TODO: Change this method to accept any Assay() instances, not just the assay first object
    def numeric_multivar_regularization(self, hole_id: str, comp_length: Union[int, float]) -> pl.DataFrame:
        df = self.filter_drillhole(hole_id)
        self.assay_numerical_cols = self.target_numerical_columns()

        # Convert Polars columns to NumPy arrays for calculations
        drill_from = df[self.from_col].to_numpy()
        drill_to = df[self.to_col].to_numpy()

        # Calculate the number of composites
        n_comp = int(np.ceil(drill_to[-1] / comp_length))
        n_vars = len(self.assay_numerical_cols)

        # Create composite intervals
        comp_from = np.arange(0, n_comp * comp_length, comp_length, dtype=np.float32)
        comp_to = comp_from + comp_length

        if n_vars == 0:
            # print("Not numeric columns founded.")
            return pl.DataFrame({
                self.dhid: hole_id,
                "FROM": comp_from,
                "TO": comp_to,
            })

        comp_vars = np.full((n_comp, n_vars), np.nan, dtype=np.float32)
        drill_variables = np.stack([df[var].to_numpy() for var in self.assay_numerical_cols], axis=1)  # Stack all variable columns

        for icomp in range(n_comp):
            comp_start = comp_from[icomp]
            comp_end = comp_to[icomp]

            # Determine overlaps
            overlap = np.minimum(drill_to, comp_end) - np.maximum(drill_from, comp_start)
            mask = overlap <= 0
            overlap[mask] = 0

            weights = overlap[~mask]
            values = drill_variables[~mask]

            # Calculate lengths and weighted sums for all variables
            weighted_sums = weights[:, None] * values  # Element-wise multiplication

            for ivar in range(n_vars):
                weighted_sum = weighted_sums[:, ivar]
                # if np.all((weighted_sum == 0) | np.isnan(weighted_sum)):
                if np.all(np.isnan(weighted_sum)):
                    comp_vars[icomp, ivar] = np.nan
                else:
                    comp_vars[icomp, ivar] = (
                        np.nansum(weighted_sum) / weights.sum() if weights.sum() > 0 else np.nan
                    )

        # Create the final Polars DataFrame
        data = {
            self.dhid: hole_id,
            "FROM": comp_from,
            "TO": comp_to,
        }
        for i, var in enumerate(self.assay_numerical_cols):
            data[f"{var}"] = comp_vars[:, i]

        return pl.DataFrame(data)


class Survey(object):
    def __init__(
            self,
            dataframe: pl.DataFrame,
            dhid: str,
            depth_col: str,
            azimuth_col: str,
            dip_col: str,
            tag_name: str = "survey"
        ):
        self.dataframe = dataframe
        self.dhid = dhid
        self.depth_col = depth_col
        self.azimuth_col = azimuth_col
        self.dip_col = dip_col
        self.tag_name = tag_name
        self.__autosort()

    def __autosort(self):
        self.dataframe = (
            self.dataframe
            .select([self.dhid, self.depth_col, self.azimuth_col, self.dip_col])
            .sort([self.dhid, self.depth_col])
        )

    def filter_drillhole(self, by: str):
        return self.dataframe.filter(pl.col(self.dhid) == by)

class Collar(object):
    def __init__(
            self,
            dataframe: pl.DataFrame,
            dhid: str,
            east_col: str,
            north_col: str,
            elev_col: str,
            length_col: str,
            tag_name: str = "collar"
        ):
        self.dataframe = dataframe
        self.dhid = dhid
        self.east_col = east_col
        self.north_col = north_col
        self.elev_col = elev_col
        self.length_col = length_col
        self.tag_name = tag_name

    def filter_drillhole(self, by: str):
        return self.dataframe.filter(pl.col(self.dhid) == by)


class DrillholesCampaign(object):
    def __init__(self, assay: Assay, survey: Survey, collar: Collar):
        self.assay = assay
        self.survey = survey
        self.collar = collar
        self.composite_dbs: List[Assay] = []
        self.__check_valid_holeids()
        # First add assay as default
        self.add_database(self.assay)

    @staticmethod
    def __calc_new_coords(x, y, z, az, dip, length):
        new_z = np.round(z + length * np.sin(np.radians(dip)), 2)
        new_y = np.round(y + length * np.cos(np.radians(dip)) * np.cos(np.radians(az)), 2)
        new_x = np.round(x + length * np.cos(np.radians(dip)) * np.sin(np.radians(az)), 2)
        return new_x, new_y, new_z

    def __check_valid_holeids(self):
        self.valid_holeids = check_unique_sets(
            self.survey.dataframe[self.survey.dhid].to_list(),
            self.collar.dataframe[self.collar.dhid].to_list()
        )
        return self.valid_holeids

    def add_database(self, databse: Assay):
        self.composite_dbs.append(databse)

    def cambio_soporte(self, apply_on: Assay, comp_length: Union[int, float]) -> pl.DataFrame:
        dhs_composited: List[pl.DataFrame] = []
        for dh in self.valid_holeids:
            _assay = apply_on.filter_drillhole(dh)

            # Skip if assay is empty
            if _assay.shape[0] == 0:
                continue

            _assay_segments_numeric = apply_on.numeric_multivar_regularization(
                hole_id = dh,
                comp_length = comp_length
            )

            _assay_segments_categoric = apply_on.categoric_multivar_regularization(
                hole_id = dh,
                comp_length = comp_length
            )

            _assay_segments = (
                _assay_segments_numeric
                .join(
                    _assay_segments_categoric,
                    left_on = [self.assay.dhid, "FROM", "TO"],
                    right_on = [self.assay.dhid, "FROM", "TO"],
                    how = "left"
                )
            )
            dhs_composited.append(_assay_segments)

        return pl.concat(dhs_composited)

    def composite(self, comp_length: Union[int, float], return_dtype: Literal["pandas", "polars"] = "polars") -> pl.DataFrame:
        regularized_dfs: List[pl.DataFrame] = []
        for assay_instance in self.composite_dbs:
            _assay_cambio_soporte = self.cambio_soporte(
                assay_instance,
                comp_length
            )
            regularized_dfs.append(_assay_cambio_soporte)

        joined_df = reduce(
            lambda left, right: left.join(
                other = right,
                on = [self.assay.dhid, "FROM", "TO"],
                how = "full",
                coalesce = True
            ),
            regularized_dfs
        )

        joined_df_filtered = (
            joined_df
            .with_columns([
                pl.col(c).fill_nan(None).alias(c)
                for c in joined_df.columns
                if joined_df[c].dtype in NUMERIC_DTYPES_POLARS
            ])
            .with_columns([
                pl.when(pl.col(c) == "nan")
                .then(None)
                .otherwise(pl.col(c))
                .alias(c)
                for c in joined_df.columns
                if joined_df[c].dtype in STRING_DTYPES_POLARS
            ])
        )
        self.regularized_columns = joined_df_filtered.columns[3:]

        print("Asignando coordenadas...")
        composites = self.__desurverying(joined_df_filtered)

        if return_dtype == "pandas":
            return composites.to_pandas() # type: ignore
        else:
            return composites

    def __desurverying(self, drillholes: pl.DataFrame) -> pl.DataFrame:
        dhs_composited: List[pl.DataFrame] = []
        for dh in self.valid_holeids:
            _tmp_assay = drillholes.filter(pl.col(self.assay.dhid) == dh)
            _collar = self.collar.filter_drillhole(dh)
            _survey = self.survey.filter_drillhole(dh)

            _assay_survey = (
                _tmp_assay
                .with_columns([pl.col("FROM").cast(pl.Float32).alias("FROM")])
                .sort("FROM")
                .join_asof(
                    _survey
                    .with_columns([pl.col(self.survey.depth_col).cast(pl.Float32).alias(self.survey.depth_col)])
                    .sort(self.survey.depth_col),
                    left_on = "FROM",
                    right_on = self.survey.depth_col,
                    strategy = "nearest"
                )
                .with_columns([(pl.col("TO") - pl.col("FROM")).alias("sample_length")])
            )
            az = _assay_survey[self.survey.azimuth_col].to_numpy()
            dip = _assay_survey[self.survey.dip_col].to_numpy()
            lengths = _assay_survey["sample_length"].to_numpy()

            coordinates = np.empty((_assay_survey.shape[0], 3), dtype = np.float64)

            for i in range(_assay_survey.shape[0]):
                if i == 0:
                    coordinates[i] = _collar[[
                        self.collar.east_col,
                        self.collar.north_col,
                        self.collar.elev_col
                    ]].to_numpy().flatten()
                else:
                    coords = coordinates[int(i - 1)]
                    coordinates[i] = self.__calc_new_coords(
                        x = coords[0],
                        y = coords[1],
                        z = coords[2],
                        az = az[i],
                        dip = dip[i],
                        length = lengths[i]
                    )

            _comp_df = (
                _assay_survey
                .with_columns(
                    x = pl.Series(coordinates[:, 0]),
                    y = pl.Series(coordinates[:, 1]),
                    z = pl.Series(coordinates[:, 2])
                )
                .select([self.assay.dhid, "x", "y", "z", "FROM", "TO"] + self.regularized_columns)
            )
            dhs_composited.append(_comp_df)

        return pl.concat(dhs_composited).rename({"FROM": "from", "TO": "to"})


