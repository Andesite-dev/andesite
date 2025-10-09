import inspect
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from scipy.sparse import spmatrix
from os import PathLike as OSPathLike
from typing import get_type_hints, Dict, Union, List, Optional, IO

# Custom Exception for Typing Mismatch
class WrongTypingException(Exception):
    def __init__(self, param_name, expected_type, actual_type):
        super().__init__(
            f"Parameter '{param_name}' is expected to be of type '{expected_type}', but got '{actual_type}'."
        )

# Define custom types
String = Union[str, np.str_, np.bytes_, pd.StringDtype]
Int = Union[int, np.int8, np.int16, np.int32, np.int64]
Float = Union[float, np.float16, np.float32, np.float64]
Number = Union[Int, Float]
StringList = List[String]
PathLike = Union[
    String,
    OSPathLike,
    Path,
    IO[str],
    IO[bytes],
    bytes,
    List[str],
    List[Path],
    List[IO[str]],
    List[IO[bytes]],
    List[bytes],
]
PathLikeSources = Union[PathLike, List[PathLike]]

StringOrList = Union[String, StringList]
Bool = Union[bool, pd.BooleanDtype, pl.Boolean]
DataframeLike = Union[pl.DataFrame, pd.DataFrame, pl.LazyFrame]
MatrixLike = Union[np.ndarray, List]

# Optional typing
OptionalInt = Optional[Int]
OptionalBool = Optional[Bool]
OptionalFloat = Optional[Float]
OptionalNumber = Optional[Number]
OptionalString = Optional[String]
OptionalPathLike = Optional[PathLike]
OptionalStringList = Optional[StringList]
OptionalDataFrame = Optional[DataframeLike]

StringDict = Dict[String, String]
IntDict = Dict[String, Int]
FloatDict = Dict[String, Float]
NumberDict = Dict[String, Number]

dict_typing_names = {
    "String": String,
    "Number": Number,
    "Int": Int,
    "Float": Float,
    "StringList": StringList,
    "PathLike": PathLike,
    "PathLikeSources": PathLikeSources,
    "StringOrList": StringOrList,
    "Bool": Bool,
    "DataframeLike": DataframeLike,
    "MatrixLike": MatrixLike,
    "OptionalInt": OptionalInt,
    "OptionalBool": OptionalBool,
    "OptionalFloat": OptionalFloat,
    "OptionalNumber": OptionalNumber,
    "OptionalString": OptionalString,
    "OptionalPathLike": OptionalPathLike,
    "OptionalStringList": OptionalStringList,
    "OptionalDataFrame": OptionalDataFrame,
    "StringDict": StringDict,
    "IntDict": IntDict,
    "FloatDict": FloatDict,
    "NumberDict": NumberDict,
}

def format_custom_typing(my_type):
    global dict_typing_names
    for k, v in dict_typing_names.items():
        if my_type == v:
            return k
    return my_type

def check_typing(func):
    def wrapper(*args, **kwargs):
        # Get the method's signature and type hints
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        # Bind arguments to the signature
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Check each parameter's type
        for param_name, value in bound_args.arguments.items():
            if param_name in type_hints:
                expected_type = type_hints[param_name]
                if not isinstance(value, expected_type) and not isinstance(value, (type(None), expected_type)):
                    expected_type = format_custom_typing(expected_type)
                    raise WrongTypingException(param_name, expected_type, type(value))
        return func(*args, **kwargs)

    return wrapper