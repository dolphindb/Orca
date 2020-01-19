# coding=utf-8
import datetime
import itertools
import json
import random
import string
from typing import Iterable, List, Optional, Tuple, Union

import dolphindb as ddb
import dolphindb.settings as types
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from .common import _warn_not_dolphindb_identifier

dolphindb_numeric_types = [
    ddb.settings.DT_BOOL,
    ddb.settings.DT_BYTE,
    ddb.settings.DT_SHORT,
    ddb.settings.DT_INT,
    ddb.settings.DT_LONG,
    ddb.settings.DT_FLOAT,
    ddb.settings.DT_DOUBLE
]

dolphindb_temporal_types = [
    ddb.settings.DT_DATE,
    ddb.settings.DT_MONTH,
    ddb.settings.DT_TIME,
    ddb.settings.DT_MINUTE,
    ddb.settings.DT_SECOND,
    ddb.settings.DT_DATETIME,
    ddb.settings.DT_TIMESTAMP,
    ddb.settings.DT_NANOTIME,
    ddb.settings.DT_NANOTIMESTAMP
]

dolphindb_literal_types = [
    ddb.settings.DT_STRING,
    ddb.settings.DT_SYMBOL
]

_TYPE_TO_FREQ = {
    ddb.settings.DT_DATE: "D",
    ddb.settings.DT_MONTH: "MS",
    ddb.settings.DT_TIME: "S",
    ddb.settings.DT_MINUTE: "MIN",
    ddb.settings.DT_SECOND: "S",
    ddb.settings.DT_DATETIME: "S",
    ddb.settings.DT_TIMESTAMP: "L",
    ddb.settings.DT_NANOTIME: "N",
    ddb.settings.DT_NANOTIMESTAMP: "N"
}


def ORCA_INDEX_NAME_FORMAT(i):
    return f"ORCA_INDEX_LEVEL_{i}_"


def ORCA_COLUMN_NAME_FORMAT(i):
    return f"ORCA_COLUMN_LEVEL_{i}_"


def _to_freq(dtype):
    freq_str = _TYPE_TO_FREQ.get(dtype, None)
    if freq_str is None:
        raise ValueError(f"Data type {dtype} is not a temporal type")
    else:
        return to_offset(freq_str)


def is_bool_indexer(key):

    # if (isinstance(key, (ABCSeries, np.ndarray, ABCIndex)) or
    #         (is_array_like(key)) and is_extension_array_dtype(key.dtype))):
    #     if key.dtype == np.object_:
    return True


def _new_orca_identifier():
    LEN = 16
    return ''.join(random.choices(string.ascii_letters + string.digits, k=LEN))


def exec_prev(sdf):
    keys = sdf.script_list[sdf.curFlag:]
    if len(keys) == 0:
        return
    sdf.curFlag = len(sdf.script_list)
    all_script = ";".join(keys)
    sdf.session.run(all_script)


# def create_series_from_dataframe(session, df, data_column, index_column):
#     from .series import Series
#     from .datastruc import DataFrameMetaData
#     metadata = DataFrameMetaData(df, column_names=data_column)
#     index_metadata = DataFrameMetaData(df, column_names=index_column)
#     return Series(metadata, index=index_metadata, session=session)


# def create_dataframe_from_dataframe(session, df, data_column, index_column):
#     from .series import Series
#     from .datastruc import DataFrameMetaData
#     data = df._inner
#     metadata = DataFrameMetaData(data, column_names=data_column)
#     index_metadata = DataFrameMetaData(data, column_names=index_column)
#     return DataFrame(metadata, index=index_metadata, session=session)


# def _get_index_select_list(obj, index):
#     from .indexes import RangeIndex
#     if not obj._in_memory and isinstance(index, RangeIndex):
#         warnings.warn("A DolphinDB table which is not in memory does not support RangeIndex. The return value will use a default index.", Warning)
#         index_select_list = []
#     else:
#         index_select_list = index._get_select_list()
#     return index_select_list


# def _as_in_memory_dataframe(session, df):
#     from .frame import DataFrame
#     from .internal import _ConstantSP, _InternalFrame
#     if df._in_memory:
#         return df
#     else:
#         odf, index = df._internal, df.index
#         select_list = itertools.chain(odf.data_columns, index.data_columns)
#         script = sql_select(select_list, self._var_name)
#         script = f"loadTableBySQL(<{script}>)"
#         var = _ConstantSP.run_script(session, script)
#         new_odf = _InternalFrame(session, var, data_columns=odf.data_columns)
#         idx_odf = _InternalFrame(session, var, data_columns=index._data_columns)
#         return DataFrame(new_odf, idx_odf, session=session)


# def _as_in_memory_series(session, df):
#     from .series import Series
#     from .internal import _ConstantSP, _InternalFrame
#     if df._in_memory:
#         return df
#     else:
#         odf, index = df._internal, df.index
#         select_list = itertools.chain(odf.data_columns, index.data_columns)
#         script = sql_select(select_list, self._var_name)
#         script = f"loadTableBySQL(<{script}>)"
#         var = _ConstantSP.run_script(session, script)
#         new_odf = _InternalFrame(session, var, data_columns=odf.data_columns)
#         idx_odf = _InternalFrame(session, var, data_columns=index._data_columns)    # TODO: does not work
#         return Series(new_odf, idx_odf, session=session)

def _get_where_list(key):
    """
    Generate a WHERE clause list to be used in the SQL query.
    
    Parameters
    ----------
    key : BooleanExpression or tuple
        An abstrct expression of WHERE condition.
    
    Returns
    -------
    list[str]
        The deduced WHERE-list starting with WHERE keyword.
    
    Raises
    ------
    TypeError
        If the key is neither a BooleanExpression nor a tuple.
    """
    from .operator import BooleanExpression
    from .merge import BooleanMergeExpression
    if key is None:    # trivial case
        return None
    elif isinstance(key, BooleanExpression):
        return key._to_where_list()
    elif isinstance(key, tuple):
        assert (all(isinstance(k, BooleanExpression) for k in key)
                or all(isinstance(k, BooleanMergeExpression) for k in key))
        return sum((k._to_where_list() for k in key), [])
    elif isinstance(key, Iterable):
        key = _try_convert_iterable_to_list(key)
        assert all(isinstance(k, str) for k in key)
        return key
    elif isinstance(key, BooleanMergeExpression):
        return key._to_where_list()
    else:
        raise TypeError('Unable to deduce where clause')

def _merge_where_expr(where_expr1, where_expr2):
    from .operator import BooleanExpression
    from .merge import BooleanMergeExpression
    def is_string_like(where_expr):
        return (isinstance(where_expr, str)
                or (not isinstance(where_expr, (tuple, BooleanExpression))
                    and isinstance(where_expr, Iterable)))

    if is_string_like(where_expr1) or is_string_like(where_expr2):
        where_expr1 = _get_where_list(where_expr1) or []
        where_expr2 = _get_where_list(where_expr2) or []
        return where_expr1 + where_expr2

    def to_tuple(where_expr):
        assert (where_expr is None
                or isinstance(where_expr, (tuple, BooleanExpression, BooleanMergeExpression)))
        if where_expr is None:
            return ()
        elif isinstance(where_expr, tuple):
            return where_expr
        else:
            return (where_expr,)

    where_expr1 = to_tuple(where_expr1)
    where_expr2 = to_tuple(where_expr2)
    where_expr = where_expr1 + where_expr2
    if len(where_expr) == 0:
        return None
    elif len(where_expr) == 1:
        return where_expr[0]
    else:
        return where_expr


def _infer_axis(obj, axis):
    if obj is None:
        if axis is None:
            axis = None
        elif axis == "columns" or axis == 1:
            axis = 1
        elif axis == "index" or axis == 0:
            axis = 0
        else:
            raise ValueError(f"No axis named {axis}")
    elif obj._is_dataframe_like:
        # DEFAULT_AXIS = 1
        if axis == "columns" or axis == 1:
            axis = 1
        elif axis == "index" or axis == 0:
            axis = 0
        else:
            # axis = DEFAULT_AXIS
            raise ValueError(f"No axis named {axis}")
    elif obj._is_series_like:
        # DEFAULT_AXIS = 0
        if axis == "index" or axis == 0 or axis is None:
            axis = 0
        else:
            # axis = DEFAULT_AXIS
            raise ValueError(f"No axis named {axis}")
    else:
        raise TypeError(f"Unsupported data type: {obj.__class__.__name__}")
    return axis


def _scale_nanos(nanos, ddb_dtype):
    if ddb_dtype != ddb.settings.DT_NANOTIMESTAMP:
        nanos //= _to_freq(ddb_dtype).nanos
    return nanos


def sql_select(select_list: List[str],
               from_clause: str,
               where_expr=None,
               groupby_list: Optional[List[str]] = None,
               is_groupby: Optional[bool] = True,
               having_list: Optional[List[str]] = None,
               orderby_list: Optional[List[str]] = None,
               asc: Optional[Union[bool, List[bool]]] = True,
               pivot_list: Optional[List[str]] = None,
               is_exec: Optional[bool] = False,
               limit: Optional[str] = None,
               hint: Optional[int] = None):
    select_clause = ",".join(select_list)
    script = "exec" if is_exec else "select"
    if hint is not None:
        assert isinstance(hint, int)
        script += f" [{hint}]"
    if limit:
        script += f" top {limit}"
    script += f" {select_clause} from {from_clause}"
    if where_expr is not None:
        where_clause = ",".join(_get_where_list(where_expr))
        script += f" where {where_clause}"
    if groupby_list is not None:
        groupby_clause = ",".join(groupby_list)
        groupby_keyword = "group by" if is_groupby else "context by"
        script += f" {groupby_keyword} {groupby_clause}"
    if having_list is not None:
        assert groupby_list is not None, "HAVING clause must be used with GROUP BY or CONTEXT BY"
        having_clause = ",".join(having_list)
        script += f" having {having_clause}"
    if orderby_list is not None:
        if isinstance(asc, bool):
            asc = "asc" if asc else "desc"
            orderby_clause = ",".join(f"{col} {asc}" for col in orderby_list)
        elif isinstance(asc, list):
            asc = ("asc" if a else "desc" for a in asc)
            orderby_clause = ",".join(f"{col} {a}" for col, a in zip(orderby_clause, asc))
        else:
            assert False
        script += f" order by {orderby_clause}"
    if pivot_list is not None:
        assert groupby_list is None
        assert orderby_list is None
        assert len(pivot_list) == 2
        script += f" pivot by {pivot_list[0]}, {pivot_list[1]}"
    return f"({script})"


def sql_update(table_name: str,
               column_names: List[str],
               new_values: List[str],
               from_table_joiner: Optional[str] = None,
               where_expr=None,
               contextby_list: Optional[List[str]] = None):
    assert len(column_names) == len(new_values)
    update_clause = ",".join(
        f"{column_name} = {new_value}"
        for column_name, new_value in zip(column_names, new_values)
    )
    script = f"update {table_name} set {update_clause}"
    if from_table_joiner is not None:
        script += f" from {from_table_joiner}"
    if where_expr is not None:
        where_clause = ",".join(_get_where_list(where_expr))
        script += f" where {where_clause}"
    if contextby_list is not None:
        contextby_clause = ",".join(contextby_list)
        script += f" context by {contextby_clause}"
    return script


def _to_index_map(index, all_columns=None):
    """
    Generate an index map from an orca or pandas Index object.
    
    Parameters
    ----------
    index : orca.Index or pandas.Index
        The index that index_map is generated from
    all_columns : List, optional
        All index in the table, by default None
        If specified, when a name in the index is in it, convert it to
        a default name.
    
    Returns
    -------
    List[IndexMap]
        The generated index map
    """
    from .indexes import Index, MultiIndex
    assert isinstance(index, (Index, pd.Index, list, np.ndarray))
    if isinstance(index, (MultiIndex, pd.MultiIndex)):
        if index.names is None:
            return [(ORCA_INDEX_NAME_FORMAT(i), None)    # TODO: the index names might conflict with the existing ones
                    for i in range(len(index.levels))]
        else:
            index_columns = index.names
    elif isinstance(index, (Index, pd.Index)):
        index_columns = [index.name]
    else:
        index_columns = index
    if all_columns is None:
        return [(ORCA_INDEX_NAME_FORMAT(i) if name is None
                    or not is_dolphindb_identifier(name) else name,
                 name if name is None or isinstance(name, tuple) else (name,))
                for i, name in enumerate(index_columns)]
    else:
        return [(name if name is not None and name not in all_columns
                      else ORCA_INDEX_NAME_FORMAT(i),
                 name if name is None or isinstance(name, tuple) else (name,))
                for i, name in enumerate(index_columns)]


def _to_column_index(columns):
    from .indexes import Index, MultiIndex
    assert isinstance(columns, (Index, pd.Index, list, np.ndarray))
    if isinstance(columns, pd.MultiIndex):
        return columns.tolist()
    else:
        return [col if isinstance(col, tuple) else (col,) for col in columns]


def _unsupport_columns_axis(self, axis):    # TODO: common.py?
    axis = _infer_axis(self, axis)
    if axis == 1:
        raise NotImplementedError("Orca does not support axis == 1")
    return axis


def get_orca_obj_from_script(
        session, script, index_map, data_columns=None, column_index=None,
        column_index_names=None, name=None, squeeze=False, squeeze_axis=None,
        as_index=False):
    """
    Create an orca object by executing a DolphinDB script. Type of the
    returned object is automatically deduced from the form of execution
    result. But users can still manually determine the return type by
    specifying the squeeze and as_index parameters.
    
    Parameters
    ----------
    session : dolphindb.Session
        The DolphinDB session to be used.
    script : str
        The script to be executed.
    index: list[str] or list[List[IndexMap]]
        The index map or index columns of the returned frame.
    data_columns: Optional[list[str]]
        The data columns of the returned frame, by default None
        If not specified, it is automatically deduced from the script
        result and index_columns.
    column_index : List[Tuple[str, ...]], optional
        The column_index of the the returned frame, by default None
    name : str, optional
        If specified, it is used as the name of the result,
        by default None
    in_memory : bool, optional
        Indicates whether the result is in memory, by default True
    squeeze: bool, optional
        Whether to squeeze 1 dimensional axis objects into scalars,
        by default False
    squeeze_axis : {0 or 'index', 1 or 'columns', None}, default None
        A specific axis to squeeze. By default, all length-1 axes are squeezed.
    as_index : bool, optional
        Whether to return an Index instead of a Series, by default False
    
    Returns
    -------
    DataFrame, Index or Series
        The orca object representing the execution result of the input script.
    
    Raises
    ------
    ValueError
        If unable to create an orca object given the unrecognized form.
    """

    from .frame import DataFrame
    from .indexes import Index
    from .series import Series
    from .internal import _ConstantSP, _InternalFrame

    var = _ConstantSP.run_script(session, script)
    var_name, form, size = var.var_name, var.form, len(var)
    axis = _infer_axis(None, squeeze_axis)
    if squeeze and axis in (0, None) and size == 1:
        return var.squeeze([], data_columns, as_index=as_index, squeeze_axis=axis)
    if index_map and isinstance(index_map[0], str):
        index_map = _to_index_map(index_map)

    if form == ddb.settings.DF_SCALAR:    # TODO: eliminate two "run" calls here to one
        return session.run(var_name)
    elif form == ddb.settings.DF_VECTOR:
        if as_index:
            return Index(var, name=name, session=session)
        else:
            odf = _InternalFrame(session, var, index_map=index_map)
            return Series(odf, name=name, session=session)
    elif form == ddb.settings.DF_TABLE:
        odf = _InternalFrame(session, var, index_map=index_map, column_index=column_index, column_index_names=column_index_names)
        if squeeze and axis in (1, None) and len(odf.data_columns) == 1:
            if as_index:
                return var.squeeze([], as_index=True, squeeze_axis=axis)
            else:
                return Series(odf, name=name, session=session)
        else:
            return DataFrame(odf, session=session)
    else:
        raise ValueError("Unable to create an orca object from DolphinDB form " + str(form))


def _try_convert_iterable_to_list(iterable):
    from .series import Series
    from .operator import BaseExpression
    if iterable is None:
        return []
    elif isinstance(iterable, list):
        return iterable
    elif isinstance(iterable, (str, Series, BaseExpression)):
        return [iterable]
    else:
        return list(iterable)


def check_key_existence(keys, data_columns):
    from .frame import DataFrame
    if isinstance(keys, DataFrame):
        raise TypeError('DataFrame cannot be a key')
    elif isinstance(keys, (str, Iterable)):
        keys = _try_convert_iterable_to_list(keys)
    else:
        raise KeyError(keys)

    non_exist_keys = [k for k in keys if k not in data_columns]
    if len(non_exist_keys) > 0:
        raise KeyError(non_exist_keys)
    dropped = [col for col in data_columns if col not in keys]
    return keys, dropped


_TYPE_TO_TYPE_STR = {
    types.DT_BOOL: "bool",
    types.DT_BYTE: "char",
    types.DT_SHORT: "short",
    types.DT_INT: "int",
    types.DT_LONG: "long",
    types.DT_FLOAT: "float",
    types.DT_DOUBLE: "double",
    types.DT_DATE: "date",
    types.DT_DATETIME: "datetime",
    types.DT_DATETIME64: "timestamp",
    types.DT_MINUTE: "minute",
    types.DT_MONTH: "month",
    types.DT_NANOTIME: "nanotime",
    types.DT_NANOTIMESTAMP: "nanotimestamp",
    types.DT_SECOND: "second",
    types.DT_TIME: "time",
    types.DT_TIMESTAMP: "timestamp",
    types.DT_STRING: "string",
    types.DT_SYMBOL: "symbol",
}

def is_datetimelike(type):
    return type in dolphindb_temporal_types

def to_dolphindb_type_name(ddb_dtype):
    type_name = _TYPE_TO_TYPE_STR.get(ddb_dtype, None)
    if type_name is None:
        raise ValueError(f"Unsupported type: {ddb_dtype}")
    else:
        return type_name


_TYPE_TO_NUMPY_TYPE = {
    types.DT_BOOL: np.dtype(np.bool),
    types.DT_BYTE: np.dtype(np.int8),
    types.DT_SHORT: np.dtype(np.int16),
    types.DT_INT: np.dtype(np.int32),
    types.DT_LONG: np.dtype(np.int64),
    types.DT_FLOAT: np.dtype(np.float32),
    types.DT_DOUBLE: np.dtype(np.float64),
    types.DT_DATE: np.dtype("=M8[D]"),
    types.DT_DATETIME: np.dtype("=M8[s]"),
    types.DT_DATETIME64: np.dtype("=M8[ms]"),
    types.DT_MINUTE: np.dtype("=M8[m]"),
    types.DT_MONTH: np.dtype("=M8[M]"),
    types.DT_NANOTIME: np.dtype("=M8[ns]"),
    types.DT_NANOTIMESTAMP: np.dtype("=M8[ns]"),
    types.DT_SECOND: np.dtype("=M8[s]"),
    types.DT_TIME: np.dtype("=M8[s]"),
    types.DT_TIMESTAMP: np.dtype("=M8[ms]"),
    types.DT_STRING: np.dtype(np.object),
    types.DT_SYMBOL: np.dtype(np.object),
}


def _get_python_object_dtype(value):
    t = np.dtype(type(value))
    if t != np.dtype('<U'):
        return t
    else:
        return np.dtype('O')


def _to_numpy_dtype(ddb_dtype):
    np_dtype = _TYPE_TO_NUMPY_TYPE.get(ddb_dtype, None)
    if np_dtype is None:
        raise ValueError(f"Unsupported type: {ddb_dtype}")
    else:
        return np_dtype


def to_dolphindb_type_string(dtype):
    # TODO: use a dictionary
    if dtype is np.bool or dtype is np.bool8 or dtype is bool:
        return "BOOL"
    elif dtype is np.int8:
        return "CHAR"
    elif dtype is np.int16:
        return "SHORT"
    elif dtype is np.int32:
        return "INT"
    elif dtype is np.int64 or dtype is int:
        return "LONG"
    elif dtype is np.float32:
        return "FLOAT"
    elif dtype is np.float64 or dtype is float:
        return "DOUBLE"
    elif dtype is np.str or dtype is np.str_ or dtype is str:
        return "STRING"
    elif dtype == "category":
        return "SYMBOL"
    elif isinstance(dtype, str):
        if dtype in ["SYMBOL", "DATE", "DATETIME", "MINUTE", "MONTH",
                     "NANOTIME", "NANOTIMESTAMP", "SECOND", "TIME",
                     "TIMESTAMP", "BOOL", "CHAR", "SHORT", "INT",
                     "LONG", "FLOAT", "DOUBLE", "STRING"]:
            return dtype
        else:
            raise ValueError("Unrecognized dtype: " + dtype)
    elif dtype is np.datetime64:
        raise NotImplementedError()
    else:
        raise ValueError("Unrecognized dtype: " + str(dtype))


def _to_data_column(column):
    return "_".join(map(str, column)) if isinstance(column, tuple) else column


def is_dolphindb_scalar(obj):
    return obj is None or \
           isinstance(obj, (bool, int, float, str, np.number, np.datetime64, pd.Timestamp))


def is_dolphindb_vector(obj):
    from dolphindb_numpy import ndarray
    return (isinstance(obj, (list, tuple)) or
            isinstance(obj, (ndarray, np.ndarray)) and obj.ndim == 1)


def is_dolphindb_integral(obj):
    return isinstance(obj, (int, np.signedinteger))


def is_dolphindb_floating(obj):
    return isinstance(obj, (float, np.float32, np.float64))


def is_dolphindb_uploadable(obj):
    return isinstance(obj, (bool, int, float, str, list, dict, set,
                            np.ndarray, np.number, pd.DataFrame))


def is_dolphindb_identifier(var_name):
    is_id = (isinstance(var_name, str)
             and not var_name.startswith("_")
             and var_name.isidentifier())
    if not is_id:
        _warn_not_dolphindb_identifier()
    return is_id


def to_dolphindb_literal(obj):
    if obj is float('nan') or obj is np.NaN:
        return "NULL"
    elif isinstance(obj, str):
        return json.dumps(obj)
    elif (isinstance(obj, (list, tuple))
            and all(isinstance(o, str) for o in obj)):
        return "[" + ",".join(json.dumps(o) for o in obj) + "]"
    else:
        return str(obj)

def _get_time_str(time):
    if isinstance(time, str):
        if len(time) == 4:
            timestr = f"0{time}"
        else:
            timestr = time
    elif isinstance(time, datetime.time):
        timestr = time.strftime('%H:%M')
    else:
        raise TypeError("datetime.time or str")
    return timestr

def _infer_level(level, index_map):
    """
    Returns (index_columns, groupby_list, index_names, level_idx) by infering level from index_map
    """
    def check_level_existence(levels, index_map):
        index_columns = []
        index_names = []
        index_level_num = len(index_map)
        level_idx = []
        for level in levels:
            if isinstance(level, str):
                for i, (index_column, index_name) in enumerate(index_map):
                    if index_name is not None and index_name[0] == level:
                        index_columns.append(index_column)
                        index_names.append(level)
                        level_idx.append(i)
                        break
                else:
                    raise ValueError(f"level name {level} is not the name of the index")
            elif isinstance(level, int):
                if level < 0:    # TODO: level < 0 with non-MultiIndex might be invalid?
                    level += index_level_num
                if level >= index_level_num:
                    raise ValueError("level > 0 or level < -1 only valid with MultiIndex")
                index_columns.append(index_map[level][0])
                index_name = index_map[level][1]
                index_names.append(None if index_name is None else index_name[0])
                level_idx.append(level)
            else:
                raise ValueError("level must be a string, an integer, or a sequence of such")
        return index_columns, index_names, level_idx

    if level is None:
        index_columns = [index_column for index_column, _ in index_map]
        index_names = [None if index_name is None else index_name[0]
                       for _, index_name in index_map]
        groupby_list = None
        level_idx = list(range(len(index_map)))
    else:
        if isinstance(level, (str, int)):
            level = [level]
        elif not isinstance(level, Iterable):
            raise ValueError("level must be a string, an integer, or a sequence of such")
        index_columns, index_names, level_idx = check_level_existence(level, index_map)
        groupby_list = index_columns
    return index_columns, groupby_list, index_names, level_idx


_TYPE_MAP = {
    'categorical': 'categorical',
    'category': 'categorical',
    'int': 'int',
    'int8': 'integer',
    'int16': 'integer',
    'int32': 'integer',
    'int64': 'integer',
    'i': 'integer',
    'uint8': 'integer',
    'uint16': 'integer',
    'uint32': 'integer',
    'uint64': 'integer',
    'u': 'integer',
    'float32': 'floating',
    'float64': 'floating',
    'f': 'floating',
    'complex64': 'complex',
    'complex128': 'complex',
    'c': 'complex',
    'string': 'bytes',
    'S': 'bytes',
    'U': 'string',
    'bool': 'boolean',
    'b': 'boolean',
    'datetime64[ns]': 'datetime64',
    'M': 'datetime64',
    'timedelta64[ns]': 'timedelta64',
    'm': 'timedelta64',
    'interval': 'interval',
}


def _try_infer_map(v):
    for attr in ['name', 'kind', 'base']:
        val = getattr(v.dtype, attr)
        if val in _TYPE_MAP:
            return _TYPE_MAP[val]
    return None


def _infer_dtype(value=None, skipna=None):
    '''Value : scalar, list, ndarray, or pandas type
    skipna : bool, default False
    Ignore NaN values when inferring the type.

    :raise
        TypeError if ndarray-like but cannot infer the dtype
    '''
    if skipna is None:
        skipna = False

    if isinstance(value,np.ndarray):
        values = value
    elif hasattr(value,'dtype'):
        try:
            values = getattr(value,'_values',getattr(value,'values',value))
        except TypeError:
            value = _try_infer_map(value)
            if value is not None:
                return value
            raise ValueError("can not infer type")
    else:
        if not isinstance(value,list):
            value = list(value)
        from pandas.core.dtypes.cast import (
            construct_1d_object_array_from_listlike)
        values = construct_1d_object_array_from_listlike(value)

    values = values.ravel()

    val = _try_infer_map(values)
    if val is not None :
        return val

    n = len(values)
    if n==0:
        return 'empty'

    data_type = type(values[0])

    for i in range(n):
        val = values[i]

        if val is None:
            pass

        if data_type != type(val):
            return 'mixed'

    return data_type.__name__
