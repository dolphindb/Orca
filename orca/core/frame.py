import itertools
import warnings
from collections import OrderedDict
from typing import Iterable, List, Set

import dolphindb as ddb
import numpy as np
import pandas as pd
import pandas.plotting._core as gfx

from .base import _Frame
from .common import (MixedTypeWarning, _raise_must_compute_error,
                     default_session)
from .groupby import ContextByExpression
from .indexes import Index, MultiIndex, RangeIndex
from .indexing import _iLocIndexer, _LocIndexer
from .internal import _ConstantSP, _InternalFrame
from .merge import _generate_joiner, merge, merge_window
from .operator import (
    ArithExpression, BooleanExpression, DataFrameLike,
    IOOpsMixin)
from .series import Series
from .utils import (
    ORCA_COLUMN_NAME_FORMAT, ORCA_INDEX_NAME_FORMAT, _infer_axis, _infer_level,
    _to_column_index, _to_index_map, _to_numpy_dtype,
    _try_convert_iterable_to_list, _unsupport_columns_axis,
    check_key_existence, dolphindb_numeric_types, get_orca_obj_from_script,
    is_dolphindb_identifier, is_dolphindb_scalar, is_dolphindb_vector,
    sql_select, to_dolphindb_literal, to_dolphindb_type_name,
    dolphindb_literal_types, dolphindb_temporal_types, _get_python_object_dtype)


class DataFrame(DataFrameLike, _Frame, IOOpsMixin):

    _internal_names: List[str] = [
        "_internal",
        "_where_expr",
        "_index",
        "_columns",
        "_is_snapshot",
        "_name",
        "_session",
    ]
    _internal_names_set: Set[str] = set(_internal_names)

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False, session=default_session()):
        if isinstance(data, _InternalFrame):
            assert index is None
            assert columns is None
            assert dtype is None
            assert not copy
            _Frame.__init__(self, data, session)
        elif isinstance(data, DataFrame):
            # TODO: check arguments
            odf = data._internal
            new_odf = odf.copy_as_in_memory_table() if copy else odf
            _Frame.__init__(self, new_odf, session)
            self._where_expr = data._where_expr
        elif isinstance(data, Series):
            new_odf = data._internal.copy_as_in_memory_table()
            _Frame.__init__(self, new_odf, session)
            self._where_expr = data._where_expr
        else:
            if isinstance(data, pd.DataFrame):
                # TODO: orca.from_pandas top function becasue this might be an unnecessary copy
                data = (data if index is None and columns is None and dtype is None
                        else pd.DataFrame(data, index=index, columns=columns, dtype=dtype))    # TODO: copy?
                odf = _InternalFrame.from_pandas(session, data)
            else:
                # TODO: there's a to_pandas operation and it's time consuming
                # FIXME: uploading an orca Index and all-empty columns
                pd_index = pd.RangeIndex(0, len(index)) if isinstance(index, Index) else index
                pd_columns = columns.to_pandas() if isinstance(columns, Index) else columns
                data = pd.DataFrame(data=data, index=pd_index,    # TODO: write your own Series parsing function
                                    columns=pd_columns, dtype=dtype, copy=False)    # TODO: copy = True or False ?
                odf = _InternalFrame.from_pandas(session, data)
                if isinstance(index, Index):
                    odf.attach_index(index)
            _Frame.__init__(self, odf, session)
        self._name = None
        self._update_metadata(index)        

    def _update_metadata(self, index=None):
        self._index = Index._from_internal(self._internal, index)
        if self._column_index_level > 1:
            columns = pd.MultiIndex.from_tuples(self._column_index)
        else:
            columns = pd.Index([idx[0] for idx in self._column_index])
        if self._internal.column_index_names is not None:
            columns.names = self._internal.column_index_names
        self._columns = columns

    def __getitem__(self, key):
        if key is None:
            return self
        elif isinstance(key, (int, slice)):
            return self.iloc[key]
        elif isinstance(key, BooleanExpression):
            return self._get_rows_from_boolean_expression(key)
        elif isinstance(key, tuple):
            if not all(isinstance(k, BooleanExpression) for k in key):
                raise TypeError("All elements in the key tuple must be a boolean expression")
            return self._get_rows_from_boolean_expression(key)
        elif isinstance(key, (str, Iterable)):
            return self._get_columns(key)
        elif isinstance(key, Series):    # TODO: df[orca.Series([True, False, True])]
            if key._ddb_dtype == ddb.settings.DT_BOOL:
                return self._get_rows_from_boolean_series(key)
            elif key._ddb_dtype == ddb.settings.DT_STRING:
                key = key.values
                return self._get_columns(key)
        elif isinstance(key, DataFrame):
            raise NotImplementedError()
        else:
            raise KeyError(key)

    def __setitem__(self, key, value):
        if (isinstance(key, BooleanExpression)
                or (isinstance(key, Series) and key._ddb_dtype == ddb.settings.DT_BOOL)):
            self.loc[key] = value
            return
        elif isinstance(key, str):
            return self._set_columns(key, value)
        elif isinstance(key, Iterable):
            key = _try_convert_iterable_to_list(key)
            if all(isinstance(k, str) for k in key):
                return self._set_columns(key, value)
            else:
                raise ValueError("Each element in the key must be a string")
        else:
            raise KeyError(key)

    def __getattr__(self, key: str):
        if key in self._data_columns:
            return self[key]
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def __setattr__(self, name, value):
        try:
            object.__getattribute__(self, name)
            return object.__setattr__(self, name, value)
        except AttributeError:
            pass

        if name in self._internal_names_set:
            object.__setattr__(self, name, value)
        else:
            try:
                self.__getitem__(name)
                return self.__setitem__(name, value)
            except (AttributeError, TypeError):
                warnings.warn("Orca doesn't allow columns to be created via "
                              "a new attribute name")

    # iteration

    def __iter__(self):
        return iter(self.columns)

    def items(self):
        return zip(self.columns, (self[col] for col in self._data_columns))

    iteritems = items

    def iterrows(self):
        if self._segmented:
            raise NotImplementedError("Cannot iterate over a segmented orca object")
        session = self._session
        ref = self.compute()
        for i in range(len(ref)):
            select_list = itertools.chain(ref._index_columns, ref._data_columns)
            from_clause = f"{ref._var_name}[{i}:{i+1}]"
            script = sql_select(select_list, from_clause)
            row = ref._get_from_script(session, script, squeeze=True, squeeze_axis=None)
        raise NotImplementedError()    # TODO: implement

    def itertuples(self, index=True, name='Orca'):
        if self._segmented:
            raise NotImplementedError("Cannot iterate over a segmented orca object")

    def to_numpy(self):
        return self.to_pandas().to_numpy()

    @property
    def loc(self):
        return _LocIndexer(self)

    @property
    def iloc(self):
        return _iLocIndexer(self)

    @property
    def columns(self):
        """
        The column labels of the DataFrame.
        
        Returns
        -------
        orca.Index
            The column labels of each column.
        """
        return self._columns
        # return Index(self._data_columns)

    @columns.setter
    def columns(self, columns):
        if isinstance(columns, pd.MultiIndex):
            raise NotImplementedError()
        else:
            old_names = self._column_index
            if all(isinstance(col, str) for col in columns):
                columns = {old_col: new_col for old_col, new_col in zip(self._data_columns, columns)}
                self.rename(columns=columns, level=None, inplace=True)
                return
            if len(old_names) != len(columns):
                raise ValueError(
                    f"Length mismatch: Expected axis has {len(old_names)} " \
                    f"elements, new values have {len(columns)} elements")
            # self._internal = self._internal.copy_as_in_memory_table(columns)
            self._internal.set_columns(columns)

    @property
    def ndim(self):
        """
        Return an int representing the number of axes / array dimensions.
        
        Returns
        -------
        int
            Return 1 if Series. Otherwise return 2 if DataFrame.
        """
        return 2

    @property    # TODO: @lazyproperty
    def dtypes(self):
        """
        Return the dtypes in the DataFrame.

        This returns a Series with the data type of each column. The
        result’s index is the original DataFrame’s columns. Columns with
        mixed types are stored with the object dtype. See the User Guide
        for more.

        Returns
        -------
        pandas.Series
            The data type of each column.
        """
        ddb_dtypes = self._ddb_dtypes
        dtypes = (_to_numpy_dtype(ddb_dtypes[col])
                  for col in self._data_columns)
        return pd.Series(dtypes, index=self._data_columns)

    @property
    def shape(self):
        row_num, col_num = len(self), len(self._data_columns)
        return (row_num, col_num)

    @property
    def size(self):
        row_num, col_num = len(self), len(self._data_columns)
        return row_num * col_num

    @property
    def axes(self):
        return [self.index, self.columns]

    @property
    def empty(self):
        if self.size > 0:
            return False
        else:
            return True

    @property
    def _is_empty_dataframe(self):
        return len(self._data_columns) == 0 and len(self) == 0

    # Conversion

    def to_pandas(self):
        pdf = _Frame.to_pandas(self)
        pdf.columns = self.columns
        return pdf

    def _uses_internal_index(self):
        index = self.index
        if isinstance(index, RangeIndex):
            return False
        else:
            return self._var_name == index._var_name

    def _binary_op_on_different_indices(self, other, func, axis):    # TODO: add axis check
        """
        Implementation of binary operator between DataFrames on different
        indices. A new DataFrame representing an in-memory DolphinDB table
        is returned. It is garenteed that both DataFrames have no where_expr.
        
        Parameters
        ----------
        other : _Frame
            Right hand side of the operator.
        func : str
            Fuction name.
        
        Returns
        -------
        orca.DataFrame
            The result of the operation.
        
        Raises
        ------
        NotImplementedError
            To be implemented.
        """

        def merge_columns(self_columns, other_columns):
            """
            Align the input columns, filling the missing columns with None
            --------
            
            Examples
            --------
            >>> merge_columns(
            ...     ["a", "b", "ba", "d", "f"],
            ...     ["e", "c", "d", "g", "ga", "a"]
            ... )
            (('a','a'),('b',None),('ba',None),(None,c),('d','d'),(None,'e'),('f',None),(None,'g'),(None,'ga'))
            """
            sorted_self_columns, sorted_other_columns = sorted(self_columns), sorted(other_columns)
            self_idx = other_idx = 0
            self_len, other_len = len(self_columns), len(other_columns)
            while self_idx < self_len and other_idx < other_len:
                curr_self_column, curr_other_column = sorted_self_columns[self_idx], sorted_other_columns[other_idx]
                if curr_self_column == curr_other_column:
                    yield curr_self_column, curr_other_column
                    self_idx += 1
                    other_idx += 1
                elif curr_self_column < curr_other_column:
                    yield curr_self_column, None
                    self_idx += 1
                else:
                    yield None, curr_other_column
                    other_idx += 1
            while self_idx < self_len:
                yield sorted_self_columns[self_idx], None
                self_idx += 1
            while other_idx < other_len:
                yield None, sorted_other_columns[other_idx]
                other_idx += 1

        assert isinstance(self, _Frame)
        assert isinstance(other, _Frame)
        if ((not self._in_memory and len(self._index_columns) == 0)
                or (not other._in_memory and len(other._index_columns) == 0)):
            raise ValueError("Frame has no default index if it is not in memory")
        session = self._session
        self_var_name, other_var_name = self._var_name, other._var_name
        if other._is_dataframe_like:
            self_data_columns = self._data_columns
            other_data_columns = other._data_columns
            index_list, from_clause = _generate_joiner(
                self_var_name, other_var_name, self._index_columns, other._index_columns)
            if self_data_columns == other_data_columns:
                select_list = (f"{func}({self_var_name}.{c}, {other_var_name}.{c}) as {c}"
                               for c in self_data_columns)
                data_columns = self_data_columns
            else:
                merged_columns = list(merge_columns(self_data_columns, other_data_columns))
                select_list = (f"00f as {s if o is None else o}" if s is None or o is None
                               else f"{func}({self_var_name}.{s}, {other_var_name}.{s}) as {s}"
                               for s, o in merged_columns)
                data_columns = [s if o is None else o for s, o in merged_columns]
            select_list = itertools.chain(index_list, select_list)
            script = sql_select(select_list, from_clause)
        elif other._is_series_like:
            self_data_columns = self._data_columns
            other_data_column = other._data_columns[0]
            index_list, from_clause = _generate_joiner(
                self._var_name, other._var_name, self._index_columns, other._index_columns)
            select_list = (f"{func}({self_var_name}.{c}, {other_var_name}.{other_data_column}) as {c}"
                           for c in self_data_columns)
            data_columns = self_data_columns
            select_list = itertools.chain(index_list, select_list)
            script = sql_select(select_list, from_clause)
        return self._get_from_script(
            session, script, data_columns=data_columns, index_map=self._index_map, index=self._index)

    def _get_columns(self, key):
        new_odf = self._internal[key]
        if isinstance(key, str):
            return Series._with_where_expr(self._where_expr, new_odf, name=key)
        elif isinstance(key, Iterable):
            # TODO: sys:1: SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame
            return DataFrame._with_where_expr(self._where_expr, new_odf)
        else:    # TODO: support other column types
            raise NotImplementedError()

    # Reshaping / sorting / transposing

    def explode(self, column):
        warnings.warn("orca columns cannot be list-like. explode will not work", MixedTypeWarning)
        return self

    # Combining / joining / merging

    def join(self, other, on=None, how="left", lsuffix="", rsuffix="", sort=False, lazy=False):
        return merge(self, other, how=how, left_on=on, left_index=(not on),
                     right_index=True, suffixes=(lsuffix, rsuffix), sort=sort, lazy=lazy)

    def merge(self, right, how='inner', on=None, left_on=None, right_on=None,
              left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'),
              copy=True, indicator=False, validate=None, lazy=False):
        return merge(self, right, how=how, on=on, left_on=left_on, right_on=right_on,
                     left_index=left_index, right_index=right_index, sort=sort,
                     suffixes=suffixes, copy=copy, indicator=indicator, validate=validate, lazy=lazy)

    def merge_window(self, right, window_lower, window_upper, prevailing=False, on=None,
                     left_on=None, right_on=None, left_index=False, right_index=False):
        return merge_window(self, right, window_lower=window_lower, window_upper=window_upper,
                            prevailing=prevailing, on=on, left_on=left_on, right_on=right_on,
                            left_index=left_index, right_index=right_index)

    # Methods that are trying to modify the data
    def append(self, other, ignore_index=False, verify_integrity=False, sort=None, inplace=False):
        from .operator import ArithExpression, BooleanExpression
        if isinstance(other, (DataFrame, ArithExpression, BooleanExpression)):
            other = other.compute()
            if self._is_empty_dataframe:
                if not inplace:
                    return DataFrame(other, copy=True, session=self._session)
                else:
                    self._internal = other._internal.copy_as_in_memory_table()
                    return
            else:
                if not inplace:
                    new_df = self._copy_as_in_memory_frame(ignore_index=ignore_index)
                    new_df._internal.append(other, ignore_index, sort)
                    return new_df
                else:
                    if ignore_index:
                        raise ValueError("in-place append cannot ignore index")
                    if sort:
                        raise ValueError("in-place append does not support sort indexes")
                    self._internal.append(other, False, None)
                    return
        else:
            raise NotImplementedError()

    def _set_columns(self, key, value):
        self._prepare_for_update()
        session = self._session

        def update_different_frame(var, value, key):
            ref = value.compute()
            column_names = ref._data_columns
            ref_var_name = ref._var_name
            if self._index_columns == [] and ref._index_columns == []:
                new_values = [f"{ref_var_name}.{column_name}" for column_name in column_names]
                var._sql_update(key, new_values)
            else:
                _, from_table_joiner = _generate_joiner(
                    self._var_name, ref._var_name,
                    self._index_columns, ref._index_columns, how="left_semi")
                new_values = [f"{ref_var_name}.{column_name}" for column_name in column_names]
                var._sql_update(key, new_values, from_table_joiner)

        var = self._var
        key = [key] if isinstance(key, str) else key
        new_data_columns = list(self._data_columns)
        has_new_key = False
        for k in key:
            if not is_dolphindb_identifier(k):
                raise ValueError(f"'{k}' is not a valid DolphinDB identifier")
            if k not in new_data_columns:
                new_data_columns.append(k)
                has_new_key = True
        if is_dolphindb_scalar(value) or is_dolphindb_vector(value):
            upload_var = _ConstantSP.upload_obj(session, value)
            new_values = [f"{upload_var._var_name}"]
            var._sql_update(key, new_values)
        elif isinstance(value, Series):
            if len(key) != 1:
                raise ValueError("Columns must be same length as key")
            if self._is_empty_dataframe:
                data_column = value._data_columns[0]
                odf = value._internal.copy_as_in_memory_table()
                odf.rename(columns={data_column: key[0]}, level=None)
                self._internal = odf
                self._update_metadata()
                return
            if value._var_name == self._var_name:
                if value._where_expr is not self._where_expr:
                    _raise_must_compute_error("Setting a value to a DataFrame "
                                              "with a WHERE clause is not supported")
                    # update t set key = NULL
                    # type_name = to_dolphindb_type_name(value._ddb_dtype)
                    # null_values = [f"{type_name}(NULL)"]
                    # var._sql_update(key, null_values)
                # update t set key = new_key where ...
                new_values = [value._data_columns[0]]
                var._sql_update(key, new_values, where_expr=value._where_expr)
            else:
                update_different_frame(var, value, key)
        elif isinstance(value, DataFrame):
            data_columns = value._data_columns
            if len(key) != len(data_columns):
                raise ValueError("Columns must be same length as key")
            if value._var_name == self._var_name:
                if value._where_expr is not None:
                    type_names = (to_dolphindb_type_name(ddb_dtype)
                                  for ddb_dtype in value._ddb_dtypes)
                    null_values = (f"{type_name}(NULL)" for type_name in type_names)
                    var._sql_update(key, null_values)
                var._sql_update(key, data_columns, where_expr=value._where_expr)
            else:
                update_different_frame(var, value, key)
        elif isinstance(value, ContextByExpression):
            if (value._var_name == self._var_name
                    and value._where_expr is self._where_expr):
                new_values = value._get_data_select_list()
                if len(key) != len(new_values):
                    raise ValueError("Columns must be same length as key")
                var._sql_update(key, new_values, where_expr=value._where_expr,
                                contextby_list=value._get_contextby_list())
            else:
                update_different_frame(var, value, key)
        elif isinstance(value, (ArithExpression, BooleanExpression)):
            if (value._var_name == self._var_name
                    and value._where_expr is self._where_expr):
                new_values = value._get_data_select_list()
                if len(key) != len(new_values):     # TODO: align columns?
                    raise ValueError("Columns must be same length as key")
                var._sql_update(key, new_values, where_expr=value._where_expr)
            else:
                update_different_frame(var, value, key)
        else:
            raise ValueError("value must be a scalar or a Series")

        if has_new_key:
            new_odf = _InternalFrame(session, self._var, index_map=self._index_map, data_columns=new_data_columns)
            self._internal = new_odf
        self._update_metadata()

    # Reindexing / selection / label manipulation

    def drop(self, labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise'):
        axis = _infer_axis(self, axis)
        if axis == 0 and index is None:
            index = labels
        elif axis == 1 and columns is None:
            columns = labels
        elif axis not in [0, 1]:
            raise ValueError('axis must be either 0 or 1')

        if index is not None:
            raise NotImplementedError()

        session = self._session
        var = self._var
        if columns is not None:
            data_columns = self._data_columns
            columns, dropped = check_key_existence(columns, data_columns)
            new_odf = _InternalFrame(session, var, index_map=self._index_map,
                                     data_columns=dropped)
            if inplace:
                self._internal = new_odf
                return
            else:
                return DataFrame._with_where_expr(self._where_expr, new_odf)

    def rename(self, mapper=None, index=None, columns=None, axis='index', copy=True, inplace=False, level=None, errors='ignore'):
        if inplace and self._segmented:
            raise ValueError("A segmented table is not allowed to be renamed inplace")
        axis = _infer_axis(None, axis)
        if axis == 0:
            index = mapper
        elif axis == 1:
            columns = mapper
        else:
            raise ValueError(f"No axis named {axis} for object type {type(self)}")

        if index is not None:    # TODO: support more args; implement rename index
            if not isinstance(index, dict):
                raise NotImplementedError()
            raise NotImplementedError()
        if columns is not None:
            if not isinstance(columns, dict):
                raise NotImplementedError()
            column_index_level = self._column_index_level
            if level is not None:
                if not isinstance(level, int):    # TODO: support level
                    raise NotImplementedError()
                elif level < 0:
                    level += column_index_level
                if level < 0:
                    raise IndexError(f"Too many levels: Index has only "
                                     f"{column_index_level} level, {level} "
                                     f"is not a valid level number")
            if not inplace:
                odf = self._internal.copy_as_in_memory_table()
                odf.rename(columns=columns, level=level)
                return self._with_where_expr(self._where_expr, odf, session=self._session)
            else:
                self._internal.rename(columns=columns, level=level)
                self._update_metadata()

    def reindex(self, labels=None, index=None, columns=None, axis=None, method=None, copy=True, level=None, fill_value=None, limit=None, tolerance=None):
        if axis is not None and (index is not None or columns is not None):
            raise TypeError("Cannot specify both 'axis' and any of 'index' or 'columns'.")
        axis = _infer_axis(None, axis)
        if labels is not None:
            if axis in (0, None):
                index = labels
            else:
                columns = labels

        if index is not None and not is_dolphindb_vector(index):
            raise TypeError(f"Index must be called with a collection of some kind, "
                            f"{type(index)} was passed")
        if columns is not None and not is_dolphindb_vector(columns):
            raise TypeError(f"Columns must be called with a collection of some kind, "
                            f"{type(columns)} was passed")
        
        if not copy:    # TODO: handle not copy
            raise NotImplementedError()
        df = self.copy()
        if index is not None:
            df = df._reindex_index(index)
        if columns is not None:
            df = df._reindex_columns(columns)
        if fill_value is not None:
            df = df.fillna(fill_value)
        return df

    def _reindex_index(self, index):
        assert len(self._index_columns) <= 1, "Index should be single column or not set."
        index_column = self._index_columns[0]
        if index_column == ORCA_INDEX_NAME_FORMAT(0):
            index_column = None
        labels = DataFrame(pd.DataFrame(index=pd.Index(list(index), name=index_column)))
        return self.join(labels, how="right")

    def _reindex_columns(self, columns):
        level = self._column_index_level
        if level > 1:
            label_columns = list(columns)
            for col in label_columns:
                if not isinstance(col, tuple):
                    raise TypeError(f"Expected tuple, got {type(col)}")
        else:
            label_columns = [(col,) for col in columns]
        for col in label_columns:
            if len(col) != level:
                raise ValueError(f"shape (1,{len(col)}) doesn't match the shape (1,{level})")
        
        columns, idx = [], []
        for label in label_columns:
            if label in self._column_index:
                columns.append(self._column_name_for(label))
            else:
                raise NotImplementedError("Cannot add a new column filled with nan to a DataFrame")
            idx.append(label)
        internal = _InternalFrame(self._session, self._var, data_columns=columns, column_index=idx)
        return DataFrame(internal, session=self._session)

    def set_index(self, keys,
                  drop: bool = True,
                  append: bool = False,
                  inplace: bool = False,    # TODO: copy ddb df
                  verify_integrity: bool = False):
        if not inplace:
            new_df = self._copy_as_in_memory_frame()
            new_df.set_index(keys=keys, drop=drop, append=append,
                             inplace=True, verify_integrity=verify_integrity)
            return new_df
        else:
            data_columns = self._data_columns
            keys, dropped = check_key_existence(keys, data_columns)
            if drop:
                data_columns = dropped
            index_map = [(key, (key,)) for key in keys]

            new_odf = _InternalFrame(self._session, self._var, index_map=index_map, data_columns=data_columns)    # TODO: dealing with multiple level columns
            self._internal = new_odf
            self._index = Index._from_internal(new_odf)
            self._update_metadata()
            return

    def stack(self, level=-1, dropna=True):
        if self._column_index_level != 1:
            raise ValueError("column level must be 1")
        if len(self._index_columns) > 0:
            key_col_names = to_dolphindb_literal(self._index_columns)
        else:
            key_col_names = ""
        value_col_names = to_dolphindb_literal(self._data_columns)
        if self._column_index_names is None or self._column_index_names[0] is None:
            new_index = ORCA_INDEX_NAME_FORMAT(len(self._index_columns))
        else:
            new_index = self._column_index_names[0]
        new_names = to_dolphindb_literal([new_index, ORCA_COLUMN_NAME_FORMAT(0)])
        script = f"unpivot({self._var_name},{key_col_names},{value_col_names})" \
                 f".rename!(`valueType`value, {new_names})"
        if self._column_index_names is not None:
            value_column_name = self._column_index_names[0]
        else:
            value_column_name = None
        index_map = self._index_map + [(new_index, (value_column_name,))]
        res = get_orca_obj_from_script(self._session, script, index_map, squeeze=True, squeeze_axis=1)
        if dropna:
            return res.dropna(how='all')
        else:
            return res

    def melt(self, id_vars=None, value_vars=None, var_name=None, value_name='value', col_level=None):
        if self._column_index_level != 1 and col_level is not None:
            raise NotImplementedError()
        if id_vars is None:
            id_vars = []
            key_col_names = ""
        else:
            check_key_existence(id_vars, self._data_columns)
            key_col_names = to_dolphindb_literal(id_vars)
        if value_vars is None:
            value_vars = [k for k in self._data_columns if k not in id_vars]
        else:
            check_key_existence(value_vars, self._data_columns)
        for col in value_vars:
            if self._ddb_dtypes[col] not in dolphindb_numeric_types:
                raise TypeError("The value column to melt must be numeric type")
        value_col_names = to_dolphindb_literal(value_vars)
        if var_name is None:
            var_name = "variable"
        elif var_name is not None:
            if not isinstance(var_name, str):
                raise TypeError("var_name must be a string")
        if value_name is None:
            value_name = "value"
        elif value_name is not None:
            if not isinstance(value_name, str):
                raise TypeError("value_name must be a string")
        new_names = to_dolphindb_literal([var_name, value_name])
        script = f"unpivot({self._var_name},{key_col_names},{value_col_names})" \
                 f".rename!(`valueType`value, {new_names})"

        res = get_orca_obj_from_script(self._session, script, None, squeeze=True, squeeze_axis=1)
        return res

    def pivot(self, index=None, columns=None, values=None):
        if index is None:
            raise ValueError("index cannot be empty in Orca's pivot_table")
        if columns is None:
            raise ValueError("columns cannot be empty in Orca's pivot_table")

        def get_column_key_and_script(column):
            ext_err = "Unable to pivot with an external Series"
            if isinstance(column, str):
                check_key_existence(column, self._data_columns)
                key = script = column
            elif isinstance(column, Series):
                if column._var_name != self._var_name:
                    raise ValueError(ext_err)
                key = script = column._data_columns[0]
            elif isinstance(column, Index):
                if column._var_name != self._var_name:
                    raise ValueError(ext_err)
                if isinstance(column, MultiIndex):
                    raise ValueError("Unable to pivot with MultiIndex")
                key = script = column._index_columns[0]
            elif isinstance(column, (ArithExpression, BooleanExpression)):
                if not column._is_series_like:
                    raise ValueError("Grouper is not 1-dimensional")
                if column._var_name != self._var_name:
                    raise ValueError(ext_err)
                key = column._data_columns[0]
                script = column._get_data_select_list()[0]
            else:
                raise KeyError(column)
            return key, script

        if values is not None:
            if not isinstance(values, str):
                raise TypeError("values must be a string")
            check_key_existence(values, self._data_columns)
            select_list = [f"{values}"]
        else:  # DolphinDB extension
            select_list = []
        index_key, index_script = get_column_key_and_script(index)
        columns_key, columns_script = get_column_key_and_script(columns)
        pivot_list = [f"{index_script} as {index_key}",
                      f"{columns_script} as {columns_key}"]

        script = sql_select(select_list, self._var_name, self._where_expr,
                            pivot_list=pivot_list)
        return get_orca_obj_from_script(self._session, script, [index_key], column_index_names=[columns_key])

    def pivot_table(self, values=None, index=None, columns=None, aggfunc='mean', fill_value=None, margins=False, dropna=True, margins_name='All', observed=False):
        if index is None:
            raise ValueError("index cannot be empty in Orca's pivot_table")
        if columns is None:
            raise ValueError("columns cannot be empty in Orca's pivot_table")
        if not isinstance(aggfunc, str):
            raise TypeError("aggfunc must be a string in Orca's pivot_table")

        def get_column_key_and_script(column):
            ext_err = "Unable to pivot with an external Series"
            if isinstance(column, str):
                check_key_existence(column, self._data_columns)
                key = script = column
            elif isinstance(column, Series):
                if column._var_name != self._var_name:
                    raise ValueError(ext_err)
                key = script = column._data_columns[0]
            elif isinstance(column, Index):
                if column._var_name != self._var_name:
                    raise ValueError(ext_err)
                if isinstance(column, MultiIndex):
                    raise ValueError("Unable to pivot with MultiIndex")
                key = script = column._index_columns[0]
            elif isinstance(column, (ArithExpression, BooleanExpression)):
                if not column._is_series_like:
                    raise ValueError("Grouper is not 1-dimensional")
                if column._var_name != self._var_name:
                    raise ValueError(ext_err)
                key = column._data_columns[0]
                script = column._get_data_select_list()[0]
            else:
                raise KeyError(column)
            return key, script
        
        if values is not None:
            if not isinstance(values, str):
                raise TypeError("values must be a string")
            check_key_existence(values, self._data_columns)
            select_list = [f"{aggfunc}({values})"]
        else:    # DolphinDB extension
            select_list = [aggfunc]
        index_key, index_script = get_column_key_and_script(index)
        columns_key, columns_script = get_column_key_and_script(columns)
        pivot_list = [f"{index_script} as {index_key}",
                      f"{columns_script} as {columns_key}"]

        script = sql_select(select_list, self._var_name, self._where_expr,
                            pivot_list=pivot_list)
        return get_orca_obj_from_script(self._session, script, [index_key], column_index_names=[columns_key])

    def transpose(self, copy=True, *args, **kwargs):
        # TODO: support MultiIndex
        if not copy:
            raise ValueError("copy must be True for an Orca DataFrame to be transposed")
        if not self._is_matrix_like():
            raise ValueError("A DataFrame with mixed-type or string-type columns cannot be transposed")
        if self._column_index_level > 1 or len(self._index_columns) > 1:
            raise ValueError("Only DataFrames with a one-level column and index can be transposed")

        session = self._session
        ref = self.compute()

        new_index = np.array([col_idx[0] for col_idx in self._column_index])
        new_index_var = _ConstantSP.upload_obj(session, new_index)
        new_index_map = _to_index_map([self._columns.name])
        new_index_column_name = new_index_map[0][0]
        new_column_script = sql_select(ref._index_columns, ref._var_name, ref._where_expr, is_exec=True)
        new_columns = session.run(new_column_script)
        new_column_index_names = [self._index.name]
        new_column_index = _to_column_index(new_columns)

        script = sql_select(ref._data_columns, ref._var_name, ref._where_expr)
        script = f"{script}.matrix().transpose().table()"
        var = _ConstantSP.run_script(session, script)
        var._sql_update([new_index_column_name], [new_index_var._var_name])
        odf = _InternalFrame(
            session, var, index_map=new_index_map, column_index=new_column_index,
            column_index_names=new_column_index_names)
        return DataFrame(odf, session=session)

    @property
    def T(self):
        return self.transpose()

    def dot(self, other):
        if isinstance(other, (ArithExpression, BooleanExpression)):
            _raise_must_compute_error("Unable to call 'dot' on an Expression")
        if not self._is_matrix_like() or not other._is_matrix_like():
            raise ValueError("Unable to call 'dot' on a DataFrame with mixed-type columns")
        if len(self.columns) != len(other):
            raise ValueError("matrices are not aligned")

        session = self._session
        ref = self.compute()
        self_script = ref._to_script(ignore_index=True)
        other_script = other._to_script(ignore_index=True)
        script = f"dot(matrix({self_script}),matrix({other_script})).table()"
        var = _ConstantSP.run_script(session, script)
        data_columns = [f"C{i}" for i in range(len(other.columns))]
        column_index = other._column_index
        odf = _InternalFrame(session, var, data_columns=data_columns, column_index=column_index)
        odf.attach_index(ref.index)
        return DataFrame(odf, session=session)

    def diag(self, ignore_index=False):
        if isinstance(self, (ArithExpression, BooleanExpression)):
            _raise_must_compute_error("Unable to call 'diag' on an Expression")
        if not self._is_matrix_like():
            raise ValueError("Unable to call 'diag' on a DataFrame with mixed-type columns")
        ref = self.compute()
        size = len(ref)
        if size != len(ref.columns):
            raise ValueError("The DataFrame is not a square matrix")
        
        self_script = ref._to_script(ignore_index=True)
        script = f"diag(matrix({self_script}))"
        if ignore_index:
            script = f"table({script} as {ORCA_COLUMN_NAME_FORMAT(0)})"
            index_map = None
        else:
            index_list = (f"{script} as {col}"
                          for (col, _), script
                          in zip(ref._index_map, ref.index._to_script_list()))
            table_columns = itertools.chain([f"{script} as {ORCA_COLUMN_NAME_FORMAT(0)}"], index_list)
            script = ",".join(table_columns)
            script = f"table({script})"
            index_map = ref._index_map
        return Series._get_from_script(self._session, script, index_map=index_map)

    def xs(self, key, axis=0, level=None, drop_level=True):
        if is_dolphindb_scalar(key):
            key = (key,)
        elif not isinstance(key, tuple):
            raise TypeError(f"{key} is an invalid key")
        _unsupport_columns_axis(self, axis)
        if level is not None:
            raise NotImplementedError()
        var_name = self._var_name
        where_list = (
            f"{var_name}.{i} == {to_dolphindb_literal(k)}"
            for k, i in zip(key, self._index_columns)
        )
        select_list = itertools.chain(self._index_columns, self._data_columns)
        script = sql_select(select_list, var_name, where_list)
        res = self._get_from_script(self._session, script, self)
        if drop_level:
            return res.droplevel(range(len(key)))
        else:
            return res

    # Time series-related

    def asof(self, where, subset=None):
        pass

    def pop(self, key):
        if not isinstance(key, str):
            raise KeyError('Key must be str')
        col = self[key]
        if col is None:
            raise KeyError("Column not found")
        self.drop(labels = key, axis = 1, inplace = True)
        return col

    def get(self, key, default = None):
        try:
            return self[key]
        except (KeyError, ValueError, IndexError):
            return default

    def lookup(self, row_labels, col_labels):
        l = len(row_labels)
        if l != len(col_labels):
            raise ValueError("Row labels must have same size as column labels")

        thresh = 1000
        if (not self._is_mixed_type()) or (l > thresh):
            t = list(set(self._ddb_dtypes[self._data_columns]))[0]
            result = np.empty(l, dtype=_to_numpy_dtype(t))
            for i, (r, c) in enumerate(zip(row_labels, col_labels)):
                result[i] = self.loc[r, c]
        else:
            result = np.empty(l, dtype="O")
            for i, (r, c) in enumerate(zip(row_labels, col_labels)):
                result[i] = self.loc[r, c]
        return result

    # TODO: @property and @lazyproperty
    def _is_mixed_type(self):
        return len(set(self._ddb_dtypes[self._data_columns])) > 1

    def _is_matrix_like(self):
        return not self._is_mixed_type() \
            and self._ddb_dtypes[self._data_columns[0]] in dolphindb_numeric_types

    @classmethod
    def from_dict(cls, *args, **kwargs):
        return DataFrame(pd.DataFrame.from_dict(*args, **kwargs))

    @classmethod
    def from_items(cls, *args, **kwargs):
        return DataFrame(pd.DataFrame.from_items(*args, **kwargs))

    @classmethod
    def from_records(cls, *args, **kwargs):
        return DataFrame(pd.DataFrame.from_records(*args, **kwargs))

    def to_csv(
            self,
            path_or_buf=None,
            sep=",",
            na_rep="",
            float_format=None,
            columns=None,
            header=True,
            index=True,
            index_label=None,
            mode="w",
            encoding=None,
            compression="infer",
            quoting=None,
            quotechar='"',
            line_terminator=None,
            chunksize=None,
            date_format=None,
            doublequote=True,
            escapechar=None,
            decimal=".",
            engine="dolphindb",
            append=False,
    ):
        r"""
        Write object to a comma-separated values (csv) file.

        Parameters
        ----------
        path_or_buf : str or file handle, default None
            File path or object, if None is provided the result is returned as
            a string.  If a file object is passed it should be opened with
            `newline=''`, disabling universal newlines.

        sep : str, default ','
            String of length 1. Field delimiter for the output file.
        na_rep : str, default ''
            Missing data representation.
        float_format : str, default None
            Format string for floating point numbers.
        columns : sequence, optional
            Columns to write.
        header : bool or list of str, default True
            Write out the column names. If a list of strings is given it is
            assumed to be aliases for the column names.

        index : bool, default True
            Write row names (index).
        index_label : str or sequence, or False, default None
            Column label for index column(s) if desired. If None is given, and
            `header` and `index` are True, then the index names are used. A
            sequence should be given if the object uses MultiIndex. If
            False do not print fields for index names. Use index_label=False
            for easier importing in R.
        mode : str
            Python write mode, default 'w'.
        encoding : str, optional
            A string representing the encoding to use in the output file,
            defaults to 'utf-8'.
        compression : str, default 'infer'
            Compression mode among the following possible values: {'infer',
            'gzip', 'bz2', 'zip', 'xz', None}. If 'infer' and `path_or_buf`
            is path-like, then detect compression from the following
            extensions: '.gz', '.bz2', '.zip' or '.xz'. (otherwise no
            compression).
        quoting : optional constant from csv module
            Defaults to csv.QUOTE_MINIMAL. If you have set a `float_format`
            then floats are converted to strings and thus csv.QUOTE_NONNUMERIC
            will treat them as non-numeric.
        quotechar : str, default '\"'
            String of length 1. Character used to quote fields.
        line_terminator : str, optional
            The newline character or character sequence to use in the output
            file. Defaults to `os.linesep`, which depends on the OS in which
            this method is called ('\n' for linux, '\r\n' for Windows, i.e.).

        chunksize : int or None
            Rows to write at a time.
        date_format : str, default None
            Format string for datetime objects.
        doublequote : bool, default True
            Control quoting of `quotechar` inside a field.
        escapechar : str, default None
            String of length 1. Character used to escape `sep` and `quotechar`
            when appropriate.
        decimal : str, default '.'
            Character recognized as decimal separator. E.g. use ',' for
            European data.

        Returns
        -------
        None or str
            If path_or_buf is None, returns the resulting csv format as a
            string. Otherwise returns None.

        See Also
        --------
        read_csv : Load a CSV file into a DataFrame.
        to_excel : Write DataFrame to an Excel file.
        """
        if engine == "pandas":
            df = self.to_pandas()
            from pandas.io.formats.csvs import CSVFormatter

            formatter = CSVFormatter(
                df,
                path_or_buf,
                line_terminator=line_terminator,
                sep=sep,
                encoding=encoding,
                compression=compression,
                quoting=quoting,
                na_rep=na_rep,
                float_format=float_format,
                cols=columns,
                header=header,
                index=index,
                index_label=index_label,
                mode=mode,
                chunksize=chunksize,
                quotechar=quotechar,
                date_format=date_format,
                doublequote=doublequote,
                escapechar=escapechar,
                decimal=decimal,
            )
            formatter.save()

            if path_or_buf is None:
                return formatter.path_or_buf.getvalue()
        elif engine == "dolphindb":
            append = 'true' if append else 'false'
            self_script = self._to_script(ignore_index=True)
            script = f"saveText({self_script},'{path_or_buf}', '{sep}', {append})"
            self._session.run(script)
        else:
            raise ValueError("Unsupport type engine " + engine)
        return

    def select_dtypes(self, include = None, exclude = None):

        df = self
        data_columns = df._data_columns
        col_dtypes = self.dtypes
        vlist = []

        if exclude is None:
            if isinstance(include, str):
                include = [include]
            elif not isinstance(include, list):
                include = list(include)
            for i in range(0, len(data_columns)):
                if col_dtypes[i] in include:
                    vlist.append(data_columns[i])
        else:
            if isinstance(exclude, str):
                exclude = [exclude]
            elif not isinstance(exclude, list):
                exclude = list(exclude)
            for i in range(0, len(data_columns)):
                if not (col_dtypes[i] in exclude):
                    vlist.append(data_columns[i])
        return df.loc[:,vlist]

    def set_axis(self, labels, axis=0, inplace=None):
        axis = _infer_axis(self, axis)
        mapper = dict()
        if axis == 0:
            if len(self.index.values) != len(labels):
                raise ValueError(f" Length mismatch: Expected axis has {len(self.index.values)} elements, new values have {len(labels)} elements")
            for idx, new_idx in zip(self.index.values, labels):
                mapper[idx] = new_idx
            raise NotImplementedError()  # TODO: support more args; implement set_axis index
        elif axis == 1:
            if len(self.columns.values) != len(labels):
                raise ValueError(f" Length mismatch: Expected axis has {len(self.columns.values)} elements, new values have {len(labels)} elements")
            for col, new_col in zip(self.columns.values, labels):
                mapper[col] = new_col
            column_index_level = self._column_index_level
            if not inplace:
                odf = self._internal.copy_as_in_memory_table()
                odf.rename(columns=mapper, level=None)
                return self._with_where_expr(self._where_expr, odf, session=self._session)
            else:
                self._internal.rename(columns=mapper, level=None)
        elif axis not in [0, 1]:
            raise ValueError('axis must be either 0 or 1')

    def rename_axis(self, mapper=None, index=None, columns=None, axis=0, copy=True, inplace=False):
        if inplace and self._segmented:
            raise ValueError("A segmented table is not allowed to be rename axis inplace")
        axis = _infer_axis(None, axis)
        if axis == 0:
            if mapper is not None:
                index = mapper
        elif axis == 1:
            if mapper is not None:
                columns = mapper
        else:
            raise ValueError(f"No axis named {axis} for object type {type(self)}")
        if index == str.upper:
            index = str.upper(self._internal.index_name)
        elif index == str.lower:
            index = str.lower(self._internal.index_name)
        if columns == str.upper:
            columns = str.upper(self.columns.name)
        elif columns == str.lower:
            columns = str.lower(self.columns.name)
        if index is not None:    # TODO: support more args; implement rename axis index
            if isinstance(index, dict):
                raise NotImplementedError()
            else:
                if isinstance(index, str) or isinstance(index, int) or isinstance(index, float) or isinstance(index, bool) or index is None:
                    index = [index]
                if len(index) != 1:
                    raise ValueError(f"Length of new names must be 1, got {len(index)}")
                if not inplace:
                    odf = self._internal.copy_as_in_memory_table()
                    new_df = self._with_where_expr(self._where_expr, odf, session=self._session)
                    new_df.rename_axis(index[0],axis=0,inplace=True)
                    return new_df
                else:
                    index_map = [i for i in self._index_map]
                    index_map.remove(index_map[0])
                    tup = (self._index_map[0][0], (index[0],))
                    index_map.insert(0, tup)
                    new_odf = _InternalFrame(self._session, self._var, index_map=index_map)
                    self._internal = new_odf
                    self._index = Index._from_internal(new_odf)
                    self._update_metadata()
        if columns is not None:
            if isinstance(columns, dict):
                if not inplace:
                    odf = self._internal.copy_as_in_memory_table()
                    odf.rename(columns=columns, level=None)
                    return self._with_where_expr(self._where_expr, odf, session=self._session)
                else:
                    self._internal.rename(columns=columns, level=None)
                    self._update_metadata()
            else:
                if isinstance(columns, str) or isinstance(columns, int) or isinstance(columns, float) \
                        or isinstance(columns, bool) or columns is None:
                    columns = [columns]
                if len(columns) != 1:
                    raise ValueError(f"Length of new names must be 1, got {len(columns)}")
                if not inplace:
                    odf = self._internal.copy_as_in_memory_table()
                    new_df = self._with_where_expr(self._where_expr, odf, session=self._session)
                    new_df.rename_axis(columns[0], axis =1, inplace=True)
                    return new_df
                else:
                    col = self.columns
                    col.name = columns[0]
                    self._internal.set_columns(columns=col)

    def describe(self,  percentiles=None, include=None, exclude=None):
        df = self
        col_list = []
        data_columns = df._data_columns
        #dtype_list = dt
        if include is None and exclude is None:
            col_list = [x for x in data_columns if df[x]._ddb_dtype in dolphindb_numeric_types]
            #col_list = [data_columns[i] for i in range(0, len(data_columns)) if (df[data_columns[i]]._ddb_dtype in dolphindb_numeric_types)]
        elif include == 'all':
            col_list = data_columns
        #elif include is None:
        #    col_list = [data_columns[i] for i in range(0, len(data_columns)) if
        #                (df[data_columns[i]]._ddb_dtype not in exclude)]
        else:
            clude = include or exclude
            tlist = []
            if not isinstance(clude, list):
                clude = [clude]

            if np.number in clude or 'number' in clude:
                tlist = tlist + [x for x in data_columns if df[x]._ddb_dtype in dolphindb_numeric_types]
            if 'O' in clude or np.object in clude or 'object' in clude:
                tlist = tlist  + [x for x in data_columns if df[x]._ddb_dtype in dolphindb_literal_types]
            if 'datetime64' in clude or 'datetime' in clude or np.datetime64 in clude:
                tlist = tlist + [x for x in data_columns if df[x]._ddb_dtype in dolphindb_temporal_types]
            # TODO: time types
            tlist = tlist + [x for x in data_columns if df[x].dtype in clude]

            if exclude is None:
                col_list = [x for x in data_columns if x in tlist]
            else:
                col_list = [x for x in data_columns if x not in tlist]
        session = self._session
        series_list = []
        for i in col_list:
            s = df[i]
            series_list = series_list + [s.describe(percentiles)]

        #     series_list = series_list + [pd.Series(data=data_list, index=index_list)]

        d = pd.concat([x for x in series_list], axis=1, sort=False)
        d.columns = col_list
        return d

    def add_prefix(self, prefix):
        df = self
        data_columns = [str(x[0]) for x in df._column_index]
        coln_list = [prefix + i for i in data_columns]
        ture_name = []
        for i in range(0, len(coln_list)):
            var_name = coln_list[i]
            is_id = (isinstance(var_name, str)
             and not var_name.startswith("_")
             and var_name.isidentifier())
            if is_id:
                ture_name.append(coln_list[i])
            else:
                ture_name.append(ORCA_COLUMN_NAME_FORMAT(i))
        internal = _InternalFrame(self._session, self._var, data_columns=self._data_columns, column_index=self._column_index)
        internal.set_columns(coln_list)
        return DataFrame(internal, session=self._session)


    def add_suffix(self, suffix):
        df = self
        data_columns = [str(x[0]) for x in df._column_index]
        coln_list = [i+suffix for i in data_columns]
        ture_name = []
        for i in range(0, len(coln_list)):
            var_name = coln_list[i]
            is_id = (isinstance(var_name, str)
             and not var_name.startswith("_")
             and var_name.isidentifier())
            if is_id:
                ture_name.append(coln_list[i])
            else:
                ture_name.append(ORCA_COLUMN_NAME_FORMAT(i))
        internal = _InternalFrame(self._session, self._var, data_columns=self._data_columns, column_index=self._column_index)
        internal.set_columns(coln_list)
        return DataFrame(internal, session=self._session)

    def replace(self, to_replace=None, value=None, inplace=False, limit=None, regex=False, method='pad'):
        if inplace:
            replaced = self.replace(value=value, to_replace=to_replace, inplace=False, limit=limit, regex=regex, method=method)
            self[self._data_columns] = replaced
            return
        if regex:
            raise NotImplementedError()
        df = self
        session = self._session
        data_columns = self._data_columns
        select_script = []
        if (value is None) and (isinstance(to_replace, list) or isinstance(to_replace, tuple) or is_dolphindb_scalar(to_replace)):
            if isinstance(to_replace, tuple):
                to_replace = list(to_replace)
            elif is_dolphindb_scalar(to_replace):
                to_replace = [to_replace]
            #ref = _ConstantSP.upload_obj(session, to_replace)
            if method == 'pad':
                method = 'ffill'
            for col in data_columns:
                same_type_list = [x for x in to_replace if _get_python_object_dtype(x) == df[col].dtype]
                if len(same_type_list) > 0:
                    select_script.append(f"iif({col} in {to_dolphindb_literal(same_type_list)}, {self._ddb_dtypestr[col]}(NULL), {col}).{method}() as {col}")
                else:
                    select_script.append(f"{col}")

            #select_script = [f"iif({col} in {ref.var_name}, {self._ddb_dtypestr[col]}(NULL), {col}).{method}() as {col}" for col in data_columns]

        elif is_dolphindb_scalar(to_replace) and is_dolphindb_scalar(value):
            for col in data_columns:
                if df[col].dtype == _get_python_object_dtype(to_replace):
                    select_script.append(f"iif({col}=={to_dolphindb_literal(to_replace)}, {to_dolphindb_literal(value)}, {col}) as {col}")
                else:
                    select_script.append(f"{col}")
            #select_script = [f"iif({col}=={to_dolphindb_literal(to_replace)}, {to_dolphindb_literal(value)}, {col}) as {col}" for col in data_columns]

        elif isinstance(to_replace, list) and isinstance(value, list):
            raise NotImplementedError("value can not be list now ")       # TODO :  value is list
        elif isinstance(to_replace, list) and is_dolphindb_scalar(value):
            #ref = _ConstantSP.upload_obj(session, to_replace)
            #select_script = [f"iif({col} in {ref.var_name}, {to_dolphindb_literal(value)}, {col}) as {col}" for col in data_columns]
            for col in data_columns:
                same_type_list = [x for x in to_replace if _get_python_object_dtype(x) == df[col].dtype]
                if len(same_type_list) > 0:
                    select_script.append(f"iif({col} in {to_dolphindb_literal(same_type_list)}, {to_dolphindb_literal(value)}, {col}) as {col}")
                else:
                    select_script.append(f"{col}")
                #select_script = [f"iif({col} in {}, {to_dolphindb_literal(value)}, {col}) as {col}" for col in data_columns]

        elif isinstance(to_replace, dict) and value is None:
            raise NotImplementedError()                         # TODO :  key-value is more than one
        elif isinstance(to_replace, dict) and is_dolphindb_scalar(value):
            for col in data_columns:
                if col in to_replace.keys() and df[col].dtype == _get_python_object_dtype(to_replace[col]):
                    select_script.append(f"iif({col}=={to_dolphindb_literal(to_replace[col])}, {value}, {col}) as {col}")
                else:
                    select_script.append(f"{col}")

        select_script = itertools.chain(self._index_columns, select_script)
        script = sql_select(select_script, self._var_name)
        return self._get_from_script(session, script)
