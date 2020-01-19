import abc
import itertools
import warnings

import numpy as np
import pandas as pd
import pandas.plotting._core as gfx

from .accessor import CachedAccessor
from .common import CopiedTableWarning, _get_verbose
from .internal import _ConstantSP, _InternalAccessor, _InternalFrame
from .operator import (ArithExpression, ArithOpsMixin, BooleanExpression,
                       LogicalOpsMixin, StatOpsMixin)
from .utils import (_infer_axis, _infer_level, _merge_where_expr, _to_data_column,
                    check_key_existence, get_orca_obj_from_script,
                    is_dolphindb_integral, sql_select, sql_update,
                    to_dolphindb_literal, is_dolphindb_scalar,
                    _try_convert_iterable_to_list, _unsupport_columns_axis, _get_time_str,
                    dolphindb_temporal_types)


class _Frame(_InternalAccessor, ArithOpsMixin, LogicalOpsMixin, StatOpsMixin, metaclass=abc.ABCMeta):

    """
    The base class for both DataFrame and Series.
    """

    # def __init__(self, session, internal, index):
    def __init__(self, internal, session, copy=False):
        from .indexes import Index
        self._internal = internal
        self._session = session
        self._where_expr = None
        self._is_snapshot = False
        self._name = None
        self._index = None

    @classmethod
    def _with_where_expr(cls, where_expr, odf, *args, **kwargs):
        if "session" not in kwargs:
            kwargs["session"] = odf._session
        new_df = cls(odf, *args, **kwargs)
        new_df._where_expr = where_expr
        new_df._index._where_expr = where_expr
        new_df._is_snapshot = True
        return new_df

    def _get_rows_from_boolean_expression(self, key):
        assert isinstance(key, (BooleanExpression, tuple))
        where_expr = _merge_where_expr(self._where_expr, key)
        df = self._with_where_expr(where_expr, self._internal)
        if df._is_series_like:
            df._name = self._name
        return df

    def _get_rows_from_boolean_series(self, key):
        # TODO: align index
        if key._segmented:
            raise ValueError("A segmented table cannot be used as a key")
        where_clause = key._to_script(ignore_index=True, is_exec=True) + " = true"
        where_expr = _merge_where_expr(self._where_expr, where_clause)
        try:
            return self._with_where_expr(where_expr, self._internal).compute()
        except RuntimeError as ex:
            ex_msg = str(ex)
            if ex_msg.find("The resulting dimension of WHERE clause") != -1:
                raise IndexError("Unalignable boolean Series provided as indexer (index of the boolean Series and of the indexed object do not match)")
            else:
                raise ex

    @classmethod
    def _get_from_script(cls, session, script, df=None,
                         data_columns=None, index_map=None,
                         column_index=None, column_index_names=None, index=None,
                         name=None, squeeze=False, squeeze_axis=None):
        """
        Get a _Frame from a script. Users can specify the column names and index
        names of the _Frame, or provide a template df.
        
        Parameters
        ----------
        session : dolphindb.Session
            The DolphinDB session to be used.
        script : str
            The script string to be executed.
        df : orca._Frame, optional
            The template _Frame to be used, by default None
        data_columns : list[str], optional
            The list of column name of the _Frame, by default None
        index_map : list[tuple[str, tuple[str, ...]]], optional
            The index_map the _Frame, by default None
        column_index : List[Tuple[str, ...]], optional
            The column_index of the _Frame
        column_index_names : List[str], optional
            The list of column index names of the _Frame, by default None
        name : str, optional
            The name of Series object. Note that DataFrame does not need it.
        squeeze: bool, optional
            Whether to squeeze 1 dimensional axis objects into scalars,
            by default False
        squeeze_axis : {0 or 'index', 1 or 'columns', None}, default None
            A specific axis to squeeze. By default, all length-1 axes are squeezed.
        
        Returns
        -------
        orca._Frame
            The _Frame representing the result of the script. Note that _Frame is
            an abstrct base class. The actual return type is DataFrame or Series.
        """
        from .series import Series
        if df is not None:
            data_columns = df._data_columns if data_columns is None else data_columns
            index_map = df._index_map if index_map is None else index_map
            column_index = df._column_index if column_index is None else column_index
            column_index_names = df._column_index_names if column_index_names is None else column_index_names
            index = df._index if index is None else index
            name = df._name
        var = _ConstantSP.run_script(session, script)
        if squeeze and len(var) == 1:
            if cls is Series:
                data_column = to_dolphindb_literal(data_columns[0])
                script = f"{var._var_name}[{data_column}, 0]"
                if _get_verbose():
                    print(script)
                return session.run(script)
            else:
                index_columns = [index_column for index_column, _ in index_map]
                return var.squeeze(index_columns, squeeze_axis=squeeze_axis)
        odf = _InternalFrame(session, var, data_columns=data_columns,
                             index_map=index_map, column_index=column_index,
                             column_index_names=column_index_names)
        if squeeze and squeeze_axis in (1, None) and len(odf.data_columns) == 1:
            return Series(odf, name=name, session=session)
        obj = cls(odf, session=session)
        if obj._is_series_like:
            obj.name = name
        return obj

    def __repr__(self):
        type_str = str(type(self))[7:-1]
        if self._where_expr is not None:
            return f"<{type_str} object with a WHERE clause>"
        if self._segmented:
            return f"<{type_str} object representing a column in a DolphinDB segmented table>"
        else:
            return self.to_pandas().__repr__()

    def __len__(self):
        if self._where_expr is None:
            return len(self._internal)
        else:
            script = sql_select(["count(*)"], self._var_name, self._where_expr, is_exec=True)
            if _get_verbose():
                print(script)
            return self._session.run(script)

    @property
    def empty(self):
        if len(self) > 0:
            return False
        else:
            return True

    @property
    def is_copy(self):
        return None
    
    def to_pandas(self):    # TODO: optimize
        script = self._to_script()
        if _get_verbose():
            print(script)
        data = self._session.run(script)
        index_columns = self._index_columns
        names = self.index.names if len(index_columns) > 1 else self.index.name
        if index_columns:
            data.set_index(index_columns, inplace=True)
        data.index.rename(names, inplace=True)
        # if isinstance(data_index, pd.DatetimeIndex):
            # self_index = self._index
            # TODO: set dtype and tz
            # data_index.dtype = self_index.dtype
            # data_index.freq = self_index.freq
            # data_index.tz = self_index.tz
        return data

    def to_numpy(self):
        return self.to_pandas().to_numpy()

    def _to_script(self, data_columns=None, limit=None, ignore_index=False,
                  orderby_list=None, asc=True, is_exec=False):
        index_columns = [] if ignore_index else self._index_columns
        if data_columns is not None:
            data_columns, _ = check_key_existence(data_columns, self._data_columns)
        data_columns = data_columns or self._data_columns
        select_list = itertools.chain(index_columns, data_columns)
        script = sql_select(select_list, self._var_name, self._where_expr,
                            orderby_list=orderby_list, asc=asc, limit=limit, is_exec=is_exec)
        return script

    def compute(self, data_columns=None, as_non_segmented=False, limit=None, squeeze=False, squeeze_axis=None):
        if (self._where_expr is None
            and (data_columns is None or data_columns == [])
            and not (as_non_segmented and self._segmented)
            and limit is None
            and not squeeze):
            return self
        else:
            return self._copy_as_in_memory_frame(data_columns, limit, squeeze=squeeze, squeeze_axis=squeeze_axis)

    def copy(self, deep=True):
        if deep:
            return self._copy_as_in_memory_frame()
        else:
            raise NotImplementedError()    # TODO: shallow copy

    def _copy_as_in_memory_frame(self, data_columns=None, limit=None, ignore_index=False, squeeze=False, squeeze_axis=None):
        """
        Return a copy of the Frame.
        
        Parameters
        ----------
        data_columns : List[str], optional
            The data columns of the original frame to be copied, by default None
            If not specified, copy all columns
        limit : str, optional
            The limit script (top clause), by default None
        ignore_index : bool, optional
            Whether to ignore the index columns, by default None
        
        Returns
        -------
        DataFrame or Series
            A copy of the original frame.

        .. note:: Difference with _InternalFrame.copy_as_in_memory_table:
            _InternalFrame.copy_as_in_memory_table will keep the segments
            while this function always returns an in-memory non-segmented table
        """
        session = self._session
        if self._internal.is_any_vector:
            return session.run(f"{self._var_name}[0]")
        script = self._to_script(data_columns=data_columns, limit=limit, ignore_index=ignore_index)
        if data_columns is not None:
            column_index = [(col,) for col in data_columns]
        else:
            column_index = None
        index_map = [] if ignore_index else None
        name = self._name if self._is_series_like else None
        return self._get_from_script(
            session, script, self, index_map=index_map,
            data_columns=data_columns, column_index=column_index,
            name=name, squeeze=squeeze, squeeze_axis=squeeze_axis)

    def _prepare_for_update(self):
        if not self._in_memory:
            warnings.warn("A table is copied as an in-memory table.",
                          CopiedTableWarning)
            # self._internal = self._internal.copy_as_in_memory_table()
            self._internal.copy_as_in_memory_table(inplace=True)

    def head(self, n=5):
        if not is_dolphindb_integral(n):
            raise TypeError(f"Incorrect type of n: 'int' expected, "
                            f"got '{n.__class__.__name__}'")
        elif n == 0:
            raise NotImplementedError()
        elif n < 0:
            return self.iloc[:n]
        else:
            return self.compute(limit=str(n))

    def tail(self, n=5):
        if not is_dolphindb_integral(n):
            raise TypeError(f"Incorrect type of n: 'int' expected, "
                            f"got '{n.__class__.__name__}'")
        elif n == 0:
            raise NotImplementedError()
        else:
            return self.iloc[-n:]

    @property
    @abc.abstractmethod
    def iloc(self):
        pass

    @property
    @abc.abstractmethod
    def loc(self):
        pass

    @property
    @abc.abstractmethod
    def _is_dataframe_like(self):
        pass

    @property
    @abc.abstractmethod
    def _is_series_like(self):
        pass

    @property
    def array(self):
        return self.to_numpy().tolist()

    def __array__(self, dtype=None):
        # this function is to support some library like numpy to init data
        return np.asarray(self.array, dtype)

    __array_priority__ = 10    # support operation with NumPy

    @property
    def index(self):
        return self._index

    @property
    def values(self):
        warnings.warn("orca objects does not store data in numpy arrays. Accessing values will retrive whole data from the remote node.", Warning)
        return self.to_numpy()

    def _unary_op(self, *args, **kwargs):
        return StatOpsMixin._unary_op(self, *args, **kwargs)

    def _unary_agg_op(self, *args, **kwargs):
        return StatOpsMixin._unary_agg_op(self, *args, **kwargs)

    def _binary_op(self, *args, **kwargs):
        return ArithOpsMixin._binary_op(self, *args, **kwargs)

    def _binary_agg_op(self, *args, **kwargs):
        return StatOpsMixin._binary_agg_op(self, *args, **kwargs)

    def _extended_binary_op(self, *args, **kwargs):
        return ArithOpsMixin._extended_binary_op(self, *args, **kwargs)

    def _logical_op(self, *args, **kwargs):
        return LogicalOpsMixin._logical_op(self, *args, **kwargs)

    def _logical_unary_op(self, *args, **kwargs):
        return LogicalOpsMixin._logical_unary_op(self, *args, **kwargs)

    def _update_metadata(self):
        pass

    def _concat_script(self, merged_columns, ignore_index, inner_join=False):
        def find_column_name(col_idx):
            for data_column, column_index in zip(self._data_columns, self._column_index):
                if col_idx == column_index:
                    return data_column
            else:
                raise ValueError(f"The DataFrame or Series does not contain column_index {col_idx}")
        if inner_join:
            select_list = map(find_column_name, merged_columns)
        else:
            length = len(self)
            select_list = (find_column_name(column) if column in self._column_index
                           else f"{typestr}(NULL) as {_to_data_column(column)}"
                           if typestr != "symbol"
                           else f'symbol(take("", {length})) as {_to_data_column(column)}'
                           for column, typestr in merged_columns)
        if not ignore_index:
            select_list = itertools.chain(self._index_columns, select_list)
        script = sql_select(select_list, self._var_name, self._where_expr)
        if not self._in_memory:
            script = f"loadTableBySQL(<{script}>)"
        return f"({script})"
    
    def _get_data_select_list(self):
        return self._internal.data_select_list

    def reset_index(self, level=None, drop=False, inplace=False, col_level=0, col_fill=''):
        from orca.core.frame import DataFrame
        from orca.core.indexes import Index

        index_columns, _, _, _ = _infer_level(level, self._index_map)
        index_column_size = len(self._index_columns)
        column_index_level = self._column_index_level
        if col_level >= column_index_level:
            raise IndexError(f"Too many levels: Index has only {column_index_level}, not {col_level+1}")
        index_map = []
        column_index = []
        data_columns = []
        for level_idx, col, idx_map in zip(range(index_column_size), self._index_columns, self._index_map):
            if col not in index_columns:
                index_map.append(idx_map)
            elif not drop:
                if index_column_size == 1:
                    index_name = "index" if idx_map[1] is None else idx_map[1][0]
                else:
                    index_name = f"level_{level_idx}" if idx_map[1] is None else idx_map[1][0]
                new_column_index = [col_fill] * column_index_level
                new_column_index[col_level] = index_name
                column_index.append(tuple(new_column_index))
                data_columns.append(col)
        column_index += self._column_index
        data_columns += self._data_columns
        new_odf = _InternalFrame(self._session, self._var, index_map=index_map,
                                 data_columns=data_columns, column_index=column_index)

        if inplace:
            if self._is_series_like:
                raise TypeError("Cannot reset_index inplace on a Series to create a DataFrame")
            self._internal = new_odf
            self._index = Index._from_internal(new_odf)
            self._update_metadata()
            return

        return DataFrame._with_where_expr(self._where_expr, new_odf)

    def unstack(self, level=-1, fill_value=None):
        if self._is_dataframe_like:
            raise NotImplementedError()
        if len(self._index_columns) != 2:
            raise ValueError("Index levels must be exactly 2 due to the limitations of DolphinDB. Use droplevel to drop unneeded levels")
        index_columns, _, _, level_idx = _infer_level(level, self._index_map)
        assert len(level_idx) == 1
        level_idx = level_idx[0]
        if len(index_columns) != 1:
            raise ValueError("Only one level is supported in unstack")
        pivot_column = index_columns[0]
        select_list = [self._data_columns[0]]
        pivot_index = None
        other_index = []
        pivot_index_level = 0
        for i, idx in enumerate(self._index_columns):
            if i != level_idx:
                if pivot_index is None:
                    pivot_index = idx
                    pivot_index_level = i
                else:
                    other_index.append(idx)
        assert len(other_index) == 0

        pivot_list = [pivot_index, pivot_column]
        script = sql_select(select_list, self._var_name, self._where_expr,
                            pivot_list=pivot_list)
        index_map = [self._index_map[pivot_index_level]]
        column_index_names = [self._index_names[level_idx]]
        return get_orca_obj_from_script(self._session, script, index_map, column_index_names=column_index_names)

    if pd.__version__.startswith("0.24"):
        pd_plot = gfx.FramePlotMethods
        pd_hist_frame = gfx.hist_frame
        pd_boxplot_frame = gfx.boxplot_frame
    elif pd.__version__.startswith("0.25"):
        pd_plot = pd.plotting.PlotAccessor
        pd_hist_frame = pd.plotting.hist_frame
        pd_boxplot_frame = pd.plotting.boxplot_frame
    plot = CachedAccessor("plot", pd_plot)

    def hist(self, *args, **kwargs):
        return self.to_pandas().hist(*args, **kwargs)

    def boxplot(self, *args, **kwargs):
        return self.to_pandas().boxplot(*args, **kwargs)

    def _pandas_where(self, cond, other=np.nan, inplace=False, axis=None, level=None, errors="raise", try_cast=False):
        if isinstance(cond,_Frame) or isinstance(cond,BooleanExpression) or isinstance(cond,ArithExpression):
            cond = cond.to_pandas()
        if isinstance(other,_Frame) or isinstance(other, BooleanExpression) or isinstance(other,ArithExpression):
            other = other.to_pandas()

        if inplace:
            self = self.__class__(self.to_pandas().where(
                cond, other, inplace, axis, level, errors=errors, try_cast=try_cast))
            return self
        else:
            return self.__class__(self.to_pandas().where(
                cond, other, inplace, axis, level, errors=errors, try_cast=try_cast))

    def where(self, cond, other=np.nan, inplace=False, axis=None, level=None, errors="raise", try_cast=False):
        from .frame import DataFrame
        from .series import Series
        if callable(cond) or callable(other) or axis !=None or level !=None:
            return self._pandas_where(cond, other, inplace, axis, level, errors=errors, try_cast=try_cast)
        elif isinstance(cond, BooleanExpression):
            cond_script = " and ".join(cond._to_where_list())
            if isinstance(self,DataFrame):
                # when self is DataFrame, other can not be series, or the axis must be setted, the pandas will process
                # this situation
                cond_list = cond._to_where_list()
                if isinstance(other, ArithExpression) or isinstance(other, DataFrame):
                    if self._var_name == other._var_name:
                        other_data = other._get_data_select_list()
                        select_list = [f"iif({cond_col},{self_col},{other_col}) as {self_col}"
                                       for cond_col, self_col, other_col in zip(cond_list, self._data_columns, other_data)]
                        update_value = [f"iif({cond_col},{self_col},{other_col})"
                                        for cond_col, self_col, other_col in zip(cond_list, self._data_columns, other_data)]
                    else:
                        # We do not support self and other are not the same dataframe. So, just use pandas to support.
                        return self._pandas_where(cond, other, inplace, axis, level, errors=errors, try_cast=try_cast)
                elif other is None or other is np.nan or other is float('nan'):
                    select_list = [f"iif({cond_col},{self_col},{self._ddb_dtypestr[self_col]}(NULL)) as {self_col}"
                                   for cond_col, self_col in zip(cond_list, self._data_columns)]
                    update_value = ["iif({cond_col},{self_col},{self._ddb_dtypestr[self_col]}(NULL))"
                                    for cond_col, self_col in zip(cond_list, self._data_columns)]
                else:
                    other = to_dolphindb_literal(other)
                    select_list = [f"iif({cond_col},{self_col},{other}) as {self_col}"
                                   for cond_col, self_col in zip(cond_list, self._data_columns)]
                    update_value = [f"iif({cond_col},{self_col},{other})"
                                    for cond_col, self_col in zip(cond_list, self._data_columns)]
                return self._execute_select_and_update(select_list, update_value, inplace, errors)
            elif isinstance(self, Series):
                if isinstance(other, ArithExpression) or isinstance(other, Series):
                    if self._var_name == other._var_name:
                        other_data = other._get_data_select_list()
                        select_list = [f"iif({cond_script},{self_col},{other_col}) as {self_col}"
                                       for self_col, other_col in zip(self._data_columns, other_data)]
                        update_value = [f"iif({cond_script},{self_col},{other_col})"
                                        for self_col, other_col in zip(self._data_columns, other_data)]
                    else:
                        return self._pandas_where(cond, other, inplace, axis, level, errors=errors, try_cast=try_cast)
                elif other is None or other is np.nan or other is float('nan'):
                    select_list = [f"iif({cond_script},{self_col},{self._ddb_dtypestr[self_col]}(NULL)) as {self_col}"
                                   for self_col in self._data_columns]
                    update_value = ["iif({cond_script},{self_col},{self._ddb_dtypestr[self_col]}(NULL))"
                                    for self_col in self._data_columns]
                else:
                    other = to_dolphindb_literal(other)
                    select_list = [f"iif({cond_script},{self_col},{other}) as {self_col}"
                                   for self_col in self._data_columns]
                    update_value = [f"iif({cond_script},{self_col},{other})"
                                    for self_col in self._data_columns]
                return self._execute_select_and_update(select_list, update_value, inplace, errors)
        else:
            # TODO : if cond is _Frame, this function can be optimized
            return self._pandas_where(cond, other, inplace, axis, level, errors=errors, try_cast=try_cast)

    def _execute_select_and_update(self, select_script=None, update_value=None, inplace=False, errors="raise"):
        select_script = itertools.chain(self._index_columns, select_script)
        script = sql_select(select_script, self._var_name)
        try:
            data = self._get_from_script(self._session, script, df=self)
            if inplace:
                update_columns = ["{}".format(col) for col in self._data_columns]
                update_script = sql_update(self._var_name, update_columns, update_value)
                self._session.run(update_script)
            return data
        except:
            if errors == "ignore":
                return self
            else:
                raise ValueError("Can not get data by where function")

    def mask(self, cond, other=np.nan, inplace=False, axis=None, level=None, errors="raise", try_cast=False):
        return self.where(~cond, other=other, inplace=inplace, axis=axis,
                          level=level, errors=errors, try_cast=False)

    def take(self, indices, axis=0, is_copy=False, **kwargs):
        axis = _infer_axis(self, axis)
        if axis == 0 or axis == 'index':
            return self.iloc[indices]
        elif axis == 1 or axis == 'columns':
            return self.iloc[:, indices]
        elif axis not in [0, 1]:
            raise ValueError('axis must be either 0 or 1')

    @property
    def at(self):
        from .indexing import _LocIndexer
        return _LocIndexer(self)

    @property
    def iat(self):
        from .indexing import _iLocIndexer
        return _iLocIndexer(self)

    def droplevel(self, level, axis=0):
        session = self._session
        if is_dolphindb_scalar(level):
            level = (level,)
        else:
            level = _try_convert_iterable_to_list(level)

        _unsupport_columns_axis(self, axis)

        _, _, _, level_idx = _infer_level(level, self._index_map)
        index_map = [m for i, m in enumerate(self._index_map) if i not in level_idx]
        new_odf = _InternalFrame(session, self._var, index_map, self._data_columns,
                                 self._column_index, self._column_index_names)
        return self._with_where_expr(self._where_expr, new_odf)

    def truncate(self, before=None, after=None, axis=None, copy=True):
        def gettimestr(time):
            if time is None:
                return ""
            p = "0000.01.01T00:00:00.000000000"
            temp = str(time)
            temp = temp + p[len(temp):]

            return temp.replace('-', '.')

        if axis == "columns" or axis == 1:
            return self.loc[:, before:after]

        else:
            if self.index._ddb_dtype not in dolphindb_temporal_types:
                return self.loc[before:after]
            else:
                start_str = gettimestr(before)
                end_str = gettimestr(after)
                where_list = f"between(nanotimestamp({self._var_name}.{self._index_columns[0]}), {start_str}:{end_str})"
                return self._with_where_expr(where_list, self)



    def reorder_levels(self, order, axis=0):
        session = self._session
        if is_dolphindb_scalar(order):
            level = (order,)
        else:
            level = _try_convert_iterable_to_list(order)
            if len(level) > len(set(level)):
                raise IndexError("Cannot support same Index")

        _unsupport_columns_axis(self, axis)
        _, _, _, level_idx = _infer_level(level, self._index_map)

        index_map = [self._index_map[i] for i in level_idx]
        new_odf = _InternalFrame(session, self._var, index_map, self._data_columns,
                                 self._column_index, self._column_index_names)
        return self._with_where_expr(self._where_expr, new_odf)

    def swaplevel(self, i=-2, j=-1, axis=0):
        session = self._session
        level = [i, j]

        _unsupport_columns_axis(self, axis)
        _, _, _, level_idx = _infer_level(level, self._index_map)

        index_map = self._index_map
        temp = index_map[level_idx[0]]
        index_map[level_idx[0]] = index_map[level_idx[1]]
        index_map[level_idx[1]] = temp
        new_odf = _InternalFrame(session, self._var, index_map, self._data_columns,
                                 self._column_index, self._column_index_names)
        return self._with_where_expr(self._where_expr, new_odf)

    def at_time(self, time, asof=False, axis=None):
        timestr = _get_time_str(time)
        where_list = f"minute({self._var_name}.{self._index_columns[0]}) = {timestr}m"
        return self._with_where_expr(where_list, self)

    def between_time(self, start_time, end_time, include_start=True, include_end=True, axis=None):
        if include_start == False or include_end == False:
            raise NotImplementedError("Do not support include")
        start_str = _get_time_str(start_time)
        end_str = _get_time_str(end_time)

        session = self._session
        flag = session.run(f"{start_str}m <= {end_str}m")
        if flag:
            where_list = f"between(minute({self._var_name}.{self._index_columns[0]}), {start_str}m:{end_str}m)"
        else:
            where_list = f"not between(minute({self._var_name}.{self._index_columns[0]}), {end_str}m:{start_str}m)"

        return self._with_where_expr(where_list, self)

    def equals(self, other):
        #list_self = [f"{self._var_name}.{x}" for x in self._data_columns]
        #list_other = [f"{other._var_name}.{x}" for x in other._data_columns]
        script = "eqObj(["
        for i in range(0, len(self._data_columns)-1):
            script = script + f"{self._var_name}.{self._data_columns[i]}, "
        script = script + f"{self._var_name}.{self._data_columns[len(self._data_columns)-1]}], ["
        for i in range(0, len(other._data_columns)-1):
            script = script + f"{other._var_name}.{other._data_columns[i]}, "
        script = script + f"{other._var_name}.{other._data_columns[len(self._data_columns) - 1]}])"
        #script = f"eqObj( {to_dolphindb_literal(list_self)} , {to_dolphindb_literal(list_other)})"
        #script = f"each(eqObj, {self._var_name}.values(), {other._var_name}.values())"
        #script = f"each(eqObj, {self._var_name}.{self._data_columns}.values(), {other._var_name}.{other._data_columns}.values())"

        res = self._session.run(script)
        if res:
            return True
        else:
            return False
