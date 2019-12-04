import abc
import itertools
import warnings
from typing import Iterable

import dolphindb as ddb
import numpy as np
import pandas as pd
import pandas.plotting._core as gfx

from .accessor import CachedAccessor
from .common import CopiedTableWarning
from .internal import _ConstantSP, _InternalAccessor, _InternalFrame
from .operator import (ArithOpsMixin, BooleanExpression,
                                LogicalOpsMixin, StatOpsMixin)
from .utils import (_merge_where_expr, check_key_existence, _to_data_column,
                             get_orca_obj_from_script, is_dolphindb_integral,
                             is_dolphindb_scalar, is_dolphindb_vector,
                             sql_select, to_dolphindb_literal,sql_update)


class _Frame(_InternalAccessor, ArithOpsMixin, LogicalOpsMixin, StatOpsMixin, metaclass=abc.ABCMeta):

    """
    The base class for both DataFrame and Series.
    """

    # def __init__(self, session, internal, index):
    def __init__(self, internal, session, index=None, copy=False):
        from .indexes import Index
        self._internal = internal
        self._session = session
        self._where_expr = None
        self._is_snapshot = False
        self._name = None
        self._index = Index._from_internal(internal, index)

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
        return self._with_where_expr(where_expr, self._internal)

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
                data_column = data_columns[0]
                script = f"{var._var_name}['{data_column}', 0]"
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
        if obj.is_series_like:
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

    # def __getattr__(self, name):
    #     try:
    #         return self.__getitem__(name)
    #     except KeyError:
    #         raise AttributeError(
    #             f"'{self.__class__.__name__}' object has no attribute '{name}'")
    #     except RuntimeError:
    #         return object.__getattribute__(self, name)

    def __len__(self):
        if self._where_expr is None:
            return len(self._internal)
        else:
            script = sql_select(["count(*)"], self._var_name, self._where_expr, is_exec=True)
            return self._session.run(script)
    
    def to_pandas(self):    # TODO: optimize
        data = self._session.run(self._to_script())
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
        # print(script)    # TODO: debug info
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
        name = self._name if self.is_series_like else None
        return self._get_from_script(session, script, self, index_map=index_map,
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

    @abc.abstractproperty
    def iloc(self):
        pass

    @abc.abstractproperty
    def is_dataframe_like(self):
        pass

    @abc.abstractproperty
    def is_series_like(self):
        pass

    @property
    def array(self):
        return self.to_numpy().tolist()

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


    @staticmethod
    def _generate_joiner(left_var_name, right_var_name,
                         left_join_columns, right_join_columns, 
                         how="outer", sort=False, use_right_index=False):
        """
        Returns the index_list and from_clause used in binary operations on
        Frames with different indices.
        """
        if len(left_join_columns) != len(right_join_columns):
            raise ValueError("cannot join with no overlapping index names")
        if sort:
            methods = {"left_semi": "slsj", "left": "slj", "right": "slj",
                       "outer": "sfj", "inner": "sej"}
        else:
            methods = {"left_semi": "lsj", "left": "lj", "right": "lj",
                       "outer": "fj", "inner": "ej", "asof": "aj"}
        method = methods.get(how, None)
        if method is None:
            raise ValueError(f"do not recognize join method {how}")

        left_join_literal = to_dolphindb_literal(left_join_columns)
        right_join_literal = to_dolphindb_literal(right_join_columns)
        if how == "right":
            index_list = (f"tmp2.{col} as {col}" for col in right_join_columns)
        elif how in ("left", "left_semi"):
            index_list = (f"tmp1.{col} as {col}" for col in left_join_columns)
        else:
            index_list = (
                f"iif(isValid(tmp1.{left_col}), "
                f"tmp1.{left_col}, "
                f"tmp2.{right_col}) "
                f"as {right_col if use_right_index else left_col}"
                for left_col, right_col
                in zip(left_join_columns, right_join_columns)
            )
        if how == "right":
            from_clause = f"{method}({right_var_name} as tmp2, " \
                          f"{left_var_name} as tmp1, " \
                          f"{right_join_literal}, {left_join_literal})"
        else:
            from_clause = f"{method}({left_var_name} as tmp1, " \
                          f"{right_var_name} as tmp2, " \
                          f"{left_join_literal}, {right_join_literal})"
        return index_list, from_clause

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

    # def _get_script_with_binary_op_on_table_like_and_constant(self, other_name, func, reversed):
    #     select_list = ((f"{func}({col},{other_name}) as {col}"
    #                     if not reversed
    #                     else f"{func}({other_name},{col}) as {col}")
    #                    for col in self._column_names)
    #     from_clause = self._var_name
    #     return sql_select(select_list, from_clause, is_exec=True)

    # def _get_script_with_binary_op_on_dataframe_and_constant(self, other, func, reversed=False):
    #     assert self.is_dataframe_like
    #     assert isinstance(other, _ConstantSP)
    #     other_name, other_form = other._var_name, other.form
    #     if other_form == ddb.settings.DF_SCALAR:
    #         return self._get_script_with_binary_op_on_table_like_and_constant(other_name, func, reversed)
    #     elif other_form == ddb.settings.DF_VECTOR:
    #         raise NotImplementedError()
    #     else:
    #         raise TypeError("Parameter must be a DolphinDB scalar or vector")

    # def _get_script_with_binary_op_on_series_and_constant(self, other, func, reversed=False):
    #     assert self.is_series_like
    #     assert isinstance(other, _ConstantSP)
    #     self_name, other_name = self._var_name, other._var_name
    #     if self.is_table_like:
    #         return self._get_script_with_binary_op_on_table_like_and_constant(other_name, func, reversed)
    #     elif not reversed:
    #         return f"{func}({self_name},{other_name})"
    #     else:
    #         return f"{func}({other_name},{self_name})"

    # def _get_script_with_binary_op_on_series_and_series(self, other, func):
    #     assert self.is_series_like
    #     assert other.is_series_like
    #     raise NotImplementedError()

    # def _get_script_with_binary_op_on_dataframe_and_series(self, other, func):
    #     if self.is_dataframe_like and other.is_series_like:
    #         raise NotImplementedError()
    #     elif self.is_series_like and other.is_dataframe_like:
    #         raise NotImplementedError()

    # def _get_script_with_binary_op_on_dataframe_and_dataframe(self, other, func):
    #     assert self.is_dataframe_like
    #     assert other.is_dataframe_like
    #     raise NotImplementedError()

    # def _get_series_script_within_sql(self):
    #     odf = self._internal
    #     var_name, column_name = self._var_name, odf.column_names[0]
    #     return f"{var_name}.{column_name}"
    
    def _get_data_select_list(self):
        return self._internal.data_select_list

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

    @property
    def empty(self):
        if self.size > 0:
            return False
        else:
            return True

    @property
    def is_copy(self):
        return None

    def where(self, cond, other=np.nan, inplace=False, axis=None, level=None, errors="raise", try_cast=False):
        # TODO: support inplace

        if callable(cond) or callable(other) or axis !=None or level !=None:
            return self.to_pandas().where(
                cond, other, inplace, axis, level, errors=errors, try_cast=try_cast)
            # TODO: if inplace, how to change data
        elif isinstance(cond, BooleanExpression):
            cond_script = " and ".join(cond.to_where_list())
            if isinstance(other, _Frame):
                select_list = [f"iif({cond_script},{self_col},{other_col}) as {self_col}"
                               for self_col, other_col in zip(self._data_columns, other._data_columns)]
                update_value = [f"iif({cond_script},{self_col},{other_col})"
                                for self_col, other_col in zip(self._data_columns, other._data_columns)]
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
        elif isinstance(cond, _Frame) or is_dolphindb_vector(cond):
            if is_dolphindb_vector(cond):
                from .series import Series
                cond = Series(cond)

            cond_var_name = cond._var_name
            if isinstance(other, _Frame):
                select_list = [f"iif({cond_var_name}.{cond_col},{self_col},{other_col}) as {self_col}"
                               for cond_col, self_col, other_col in zip(cond._data_columns, self._data_columns, other._data_columns)]
                update_value = [f"iif({cond_var_name}.{cond_col},{self_col},{other_col})"
                                for cond_col, self_col, other_col in zip(cond._data_columns, self._data_columns, other._data_columns)]
            elif other is None or other is np.nan or other is float('nan'):
                select_list = [f"iif({cond_var_name}.{cond_col},{self_col},{self._ddb_dtypestr[self_col]}(NULL)) as {self_col}"
                               for cond_col, self_col in zip(cond._data_columns, self._data_columns)]
                update_value = [f"iif({cond_var_name}.{cond_col},{self_col},{self._ddb_dtypestr[self_col]}(NULL))"
                                for cond_col, self_col in zip(cond._data_columns, self._data_columns)]
            else:
                select_list = [f"iif({cond_var_name}.{cond_col},{self_col},{other}) as {self_col}"
                               for cond_col, self_col in zip(cond._data_columns, self._data_columns)]
                update_value = [f"iif({cond_var_name}.{cond_col},{self_col},{other})"
                                for cond_col, self_col in zip(cond._data_columns, self._data_columns)]
            return self._execute_select_and_update(select_list, update_value, inplace, errors)
        else:
            raise ValueError(f"unsupport cond type {type(cond)}")

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
