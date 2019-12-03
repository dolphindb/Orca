import abc
import itertools
from typing import Iterable

import dolphindb as ddb
from pandas.tseries.frequencies import to_offset

from .indexes import DatetimeIndex
from .internal import _InternalAccessor
from .operator import DataFrameLike, SeriesLike, WindowOpsMixin, EwmOpsMixin
from .utils import (
    _scale_nanos, check_key_existence, dolphindb_numeric_types,
    dolphindb_temporal_types, get_orca_obj_from_script, sql_select,
    to_dolphindb_literal)


class Rolling(_InternalAccessor, WindowOpsMixin, metaclass=abc.ABCMeta):

    _ROLLING_COLUMN = "ORCA_ROLLING_COLUMN"

    def __init__(self, session, internal, index, window, on, where_expr, name):
        self._session = session
        self._internal = internal
        self._index = index
        self._on = on
        self._where_expr = where_expr
        self._name = name

        if on is not None:
            check_key_existence(on, self._data_columns)
        if isinstance(window, int):
            self._window = window
            self._rolling_on_temporal = False
        else:
            self._window = self._match_offset(window)
            if on is not None:
                if self._ddb_dtypes[on] not in dolphindb_temporal_types:
                    raise ValueError("window must be an integer")
            elif not isinstance(self._index, DatetimeIndex):
                raise ValueError("window must be an integer")
            self._rolling_on_temporal = True

    @abc.abstractproperty
    def is_series_like(self):
        pass

    @abc.abstractproperty
    def is_dataframe_like(self):
        pass

    @property
    def _index_column(self):
        return self._index_columns[0]

    def __getitem__(self, key):
        if isinstance(key, str):
            klass = SeriesRolling
            name = key
        elif isinstance(key, Iterable):
            klass = DataFrameRolling
            name = self._name
        else:
            raise KeyError(key)
        new_odf = self._internal[key]
        return klass(self._session, new_odf, self._index, self._window, self._on, self._where_expr, name)

    def _match_offset(self, rule):
        offset = to_offset(rule)
        offset_fixed = offset - offset.base
        on = self._on
        if on is None:
            col = self._index_column
        else:
            col = on
        ddb_dtype = self._ddb_dtypes[col]
        nanos = _scale_nanos(offset_fixed.nanos, ddb_dtype)
        return nanos

    def _cols_with_and_without_on(self, numeric_only):
        on, ddb_dtypes, data_columns = self._on, self._ddb_dtypes, self._data_columns
        cols_with_on = [col for col in data_columns
                        if (not numeric_only
                            or ddb_dtypes[col] in dolphindb_numeric_types
                            or col == on)]
        cols = [col for col in cols_with_on if col != on]
        return cols, cols_with_on

    def _window_op(self, func, numeric_only, use_moving_template):
        if self._rolling_on_temporal:
            data = self._window_op_on_temporal(func, numeric_only)
        else:
            data = self._window_op_on_non_temporal(func, numeric_only, use_moving_template)
        if self.is_series_like:
            column = data._data_columns[0]
            return data[column]
        else:
            return data

    def _window_op_on_non_temporal(self, func, numeric_only, use_moving_template):
        on = self._on
        _, cols_with_on = self._cols_with_and_without_on(numeric_only)
        if use_moving_template:
            gen_moving_col = "moving({func}, {col}, {window}) as {col}"
        else:
            gen_moving_col = "m{func}({col}, {window}) as {col}"
        select_list = (gen_moving_col.format(func=func, col=col, window=self._window)
                       if col != on else col
                       for col in cols_with_on)
        select_list = itertools.chain(self._index_columns, select_list)
        script = sql_select(select_list, self._var_name, self._where_expr)
        return get_orca_obj_from_script(
            self._session, script, self._index_map, name=self._name)

    def _window_op_on_temporal(self, func, numeric_only):
        on = self._on
        cols, cols_with_on = self._cols_with_and_without_on(numeric_only)
        agg_list = [f"{func}({col}) as {self._ROLLING_COLUMN}_{col}" for col in cols]
        agg_script = "<[" + ",".join(agg_list) + "]>"
        if self._where_expr is not None:
            raise NotImplementedError()
        var_name = self._var_name
        window = self._window
        on_literal = to_dolphindb_literal(self._on or self._index_column)
        from_clause = f"wj({var_name}, {var_name}, -{window}:0, {agg_script}, {on_literal})"
        select_list = (f"{self._ROLLING_COLUMN}_{col} as {col}"
                       if col != on else col
                       for col in cols_with_on)
        select_list = itertools.chain(self._index_columns, select_list)
        script = sql_select(select_list, from_clause)
        # print(script)    # TODO: debug info
        return get_orca_obj_from_script(
            self._session, script, self._index_map, name=self._name)

class DataFrameRolling(DataFrameLike, Rolling):

    pass


class SeriesRolling(SeriesLike, Rolling):

    pass


class Ewm(_InternalAccessor, EwmOpsMixin, metaclass=abc.ABCMeta):

    def __init__(self, session, internal, index, com, span, halflife, alpha, min_periods, adjust, ignore_na, where_expr):
        self._session = session
        self._internal = internal
        self._index = index

        if com is None:
            com = ""
        elif com < 0:
            raise ValueError("com must greater than 0")
        if span is None:
            span = ""
        elif span < 1:
            raise ValueError("span must greater than 1")
        if halflife is None:
            halflife = ""
        elif halflife <= 0:
            raise ValueError("halflife must greater than 0")
        if alpha is None:
            alpha = ""
        elif alpha < 0 or alpha > 1:
            raise ValueError("alpha must between 0 and 1")
        if adjust:
            adjust = "true"
        else:
            adjust = "false"
        if ignore_na:
            ignore_na = "true"
        else:
            ignore_na = "false"
        self._com = com
        self._span = span
        self._halflife = halflife
        self._alpha = alpha
        self._min_periods = min_periods
        self._adjust = adjust
        self._ignore_na = ignore_na
        self._where_expr = where_expr

    def __getitem__(self, key):
        if isinstance(key, str):
            klass = SeriesEwm
        elif isinstance(key, Iterable):
            klass = DataFrameEwm
        else:
            raise KeyError(key)
        new_odf = self._internal[key]
        return klass(self._session, new_odf, self._index, self._com, self._span, self._halflife, self._alpha, self._min_periods, self._adjust, self._ignore_na)

    def _cols_with_and_without_on(self, numeric_only):
        on, ddb_dtypes, data_columns = None, self._ddb_dtypes, self._data_columns
        cols_with_on = [col for col in data_columns
                        if (not numeric_only
                            or ddb_dtypes[col] in dolphindb_numeric_types
                            or col == on)]
        cols = [col for col in cols_with_on if col != on]
        return cols, cols_with_on

    def _ewm_op(self, func):
        _, cols_with_on = self._cols_with_and_without_on(True)
        gen_moving_col = "{func}({col}, {com}, {span}, {halflife}, {alpha}, {min_periods}, {adjust}, {ignore_na}) as {col}"
        select_list = (gen_moving_col.format(func=func, col=col, com=self._com, span=self._span,halflife=self._halflife, alpha=self._alpha, min_periods=self._min_periods, adjust=self._adjust, ignore_na=self._ignore_na) for col in cols_with_on)

        select_list = itertools.chain(self._index_columns, select_list)

        script = sql_select(select_list, self._var_name, self._where_expr)

        data = get_orca_obj_from_script(
            self._session, script, self._index_map, name=None)

        if self.is_series_like:
            column = data._data_columns[0]
            return data[column]
        else:
            return data

    @property
    def _index_column(self):
        return self._index_columns[0]

    @abc.abstractproperty
    def is_series_like(self):
        pass

    @abc.abstractproperty
    def is_dataframe_like(self):
        pass



class DataFrameEwm(DataFrameLike, Ewm):

    pass


class SeriesEwm(SeriesLike, Ewm):

    pass
