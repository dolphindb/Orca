import abc
import itertools
import warnings
from enum import Enum
from typing import Callable, Iterable

import dolphindb as ddb
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from .common import _raise_must_compute_error, _warn_apply_callable
from .datetimes import Timestamp
from .internal import _ConstantSP, _InternalAccessor, _InternalFrame
from .utils import (ORCA_INDEX_NAME_FORMAT, _infer_axis, _infer_level,
                    _merge_where_expr, _scale_nanos,
                    _try_convert_iterable_to_list, _unsupport_columns_axis,
                    check_key_existence, dolphindb_numeric_types,
                    dolphindb_temporal_types, get_orca_obj_from_script,
                    is_dolphindb_scalar, is_dolphindb_uploadable,
                    is_dolphindb_vector, sql_select, to_dolphindb_literal)


def _default_axis(obj):
    return 0 if obj._is_series_like else 1


def _orca_unary_op(func, numeric_only):
    def ufunc(self):
        return self._unary_op(func, numeric_only)
    return ufunc


def _orca_unary_agg_op(func, numeric_only):
    def ufunc(self, axis=0, level=None):
        return self._unary_agg_op(func, axis, level, numeric_only)
    return ufunc


def _orca_binary_op(func):
    def bfunc(self, other):
        return self._binary_op(other, func)
    return bfunc


def _orca_binary_agg_op(func):
    def bfunc(self, other):
        return self._binary_agg_op(other, func)
    return bfunc


def _orca_reversed_binary_op(func):
    def rbfunc(self, other):
        return type(self)._binary_op(other, self, func)
    return rbfunc


def _orca_extended_binary_op(func):
    def bfunc(self, other, axis=None, level=None, fill_value=None):
        if axis is None:
            axis = _default_axis(self)
        return self._extended_binary_op(other, func, axis, level, fill_value)
    return bfunc


def _orca_extended_reversed_binary_op(func):
    def rbfunc(self, other, axis=None, level=None, fill_value=None):
        if axis is None:
             axis = _default_axis(self)
        return type(self)._extended_binary_op(other, self, func, axis, level, fill_value)
    return rbfunc


def _orca_logical_op(func):
    def lfunc(self, other):
        return self._logical_op(other, func)
    return lfunc


def _orca_logical_unary_op(func):
    def lufunc(self):
        return self._logical_unary_op(func)
    return lufunc


def _orca_ewm_op(func):
    def efunc(self, bias=False):
        return self._ewm_op(func, bias)
    return efunc


def _orca_ewm_covcorr(func):
    def efunc(self, other=None, bias=False, method='pearson', min_periods=1):
        return self._ewm_covcorr(func, other, bias, method, min_periods)
    return efunc


def _binary_op_compute(session, left, right, func, axis):
    left_obj = left.compute()
    right_obj = right.compute()
    return left_obj._binary_op_on_different_indices(right_obj, func, axis)


def _get_numeric_only_columns_ref(obj, numeric_only, data_columns=None):
    data_columns = data_columns or obj._data_columns
    if numeric_only:
        ddb_dtypes = obj._ddb_dtypes
        data_columns = [col for col in data_columns
                        if ddb_dtypes[col] in dolphindb_numeric_types]

    if data_columns == obj._data_columns:
        return obj
    else:
        return obj[data_columns]


def _get_expr_with_binary_op(ExpressionType, left, right, func):
    from .indexes import Index, MultiIndex, DatetimeIndex
    from .frame import DataFrame
    from .series import Series
    from .merge import MergeExpression

    if (isinstance(left, (pd.DataFrame, pd.Series, pd.Index))
        or isinstance(right, (pd.DataFrame, pd.Series, pd.Index))):
        raise TypeError("Unable to run a binary operation on an orca object and a pandas object")
    elif isinstance(left, MultiIndex) or isinstance(right, MultiIndex):
        raise TypeError(f"cannot perform {func} with index type: MultiIndex")
    elif is_dolphindb_uploadable(left):
        if isinstance(left, (str, pd.Timestamp)):
            try:
                left_obj, right_obj = Timestamp(left, session=right._session), right
            except:
                left_obj, right_obj = _ConstantSP.upload_obj(right._session, left), right
        else:
            left_obj, right_obj = _ConstantSP.upload_obj(right._session, left), right
        axis = _default_axis(right)
    elif isinstance(left, (_ConstantSP, Timestamp)):
        left_obj, right_obj = left, right
        axis = _default_axis(right)
    elif is_dolphindb_uploadable(right):
        if isinstance(right, (str, pd.Timestamp)):
            try:
                left_obj, right_obj = left, Timestamp(right, session=left._session)
            except:
                left_obj, right_obj = left, _ConstantSP.upload_obj(left._session, right)
        else:
            left_obj, right_obj = left, _ConstantSP.upload_obj(left._session, right)
        axis = _default_axis(left)
    elif isinstance(right, (_ConstantSP, Timestamp)):
        left_obj, right_obj = left, right
        axis = _default_axis(left)
    elif isinstance(left, MergeExpression) or isinstance(right, MergeExpression):
        if ((isinstance(left, MergeExpression)
                and isinstance(right, MergeExpression)
                and left._from_clause != right._from_clause)
            or (isinstance(left, MergeExpression)
                    and not isinstance(right, MergeExpression)
                    and left._left_var_name != right._var_name
                    and left._right_var_name != right._var_name)
            or (isinstance(right, MergeExpression)
                    and not isinstance(left, MergeExpression)
                    and left._var_name != right._left_var_name
                    and left._var_name != right._right_var_name)):
            raise ValueError("left and right object must be the same MergeExpression")
    elif (not isinstance(left, (DataFrame, Series, ArithExpression, BooleanExpression))
          or not isinstance(right, (DataFrame, Series, ArithExpression, BooleanExpression))):
         raise TypeError(f"Unsupported operand types: '{left.__class__.__name__}' "
                         f"and '{right.__class__.__name__}'") 
    elif (left._index_map != right._index_map
          or left._var_name != right._var_name
          or left._where_expr is not right._where_expr):
        axis = _default_axis(left)
        return _binary_op_compute(left._session, left, right, func, axis)
    else:
        left_obj, right_obj = left, right
        axis = _default_axis(left)
    return ExpressionType(left_obj, right_obj, func, axis)


def _check_rank_arguments(axis, method, na_option, ascending, pct, rank_from_zero, group_num):
    _unsupport_columns_axis(None, axis)

    if method != "min":
        raise ValueError("method must be 'min'")
    if na_option != "top":
        raise ValueError("na_option must be 'top'")
    if pct:
        raise NotImplementedError()

    asc = "true" if ascending else "false"
    if group_num is not None:
        if not isinstance(group_num, int):
            raise TypeError("group_num must be an integer")
        elif group_num < 0:
            raise ValueError("group_num must be greater than 0")

    if rank_from_zero:
        if group_num is not None:
            func = f"rank{{,{asc},{group_num}}}"
        else:
            func = f"rank{{,{asc}}}"
    else:
        if group_num is not None:
            func = f"(x->rank(x,{asc},{group_num})+1)"
        else:
            func = f"(x->rank(x,{asc})+1)"
    return func


class DataFrameLike(object):

    @property
    def _is_index_like(self):
        return False

    @property
    def _is_series_like(self):
        return False

    @property
    def _is_dataframe_like(self):
        return True


class SeriesLike(object):

    @property
    def _is_index_like(self):
        return False

    @property
    def _is_series_like(self):
        return True

    @property
    def _is_dataframe_like(self):
        return False


class IndexLike(object):
    
    @property
    def _is_series_like(self):
        return True

    @property
    def _is_dataframe_like(self):
        return False

    @property
    def _is_index_like(self):
        return True


class ArithOpsMixin(metaclass=abc.ABCMeta):

    add = _orca_extended_binary_op("add")
    sub = _orca_extended_binary_op("sub")
    mul = _orca_extended_binary_op("mul")
    div = _orca_extended_binary_op("ratio")
    truediv = div
    floordiv = _orca_extended_binary_op("div")
    mod = _orca_extended_binary_op("mod")
    pow = _orca_extended_binary_op("pow")

    radd = _orca_extended_reversed_binary_op("add")
    rsub = _orca_extended_reversed_binary_op("sub")
    rmul = _orca_extended_reversed_binary_op("mul")
    rdiv = _orca_extended_reversed_binary_op("ratio")
    rtruediv = rdiv
    rfloordiv = _orca_extended_reversed_binary_op("div")
    rmod = _orca_extended_reversed_binary_op("mod")
    rpow = _orca_extended_reversed_binary_op("pow")

    lshift = _orca_extended_binary_op("lshift")
    rshift = _orca_extended_binary_op("rshift")
    rlshift = _orca_extended_reversed_binary_op("lshift")
    rrshift = _orca_extended_reversed_binary_op("rshift")

    __add__ = _orca_binary_op("add")
    __sub__ = _orca_binary_op("sub")
    __mul__ = _orca_binary_op("mul")
    __div__ = _orca_binary_op("ratio")
    __truediv__ = __div__
    __floordiv__ = _orca_binary_op("div")
    __mod__ = _orca_binary_op("mod")
    __pow__ = _orca_binary_op("pow")

    __radd__ = _orca_reversed_binary_op("add")
    __rsub__ = _orca_reversed_binary_op("sub")
    __rmul__ = _orca_reversed_binary_op("mul")
    __rdiv__ = _orca_reversed_binary_op("ratio")
    __rtruediv__ = __rdiv__
    __rfloordiv__ = _orca_reversed_binary_op("div")
    __rmod__ = _orca_reversed_binary_op("mod")
    __rpow__ = _orca_reversed_binary_op("pow")

    __lshift__ = _orca_binary_op("lshift")
    __rshift__ = _orca_binary_op("rshift")
    __rlshift__ = _orca_reversed_binary_op("lshift")
    __rrshift__ = _orca_reversed_binary_op("rshift")

    __neg__ = _orca_unary_op("neg", numeric_only=False)
    
    def __pos__(self):
        return self

    def __divmod__(self, other):
        return self // other, self % other

    def __rdivmod__(self, other):
        return other // self, other % self

    def divmod(self, other, axis=None, level=None, fill_value=None):
        return (self.floordiv(other, axis=axis, level=level, fill_value=fill_value),
                self.mod(other, axis=axis, level=level, fill_value=fill_value))

    def rdivmod(self, other, axis=None, level=None, fill_value=None):
        return (self.rfloordiv(other, axis=axis, level=level, fill_value=fill_value),
                self.rmod(other, axis=axis, level=level, fill_value=fill_value))

    @abc.abstractmethod
    def _binary_op(self, other, func):
        return _get_expr_with_binary_op(ArithExpression, self, other, func)

    @abc.abstractmethod
    def _extended_binary_op(self, other, func, axis, level, fill_value):
        if is_dolphindb_uploadable(self):    # pandas.DataFrame is also uploadable
            axis = 0
        else:
            axis = _infer_axis(self, axis)
        if fill_value is None:
            if is_dolphindb_uploadable(self):
                return type(other)._binary_op(self, other, func)
            elif is_dolphindb_uploadable(other):
                return type(self)._binary_op(self, other, func)
            if (axis == 0
                and (self._is_series_like or self._is_dataframe_like)
                and other._is_series_like):
                return self._binary_op(other, func)
            elif (axis == 1
                  and self._is_dataframe_like
                  and other._is_dataframe_like):
                return self._binary_op(other, func)
            else:
                raise NotImplementedError()
        else:
            if is_dolphindb_uploadable(self):
                return type(other)._binary_op(self, other.fillna(fill_value), func)
            elif is_dolphindb_uploadable(other):
                return type(self)._binary_op(self.fillna(fill_value), other, func)
            if (axis == 0
                and (self._is_series_like or self._is_dataframe_like)
                and other._is_series_like):
                return self.fillna(fill_value)._binary_op(other.fillna(fill_value), func)
            elif (axis == 1
                  and self._is_dataframe_like
                  and other._is_dataframe_like):
                return self.fillna(fill_value)._binary_op(other.fillna(fill_value), func)
            else:
                raise NotImplementedError()
    
    @abc.abstractmethod
    def _unary_op(self, func, numeric_only):
        return StatOpsMixin._unary_op(self, func, numeric_only)


class LogicalOpsMixin(metaclass=abc.ABCMeta):

    lt = _orca_logical_op("lt")
    gt = _orca_logical_op("gt")
    le = _orca_logical_op("le")
    ge = _orca_logical_op("ge")
    eq = _orca_logical_op("eq")
    ne = _orca_logical_op("ne")

    __lt__ = lt
    __gt__ = gt
    __le__ = le
    __ge__ = ge
    __eq__ = eq
    __ne__ = ne
    __and__ = _orca_logical_op("and")           # logical_and
    __or__ = _orca_logical_op("or")             # logical_or
    __xor__ = _orca_logical_op("bitXor")        # bitwise_xor
    __invert__ =_orca_logical_unary_op("not")   # logical_not

    bitwise_and = _orca_logical_op("bitAnd")
    bitwise_or = _orca_logical_op("bitOr")
    bitwise_not = _orca_logical_unary_op("bitNot")
    logical_xor = _orca_logical_op("xor")

    isnull = _orca_logical_unary_op("isNull")
    notnull = _orca_logical_unary_op("isValid")
    isna = isnull
    notna = notnull

    def isin(self, values):
        from orca.core.series import Series
        from orca.core.indexes import Index, MultiIndex
        # TODO: this operator does not care about indexing
        if isinstance(values, dict):
            raise NotImplementedError()
        elif is_dolphindb_vector(values):
            return self._logical_op(values, "in")
        elif isinstance(values, set):
            return self._logical_op(list(values), "in")
        elif isinstance(values, (Series, Index)):
            if values._segmented:
                raise ValueError("values cannot be a segmented Series or Index")
            if isinstance(values, MultiIndex):
                raise ValueError("values cannot be a MultiIndex")
            return BooleanExpression(self, values, "in", 0)
        else:
            raise TypeError("only list-like objects are allowed to be passed to isin()")

    def duplicated(self, subset=None, keep="first"):
        if subset is not None and self._is_series_like:
            raise TypeError("duplicated() got an unexpected keyword argument 'subset'")

        data_columns = self._data_columns
        if subset is None:
            subset = data_columns
        else:
            subset, _ = check_key_existence(subset, data_columns)

        if keep == "first":
            keep_rule = "FIRST"
        elif keep == "last":
            keep_rule = "LAST"
        elif keep == False:
            keep_rule = "NONE"
        else:
            raise ValueError('keep must be either "first", "last" or False')
        subset_script = ",".join(subset)
        func = f"isDuplicated(({subset_script}), {keep_rule})"
        return self._logical_unary_op(func)

    def between(self, left, right, inclusive=True):
        if self._is_dataframe_like:
            raise AttributeError("'DataFrame' object has no attribute 'between'")
        if not inclusive:
            raise NotImplementedError()
        left_is_scalar = is_dolphindb_scalar(left)
        right_is_scalar = is_dolphindb_scalar(right)
        session = self._session
        if left_is_scalar and right_is_scalar:
            if isinstance(left, (str, pd.Timestamp)) and isinstance(right, (str, pd.Timestamp)):
                try:
                    left_dt = Timestamp(left, session=session)
                    right_dt = Timestamp(right, session=session)
                    if getattr(self, "_infered_ddb_dtypestr", None) is not None:
                        typestr = self._infered_ddb_dtypestr
                    else:
                        typestr = self._ddb_dtypestr[self._data_columns[0]]
                    pair = _ConstantSP.run_script(session, f"{typestr}({left_dt._var_name}):{typestr}({right_dt._var_name})")
                except:
                    left_script = to_dolphindb_literal(left)
                    right_script = to_dolphindb_literal(right)
                    pair = _ConstantSP.run_script(session, f"{left_script}:{right_script}")
            else:
                left_script = to_dolphindb_literal(left)
                right_script = to_dolphindb_literal(right)
                pair = _ConstantSP.run_script(session, f"{left_script}:{right_script}")
        elif isinstance(left, Timestamp) and isinstance(right, Timestamp):
            if getattr(self, "_infered_ddb_dtypestr", None) is not None:
                typestr = self._infered_ddb_dtypestr
            else:
                typestr = self._ddb_dtypestr[self._data_columns[0]]
            pair = _ConstantSP.run_script(session, f"{typestr}({left._var_name}):{typestr}({right._var_name})")
        else:
            raise NotImplementedError()    # TODO: between two vectors
        return self._logical_op(pair, "between")

    @abc.abstractmethod
    def _logical_op(self, other, func):
        return _get_expr_with_binary_op(BooleanExpression, self, other, func)

    @abc.abstractmethod
    def _logical_unary_op(self, func):
        return BooleanExpression(self, None, func, 0)


class StatOpsMixin(metaclass=abc.ABCMeta):

    sin = _orca_unary_op("sin", numeric_only=True)
    cos = _orca_unary_op("cos", numeric_only=True)
    tan = _orca_unary_op("tan", numeric_only=True)
    arcsin = _orca_unary_op("asin", numeric_only=True)
    arccos = _orca_unary_op("acos", numeric_only=True)
    arctan = _orca_unary_op("atan", numeric_only=True)
    exp = _orca_unary_op("exp", numeric_only=True)
    log = _orca_unary_op("log", numeric_only=True)
    floor = _orca_unary_op("floor", numeric_only=True)
    ceil = _orca_unary_op("ceil", numeric_only=True)
    sqrt = _orca_unary_op("sqrt", numeric_only=True)
    ffill = _orca_unary_op("ffill", numeric_only=False)
    bfill = _orca_unary_op("bfill", numeric_only=False)

    abs = _orca_unary_op("abs", numeric_only=True)

    cumsum = _orca_unary_op("cumsum", numeric_only=True)
    cummax = _orca_unary_op("cummax", numeric_only=False)
    cummin = _orca_unary_op("cummin", numeric_only=False)
    cumprod = _orca_unary_op("cumprod", numeric_only=True)
    pct_change = _orca_unary_op("percentChange", numeric_only=True)

    any = _orca_unary_agg_op("any", numeric_only=False)
    all = _orca_unary_agg_op("all", numeric_only=False)
    count = _orca_unary_agg_op("count", numeric_only=False)
    sum = _orca_unary_agg_op("sum", numeric_only=True)
    sum2 = _orca_unary_agg_op("sum2", numeric_only=True)
    prod = _orca_unary_agg_op("prod", numeric_only=True)
    product = prod
    mean = _orca_unary_agg_op("mean", numeric_only=True)
    median = _orca_unary_agg_op("median", numeric_only=True)
    mode = _orca_unary_agg_op("mode", numeric_only=True)
    min = _orca_unary_agg_op("min", numeric_only=False)
    max = _orca_unary_agg_op("max", numeric_only=False)
    std = _orca_unary_agg_op("std", numeric_only=True)
    var = _orca_unary_agg_op("var", numeric_only=True)
    sem = _orca_unary_agg_op("sem", numeric_only=False)
    first = _orca_unary_agg_op("first", numeric_only=False)
    last = _orca_unary_agg_op("last", numeric_only=False)
    mad = _orca_unary_agg_op("mad", numeric_only=False)

    nunique = _orca_unary_agg_op("nunique", numeric_only=False)

    wavg = _orca_binary_agg_op("wavg")
    wsum = _orca_binary_agg_op("wsum")

    sinh = _orca_unary_op("sinh", numeric_only=True)
    cosh = _orca_unary_op("cosh", numeric_only=True)
    tanh = _orca_unary_op("tanh", numeric_only=True)
    arcsinh = _orca_unary_op("arcsinh", numeric_only=True)
    arccosh = _orca_unary_op("arccosh", numeric_only=True)
    arctanh = _orca_unary_op("arctanh", numeric_only=True)
    deg2rad = _orca_unary_op("deg2rad", numeric_only=True)
    rad2deg = _orca_unary_op("rad2deg", numeric_only=True)

    _ROW_WISE_OPS = {
        "sum": "rowSum",
        "sum2": "rowSum2",
        "max": "rowMax",
        "count": "rowCount",
        "var": "rowVar",
        "mean": "rowAvg",
        "std": "rowStd",
        "min": "rowMin",
    }

    @abc.abstractmethod
    def _unary_op(self, func, numeric_only):
        if isinstance(func, dict):
            data_columns, _ = check_key_existence(func.keys(), self._data_columns)
        elif isinstance(func, list):
            raise NotImplementedError("list-like func not implemented")
        else:
            data_columns = self._data_columns

        ref = _get_numeric_only_columns_ref(self, numeric_only, data_columns)
        return ArithExpression(ref, None, func, 0)

    @abc.abstractmethod
    def _binary_op(self, other, func):
        return _get_expr_with_binary_op(ArithExpression, self, other, func)

    @abc.abstractmethod
    def _unary_agg_op(self, func, axis, level, numeric_only):
        axis = _infer_axis(None, axis)

        select_columns = self._get_data_select_list()
        ddb_dtypes = self._ddb_dtypes
        data_columns = self._data_columns
        index_columns, groupby_list, _, level_idx = _infer_level(level, self._index_map)
    
        if axis == 1:
            row_func = self._ROW_WISE_OPS.get(func)
            if row_func is None:
                raise ValueError(f"Function {func} does not support 'axis = 1'")
            columns = ",".join(
                select_col for select_col, col in zip(select_columns, data_columns)
                if ddb_dtypes[col] in dolphindb_numeric_types
            )
            row_func_script = [f"{row_func}({columns})"]
            select_list = itertools.chain(index_columns, row_func_script)
            script = sql_select(select_list, self._var_name, self._where_expr)
            index_map = [self._index_map[i] for i in level_idx]
            return self._get_from_script(
                self._session, script, index_map=index_map,
                squeeze=True, squeeze_axis=1)
        else:
            select_list = [f"{func}({select_col}) as {col}"
                           for select_col, col in zip(select_columns, data_columns)
                           if not numeric_only or ddb_dtypes[col] in dolphindb_numeric_types]
            script = sql_select(select_list, self._var_name, self._where_expr,
                                groupby_list, is_exec=self._is_series_like)
            name = self._name if self._is_series_like else None
        return get_orca_obj_from_script(
            self._session, script, index_columns, name=name,
            squeeze=True, squeeze_axis=0, as_index=self._is_index_like)

    @abc.abstractmethod
    def _binary_agg_op(self, other, func):
        session = self._session
        if not self._is_series_like or not other._is_series_like:
            raise NotImplementedError()
        self_select_columns = self._get_data_select_list()
        other_select_columns = other._get_data_select_list()
        data_columns = self._data_columns
        select_list = [f"{func}({self_select_col},{other_select_col}) as {col}"
                       for self_select_col, other_select_col, col
                       in zip(self_select_columns, other_select_columns, data_columns)]
        script = sql_select(select_list, self._var_name, self._where_expr,
                            is_exec=self._is_series_like)
        name = self._name if self._is_series_like else None
        return get_orca_obj_from_script(
            session, script, self._index_columns, name=name, squeeze=True, squeeze_axis=0)

    def skew(self, axis=None, skipna=None, level=None, numeric_only=None):
        return self._unary_agg_op("skew{,false}", None, level, False)

    def kurt(self, axis=None, skipna=None, level=None, numeric_only=None):
        return self._unary_agg_op("(x->kurtosis(x,false)-3)", None, level, False)
    
    kurtosis = kurt

    def _cov_corr(self, func, other=None, min_periods=1):
        from .frame import DataFrame
        if min_periods != 1:
            raise NotImplementedError()
        if self._is_series_like and other is None:
            raise TypeError(f"{func}() missing 1 required positional argument: 'other'")
        session = self._session
        if self._is_dataframe_like:
            ref = self.compute()
            ddb_dtypes = ref._ddb_dtypes
            numeric_columns = [col for col in ref._data_columns if ddb_dtypes[col] in dolphindb_numeric_types]
            numeric_columns_literal = to_dolphindb_literal(numeric_columns)
            script = f"pcross({func}, {ref._var_name}[{numeric_columns_literal}]).table()"
            corr_table = _ConstantSP.run_script(session, script)
            corr_var_name = corr_table._var_name
            session.run(f"rename!({corr_var_name}, {numeric_columns_literal}); "
                        f"update {corr_var_name} set {ORCA_INDEX_NAME_FORMAT(0)} = {numeric_columns_literal}")
            corr_table._update_metadata()
            index_map = [(ORCA_INDEX_NAME_FORMAT(0), None)]
            data_columns = numeric_columns
            odf = _InternalFrame(session, corr_table, index_map=index_map, data_columns=data_columns)
            return DataFrame(odf, session=session)
        else:
            return self._binary_agg_op(other, func)

    def corr(self, other=None, method='pearson', min_periods=1):
        if method != "pearson":
            raise ValueError(f"method must be 'pearson', '{method}' was supplied")
        return self._cov_corr("corr", other, min_periods)

    def cov(self, other=None, min_periods=1):
        return self._cov_corr("cov", other, min_periods)

    def round(self, decimals=0):
        if not isinstance(decimals, int):
            raise TypeError("decimals must be an integer")
        return self._unary_op(f"round{{,{decimals}}}", numeric_only=True)

    def diff(self, periods=1, axis=0):
        _unsupport_columns_axis(self, axis)
        if periods != 1:
            raise ValueError("periods must be 1")
        return self._unary_op("deltas", numeric_only=True)

    def rank(self, axis=0, method='min', numeric_only=None, na_option='top', ascending=True, pct=False, rank_from_zero=False, group_num=None):
        func = _check_rank_arguments(axis, method, na_option, ascending, pct, rank_from_zero, group_num)
        return self._unary_op(func, numeric_only=numeric_only)

    def quantile(self, q=0.5, axis=0, numeric_only=True, interpolation='linear'):
        _unsupport_columns_axis(self, axis)
        if interpolation not in ("linear", "lower", "higher", "midpoint", "nearest"):
            raise ValueError("interpolation can only be 'linear', 'lower' 'higher', 'midpoint', or 'nearest'")
        interpolation = to_dolphindb_literal(interpolation)
        if is_dolphindb_scalar(q) and q >= 0 and q <= 1:
            func = f"quantile{{,{q},{interpolation}}}"
            return self._unary_agg_op(func, None, level=None, numeric_only=True)
        elif isinstance(q, Iterable) and all(qi >= 0 and qi <= 1 for qi in q):
            session, ddb_dtypes = self._session, self._ddb_dtypes
            var = _ConstantSP.upload_obj(session, np.array(q))
            # func = f"quantileSeries{{,{var.var_name},{interpolation}}}"
            self_select_columns = self._get_data_select_list()
            data_columns = self._data_columns
            select_list = [f"quantileSeries({self_select_col},{var.var_name},{interpolation}) as {col}"
                           for self_select_col, col
                           in zip(self_select_columns, data_columns)
                           if ddb_dtypes[col] in dolphindb_numeric_types]
            select_list += [f"{var.var_name} as {ORCA_INDEX_NAME_FORMAT(0)}"]
            script = sql_select(select_list, self._var_name, self._where_expr)
            name = self._name if self._is_series_like else None
            index_map = [(ORCA_INDEX_NAME_FORMAT(0), None)]
            return get_orca_obj_from_script(
                session, script, index_map, name=name, squeeze=True, squeeze_axis=1)
        else:
            raise ValueError("percentiles should all be in the interval [0, 1]. Try [0.005 0.03 ] instead.")

    def clip(self, lower=None, upper=None, axis=0, inplace=False, *args, **kwargs):
        _unsupport_columns_axis(self, axis)
        if lower is None and upper is None:
            return self
        if inplace:
            raise NotImplementedError()

        ret = self
        if lower is not None:
            if not is_dolphindb_scalar(lower) and not isinstance(lower, Timestamp):
                raise NotImplementedError()
            ret = ret._binary_op(lower, "max")
        if upper is not None:
            if not is_dolphindb_scalar(upper) and not isinstance(upper, Timestamp):
                raise NotImplementedError()
            ret = ret._binary_op(upper, "min")
        return ret

    def clip_lower(self, threshold, axis=0, inplace=False):
        return self.clip(lower=threshold, axis=axis, inplace=inplace)

    def clip_upper(self, threshold, axis=0, inplace=False):
        return self.clip(upper=threshold, axis=axis, inplace=inplace)

    _fmax = _orca_binary_op("max")
    _fmin = _orca_binary_op("min")

    def idxmax(self, axis=0, skipna=True):
        _unsupport_columns_axis(self, axis)
        index_column = self._index_columns[0]
        return self._unary_agg_op(f"atImax{{, {index_column}}}", None, None, False)

    def idxmin(self, axis=0, skipna=True):
        _unsupport_columns_axis(self, axis)
        index_column = self._index_columns[0]
        return self._unary_agg_op(f"atImin{{, {index_column}}}", None, None, False)

    def drop_duplicates(self, subset=None, keep='first', inplace=False):
        if subset is not None and self._is_series_like:
            raise TypeError("drop_duplicates() got an unexpected keyword argument 'subset'")

        if inplace:
            raise NotImplementedError()

        return self[~self.duplicated(subset=subset, keep=keep)]

    def merge(self, other, *args, **kwargs):
        return self.compute().merge(other, *args, **kwargs)

    def join(self, other, *args, **kwargs):
        return self.compute().join(other, *args, **kwargs)

    # Function application

    def _convert_and_apply_func(self, apply_func, func, **kwargs):
        from .series import Series
        from .frame import DataFrame
        _warn_apply_callable()
        pdf = self.to_pandas().__getattr__(apply_func)(func=func, **kwargs)
        if self._is_series_like:
            return Series(pdf, session=self._session)
        else:
            return DataFrame(pdf, session=self._session)

    def apply(self, func, numeric_only=False, **kwargs):
        if isinstance(func, str):
            _unsupport_columns_axis(self, kwargs.get("axis"))
            return self._unary_op(func, numeric_only=numeric_only)
        else:
            return self._convert_and_apply_func("apply", func, **kwargs)

    def applymap(self, func):
        if self._is_series_like:
            raise AttributeError("'Series' object has no attribute 'applymap'")
        if isinstance(func, str):
            return self._unary_op(func, numeric_only=False)
        else:
            return self._convert_and_apply_func("applymap", func)

    def transform(self, func, axis=0, numeric_only=False, *args, **kwargs):
        if (isinstance(func, str)
                or (isinstance(func, list) and all(isinstance(e, str) for e in func))
                or isinstance(func, dict) and all(isinstance(e, str) for e in func.values())):
            axis = _unsupport_columns_axis(self, axis)
            return self._unary_op(func, numeric_only=numeric_only)
        else:
            return self._convert_and_apply_func("transform", func, axis=axis, *args, **kwargs)

    def aggregate(self, func, axis=0, numeric_only=False, *args, **kwargs):
        if isinstance(func, str):
            return self._unary_agg_op(func, None, level=None, numeric_only=numeric_only)
        else:
            return self._convert_and_apply_func("aggregate", func, **kwargs)

    agg = aggregate

    # Time series-related

    # def asof(self, where, subset=None):
    #     ref = self.compute()
    #     data_columns = ref._data_columns
    #     if subset is not None and ref._is_dataframe_like:
    #         subset, _ = check_key_existence(subset, data_columns)
    #     else:
    #         subset = data_columns
    #     from_clause = "aj({})"
    #     sql_select(data_columns, )

    def shift(self, periods=1, freq=None, axis=0, fill_value=None):
        _unsupport_columns_axis(self, axis)

        if not isinstance(periods, int):
            raise TypeError("periods must be an integer")
        expr = self._unary_op(f"move{{,{periods}}}", numeric_only=False)
        if fill_value is not None:
            return expr.fillna(fill_value)
        else:
            return expr

    def tshift(self, periods=1, freq=None, axis=0):
        from .indexes import DatetimeIndex
        if not isinstance(periods, int):
            raise TypeError("periods must be an integer")
        if not isinstance(self._index, DatetimeIndex):
            raise ValueError("index must be a DatetimeIndex")
        index_column = self._index_columns[0]
        ddb_dtype = self._ddb_dtypes[index_column]
        freq = freq or self._index.freq
        offset = to_offset(freq)
        nanos = _scale_nanos(offset.nanos, ddb_dtype) * periods
        shifted_index = f"add({index_column}, {nanos}) as {index_column}"
        select_list = itertools.chain([shifted_index], self._get_data_select_list())
        script = sql_select(select_list, self._var_name, self._where_expr)
        return self._get_from_script(self._session, script, index_map=self._index_map,
                                     index=self._index, name=self._name)

    def _first_last_valid_index(self, func):
        select_list = (f"{func}({col}) as {col}" for col in self._index_columns)
        where_valid_index = " or ".join(f"isValid({col})" for col in self._index_columns)
        where_list = _merge_where_expr(self._where_expr, where_valid_index)
        script = sql_select(select_list, self._var_name, where_list)
        pdf = self._session.run(script)
        return tuple(pdf.to_numpy()[0])

    def first_valid_index(self):
        return self._first_last_valid_index("first")

    def last_valid_index(self):
        return self._first_last_valid_index("last")

    # Missing data handling
    def dropna(self, axis=0, how='any', thresh=None, subset=None, inplace=False):
        ref = self.compute()
        session = ref._session
        axis = _unsupport_columns_axis(ref, axis)
        if subset is None:
            data_columns = ref._data_columns
        else:
            data_columns, _ = check_key_existence(subset, ref._data_columns)
        if inplace:
            raise NotImplementedError()

        isvalid_iter = (f"isValid({col})" for col in data_columns)
        if thresh is not None:
            if not isinstance(thresh, int):
                raise TypeError("thresh must be an integer")
            isvalid_list = ",".join(isvalid_iter)
            where_list = f"sum([{isvalid_list}]) >= {thresh}"
        elif how == "all":
            where_list = " or ".join(isvalid_iter)
        elif how == "any":
            where_list = isvalid_iter
        else:
            raise ValueError(f"invalid how option: {how}")
        where_list = _merge_where_expr(ref._where_expr, where_list)
        script = sql_select(["*"], ref._var_name, where_list)
        return ref._get_from_script(session, script, ref)

    def fillna(self, value=None, method=None, axis=0, inplace=False, limit=None, downcast=None, **kwargs):
        _unsupport_columns_axis(self, axis)

        if inplace:
            if isinstance(self, (ArithExpression, BooleanExpression)):
                _raise_must_compute_error("Unable to inplace fillna with an Expression")
            filled = self.fillna(value=value, method=method, axis=axis, inplace=False,
                                 limit=limit, downcast=downcast, **kwargs)
            self[self._data_columns] = filled
            return

        if value is None and method is None:
            raise ValueError("Must specify a fill 'value' or 'method'.")
        if value is not None and method is not None:
            raise ValueError("Cannot specify both 'value' and 'method'.")
        if limit is not None:
            if not isinstance(limit, int):
                raise TypeError("limit must be an integer")
            elif limit <= 0:
                raise ValueError("limit must be greater than 0")
        if method is None:
            if is_dolphindb_scalar(value):
                value_literal = to_dolphindb_literal(value)
                return self._unary_op(f"nullFill{{,{value_literal}}}", numeric_only=False)
            else:
                raise NotImplementedError()    # TODO: fillna with dict
        elif method in ["backfill", "bfill"]:
            if limit is not None:
                return self._unary_op(f"bfill{{,{limit}}}", numeric_only=False)
            else:
                return self._unary_op(f"bfill", numeric_only=False)
        elif method in ["pad", "ffill"]:
            if limit is not None:
                return self._unary_op(f"ffill{{,{limit}}}", numeric_only=False)
            else:
                return self._unary_op(f"ffill", numeric_only=False)
        else:
            raise ValueError(f"Invalid fill method. Expecting pad (ffill) or backfill (bfill). Got {method}")

    def interpolate(self, method='linear', axis=0, limit=None, inplace=False,
                    limit_direction='forward', limit_area=None, downcast=None, **kwargs):
        _unsupport_columns_axis(self, axis)

        if inplace:
            raise NotImplementedError()

        if method not in ("linear", "time", "index", "values", "pad", "nearest",
                          "zero", "slinear", "quadratic", "cubic", "spline",
                          "barycentric", "polynomial", "krogh", "piecewise_polynomial",
                          "pchip", "akima", "from_drivatives"):
            raise ValueError(
                f"method must be one of ['linear', 'time', 'index', 'values', " \
                f"'nearest', 'zero', 'slinear', 'quadratic', 'cubic', " \
                f"'barycentric', 'polynomial', 'krogh', 'piecewise_polynomial', " \
                f"'pchip', 'akima', 'spline', 'from_derivatives']. Got '{method}' instead.")
        if not isinstance(limit, int):
            return TypeError("Limit must be an integer")
        if limit < 0:
            return ValueError("Limit must be greater than 0")
        if limit_direction not in ('forward', 'backward', 'both'):
            raise ValueError(f"Invalid limit_direction: expecting one of ['forward', 'backward', 'both'], got '{limit_direction}'.")
        if limit_area not in (None, 'inside', 'outside'):
            raise ValueError(f"Invalid limit_area: expecting one of ['inside', 'outside'], got '{limit_area}'.")
        if downcast not in (None, "infer"):
            raise ValueError(f"downcast must have a dictionary or 'infer' as its argument")
        method = f"'{method}'"
        limit_direction = f"'{limit_direction}'"
        limit_area = "" if limit_area is None else f"'{limit_area}'"
        # TODO: downcast, more args
        return self._unary_op(f"interpolate{{,{method},{limit},,{limit_direction},{limit_area}}}", numeric_only=True)

    def squeeze(self, axis=None):
        return self.compute(squeeze=True, squeeze_axis=axis)

    @staticmethod
    def _validate_resample_arguments(how=None, axis=0, fill_method=None, closed=None,
                                     label=None, convention='start', kind=None, loffset=None,
                                     limit=None, base=0, on=None, level=None):
        if on is not None and not isinstance(on, str):
            raise TypeError("on must be a string")
        if closed is not None:
            raise NotImplementedError()
        if label is not None:
            raise NotImplementedError()
        if convention != "start":
            raise NotImplementedError()
        if kind != None:
            raise NotImplementedError()

    def groupby(self, by=None, axis=0, level=None, as_index=True, sort=True,
                group_keys=True, squeeze=False, observed=False, ascending=True,
                lazy=False, **kwargs):
        from .groupby import DataFrameGroupBy, SeriesGroupBy
        from .merge import MergeExpression
        if by is not None:
            by = _try_convert_iterable_to_list(by)
        klass = SeriesGroupBy if self._is_series_like else DataFrameGroupBy
        if not lazy and isinstance(self, (ArithExpression, BooleanExpression)):
            internal = self.compute()
        elif isinstance(self, (ArithExpression, BooleanExpression, MergeExpression)):
            internal = self
        else:
            internal = self._internal
        return klass(self._session, internal, self._index, by, level, as_index, sort, ascending, self._where_expr, self._name)

    def resample(self, rule, how=None, axis=0, fill_method=None, closed=None,
                 label=None, convention='start', kind=None, loffset=None,
                 limit=None, base=0, on=None, level=None, lazy=False, **kwargs):
        from .resample import DataFrameResampler, SeriesResampler
        self._validate_resample_arguments(how=how, axis=axis, fill_method=fill_method, closed=closed,
                                          label=label, convention=convention, kind=kind, loffset=loffset,
                                          limit=limit, base=base, on=on, level=level)
        klass = SeriesResampler if self._is_series_like else DataFrameResampler
        if not lazy and isinstance(self, (ArithExpression, BooleanExpression)):
            internal = self.compute()
        elif isinstance(self, (ArithExpression, BooleanExpression)):
            internal = self
        else:
            internal = self._internal
        return klass(self._session, internal, self._index, rule, on, level, self._where_expr, self._name)

    def rolling(self, window, min_periods=None, center=False, win_type=None, on=None, axis=0, closed=None):
        from .window import DataFrameRolling, SeriesRolling
        klass = SeriesRolling if self._is_series_like else DataFrameRolling
        if on is not None and not isinstance(on, str):
            raise TypeError("on must be a string")
        ref = self.compute() if isinstance(self, (ArithExpression, BooleanExpression)) else self
        return klass(ref._session, ref._internal, ref._index, window, on=on, where_expr=ref._where_expr, name=None)

    def ewm(self, com=None, span=None, halflife=None, alpha=None, min_periods=0, adjust=True, ignore_na=False, axis=0):
        from .window import DataFrameEwm, SeriesEwm
        klass = SeriesEwm if self._is_series_like else DataFrameEwm
        ref = self.compute() if isinstance(self, (ArithExpression, BooleanExpression)) else self
        return klass(ref._session, ref._internal, ref._index, com, span, halflife, alpha, min_periods, adjust, ignore_na, where_expr=ref._where_expr)

    def _sorting(self, orderby_list, asc):
        name = None if self._is_dataframe_like else self._name
        script = self._to_script(orderby_list=orderby_list, asc=asc)
        return self._get_from_script(
            self._session, script, index_map=self._index_map, name=name)

    def sort_values(self, by=None, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='first'):
        _unsupport_columns_axis(self, axis)

        if inplace:
            raise ValueError("Orca does not support inplace sort")
        if na_position != 'first':
            raise NotImplementedError()
        if self._is_dataframe_like:
            if by is None:
                raise ValueError("by must be provided for DataFrame")
            by, _ = check_key_existence(by, self._data_columns)
        else:
            by = self._data_columns    # sort Series with the only column
        return self._sorting(by, ascending)

    def sort_index(self, axis=0, level=None, ascending=True, inplace=False, kind='quicksort', na_position='first', sort_remaining=True, by=None):
        _unsupport_columns_axis(self, axis)

        if inplace:
            raise ValueError("Orca does not support inplace sort")
        if na_position != 'first':
            raise NotImplementedError()
        orderby_list, _, _, _ = _infer_level(level, self._index_map)
        if sort_remaining:
            _, remaining_list = check_key_existence(orderby_list, self._index_columns)
            orderby_list = orderby_list + remaining_list
        return self._sorting(orderby_list, ascending)

    @property
    def str(self):
        from .strings import StringMethods
        if not self._is_series_like:
            raise AttributeError("'DataFrame' object has no attribute 'str'")
        if self._ddb_dtypes[self._data_columns[0]] not in (ddb.settings.DT_STRING, ddb.settings.DT_SYMBOL):
            raise AttributeError("Can only use .str accessor with string values!")
        return StringMethods(self)

    @property
    def dt(self):    # TODO: cache reference
        from .datetimes import DatetimeMethods
        if not self._is_series_like:
            raise AttributeError("'DataFrame' object has no attribute 'dt'")
        if self._ddb_dtypes[self._data_columns[0]] not in dolphindb_temporal_types:
            raise AttributeError("Can only use .dt accessor with datetimelike values!")
        return DatetimeMethods(self)


class EwmOpsMixin(metaclass=abc.ABCMeta):

    mean = _orca_ewm_op("ewmMean")
    std = _orca_ewm_op("ewmStd")
    var = _orca_ewm_op("ewmVar")
    corr = _orca_ewm_covcorr("ewmCorr")
    cov = _orca_ewm_covcorr("ewmCov")

class IOOpsMixin(metaclass=abc.ABCMeta):

    def _orca_io_op(func):
        def wfunc(self, *args, **kwargs):
            return self._io_op(func, *args, **kwargs)
        return wfunc

    to_parquet = _orca_io_op("to_parquet")
    to_pickle = _orca_io_op("to_pickle")
    to_hdf = _orca_io_op("to_hdf")
    to_sql = _orca_io_op("to_sql")
    to_dict = _orca_io_op("to_dict")
    to_excel = _orca_io_op("to_excel")
    to_json = _orca_io_op("to_json")
    to_html = _orca_io_op("to_html")
    to_feather = _orca_io_op("to_feather")
    to_latex = _orca_io_op("to_latex")
    to_stata = _orca_io_op("to_stata")
    to_msgpack = _orca_io_op("to_msgpack")
    to_gbq = _orca_io_op("to_gbq")
    to_records = _orca_io_op("to_records")
    to_sparse = _orca_io_op("to_sparse")
    to_dense = _orca_io_op("to_dense")
    to_string = _orca_io_op("to_string")
    to_clipboard = _orca_io_op("to_clipboard")
    style = _orca_io_op("style")

    def _io_op(self, func, *args, **kwargs):
        return self.to_pandas().__getattr__(func)(*args, **kwargs)


class BaseExpression(_InternalAccessor):
    """
    Abastract tree-like representation of expressions on two Series
    that are in-memory vectors orcolumns of a single table. This does not
    compute the value of the expression immediately, but is rather evaluated
    in later stages.
    """

    def __init__(self, left, right, func, axis, infered_ddb_dtypestr=None):
        from .base import _Frame
        from .indexes import IndexOpsMixin
        # ._left and ._right are merely for saving references to manage memory
        # they are not used anywhere else
        self._left = left
        self._right = right
        self._func = func
        self._infered_ddb_dtypestr = infered_ddb_dtypestr

        core_obj, other_obj = (right, left) if isinstance(left, _ConstantSP) else (left, right)
        if isinstance(left, IndexOpsMixin) and isinstance(right, IndexOpsMixin):
            if func == "in":
                self._name = left.name
            else:
                self._name = left.name if left.name == right.name else None
        else:
            self._name = core_obj.name if isinstance(core_obj, IndexOpsMixin) else None
        self._index = core_obj._index
        self._where_expr = core_obj._where_expr
        self._session = core_obj._session
        # prepare the select list
        if isinstance(other_obj, (_ConstantSP, Timestamp)):
            self._is_index_like = core_obj._is_index_like
            self._is_series_like = core_obj._is_series_like
            self._is_dataframe_like = core_obj._is_dataframe_like
            self._internal = core_obj._internal
            if isinstance(left, _ConstantSP) and axis == 0:
                self._data_select_list = [f"{func}({left._var_name},{script})"
                                          for script in right._get_data_select_list()]
            elif isinstance(left, Timestamp) and axis == 0:
                if self._infered_ddb_dtypestr is not None:
                    typestrs = [self._infered_ddb_dtypestr]
                else:
                    typestrs = (right._ddb_dtypestr[col] for col in right._data_columns)
                self._data_select_list = [f"{func}({typestr}({left._var_name}),{script})"
                                          for typestr, script in zip(typestrs, right._get_data_select_list())]
            elif isinstance(left, _ConstantSP) and axis == 1:
                self._data_select_list = [f"{func}({left._var_name}[{i}],{script})"
                                          for i, script in enumerate(right._get_data_select_list())]
            elif isinstance(right, _ConstantSP) and axis == 0:
                self._data_select_list = [f"{func}({script},{right._var_name})"
                                          for script in left._get_data_select_list()]
            elif isinstance(right, Timestamp) and axis == 0:
                if self._infered_ddb_dtypestr is not None:
                    typestrs = [self._infered_ddb_dtypestr]
                else:
                    typestrs = (left._ddb_dtypestr[col] for col in left._data_columns)
                self._data_select_list = [f"{func}({script},{typestr}({right._var_name}))"
                                          for typestr, script in zip(typestrs, left._get_data_select_list())]
            elif isinstance(right, _ConstantSP) and axis == 1:
                self._data_select_list = [f"{func}({script},{right._var_name}[{i}])"
                                          for i, script in enumerate(left._get_data_select_list())]
        elif right is None:
            self._internal = left._internal
            self._is_index_like = left._is_index_like
            self._is_series_like = left._is_series_like
            self._is_dataframe_like = left._is_dataframe_like

            if isinstance(func, dict):
                left_data_columns = left._data_columns
                left_scripts = left._get_data_select_list()
                self._data_select_list = [f"{func[col]}({left_script})" for col, left_script in zip(left_data_columns, left_scripts)]
            elif func.startswith("isDuplicated"):
                self._is_index_like = False
                self._is_series_like = True
                self._is_dataframe_like = False
                self._data_select_list = [func]
            else:
                left_scripts = left._get_data_select_list()
                self._data_select_list = [f"{func}({left_script})" for left_script in left_scripts]
        else:
            self._is_index_like = core_obj._is_index_like and other_obj._is_index_like
            self._is_series_like = core_obj._is_series_like and other_obj._is_series_like
            self._is_dataframe_like = core_obj._is_dataframe_like or other_obj._is_dataframe_like
            self._internal = core_obj._internal
            if core_obj._is_dataframe_like:
                self._internal = core_obj._internal
            elif core_obj._is_series_like and other_obj._is_series_like:
                self._internal = core_obj._internal
            else:
                self._internal = other_obj._internal
            if left._is_series_like and right._is_series_like:
                left_script = left._get_data_select_list()[0]
                right_script = right._get_data_select_list()[0]
                self._data_select_list = [f"{func}({left_script},{right_script})"]
            elif left._is_series_like and right._is_dataframe_like:
                left_script = left._get_data_select_list()[0]
                right_scripts = right._get_data_select_list()
                self._data_select_list = [f"{func}({left_script},{right_script})"
                                          for right_script in right_scripts]
            elif left._is_dataframe_like and right._is_series_like:
                left_scripts = left._get_data_select_list()
                right_script = right._get_data_select_list()[0]
                self._data_select_list = [f"{func}({left_script},{right_script})"
                                          for left_script in left_scripts]
            elif left._is_dataframe_like and right._is_dataframe_like:
                left_scripts = left._get_data_select_list()
                right_scripts = right._get_data_select_list()
                self._data_select_list = [f"{func}({left_script},{right_script})"
                                          for left_script, right_script
                                          in zip(left_scripts, right_scripts)]
            else:
                raise TypeError("Left and right operands must be of Series or DataFrame type")

    @classmethod
    def _copy_with_columns_kept(cls, expr, keys=None, where_expr=None):
        obj = cls.__new__(cls)
        obj._left = expr._left
        obj._right = expr._right
        obj._func = expr._func
        obj._index = expr._index
        obj._name = expr._name
        obj._is_index_like = expr._is_index_like
        obj._session = expr._session
        obj._infered_ddb_dtypestr = expr._infered_ddb_dtypestr

        if keys is not None:
            if len(keys) == 1:
                obj._is_series_like = True
                obj._is_dataframe_like = False
            else:
                obj._is_series_like = False
                obj._is_dataframe_like = True
            obj._internal = expr._internal[keys]
            obj._data_select_list = [
                script for col, script in zip(expr._data_columns, expr._data_select_list)
                if col in keys
            ]
        else:
            obj._is_series_like = expr._is_series_like
            obj._is_dataframe_like = expr._is_dataframe_like
            obj._internal = expr._internal
            obj._data_select_list = expr._data_select_list
        
        if where_expr is not None:
            obj._where_expr = where_expr
        else:
            obj._where_expr = expr._where_expr
        return obj

    def __getattr__(self, key: str):
        if key in self._data_columns:
            return self[key]
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def __getitem__(self, key):
        # TODO: more checks
        if isinstance(key, BooleanExpression) and key._var_name == self._var_name:
            where_expr = _merge_where_expr(self._where_expr, key)
            return self._copy_with_columns_kept(self, where_expr=where_expr)
        else:
            keys, _ = check_key_existence(key, self._data_columns)
            return self._copy_with_columns_kept(self, keys)

    def __setitem__(self, key, value):
        _raise_must_compute_error("Setting a value to an Expression is not supported")

    def to_pandas(self):
        return self.compute().to_pandas()

    def to_numpy(self):
        return self.to_pandas().to_numpy()

    def compute(self, squeeze=False, squeeze_axis=None):
        script = self._to_script()
        if self._is_index_like:
            squeeze = True
            squeeze_axis = 1
        return self._get_from_script(
            self._session, script, index_map=self._index_map, name=self.name,
            squeeze=squeeze, squeeze_axis=squeeze_axis)

    def _get_from_script(self, *args, **kwargs):
        from .series import Series
        from .frame import DataFrame
        if self._is_index_like:
            return get_orca_obj_from_script(*args, **kwargs, as_index=self._is_index_like)
        elif self._is_dataframe_like:
            return DataFrame._get_from_script(*args, **kwargs)
        elif self._is_series_like:
            return Series._get_from_script(*args, **kwargs)

    def _get_data_select_list(self):
        return self._data_select_list

    def get_select_list(self):
        if self._is_dataframe_like:
            select_list = (f"{script} as {column_name}"
                           for script, column_name
                           in zip(self._get_data_select_list(), self._data_columns))
        elif self._is_series_like:
            select_column = self._get_data_select_list()[0]
            select_list = [f"{select_column} as ORCA_EXPRESSION_COLUMN"]
        index_select_list = self._index_columns
        return itertools.chain(index_select_list, select_list)

    def _to_script(self, orderby_list=None, asc=True):
        select_list = self.get_select_list()
        script = sql_select(select_list, self._var_name, self._where_expr,
                            orderby_list=orderby_list, asc=asc)
        return script

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

    def _unary_op(self, *args, **kwargs):
        return StatOpsMixin._unary_op(self, *args, **kwargs)

    def _unary_agg_op(self, *args, **kwargs):
        return StatOpsMixin._unary_agg_op(self, *args, **kwargs)

    @property
    def name(self):
        return self._name

    def rename(self, *args, **kwargs):
        _raise_must_compute_error("Unable to rename an Expression")

    def transpose(self, *args, **kwargs):
        _raise_must_compute_error("Unable to transpose an Expression")

    def dot(self, *args, **kwargs):
        _raise_must_compute_error("Unable to call 'dot' on an Expression")

    def reindex(self, *args, **kwargs):
        ref = self.compute()
        return ref.reindex(*args, **kwargs)

    def plot(self, *args, **kwargs):
        ref = self.compute()
        return ref.plot(*args, **kwargs)


class ArithExpression(BaseExpression, ArithOpsMixin, LogicalOpsMixin, StatOpsMixin, IOOpsMixin):
    """
    Subclass of BaseExpression dealing with arithmetic expressions.
    """

    pass


class BooleanExpression(BaseExpression, ArithOpsMixin, LogicalOpsMixin, StatOpsMixin, IOOpsMixin):
    """
    Subclass of BaseExpression dealing with logical expressions.
    """

    def _to_where_list(self):
        return self._data_select_list
